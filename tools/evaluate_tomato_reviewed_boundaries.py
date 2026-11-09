"""CPU-only frozen-probability audit. Proposals never overwrite source labels."""
from copy import deepcopy
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import numpy as np
from . import tomato_dataset as ds, tomato_metrics as metrics, tomato_boundary_metrics as boundary

AUDIT=Path('runs/tomato/transition_audit_20261109')
CAPTURE=Path('docs/validation/tomato_boundary_review_20261109/completed_review_capture.json')
POLICY=Path('configs/tomato_curated91_20261109/quality_exclusions.json')
MODEL_HORIZONS={'joint':1.,'independent_h10':1.,'independent_h15':1.5}


def match_events(events, truth, head, horizon):
    used=set(); rows=[]
    for b in truth:
        lo=b['time'] if head=='current' else b['time']-horizon
        hi=min(b['end'],b['time']+2.) if head=='current' else b['time']
        hit=next((i for i,p in enumerate(events) if i not in used and p['previous']==b['previous'] and p['label']==b['label'] and lo<=p['time']<hi),None)
        if hit is not None:used.add(hit)
        t=events[hit]['time'] if hit is not None else None
        rows.append(dict(**b,detected=hit is not None,event_time=t,offset_s=t-b['time'] if t is not None else None))
    return rows,[p for i,p in enumerate(events) if i not in used]


def revise_episode(episode, proposals):
    e=deepcopy(episode); seg=e['annotation']['segments'];seen={}
    for r in proposals:
        old=r['original_boundary_frame'];new=r['boundary_frame_reconstructed_from_visible_time']
        if old in seen and seen[old]!=new:raise ValueError('Conflicting duplicate proposals')
        seen[old]=new
    # Match against the frozen original segment starts, never previously edited starts.
    starts={s['start_frame']:i for i,s in enumerate(episode['annotation']['segments'])}
    for old,new in seen.items():
        i=starts.get(old)
        if i is None or i==0 or new is None:raise ValueError('Unknown/noninternal boundary')
        seg[i]['start_frame']=new;seg[i-1]['end_frame']=new
        seg[i]['start_seconds']=new/e['fps'];seg[i-1]['end_seconds']=new/e['fps']
    for a,b in zip(seg,seg[1:]):
        if a['start_frame']>=a['end_frame'] or a['end_frame']!=b['start_frame']:raise ValueError('Noncausal or overlapping label intervals')
    if seg[-1]['start_frame']>=seg[-1]['end_frame']:raise ValueError('Empty final phase')
    return e


def summarize(per,head):
    vals=[r[head] for r in per.values()];rows=[b for r in vals for b in r['boundaries']];hits=[b for b in rows if b['detected']]
    fp=sum(len(r['false_events']) for r in vals);minutes=sum(r['duration_s'] for r in vals)/60
    types=defaultdict(lambda:dict(total=0,detected=0))
    for b in rows:
        t=types[ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']]];t['total']+=1;t['detected']+=int(b['detected'])
    offsets=[b['offset_s'] for b in hits]
    return dict(total=len(rows),detected=len(hits),recall=len(hits)/len(rows),false_events=fp,false_per_minute=fp/minutes,evaluated_minutes=minutes,offset_p50_s=float(np.median(offsets)) if offsets else None,offset_p95_s=float(np.percentile(offsets,95)) if offsets else None,jitter_reversals=sum(r['reversals_within_2s'] for r in vals),by_transition=dict(types))


def run(out):
    if out.exists():raise FileExistsError('Use a new output directory; frozen outputs are not overwritten')
    protocol=ds.read_json(AUDIT/'protocol.json');assert ds.digest(protocol['manifest'])==protocol['manifest_sha256']
    m=ds.read_json(protocol['manifest']);comparison=ds.read_json(AUDIT/'comparison.json');capture=ds.read_json(CAPTURE);policy=ds.read_json(POLICY)
    excluded={r['source_recording_id'] for r in policy['excluded_episodes']}
    episodes=[e for e in m['episodes'] if e['split']=='val' and e['source_recording_id'] not in excluded]
    assert len(episodes)==9
    assert all(ds.digest(e['annotation']['path'])==e['annotation']['sha256'] for e in episodes)
    cases=ds.read_json('runs/tomato/boundary_review_20261109/review_package/cases.json')['cases'];bycase={c['case_id']:c for c in cases}
    grouped=defaultdict(list);false_review=[]
    for r in capture['rows']:
        c=bycase[r['case_id']];assert c['source_recording_id'] not in excluded
        assert c['annotation_sha256']==r['source_annotation_sha256']
        if r['kind']=='truth':grouped[c['episode_id']].append(r)
        else:
            ep=next(e for e in episodes if e['episode_id']==c['episode_id']);p=r['boundary_frame_reconstructed_from_visible_time'];bs=boundary.boundaries(dict(episode=ep))
            same=[b for b in bs if ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']]==r['transition']]
            nearest=min(same,key=lambda b:abs(b['time']-p/ep['fps'])) if same and p is not None else None
            false_review.append(dict(**r,episode_id=c['episode_id'],nearest_original_same_type_time_s=nearest['time'] if nearest else None,nearest_original_same_type_offset_s=p/ep['fps']-nearest['time'] if nearest else None,action='retain_no_transition' if p is None else 'needs_adjacent_phase_reconciliation_do_not_create_event'))
    revised={e['episode_id']:revise_episode(e,grouped[e['episode_id']]) for e in episodes}
    frozen_inputs=[str(CAPTURE),str(POLICY),str(AUDIT/'comparison.json'),protocol['manifest'],str(AUDIT/'protocol.json')]
    logit_dir=Path('runs/tomato/optimization_20261108/evaluations');result={};all_event_differences=[]
    for name,horizon in MODEL_HORIZONS.items():
        path=logit_dir/(name+'_s42_val_logits.npz');frozen_inputs.append(str(path));z=np.load(path)
        slices={};offset=0
        for e in [e for e in m['episodes'] if e['split']=='val']:
            n=len(e['anchors']);slices[e['episode_id']]=slice(offset,offset+n);offset+=n
        assert offset==len(z['current_logits'])
        native=comparison[name+'_2hz_k2']['per_episode']
        for scenario in ['excluded_package_original_boundaries','reviewed_truth14_provisional']:
            per={};buffers={h:[[],[]] for h in ['current','future']}
            for old in episodes:
                eid=old['episode_id'];e=revised[eid] if scenario=='reviewed_truth14_provisional' else old;truth=boundary.boundaries(dict(episode=e));per[eid]={};anchors=np.asarray(e['anchors']);sl=slices[eid]
                for head in ['current','future']:
                    source=native[eid][head];matched,false=match_events(source['events'],truth,head,horizon);common,_=match_events(source['events'],truth,head,1.)
                    valid=np.ones(len(anchors),bool) if head=='current' else anchors+round(horizon*e['fps'])<e['length']
                    y=np.array([ds.label_at_frame(e['annotation']['segments'],int(a)+(round(horizon*e['fps']) if head=='future' else 0)) for a in anchors[valid]])
                    zz=z[head+'_logits'][sl][valid];buffers[head][0].append(zz);buffers[head][1].append(y)
                    per[eid][head]=dict(boundaries=matched,common_1s_boundaries=common,false_events=false,duration_s=source['duration_s'],reversals_within_2s=source['reversals_within_2s'],classification=metrics.scores(zz,y))
            result[name+'/'+scenario]=dict(horizon_s=horizon,summary={h:summarize(per,h) for h in buffers},classification={h:metrics.scores(np.concatenate(v[0]),np.concatenate(v[1])) for h,v in buffers.items()},per_episode=per,common_1s_future_detected=sum(b['detected'] for r in per.values() for b in r['future']['common_1s_boundaries']))
        before=result[name+'/excluded_package_original_boundaries'];after=result[name+'/reviewed_truth14_provisional']
        for e in episodes:
            for head in ['current','future']:
                for b,a in zip(before['per_episode'][e['episode_id']][head]['boundaries'],after['per_episode'][e['episode_id']][head]['boundaries']):
                    if b['detected']!=a['detected']:all_event_differences.append(dict(model=name,head=head,episode_id=e['episode_id'],before=b,after=a))
    # Historical all10 baseline is separate; assert cached event matching exactly reproduces it.
    parity={}
    for name,h in MODEL_HORIZONS.items():
        per=comparison[name+'_2hz_k2']['per_episode'];s=summarize(per,'future');expected=comparison[name+'_2hz_k2']['summary']['future']
        assert s['detected']==expected['detected'] and s['false_events']==expected['false_events']
        for e in [e for e in m['episodes'] if e['split']=='val']:
            for head in ['current','future']:
                x=per[e['episode_id']][head];rows,fp=match_events(x['events'],boundary.boundaries(dict(episode=e)),head,h)
                assert rows==x['boundaries'] and fp==x['false_events']
        parity[name]=dict(detected=s['detected'],total=s['total'],false_events=s['false_events'],passed=True)
    # Diagnose the unchanged candidate using revised truth and cached probabilities.
    candidate=result['independent_h15/reviewed_truth14_provisional']
    candidate_logits=np.load(logit_dir/'independent_h15_s42_val_logits.npz')
    missed=[];provider_rows=[];provider=ds.read_json(AUDIT/'openpi_replay.json')['independent_h15_2hz_k2']
    frozen_inputs.append(str(AUDIT/'openpi_replay.json'))
    for old in episodes:
        eid=old['episode_id'];e=revised[eid];ts=np.asarray(e['anchors'])/e['fps']
        prob=metrics.probabilities(candidate_logits['future_logits'][slices[eid]])
        valid=np.asarray(e['anchors'])+round(1.5*e['fps'])<e['length']
        raw=comparison['independent_h15_2hz_k2']['per_episode'][eid]['future']['events']
        for ordinal,b in enumerate(candidate['per_episode'][eid]['future']['boundaries'],1):
            decision=next((x['time'] for x in provider[eid]['changes'] if x['phase']==ordinal),None)
            outcome='never_confirmed' if decision is None else ('premature' if decision<b['time'] else ('on_time' if decision<min(b['end'],b['time']+2) else 'late'))
            provider_rows.append(dict(episode_id=eid,ordinal=ordinal,boundary_s=b['time'],confirmed_s=decision,offset_s=decision-b['time'] if decision is not None else None,outcome=outcome))
            if b['detected']:continue
            mask=valid&(ts>=b['time']-1.5)&(ts<b['time'])
            adequate=mask&(prob.argmax(1)==b['label'])&(prob[:,b['label']]>=.5)
            same=[x for x in raw if x['previous']==b['previous'] and x['label']==b['label']]
            nearest=min(same,key=lambda x:abs(x['time']-b['time'])) if same else None
            prior=[x for x in raw if x['time']<b['time']-1.5]
            pre=bool(prior and prior[-1]['label']==b['label'])
            diagnosis='no_target_probability_in_allowed_window' if not adequate.any() else ('premature_event_persists_into_window' if pre else 'insufficient_confirmation_or_class_path')
            missed.append(dict(episode_id=eid,ordinal=ordinal,transition=ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']],boundary_s=b['time'],diagnosis=diagnosis,adequate_window_times_s=ts[adequate].tolist(),nearest_same_transition_offset_s=nearest['time']-b['time'] if nearest else None,window_times_s=ts[mask].tolist(),window_target_probabilities=prob[mask,b['label']].tolist(),window_argmax_labels=prob[mask].argmax(1).tolist()))
    out.mkdir(parents=True)
    ds.write_json(out/'remaining_misses.json',missed)
    ds.write_json(out/'cached_provider_confirmation.json',dict(scope='Existing historical provider cached confirmation times; no interface changes or replay code execution',outcomes=dict(Counter(r['outcome'] for r in provider_rows)),events=provider_rows))
    ds.write_json(out/'results.json',result);ds.write_json(out/'event_outcome_changes.json',all_event_differences)
    reconciled=[]
    for r in false_review:
        e=revised[r['episode_id']];frame=r['boundary_frame_reconstructed_from_visible_time'];c=bycase[r['case_id']]
        choices=[]
        if frame is not None:
            for ordinal,b in enumerate(boundary.boundaries(dict(episode=e)),1):
                if ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']]!=r['transition']:continue
                if c['clip_start_s']<=b['time']<c['clip_end_s']:
                    choices.append(dict(ordinal=ordinal,provisional_boundary_frame=round(b['time']*e['fps']),reviewed_point_offset_s=frame/e['fps']-b['time']))
        reconciled.append(dict(original_ui_number=r['original_ui_number'],case_id=r['case_id'],episode_id=r['episode_id'],same_type_existing_events_inside_clip=choices,has_frame_conflict=bool(len(choices)==1 and choices[0]['provisional_boundary_frame']!=frame),resolution='explicit_no_transition' if frame is None else ('associate_with_existing_event' if len(choices)==1 else 'needs_full_context_no_unique_existing_event')))
    ds.write_json(out/'unmatched_review_reconciliation.json',reconciled)
    ds.write_json(out/'proposals.json',dict(status='provisional_user_review_not_source_labels',truth_proposals=dict(grouped),unmatched_warning_reviews=false_review))
    ds.write_json(out/'protocol.json',dict(seed=42,threshold=.5,confirmations=2,hz=2,split='val',val_episodes=9,test_read=False,training=False,encoder_calls=0,source_labels_changed=False,matching=protocol['future_matching'],current_matching=protocol['current_matching'],native_horizons=MODEL_HORIZONS,common_future_window_s=1.,inputs={p:ds.digest(p) for p in frozen_inputs},code_sha=ds.digest(__file__),historical_parity=parity,limitations=capture['limitations']))
    lines=['# 清理后验证集与人工边界提案离线对照','','本轮只重用seed42、2Hz、阈值0.5、连续2次确认的已保存预测。9条验证录制；排除整包失败演示；没有训练、读取测试预测或修改源标签。原10条历史事件结果逐事件重现通过。','','人工复核只调整14个原真实事件的边界，未复核的真实事件保持原样。这是部分修订的敏感性分析，不是完整重标后的权威评价。14个误报片段的复核单独保留；不自动新增/删除完整Episode事件。原第25段无切换，其余13项需核对相邻阶段，避免重复计数。','','|模型|边界版本|Current F1|Future F1|Current命中|Future命中|Future误报/分钟|Future提前量中位秒|统一1秒窗命中|','|---|---|---:|---:|---|---|---:|---:|---:|']
    for k,r in result.items():
        c=r['summary']['current'];f=r['summary']['future'];lines.append(f"|{k.split('/')[0]}|{'清理原边界' if k.endswith('original_boundaries') else '14项提案'}|{r['classification']['current']['macro_f1']:.5f}|{r['classification']['future']['macro_f1']:.5f}|{c['detected']}/{c['total']}|{f['detected']}/{f['total']}|{f['false_per_minute']:.3f}|{-f['offset_p50_s']:.3f}|{r['common_1s_future_detected']}|")
    lines+=['','Future分数与原生窗口分别对应1秒/1.5秒任务，不可直接当作相同预测任务排名。统一1秒事件窗另列。分类/事件口径保持分离，Current事件不是七阶段决策器确认；本轮未重新运行OpenPI。','','所有模型输入和输出完全未变，任何差值均来自样本排除/边界版本变化，不是模型改善。旧模型训练时仍包含2条已排除录制及旧标注，不能声称这是清洁训练结果。人工复核备注均为“1”，27条为界面默认待对照状态；用户确认已完成，但不能据此推断动作理由或双人共识。','','完整逐录制、各类别Precision/Recall、误报、延迟/提前量、抖动、转移类型统计见results.json。检测状态变化见event_outcome_changes.json。无参数搜索，无新阈值选择。下一步优先依据此对照定位剩余迟报/提前持续，再制定74条清洁训练录制上的三种子对照；新独立录制仍需封存盲测。','']
    lines += ['', '误报片段关联核对：'+str(dict(Counter(r['resolution'] for r in reconciled)))+'。候选关联使用相同转移类型、原片段时间范围与相邻阶段；仅作为候选，不按距离自动合并冲突时间。详见unmatched_review_reconciliation.json。', '']
    lines+=['## 剩余错误与决策确认','',f"1.5秒候选在提案边界下剩余漏检{len(missed)}次。概率窗口归因：{dict(Counter(r['diagnosis'] for r in missed))}。这是数值时序诊断，不代表已从视频确认遮挡或动作语义原因。",'',f"adjust→pickup仍仅2/8命中，其余pickup→harvest为15/17、harvest→return为16/17、return→adjust为6/8。该困难转换没有被边界复核解决。",'',f"已有历史Provider确认时间对照修订边界：{dict(Counter(r['outcome'] for r in provider_rows))}。逐项见cached_provider_confirmation.json；不能把Current分类事件命中替代Agent实际确认。",'', '误报片段复核与原同类边界的距离单独记录于proposals.json；不把距离近就自动判定重复。原第26段与最近原adjust→pickup边界相差6.1秒，需核对是否为重复抓握/额外尝试，不能只移动原边界吞掉它。','','下一项针对性实验：在74条清洁训练录制上重新构建目标标签、拟合训练归一化，以既定独立1.5秒双头和三个种子重训对照；优先改善adjust→pickup的提前持续/缺少目标概率，再考虑固定单因素决策实验。此次不直接启动训练，因为验证边界仍为部分修订，新的训练manifest/目标缓存尚未构建。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
