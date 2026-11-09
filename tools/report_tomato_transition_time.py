"""Frozen-run analysis: censor baselines, per-type evidence and paired episode CIs."""
from pathlib import Path
from collections import Counter,defaultdict
import copy
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_boundary_metrics as bm
from .tomato_transition_time import survival_targets,hazard_nll,event_distribution,CLASSES
from .train_tomato_transition_time import replay,time_quality
from .evaluate_tomato_reviewed_boundaries import revise_episode,match_events

ROOT=Path('runs/tomato/transition_time_20261109')


def main():
    ident=ds.read_json(ROOT/'data_identity.json');old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');manifest={e['episode_id']:e for e in old['episodes']}
    approved=ds.read_json('configs/tomato_boundary_review_20261109/annotation_revision_provenance.json');proposal=ds.read_json('docs/validation/tomato_reviewed_boundaries_20261109/proposals.json')['truth_proposals']
    rows={'train':[],'val':[]}
    for rec in ident['episodes']:
        e=copy.deepcopy(manifest[rec['episode_id']]);data=np.load(rec['feature_path']);assert ds.digest(rec['feature_path'])==rec['feature_sha256']
        if e['episode_id']==approved['episode_id']:e['annotation']['segments']=ds.read_json(e['annotation']['path'])['segments']
        if e['split']=='val':e=revise_episode(e,proposal.get(e['episode_id'],[]))
        rows[e['split']].append(dict(episode=e,x=data['features'],anchors=data['anchors'],times=data['times'],targets=survival_targets(e['annotation']['segments'],data['anchors'],e['fps'],e['length'])))
    # Constant conditional hazards fit only training labels, including censor masks.
    trainy=np.concatenate([r['targets']['labels'] for r in rows['train']]);trainm=np.concatenate([r['targets']['mask'] for r in rows['train']]);rates=(trainy.sum(0)/trainm.sum(0)).clip(1e-6,1-1e-6);prior=np.log(rates/(1-rates));priorpred={r['episode']['episode_id']:dict(logits=np.tile(prior,(len(r['times']),1)),masses=event_distribution(torch.from_numpy(np.tile(prior,(len(r['times']),1)))).numpy()) for r in rows['val']}
    references=time_quality(rows['val'],priorpred);out=dict(constant_train_hazard_reference=references,train_bin_counts=dict(at_risk=trainm.sum(0).tolist(),events=trainy.sum(0).tolist(),conditional_hazards=rates.tolist()),per_seed={},state_evidence=[])
    names=old['contract']['joint_names'];arm=[i for i,n in enumerate(names) if n.startswith('right_fr3')];hand=[i for i,n in enumerate(names) if n.startswith('right_') and not n.startswith('right_fr3')];assert len(arm)==7 and len(hand)==22
    for r in rows['val']:
        e=r['episode'];q=r['x'][:,-928:].reshape(-1,16,58)[:,-1];dt=np.diff(r['times']);velocity=np.diff(q,axis=0)/dt[:,None];vtime=r['times'][1:]
        for ordinal,b in enumerate(bm.boundaries(dict(episode=e)),1):
            pre=(vtime>=b['time']-1)&(vtime<b['time']);post=(vtime>=b['time'])&(vtime<b['time']+1)
            def rms(ix,mask):return float(np.sqrt(np.mean(velocity[mask][:,ix]**2))) if mask.any() else None
            ev=dict(episode_id=e['episode_id'],ordinal=ordinal,transition=ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']],boundary_s=b['time'],right_arm_rms_before=rms(arm,pre),right_arm_rms_after=rms(arm,post),right_hand_rms_before=rms(hand,pre),right_hand_rms_after=rms(hand,post),camera_keys=e['camera_keys'])
            vis=r['x'][:,:3072].reshape(-1,3,1024);vd=np.linalg.norm(np.diff(vis,axis=0),axis=2)
            ev['visual_feature_change_before']=vd[pre].mean(0).tolist() if pre.any() else None;ev['visual_feature_change_after']=vd[post].mean(0).tolist() if post.any() else None
            out['state_evidence'].append(ev)
    allruns=[];cf_version={}
    for seed in [42,43,44]:
        result=ds.read_json(ROOT/f'seed{seed}/result.json');selected=str(result['selected_threshold']);cf={};hp={}
        for r in rows['val']:
            eid=r['episode']['episode_id'];f=np.load(ROOT/f'seed{seed}'/(eid+'.npz'));cf[eid]={h:f[h+'_probabilities'] for h in ['current','future']};hp[eid]=dict(logits=f['hazard_logits'],masses=f['transition_probabilities'])
        # Inference-only compatibility changes after training must reproduce real replay.
        run=replay(rows['val'],cf,hp,float(selected));assert run['summaries']==result['external_validation_thresholds_are_ablation_not_selection'][selected]['summaries']
        detail=dict(time_quality=result['time_quality'],classification=result['classification'],threshold=result['selected_threshold'],latency=result['head_latency_ms'],training_s=sum(result[k]['training_s'] for k in ['internal_epoch_selection','internal_refit','final_training']),epoch=result['internal_epoch_selection']['best_epoch'],parameters=result['parameters'],gpu_training_peak_bytes=max(result[k]['peak_gpu_allocated_bytes'] for k in ['internal_epoch_selection','internal_refit','final_training']),methods={},special_adjust=[])
        for method,per in run['per_episode'].items():
            categories=Counter();jitter=0
            for eid,v in per.items():
                truth=v['boundaries'];ev=v['events'];jitter+=sum(b['label']==a['previous'] and b['time']-a['time']<=2 for a,b in zip(ev,ev[1:]))
                for p in v['false_events']:
                    same=[b for b in truth if b['previous']==p['previous'] and b['label']==p['label']]
                    if not same:categories['wrong_transition_type']+=1;continue
                    b=min(same,key=lambda x:abs(x['time']-p['time']));delta=p['time']-b['time'];categories['too_early' if delta<-1.5 else ('late' if delta>=0 else 'duplicate_or_unmatched')]+=1
            detail['methods'][method]=dict(**run['summaries'][method],common_1s_hits=sum(v['common_1s_hits'] for v in per.values()),unmatched_categories=dict(categories),jitter_reversals=jitter)
        for d in run['decision_replay']:
            eid=d['episode_id'];pos=defaultdict(list)
            for t in d['trace']:
                if t.get('suggest_switch'):pos[t['phase_index']].append(t['time'])
            # Identity of repeated four-class transitions must be matched by ordered index
            # for decisions, instead of borrowing credit from another repeated cycle.
            for ordinal,b in enumerate(bm.boundaries(dict(episode=next(r['episode'] for r in rows['val'] if r['episode']['episode_id']==eid))),1):
                t=pos[ordinal][0] if pos[ordinal] else None
                outcome='never' if t is None else ('premature' if t<b['time'] else ('on_time' if t<min(b['end'],b['time']+2) else 'late'))
                detail.setdefault('decision_events',[]).append(dict(episode_id=eid,ordinal=ordinal,time=t,boundary_s=b['time'],outcome=outcome))
        detail['decision_outcomes']=dict(Counter(r['outcome'] for r in detail['decision_events']))
        for r in rows['val']:
            e=r['episode'];p=hp[e['episode_id']]['masses'];prob=cf[e['episode_id']]
            for ordinal,b in enumerate(bm.boundaries(dict(episode=e)),1):
                if b['previous']!=3 or b['label']!=0:continue
                mask=(r['times']>=b['time']-1.5)&(r['times']<b['time']);detail['special_adjust'].append(dict(episode_id=e['episode_id'],ordinal=ordinal,boundary_s=b['time'],window_times=r['times'][mask].tolist(),hazard_within15=(1-p[mask,-1]).tolist(),current_pickup=prob['current'][mask,0].tolist(),future_pickup=prob['future'][mask,0].tolist()))
        # Focal early-Prompt case: original frozen73.3, user proposal74.233..., old confirmation72.
        eid='3e645bf4e137d0f836c9d7ff';d=next(x for x in run['decision_replay'] if x['episode_id']==eid);new=next((x['time'] for x in d['trace'] if x.get('suggest_switch') and x['phase_index']==4),None)
        detail['early_prompt_case']=dict(episode_id=eid,ordinal=4,new_confirmation_s=new,original_boundary_s=73.3,provisional_boundary_s=next(b['time'] for i,b in enumerate(bm.boundaries(dict(episode=next(r['episode'] for r in rows['val'] if r['episode']['episode_id']==eid))),1) if i==4),old_confirmation_s=72.,note='Absent confirmation means missed/stalled, not successful correction of timing.')
        out['per_seed'][str(seed)]=detail;allruns.append(run)
    # Paired recording bootstrap, retaining correlation among all seeds/models in same recording.
    rng=np.random.default_rng(42);ids=[r['episode']['episode_id'] for r in rows['val']];cis={}
    for method in ['B1_gate','B2']:
        draws=[]
        for _ in range(3000):
            ix=rng.integers(0,len(ids),len(ids));diff=[]
            for run in allruns:
                den=sum(len(run['per_episode']['B0'][ids[i]]['boundaries']) for i in ix)
                delta=sum(sum(b['detected'] for b in run['per_episode'][method][ids[i]]['boundaries'])-sum(b['detected'] for b in run['per_episode']['B0'][ids[i]]['boundaries']) for i in ix)/den;diff.append(delta)
            draws.append(np.mean(diff))
        cis[method]=dict(recall_delta_ci95=np.percentile(draws,[2.5,97.5]).tolist(),scope='paired9 recording bootstrap, seed-mean difference; exploratory validation, not generalization CI')
    out['paired_recording_uncertainty']=cis
    out['aggregate']={}
    for method in ['B0','B1_gate','B2']:
        records=[v['methods'][method] for v in out['per_seed'].values()]
        out['aggregate'][method]={k:dict(mean=float(np.mean([r[k] for r in records])),sample_sd=float(np.std([r[k] for r in records],ddof=1))) for k in ['recall','precision','f1','false_per_minute']}
    out['B3_run']=False;out['decision']='retain_B0; B1/B2 do not improve recall at the false-positive cap; no external threshold selection'
    ds.write_json(ROOT/'analysis.json',out)
    # Retain all checkpoints and exact original training source snapshots; publish metrics only.
    doc=Path('docs/validation/tomato_transition_time_20261109');doc.mkdir(exist_ok=False)
    for name in ['analysis.json','protocol.json','data_identity.json','internal_split.json','integrity.json']:
        ds.write_json(doc/name,ds.read_json(ROOT/name))
    lines=['# 第四阶段：转换时间头与预测决策实测（2026-11-09）','','## 结论','','已完成代码、删失监督测试、三种子GPU训练与真实缓存预测回放。结果不支持替换原独立1.5秒双头；没有进入B3。时间头降低误报，但漏检显著增加，adjust→pickup未改善。所有新权重作为研究失败产物保留。未推送远程。','','## 数据与公平性','','HEAD起点bc449a4。原94条和76/10/8全部保留；新时间头用74条训练录制，去除失败整包2条训练和1条验证。唯一已批准训练修订e80436...的harvest→return使用新标注SHA；视觉缓存不包含标签身份，输入源、关节顺序、16帧历史和时间戳逐项核验后复用，监督重新生成。原Current/Future仍是历史模型及其原76条归一化，不宣称已完成清洁分类头重训。','','验证为固定9条、50事件，与39/50参考完全相同的14项提案边界；不是正式批准原标注或独立测试。误报片段标记未擅自新增事件。没有使用测试数组、未来观测或真实七阶段位置作为输入。','','按原始录制在74条训练中固定64/10内部划分，以内部删失NLL选epoch，再以该epoch在64条重拟合用于阈值校准，最后全74条重拟合归一化和权重。CF历史头已看过内部留出录制，这里不是完整模型的独立CV；外部9条未用于优化或阈值选择。阈值候选仅0.3/0.5/0.7，外部阈值表只作消融，不重新选赢家。','','## 模型和概率语义','','4000D输入→128 ReLU Dropout0.3→4个条件hazard。条件风险经生存乘积转为互斥5区间概率，总和1；最后区间为生存超过1.5秒。已观测事件按(0,.25],(.25,.5],(.5,1],(1,1.5]监督；边界精确等于当前时刻已属于新阶段，寻找下一次严格未来转换。尾部仅对已完整观察的区间施加生存损失，部分区间不伪造标签。相邻同类标注段不视为新类别转换；非相邻重复pickup/harvest/return仍独立事件。未记录不确定性范围的提案不假装具备精确置信区间。','','## 实验定义','','B0：固定原双头四类Future事件，阈值0.5/K2。B1_gate：同一事件流仅增加时间头概率门控；Current/Future输出完全不变。B2：七有序位置在线维护，Future/时间头预警与Current确认分离，Current≥0.8连续2次且进入已预测时间区间才建议切换。七阶段初始位置按任务协议pickup_1，未注入真值状态；相同pickup类别不能跳过中间阶段。B2的严格状态约束可能积累早期漏检，未隐藏这一代价。','','|Seed|内部选定阈值|方法|严格命中|Precision|F1|误报/分钟|共同1秒窗命中|adjust→pickup|','|---|---:|---|---|---:|---:|---:|---:|---|']
    for seed,v in out['per_seed'].items():
        for name,q in v['methods'].items():
            a=q['by_transition']['adjust->pickup'];lines.append(f"|{seed}|{v['threshold']}|{name}|{q['detected']}/50|{q['precision']:.3f}|{q['f1']:.3f}|{q['false_per_minute']:.3f}|{q['common_1s_hits']}|{a['detected']}/{a['total']}|")
    lines+=['','## 时间头质量与资源','','|Seed|epoch|删失NLL|1.5秒Brier|近事件时间MAE秒|时间bin准确率|三阶段训练秒|时间头P95毫秒|训练峰值MiB|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed,v in out['per_seed'].items():
        q=v['time_quality'];lines.append(f"|{seed}|{v['epoch']}|{q['censored_nll']:.4f}|{q['horizon_brier']:.4f}|{q['near_event_conditional_mae_s']:.3f}|{q['near_event_bin_accuracy']:.3f}|{v['training_s']:.1f}|{v['latency']['hazard_p95']:.3f}|{v['gpu_training_peak_bytes']/2**20:.1f}|")
    lines += ['',f"训练集常数风险对照：NLL {references['censored_nll']:.4f}、Brier {references['horizon_brier']:.4f}、近事件时间MAE {references['near_event_conditional_mae_s']:.3f}秒。时间误差只统计真实下一事件≤1.5秒的150个验证窗口，使用区间条件期望；没有为尾部赋伪造有限时间。该条件MAE不能当作全录制或严格事件命中指标。",'', '时间头参数512644；编码器与原双头均未训练。新增头基于同一次视频编码特征。此次实测仅新增头/原分类头GPU延迟，不把历史约43ms编码测量冒充本轮新编码实测。未重跑编码器、实时摄像头、解码或机器人链路。','','## 误差归因与专项证据','', 'analysis.json保留逐事件前后1秒右臂7D、右手22D RMS运动及三相机特征变化，均按关节名称提取；这是观测证据，不等于证明可分。只有相机键，没有充分来源证明哪路必然是右腕相机，不擅自命名。各seed的8个adjust→pickup窗口Current/Future概率及时间头风险均已保存。新门控在此类型上进一步损失命中，表明时间头在困难转换上召回不足；不能把更低误报解释为更准确的时间定位。','','每方法过早/过晚/类别错误、抖动、逐录制结果、确认漏检和原提前Prompt案例见analysis.json及每seed/result.json。未发生Prompt切换可能是卡住或漏确认，不能当作成功修复提前1.3秒。实际Prompt/Chunk不写回OpenPI；Mock及版本化增量字段测试确保默认旧接口不变。','','## 不确定性与选型','',str(cis),'', '验证仅9条、50事件，部分标注仍为提案；三种子的方差与录制bootstrap不能替代新独立录制泛化。B2以更保守决策换来大量漏检；时间头还未学到足以改进adjust→pickup的稳定先兆。继续使用B0，保留旧Joint seed42接口默认。B3的正向门槛未满足，不训练专用头。后续优先完整核对重复事件/边界规范，再研究只在训练分组内部验证的事件均衡抽样或序列风险校准；不继续依据这8个外部事件调阈值。','','本地权重：`'+str(ROOT.resolve())+'/seed42/hazard.pt`（seed43/44同理）。完整配置、训练源码快照、数据身份、标签监督SHA、预测缓存和optimizer范围都已保留。权重/特征不进Git。后续推理可显式传transition_time_checkpoint启用新增字段；默认不启用。','']
    Path('docs/TOMATO_TRANSITION_TIME_RESULTS_20261109.md').write_text('\n'.join(lines))
    print('REPORT COMPLETE',out['aggregate'],references)

if __name__=='__main__':main()
