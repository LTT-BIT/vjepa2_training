"""Full B0 decision evaluation. Labels enter only after causal replay."""
from collections import Counter
import numpy as np
from . import tomato_dataset as ds,tomato_metrics as metrics,tomato_boundary_metrics as bm
from .evaluate_tomato_complete100_history import score_arm
from .tomato_task_memory import TaskMemory,Observation


def reconstruct_shadow(events):
    latest={};version=0
    for event in events:
        if event['version']<=version:raise ValueError('Journal duplicate or out-of-order version')
        version=event['version'];latest[event['event_id']]=event
    transitions=sorted([e for e in latest.values() if e['event_type']=='transition' and e['status']=='confirmed'],key=lambda e:e['confirmation_time'])
    prefix=[0]
    for e in transitions:
        if e['from_phase']!=prefix[-1] or e['to_phase']!=prefix[-1]+1:raise ValueError('Noncontiguous shadow history')
        if e.get('physical_success')!='unobserved':raise ValueError('Stage log cannot certify physical outcomes')
        prefix.append(e['to_phase'])
    return prefix


def cascade_diagnostics(runs,phases,truth_positions,boundaries):
    rows=[]
    for run in runs:
        i=run['first_index'];root=min(int(phases[i]),int(truth_positions[i]))+1
        crossed=[ordinal for ordinal,b in enumerate(boundaries,1) if run['start_s']<b['time']<run['end_s']]
        rows.append(dict(start_s=run['start_s'],end_s=run['end_s'],inferred_root_ordinal=root,
                         boundaries_crossed=crossed,potential_downstream_ordinals=[v for v in crossed if v>root],
                         scope='Scorer-only temporal attribution; crossing the root boundary itself is not downstream cascade; causality not established'))
    return rows


def evaluate_episode(e,z,model_version):
    ts=np.asarray(z['times']);a=np.asarray(z['anchors']);c=np.asarray(z['current_probabilities']);f=np.asarray(z['future_probabilities'])
    if c.shape!=(len(ts),4) or f.shape!=c.shape or not np.isfinite(c).all() or not np.isfinite(f).all():raise ValueError('Invalid four-class timeline')
    if not np.allclose(ts,a/e['fps'],rtol=0,atol=1e-6):raise ValueError('Observation/cache timestamp mismatch')
    scored,_=score_arm(e,z,'none');targets=ds.build_phase_targets(a,e['annotation']['segments'],e['length'],e['fps'],1.5)
    classes={h:metrics.scores(np.log(np.maximum(p[valid],1e-30)),targets[h+'_labels'][valid]) for h,p,valid in [('current',c,np.ones(len(ts),bool)),('future',f,targets['future_valid_mask'])]}
    journal=[];memory=TaskMemory(e['episode_id'],model_version,mode='M1',sink=journal.append);timeline=[]
    for i,t in enumerate(ts):
        r=scored['trace'][i];view=memory.observe(Observation(e['episode_id'],i,f"{e['episode_id']}:{i}",float(t),float(t)+.01,tuple(c[i]),tuple(f[i]),model_version),b0_phase=int(r['phase']))
        assert view['confirmed_phase']==r['phase'] and not view['suggests_prompt_update'] and view['progress']['physical_completion']=='unknown'
        timeline.append(dict(observation_id=f"{e['episode_id']}:{i}",observation_timestamp=float(t),arrival_timestamp=float(t)+.01,
                             timestamp_scope='Arrival is a simulated replay schedule, not measured network timing',
                             current_probabilities=c[i].tolist(),future_probabilities=f[i].tolist(),decision=r,memory_version=view['memory_version'],
                             confirmed_history=view['history_entered_positions']))
    prefix=reconstruct_shadow(journal);assert prefix==memory.view()['history_entered_positions']
    observed=[0]+[r['phase'] for r in scored['trace'] if r['changed']]
    assert prefix==observed
    truth=bm.boundaries(dict(episode=e));joint=[]
    for ordinal,(w,p) in enumerate(zip(scored['warnings']['boundaries'],scored['prompt']['events']),1):
        joint.append(dict(ordinal=ordinal,transition=ds.LABELS[w['previous']]+'->'+ds.LABELS[w['label']],future_detected=w['detected'],future_event_time=w['event_time'],
                          prompt_outcome=p['outcome'],prompt_time=p['arrival_s'],prompt_offset_s=p['offset_s'],future_controls_prompt=False))
    phases=np.array([r['phase'] for r in scored['trace']]);gt=np.searchsorted([b['time'] for b in truth],ts,side='right')
    repeats={name:dict(positions=positions,correct_windows=int(np.sum(np.isin(gt,positions)&(gt==phases))),total_windows=int(np.isin(gt,positions).sum()),
                        confusion=[[int(np.sum((gt==u)&(phases==v))) for v in positions] for u in positions])
             for name,positions in [('pickup',[0,4]),('harvest',[1,5]),('return',[2,6])]}
    causal=cascade_diagnostics(scored['state']['persistent'],phases,gt,truth);complete=observed==list(range(7))
    result=dict(classification=classes,warnings=scored['warnings'],prompt=scored['prompt'],state=scored['state'],position_accuracy=scored['position_accuracy'],
                position_confusion=scored['position_confusion'],repeated_positions=repeats,future_prompt_relationship=joint,
                cascade_affected_boundaries=sum(len(v['potential_downstream_ordinals']) for v in causal),cascade_diagnostics=causal,
                legacy_boundaries_crossed_during_persistent_mismatch=sum(len(v['subsequent_boundaries']) for v in scored['state']['persistent']),
                decision_completion=dict(visited_positions=observed,ordered_sequence_complete=complete,all_six_prompts_on_time=all(r['outcome']=='on_time' for r in scored['prompt']['events']),
                                         physical_episode_success=None,VLA_execution_evaluated=False),
                memory=dict(mode='M1',reconstructed_prefix=prefix,journal_records=len(journal),snapshot=memory.snapshot(),faithful_to_B0=True,
                            task_complete=False,physical_success='unknown',memory_improves_decisions=False))
    return result,timeline,journal


def summarize_chain(per):
    relationships=[r for p in per.values() for r in p['future_prompt_relationship']];w=[b for p in per.values() for b in p['warnings']['boundaries']]
    classification={}
    for head in ['current','future']:
        c=np.sum([p['classification'][head]['confusion_true_rows'] for p in per.values()],axis=0);tp=np.diag(c);support=c.sum(1);pred=c.sum(0)
        precision=np.divide(tp,pred,out=np.zeros(4,float),where=pred>0);recall=np.divide(tp,support,out=np.zeros(4,float),where=support>0)
        f1=np.divide(2*tp,support+pred,out=np.zeros(4,float),where=(support+pred)>0)
        classification[head]=dict(macro_f1=float(f1.mean()),accuracy=float(tp.sum()/c.sum()),confusion_true_rows=c.tolist(),
            per_class={label:dict(precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i])) for i,label in enumerate(ds.LABELS)})
    position=np.sum([p['position_confusion'] for p in per.values()],axis=0)
    return dict(future_hits=sum(b['detected'] for b in w),future_total=len(w),false_events=sum(len(p['warnings']['false_events']) for p in per.values()),
                classification=classification,seven_position_accuracy=float(np.trace(position)/position.sum()),seven_position_confusion=position.tolist(),
                false_per_minute=sum(len(p['warnings']['false_events']) for p in per.values())/(sum(p['warnings']['duration_s'] for p in per.values())/60),
                future_miss_prompt_outcomes=dict(Counter(r['prompt_outcome'] for r in relationships if not r['future_detected'])),
                future_hit_prompt_outcomes=dict(Counter(r['prompt_outcome'] for r in relationships if r['future_detected'])),
                prompt_counts=dict(Counter(r['prompt_outcome'] for r in relationships)),mismatch_s=sum(p['state']['mismatch_s'] for p in per.values()),
                persistent_mismatch_s=sum(p['state']['persistent_mismatch_s'] for p in per.values()),cascade_affected_boundaries=sum(p['cascade_affected_boundaries'] for p in per.values()),
                legacy_boundaries_crossed_during_persistent_mismatch=sum(p['legacy_boundaries_crossed_during_persistent_mismatch'] for p in per.values()),
                ordered_sequence_complete=sum(p['decision_completion']['ordered_sequence_complete'] for p in per.values()),
                all_prompts_on_time_episodes=sum(p['decision_completion']['all_six_prompts_on_time'] for p in per.values()),
                episodes=len(per),physical_success_rate=None,causal_future_effect_on_prompt='Requires counterfactual interface replay; co-occurrence alone is not causal')


def episode_report(e,result):
    w=result['warnings'];hits=sum(r['detected'] for r in w['boundaries']);d=result['decision_completion']
    lines=[f"# Episode {e['episode_id']}",f"\nsource_recording_id: {e['source_recording_id']}",
           f"\nCurrent Macro-F1: {result['classification']['current']['macro_f1']:.6f}; Future1.5s Macro-F1: {result['classification']['future']['macro_f1']:.6f}",
           f"\n严格Future事件: {hits}/{len(w['boundaries'])}; 未匹配预警: {len(w['false_events'])}",
           f"\n七阶段位置准确率: {result['position_accuracy']:.6f}; 状态不一致: {result['state']['mismatch_s']:.3f}s",
           f"\n有序位置序列: {d['visited_positions']}; 六次Prompt均按时: {d['all_six_prompts_on_time']}",
           '\n上述为Agent离线决策；未运行VLA或机器人，物理成功未知。Memory仅忠实记录决策。',
           '\n|序号|转换|Future命中|Prompt结果|Prompt相对边界秒|','|---|---|---|---|---|']
    for row in result['future_prompt_relationship']:
        lines.append(f"|{row['ordinal']}|{row['transition']}|{row['future_detected']}|{row['prompt_outcome']}|{row['prompt_offset_s']}|")
    return '\n'.join(lines)+'\n'
