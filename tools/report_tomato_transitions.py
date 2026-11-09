"""Postprocess event evidence, provider behavior and the predeclared rate experiment."""
import collections
import subprocess
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds, tomato_metrics as metrics
from .audit_tomato_transitions import ROOT, MODELS, manifest


def enrich():
    m,episodes=manifest(); byid={e['episode_id']:e for e in episodes}
    results=ds.read_json(ROOT/'comparison.json'); replay=ds.read_json(ROOT/'openpi_replay.json')
    events=ds.read_json(ROOT/'events.json')['events']; provider_summary={}
    for name,per in replay.items():
        horizon=MODELS[name.rsplit('_',2)[0]][1]; hz=4 if '_4hz_' in name else 2
        counts=collections.Counter(); leads=[]; delays=[]; early_offsets=[]; false_categories=collections.Counter(); total=0;minutes=0.;episode_stats={}
        for eid,r in per.items():
            e=byid[eid];ev=[v for v in events if v['episode_id']==eid];total+=len(ev)
            valid_end=e['length']/e['fps']-horizon; rows=[v for v in r['rows'] if v['time']<valid_end];minutes+=len(rows)/hz/60
            hits=0;warnings=[];previous=None;false_rising=0
            for w in rows:
                key=w['phase']+1 if w['warning'] else None
                target=next((v for v in ev if v['ordinal']==key),None)
                correct=target is not None and target['boundary_s']-horizon<=w['time']<target['boundary_s']
                if key is not None:
                    warnings.append(dict(time=w['time'],phase=key,correct=correct));counts['warning_windows']+=1
                    if not correct:
                        counts['false_warning_windows']+=1
                        category='missing_ordinal' if target is None else ('late_after_boundary' if w['time']>=target['boundary_s'] else 'earlier_than_horizon')
                        false_categories[category]+=1
                    if key!=previous:
                        counts['warning_rising']+=1
                        if not correct:counts['false_rising']+=1;false_rising+=1
                previous=key
            for event in ev:
                b=event['boundary_s'];ordinal=event['ordinal'];target=[v for v in warnings if v['phase']==ordinal and b-horizon<=v['time']<b]
                phase_changes=[v for v in r['changes'] if v['phase']==ordinal]
                decision=phase_changes[0]['time'] if phase_changes else None
                if target:counts['any_warning_hits']+=1;hits+=1;leads.append(b-target[0]['time'])
                if decision is not None and b<=decision<min(b+2,e['annotation']['segments'][ordinal]['end_frame']/e['fps']):
                    counts['decision_hits']+=1;delays.append(decision-b)
                elif decision is not None and decision<b:counts['premature_prompts']+=1;early_offsets.append(decision-b)
                elif decision is None:counts['never_confirmed']+=1
                else:counts['late_prompt_outside_2s']+=1
                if name=='independent_h15_2hz_k2':
                    event['agent_decision']=dict(confirmed_time_s=decision,offset_s=decision-b if decision is not None else None,
                        any_warning_time_s=target[0]['time'] if target else None,prompt_premature=decision is not None and decision<b)
            episode_stats[eid]=dict(boundaries=len(ev),warning_hits=hits,false_warning_rising=false_rising,final_phase=r['final_phase'])
        provider_summary[name]=dict(boundaries=total,counts=dict(counts),any_warning_recall=counts['any_warning_hits']/total,
            strict_decision_recall=counts['decision_hits']/total,false_warning_rising_per_minute=counts['false_rising']/minutes,
            false_warning_categories=dict(false_categories),warning_lead_p50_s=float(np.median(leads)) if leads else None,
            confirmed_delay_p50_s=float(np.median(delays)) if delays else None,confirmed_delay_p95_s=float(np.percentile(delays,95)) if delays else None,
            premature_prompt_offsets_s=early_offsets,per_episode=episode_stats,
            every_future_counterfactual_left_prompt_unchanged=all(r['counterfactual_future_does_not_change_prompt'] for r in per.values()))
    ds.write_json(ROOT/'provider_summary.json',provider_summary)
    evidence=collections.Counter(tag for r in events for tag in r['attribution_evidence'])
    # Closest same from/to raw event to a missed boundary, plus boundary F1 at fixed neighborhoods.
    for event in events:
        e=byid[event['episode_id']];eid=e['episode_id'];j=event['ordinal']-1
        event['comparisons']={}
        raw=results['independent_h15_2hz_k2']['per_episode'][eid]['future']['events']
        same=[v for v in raw if v['previous']==ds.LABELS.index(event['from_stage']) and v['label']==ds.LABELS.index(event['to_stage'])]
        nearest=min(same,key=lambda v:abs(v['time']-event['boundary_s'])) if same else None
        event['nearest_same_transition_offset_s']=nearest['time']-event['boundary_s'] if nearest else None
        if not event['strict']['detected']:
            dt=event['nearest_same_transition_offset_s']
            event['primary_timing_diagnosis']='no_matching_class_path' if dt is None else ('premature_before_horizon' if dt < -1.5 else 'late_after_boundary' if dt >= 0 else 'matching_path_reused_or_other_constraint')
        for name,r in results.items():
            rr=r['per_episode'][eid];event['comparisons'][name]={h:rr[h]['boundaries'][j] for h in ['current','future']}
        event['review']['reason']='Automated evidence: '+', '.join(event['attribution_evidence'])+'. Human must inspect video and source semantics; no automatic label change.'
    ds.write_json(ROOT/'events.json',dict(label_order=list(ds.LABELS),events=events))
    ds.write_json(ROOT/'human_review.json',dict(original_labels_untouched=True,items=[v for v in events if v['attribution_evidence']],review_policy='Record reviewer, original/proposed frame, confidence/uncertainty interval, rationale and revision history. Test labels never revised for score. No revision proposed by this audit.'))
    neighborhoods={}
    for name,(cp,horizon) in MODELS.items():
        z=np.load(Path('runs/tomato/optimization_20261108/evaluations')/(name+'_s42_val_logits.npz'))
        offset=0;buffers={(head,side,w):[[],[]] for head in ['current','future'] for side in ['before','after'] for w in [.25,.5,1.]}
        for e in episodes:
            a=np.array(e['anchors']);t=a/e['fps'];bounds=np.array([v['boundary_s'] for v in events if v['episode_id']==e['episode_id']]);distance=t[:,None]-bounds;nearest=distance[np.arange(len(t)),np.abs(distance).argmin(1)]
            for head in ['current','future']:
                valid=np.ones(len(a),bool) if head=='current' else a+round(horizon*e['fps'])<e['length']
                y=np.full(len(a),-100);y[valid]=[ds.label_at_frame(e['annotation']['segments'],int(i)+(round(horizon*e['fps']) if head=='future' else 0)) for i in a[valid]]
                for side in ['before','after']:
                    for w in [.25,.5,1.]:
                        keep=valid&((nearest>=-w)&(nearest<0) if side=='before' else (nearest>=0)&(nearest<w))
                        b=buffers[head,side,w];b[0].append(z[head+'_logits'][offset:offset+len(a)][keep]);b[1].append(y[keep])
            offset+=len(a)
        neighborhoods[name]={str(k):metrics.scores(np.concatenate(v[0]),np.concatenate(v[1])) for k,v in buffers.items() if sum(len(x) for x in v[1])}
    ds.write_json(ROOT/'neighborhoods.json',neighborhoods)
    # Verify old runs/checkpoints unchanged, including historical tests, without inference.
    expected={}
    for source in ['runs/tomato/optimization_20261108/freeze.json','runs/tomato/temporal_20261108_v2/frozen_prior.json','runs/tomato/long_history_20261108_v2/frozen_source_baseline.json']:
        x=ds.read_json(source);expected.update(x.get('prior_artifacts',x))
    failures=[p for p,h in expected.items() if not Path(p).exists() or ds.digest(p)!=h]
    annotation_failures=[e['episode_id'] for e in m['episodes'] if ds.digest(e['annotation']['path'])!=e['annotation']['sha256']]
    integrity=dict(prior_files_checked=len(expected),changed=failures,annotation_files_checked=len(m['episodes']),changed_annotations=annotation_failures,test_inference_performed=False,
                   split_sha256=m['split_sha256'],manifest_sha256=ds.digest('runs/tomato/optimization_20261108/prepared_h10/manifest.json'),evidence_counts=dict(evidence))
    ds.write_json(ROOT/'integrity.json',integrity);assert not failures and not annotation_failures
    print('PROVIDER', {k:{n:v[n] for n in ['any_warning_recall','strict_decision_recall','false_warning_rising_per_minute']} for k,v in provider_summary.items()})
    print('EVIDENCE',dict(evidence));print('INTEGRITY',integrity)


if __name__=='__main__':enrich()
