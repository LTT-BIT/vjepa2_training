"""Observed model-evidence onset + exact confirmation waiting decomposition."""
import numpy as np
from . import tomato_dataset as ds
from .tomato_decision_attribution import ROOT,SOURCE


def run():
    traces=ds.read_json(ROOT/'interface_traces.json')['episodes'];truth=ds.read_json(SOURCE/'events.json')['events'];rows=[]
    for event in truth:
        trace=traces[event['episode_id']];j=next((i for i,r in enumerate(trace) if r['context']['changed'] and r['after']['phase']==event['ordinal']),None)
        if j is None:
            rows.append(dict(event_id=event['event_id'],confirmed=False));continue
        r=trace[j];start=j
        while start>0 and trace[start-1]['qualifies'] and trace[start-1]['before']['phase']==r['before']['phase']:start-=1
        first=trace[start]['time'];decision=r['time'];b=event['boundary_s'];wait=decision-first
        assert abs(wait-.5)<1e-6 and abs((first-b)+wait-(decision-b))<1e-9
        rows.append(dict(event_id=event['event_id'],confirmed=True,from_stage=event['from_stage'],to_stage=event['to_stage'],boundary_s=b,
            first_qualifying_run_time_s=first,model_evidence_onset_offset_s=first-b,confirmation_wait_s=wait,prompt_confirmation_s=decision,prompt_offset_s=decision-b,
            prompt_version_before=r['before']['version'],prompt_version_after=r['after']['version'],new_chunk_prompt_version=r['chunk']['prompt_version'],old_chunk_still_valid=r['previous_chunk_still_matches']))
    policy=ds.read_json(ROOT/'event_attribution.json')['events'];paired={}
    for left,right in [('current_argmax','current_conf08'),('current_conf08','current_conf08_k2'),('current_conf08_k2','ordered_current08_k2'),('ordered_current08_k2','ordered_current_future08_k2')]:
        counts={};changes=[]
        for e in policy:
            a=e['current_policy_results'][left]['detected'];b=e['current_policy_results'][right]['detected'];key=str(a)+'->'+str(b);counts[key]=counts.get(key,0)+1
            if a!=b:changes.append(e['event_id'])
        paired[left+'__'+right]=dict(counts=counts,changed_events=changes)
    result=dict(events=rows,paired_diagnostic_ablation=paired,confirmed=sum(r['confirmed'] for r in rows),
        model_onset_offset_median_s=float(np.median([r['model_evidence_onset_offset_s'] for r in rows if r['confirmed']])),
        scope='Onset is start of actual successful qualifying run, not earliest transient argmax. Truth only aligns reporting. Label uncertainty remains unresolved; offsets are to frozen annotation. All actual confirmation waits are0.5s.')
    ds.write_json(ROOT/'prompt_timing_decomposition.json',result);print('All56 eventual ordered confirmations decompose into model-evidence onset plus0.5s waiting')


if __name__=='__main__':run()
