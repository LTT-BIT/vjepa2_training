"""Frozen B0 error attribution. Subtask truth is scorer-only; no relabel/refit."""
from pathlib import Path
from collections import Counter
import argparse
import subprocess
import numpy as np
from . import tomato_dataset as ds
from .evaluate_tomato_complete100_history import freeze_dataset, SOURCE
from .tomato_state_recovery import B0State


def classify_miss(times, future, boundary, ordinal, previous_start, events):
    """Explain the existing K2 policy; never re-score with a looser window/K."""
    t=np.asarray(times);p=np.asarray(future);b=boundary['time'];target=boundary['label']
    window=(t>=b-1.5)&(t<b)
    qualifies=(p.argmax(1)==target)&(p[:,target]>=.5)
    exact=[e for e in events if e['previous']==boundary['previous'] and e['label']==target and max(0,previous_start-1.5)<=e['time']<boundary['end']]
    early=[e for e in exact if e['time']<b-1.5];late=[e for e in exact if e['time']>=b]
    wrong_source=[e for e in events if b-1.5<=e['time']<b and e['label']==target and e['previous']!=boundary['previous']]
    indices=np.flatnonzero(window&qualifies)
    if early:kind='too_early_accepted_transition'
    elif wrong_source:kind='wrong_from_class'
    elif not len(indices):kind='no_qualifying_future_in_window'
    else:
        crosses=any(i+1<len(t) and t[i+1]>=b and qualifies[i+1] for i in indices)
        kind='K2_confirmation_crosses_boundary' if crosses else 'transient_future_candidate_not_stable'
    return dict(kind=kind,qualifying_samples=int(len(indices)),window_samples=int(window.sum()),
                correct_argmax_samples=int(np.sum(window&(p.argmax(1)==target))),
                max_target_probability=float(p[window,target].max()),
                exact_related_events=[dict(**e,offset_s=e['time']-b) for e in exact],
                has_early_event=bool(early),has_late_event=bool(late),wrong_source_events=wrong_source,
                window_samples_detail=[dict(time=float(t[i]),offset_s=float(t[i]-b),probabilities=p[i].tolist(),qualifies=bool(qualifies[i])) for i in np.flatnonzero(window)],
                ordinal=ordinal)


def run(out,plots=False):
    if out.exists():raise FileExistsError('New output directory required')
    manifest,lock=freeze_dataset();baseline=ds.read_json('runs/tomato/complete100_history_20261111_final/seed42_per_episode.json')['none']
    prior=ds.read_json('runs/tomato/complete100_history_20261111_final/receipt.json');out.mkdir(parents=True);rows=[];inputs={};all_scores=[]
    for e in manifest['episodes']:
        if e['split']!='val':continue
        eid=e['episode_id'];path=SOURCE/'seed42'/(eid+'.npz');assert ds.digest(path)==prior['inputs'][str(path)];inputs[str(path)]=ds.digest(path)
        z=np.load(path);ts=z['times'];current=z['current_probabilities'];future=z['future_probabilities'];state=B0State();trace=[];events=[]
        for i,t in enumerate(ts):
            before=state.phase;r=state.update(current[i],future[i],float(t))
            trace.append(dict(time=float(t),before=before,phase=r['phase'],accepted_future=state.warnings.accepted))
            if r['warning']:events.append(r['warning'])
        old=baseline[eid];assert [r['phase'] for r in trace]==[r['phase'] for r in old['trace']]
        cutoff=float(ts[z['anchors']+45<e['length']][-1]);events=[r for r in events if r['time']<=cutoff]
        assert events==old['warnings']['events']
        all_scores+=old['warnings']['boundaries']
        for ordinal,b in enumerate(old['warnings']['boundaries'],1):
            if b['detected']:continue
            previous_start=e['annotation']['segments'][ordinal-1]['start_frame']/e['fps']
            diag=classify_miss(ts,future,b,ordinal,previous_start,events)
            qualifies=(current.argmax(1)==b['label'])&(current[:,b['label']]>=.8)
            pair=next((i for i in range(1,len(ts)) if qualifies[i] and qualifies[i-1] and previous_start<=ts[i-1] and ts[i]<b['end']),None)
            prompt=old['prompt']['events'][ordinal-1]
            independent_time=float(ts[pair]) if pair is not None else None
            state_blocked=pair is not None and trace[pair]['before']!=ordinal-1
            row=dict(episode_id=eid,source_recording_id=e['source_recording_id'],recording=str(Path(e['dataset_root']).parent.name+'/'+Path(e['dataset_root']).name)+f"/episode_{e['episode_index']:06d}",
                     transition=ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']],boundary_s=b['time'],ordinal=ordinal,
                     future_diagnostic=diag,prompt=prompt,independent_current_K2_s=independent_time,
                     independent_current_K2_offset_s=independent_time-b['time'] if independent_time is not None else None,
                     phase_before_independent_current_K2=trace[pair]['before'] if pair is not None else None,
                     expected_previous_phase=ordinal-1,current_candidate_blocked_by_order=state_blocked,
                     future_warning_filtered_by_order=False,label_action='retain_existing_subtask_no_revision')
            rows.append(row)
            keep=(ts>=b['time']-6)&(ts<=min(b['end'],b['time']+6))
            ds.write_json(out/f"{eid}_{ordinal}_curve.json",dict(times=ts[keep].tolist(),current=current[keep].tolist(),future=future[keep].tolist(),phase=[r['phase'] for r,k in zip(trace,keep) if k],truth_boundary_s=b['time'],prompt=prompt,diagnostic=diag))
            if plots:plot_case(out,eid,ordinal,ts[keep],current[keep],future[keep],b,prompt,events)
    assert len(rows)==11 and sum(r['detected'] for r in all_scores)==37 and len(all_scores)==48
    result=dict(baseline=dict(hits=37,total=48,false_events=13,false_per_minute=.6948775055679288),
                causes=dict(Counter(r['future_diagnostic']['kind'] for r in rows)),
                prompt_outcomes=dict(Counter(r['prompt']['outcome'] for r in rows)),
                by_transition=dict(Counter(r['transition'] for r in rows)),
                current_order_blocks=sum(r['current_candidate_blocked_by_order'] for r in rows),
                rows=rows,manifest_sha256=lock['manifest_sha256'],code_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                input_sha256=inputs,training=False,test_values_read=False,labels_modified=False,thresholds_modified=False,
                warning_policy='Unchanged Future .5/K2, exact edge in [b-1.5,b). B0 warnings are not gated by maintained seven-stage position.',
                diagnosis_scope='Explains timing and confidence under fixed Subtask; does not infer visual occlusion or label error from probabilities alone')
    ds.write_json(out/'analysis.json',result);print('CAUSES',result['causes'],'CURRENT ORDER BLOCKS',result['current_order_blocks'],'PROMPTS',result['prompt_outcomes'])


def plot_case(out,eid,ordinal,times,current,future,b,prompt,events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
    for ax,p,name in zip(axes,(current,future),('Current','Future1.5s')):
        for i,label in enumerate(ds.LABELS):ax.plot(times-b['time'],p[:,i],label=label)
        ax.axvspan(-1.5,0,color='green',alpha=.08);ax.axvline(0,color='black',linestyle='--');ax.set_ylabel(name);ax.set_ylim(-.03,1.03)
        if name.startswith('Future'):
            for event in events:
                if times[0]<=event['time']<=times[-1]:ax.axvline(event['time']-b['time'],color='orange',alpha=.5)
        if prompt['arrival_s'] is not None:ax.axvline(prompt['arrival_s']-b['time'],color='purple',linestyle=':')
    axes[0].legend(ncol=4,loc='upper center');axes[-1].set_xlabel('Seconds relative to fixed Subtask boundary; green = strict warning window')
    fig.suptitle(f'{eid[:8]} transition{ordinal}: black=truth, orange=Future edge, purple=Prompt')
    fig.tight_layout();fig.savefig(out/f'{eid}_{ordinal}.png',dpi=120);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--plots',action='store_true');a=p.parse_args();run(a.out,a.plots)
