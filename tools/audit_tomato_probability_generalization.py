"""Frozen S0/S1/S2 event attribution; labels only enter post-prediction scoring."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import subprocess
import numpy as np
from . import tomato_dataset as ds,tomato_boundary_metrics as bm
from .evaluate_tomato_complete100_history import freeze_dataset,SOURCE
from .evaluate_tomato_reviewed_boundaries import match_events
from .tomato_residual_event_audit import accepted_events,boundary_evidence,false_changes

S1=Path('runs/tomato/probability_residual_unseen_20261111_v1')


def dist(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()),p10=float(np.percentile(a,10)),median=float(np.median(a)),p90=float(np.percentile(a,90))) if len(a) else dict(n=0)


def run(s2,out,plots=False):
    if out.exists():raise FileExistsError('New output required')
    m,lock=freeze_dataset();out.mkdir(parents=True);old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json')
    oldtrain=set(old['splits']['train']);split=ds.read_json(S1/'split.json');current={k:set(m['splits'][k]) for k in m['splits']}
    assert not set(split['inner'])&set(split['holdout']) and set(split['inner'])|set(split['holdout'])==current['train']
    assert set(split['inner'])<=oldtrain and not set(split['holdout'])&oldtrain
    members=[];counts=Counter()
    for e in m['episodes']:
        if e['split']!='train':continue
        target=ds.build_phase_targets(np.array(e['anchors']),e['annotation']['segments'],e['length'],e['fps'],1.5);valid=target['future_valid_mask'];n=int(valid.sum());counts.update(target['future_labels'][valid].tolist())
        members.append(dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],recording=e['dataset_root']+f"/episode_{e['episode_index']:06d}",B0_seen=e['source_recording_id'] in oldtrain,
                            S1_role='fit' if e['source_recording_id'] in split['inner'] else 'epoch_selection',future_valid_windows=n,future_class_counts=dict(Counter(target['future_labels'][valid].tolist()))))
    ns=np.array([r['future_valid_windows'] for r in members if r['S1_role']=='fit']);share=ns/ns.sum()
    audit=dict(dataset_lock=lock,members=members,S1_fit_count=len(split['inner']),S1_fit_seen_by_B0=len(set(split['inner'])&oldtrain),S1_selection_seen_by_B0=len(set(split['holdout'])&oldtrain),
               fit_concentration=dict(largest_recording_share=float(share.max()),top5_share=float(np.sort(share)[-5:].sum()),effective_recording_count=float(1/(share**2).sum())),
               all_training_future_class_counts={ds.LABELS[k]:v for k,v in counts.items()},
               leakage=dict(training_development_source_overlap=len(current['train']&current['val']),training_test_source_overlap=len(current['train']&current['test']),
                            S1_direct_development_label_fitting=False,S1_threshold_search=False,
                            qualification='No direct split overlap found. S1 fits B0 in-sample predictions; B0 and method designs repeatedly selected on development, so development is not an unbiased test.10 holdout recordings select S1 epochs, not independent S1 test.'),
               S2='Nested five-fold calibration on10 B0-unseen training recordings; outer recording absent from residual fitting, normalization and epoch selection; no B0 retraining.',
               timestamp_scope='Cache times are observation reference times, not saved online prediction-completion or arrival times. Actual live age/latency cannot be verified from these arrays.')
    eventrows=[];false_rows=[];timeline=[];summaries={};oracle=[];inputs={}
    for seed in (42,43,44):
        per=defaultdict(list);effects=Counter();timing=defaultdict(list);gains=Counter();byrecord={}
        for e in m['episodes']:
            if e['split']!='val':continue
            eid=e['episode_id'];a=S1/f'seed{seed}'/(eid+'.npz');b=s2/f'seed{seed}'/(eid+'.npz');za,zb=np.load(a),np.load(b)
            for p in [a,b]:inputs[str(p)]=ds.digest(p)
            ts=za['times'];anchors=np.array(e['anchors']);assert np.allclose(ts,anchors/e['fps']) and np.array_equal(ts,zb['times'])
            assert np.array_equal(za['current'],zb['current']) and np.array_equal(za['original_future'],zb['original_future'])
            ref=np.load(SOURCE/f'seed{seed}'/(eid+'.npz'));assert np.array_equal(ts,ref['times']) and np.array_equal(za['original_future'],ref['future_probabilities'])
            targets=ds.build_phase_targets(anchors,e['annotation']['segments'],e['length'],e['fps'],1.5);valid=targets['future_valid_mask'];assert np.array_equal(valid,anchors+45<e['length']) and e['fps']==30
            truth=bm.boundaries(dict(episode=e));probs=dict(S0=za['original_future'],S1=za['calibrated_future'],S2=zb['calibrated_future']);streams={};scores={};false={}
            for name,p in probs.items():
                streams[name]=accepted_events(p[valid],ts[valid]);scores[name],false[name]=match_events(streams[name],truth,'future',1.5)
            if seed==42:
                ev=accepted_events(np.eye(4)[targets['future_labels'][valid]],ts[valid]);bs,fp=match_events(ev,truth,'future',1.5)
                oracle.extend(dict(episode_id=eid,**v) for v in bs)
            for name in ('S1','S2'):
                original=probs['S0'];candidate=probs[name];changed=original.argmax(1)!=candidate.argmax(1)
                q0=original.max(1)>=.5;q1=candidate.max(1)>=.5
                effects[(name,'windows')]+=int(valid.sum());effects[(name,'argmax_changed')]+=int((valid&changed).sum());effects[(name,'threshold_changed_same_argmax')]+=int((valid&~changed&(q0!=q1)).sum())
                changes=false_changes(false['S0'],false[name]);false_rows.append(dict(seed=seed,episode_id=eid,method=name,**changes))
            for ordinal,truthrow in enumerate(truth,1):
                row=dict(seed=seed,episode_id=eid,source_recording_id=e['source_recording_id'],recording=e['dataset_root']+f"/episode_{e['episode_index']:06d}",ordinal=ordinal,
                         transition=ds.LABELS[truthrow['previous']]+'->'+ds.LABELS[truthrow['label']],boundary_s=truthrow['time'],methods={})
                previous_start=e['annotation']['segments'][ordinal-1]['start_frame']/e['fps']
                for name,p in probs.items():
                    score=scores[name][ordinal-1];evidence=boundary_evidence(ts[valid],p[valid],truthrow,previous_start,streams[name]);row['methods'][name]=dict(score=score,evidence=evidence)
                    if score['detected']:timing[(name,row['transition'])].append(score['offset_s'])
                    if name!='S0':
                        hit0=scores['S0'][ordinal-1]['detected'];hit=score['detected'];status='retained' if hit0 and hit else 'lost' if hit0 else 'recovered' if hit else 'still_missed'
                        row['methods'][name]['status']=status;per[name].append(status)
                        if status=='recovered':gains[(name,eid)]+=1
                eventrows.append(row)
                keep=(ts>=truthrow['time']-4)&(ts<=min(truthrow['end'],truthrow['time']+4))
                timeline.append(dict(seed=seed,episode_id=eid,ordinal=ordinal,boundary_s=truthrow['time'],times=ts[keep].tolist(),current=za['current'][keep].tolist(),future={name:p[keep].tolist() for name,p in probs.items()}))
                if plots and seed==44 and row['methods']['S1'].get('status')=='recovered':plot(out,row,ts[keep],probs,keep)
            byrecord[eid]={name:dict(hits=sum(v['detected'] for v in scores[name]),total=len(truth),false_events=len(false[name]),duration_s=float(ts[valid][-1]-ts[valid][0]+.5)) for name in probs}
        summaries[str(seed)]=dict(changes={k:dict(Counter(v)) for k,v in per.items()},per_recording=byrecord,
                                 probability_changes={name:{k:effects[(name,k)] for k in ['windows','argmax_changed','threshold_changed_same_argmax']} for name in ['S1','S2']},
                                 matched_offset_s={name:{typ:dist(v) for (n,typ),v in timing.items() if n==name} for name in ['S0','S1','S2']},
                                 recovered_concentration={name:{eid:n for (nname,eid),n in gains.items() if nname==name} for name in ['S1','S2']})
    audit.update(seeds=summaries,oracle_label_only=dict(hits=sum(r['detected'] for r in oracle),total=len(oracle),lead_s=dist([-r['offset_s'] for r in oracle if r['detected']]),
                 definition='Artificial perfect future labels at t+45 frames; scorer-only timing diagnostic, never online input or attainable visual-performance claim.'),
                 input_sha256=inputs,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),historical_test_arrays_read=False,labels_changed=False)
    ds.write_json(out/'audit.json',audit);ds.write_json(out/'events.json',eventrows);ds.write_json(out/'false_changes.json',false_rows);ds.write_json(out/'curves.json',timeline)
    print('EVENTS',len(eventrows),'ORACLE',audit['oracle_label_only'],'CONCENTRATION',audit['fit_concentration'])


def plot(out,row,times,probs,keep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=row['methods']['S0']['score']['label'];fig,ax=plt.subplots(figsize=(8,3))
    for name,p in probs.items():ax.plot(times-row['boundary_s'],p[keep,target],label=name)
    ax.axvspan(-1.5,0,color='green',alpha=.08);ax.axvline(0,color='black',linestyle='--');ax.axhline(.5,color='grey',linestyle=':');ax.legend();ax.set_xlabel('Seconds relative to unchanged Subtask boundary');ax.set_ylim(-.02,1.02);ax.set_title(row['episode_id'][:8]+' '+row['transition']);fig.tight_layout();fig.savefig(out/(row['episode_id']+f"_b{row['ordinal']}.png"),dpi=120);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--s2',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--plots',action='store_true');a=p.parse_args();run(a.s2,a.out,a.plots)
