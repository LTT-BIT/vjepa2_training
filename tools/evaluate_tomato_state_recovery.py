"""Frozen validation-only, causal state recovery replay. Truth enters scoring only.

Run: .venv-tomato/bin/python -m tools.evaluate_tomato_state_recovery --out NEW_DIR
"""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import time
import resource
import platform
import numpy as np
from . import tomato_dataset as ds, tomato_boundary_metrics as bm
from .evaluate_tomato_reviewed_boundaries import revise_episode, match_events
from .tomato_state_recovery import StateRecovery, MODES, CLASSES, PHASES, StateInconsistencyDetector
from .tomato_decision_attribution import interface_trace

CONFIG = Path('configs/tomato_state_recovery_20261110/protocol.json')
SOURCE = Path('runs/tomato/transition_time_20261109')


def stretches(mask, times):
    """Sample-and-hold duration; 2Hz last sample holds for .5s, no invented frames."""
    mask = np.asarray(mask, bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [dict(start_s=float(times[a]), end_s=float(times[b]+.5), duration_s=float(times[b]-times[a]+.5), first_index=int(a), last_index=int(b), recovered=bool(b+1 < len(times))) for a,b in zip(starts, ends)]


def state_score(phase, truth, times):
    episodes = stretches(np.asarray(phase) != truth, times)
    persistent = [r for r in episodes if r['duration_s'] > 2.]
    return dict(mismatch_s=sum(r['duration_s'] for r in episodes), mismatch_runs=len(episodes),
                persistent_mismatch_s=sum(r['duration_s'] for r in persistent), persistent_runs=len(persistent),
                runs=episodes, persistent=persistent)


def prompt_score(trace, truth, times):
    changes = [dict(time=r['time'], previous_phase=r['previous_phase'], phase=r['phase'], version=r['prompt_version'], resync=r['resynchronized']) for r in trace if r['changed']]
    rows=[]
    for ordinal,b in enumerate(truth, 1):
        arrivals=[r for r in changes if r['phase']==ordinal]
        a=arrivals[0] if arrivals else None
        outcome = 'missing' if a is None else 'early' if a['time'] < b['time'] else 'on_time' if a['time'] < min(b['end'],b['time']+2.) else 'late'
        rows.append(dict(ordinal=ordinal, boundary_s=b['time'], transition=ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']], outcome=outcome, arrival_s=a['time'] if a else None, offset_s=a['time']-b['time'] if a else None))
    return dict(events=rows, counts=dict(Counter(r['outcome'] for r in rows)), updates=changes,
                resync_updates=sum(r['resync'] for r in changes), skipped_prompt_positions=sum(r['phase']-r['previous_phase']-1 for r in changes),
                # A multi-position recommendation may be proven, but is never credited as two Prompt updates.
                illegal_updates=sum(r['phase']<=r['previous_phase'] or r['phase']-r['previous_phase']>2 for r in changes))


def event_score(events, truth, duration):
    rows,false=match_events(events,truth,'future',1.5)
    return dict(boundaries=rows, false_events=false, duration_s=duration, events=events)


def summarize(per):
    bounds=[b for p in per.values() for b in p['warnings']['boundaries']]
    hit=sum(b['detected'] for b in bounds);false=sum(len(p['warnings']['false_events']) for p in per.values())
    minutes=sum(p['warnings']['duration_s'] for p in per.values())/60
    counts=Counter(r['outcome'] for p in per.values() for r in p['prompt']['events'])
    return dict(warnings=dict(hits=hit,total=len(bounds),false_events=false,false_per_minute=false/minutes,recall=hit/len(bounds),precision=hit/max(hit+false,1), by_transition={typ:dict(hits=sum(b['detected'] for b in bounds if ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']]==typ),total=sum(ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']]==typ for b in bounds)) for typ in sorted({ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']] for b in bounds})}),
                prompt_counts=dict(counts), **{key:sum(p['state'][key] for p in per.values()) for key in ['mismatch_s','mismatch_runs','persistent_mismatch_s','persistent_runs']},
                resync_updates=sum(p['prompt']['resync_updates'] for p in per.values()),skipped_prompt_positions=sum(p['prompt']['skipped_prompt_positions'] for p in per.values()),illegal_updates=sum(p['prompt']['illegal_updates'] for p in per.values()),
                cascade_events=sum(len(run['subsequent_boundaries']) for p in per.values() for run in p['state']['persistent']),
                prompt_by_transition={typ:dict(Counter(r['outcome'] for p in per.values() for r in p['prompt']['events'] if r['transition']==typ)) for typ in sorted({r['transition'] for p in per.values() for r in p['prompt']['events']})})


def annotate_state(score, truth, phase, times):
    for run in score['persistent']:
        # Count later boundaries crossed while a single uninterrupted mismatch persists.
        run['subsequent_boundaries']=[i for i,b in enumerate(truth,1) if run['start_s'] < b['time'] < run['end_s']]
        i=run['first_index'];run['online_phase_at_start']=int(phase[i])
    return score


def run(out):
    if out.exists(): raise FileExistsError('Preserve frozen output; choose a new output directory')
    config=ds.read_json(CONFIG);identity=ds.read_json(SOURCE/'data_identity.json')
    manifest=ds.read_json(config['original_manifest']);by={e['episode_id']:e for e in manifest['episodes']}
    proposals=ds.read_json(config['proposals'])['truth_proposals'];mechanisms=ds.read_json(config['mechanisms'])
    recs=[r for r in identity['episodes'] if r['split']=='val'];assert len(recs)==9
    out.mkdir(parents=True);frozen=[CONFIG,Path(config['original_manifest']),Path(config['proposals']),Path(config['mechanisms']),SOURCE/'data_identity.json']
    results={};all_attribution=[];bench={};root_rows=[]
    for seed in config['seeds']:
        old=ds.read_json(SOURCE/f'seed{seed}/result.json');op=old['external_validation_thresholds_are_ablation_not_selection'][str(old['selected_threshold'])]
        b2by={r['episode_id']:r for r in op['decision_replay']};per={k:{} for k in MODES};root_by={}
        frozen.append(SOURCE/f'seed{seed}/result.json')
        for rec in recs:
            eid=rec['episode_id'];original=by[eid];e=revise_episode(original,proposals.get(eid,[]));truth=bm.boundaries(dict(episode=e))
            # Ordered phase ordinal is used only below, after online decisions have been generated.
            assert [ds.LABELS.index(s['label']) for s in e['annotation']['segments']] == list(CLASSES[:len(e['annotation']['segments'])])
            path=SOURCE/f'seed{seed}'/(eid+'.npz');frozen.append(path);z=np.load(path)
            valid=z['anchors']+45<e['length'];ts=z['times'][valid];c=z['current_probabilities'][valid];f=z['future_probabilities'][valid]
            duration=float(ts[-1]-ts[0]+.5);ph_truth=np.searchsorted([b['time'] for b in truth],ts,side='right')
            # Historical provider is isolated from the user's OpenPI checkout; immutable Git blobs.
            if seed==42:
                historical=interface_trace(original,z['anchors'],{h+'_logits':np.log(np.maximum(z[h+'_probabilities'],1e-30)) for h in ['current','future']},'frozen-b0-state-recovery-parity')
                historical=[r for r,keep in zip(historical,valid) if keep]
            for mode in MODES:
                state=StateRecovery(mode,session_id=eid,model_id=f'independent-h15-s{seed}')
                latencies=[];trace=[]
                for t,cp,fp in zip(ts,c,f):
                    start=time.perf_counter_ns();row=state.update(cp,fp,float(t));latencies.append((time.perf_counter_ns()-start)/1e6);trace.append(row)
                events=[r['warning'] for r in trace if r['warning']]
                assert events==op['per_episode']['B0'][eid]['events'], 'Frozen B0 warnings changed'
                if seed==42 and mode=='baseline':
                    assert [r['phase'] for r in trace]==[r['after']['phase'] for r in historical], 'Historical provider parity failure'
                if mode=='detect':
                    base=per['baseline'][eid]['trace'];assert all((a['phase'],a['warning'],a['prompt_version'])==(b['phase'],b['warning'],b['prompt_version']) for a,b in zip(trace,base))
                phases=[r['phase'] for r in trace]
                state_metrics=annotate_state(state_score(phases,ph_truth,ts),truth,phases,ts)
                # Additional separate posterior accuracy; never fed back through truth.
                posterior=[r['estimated_phase'] for r in trace]
                estimated=state_score(posterior,ph_truth,ts) if mode!='baseline' else None
                alert=np.array([r['inconsistent'] for r in trace]);mismatch=np.asarray(phases)!=ph_truth
                per[mode][eid]=dict(trace=trace,warnings=event_score(events,truth,duration),state=state_metrics,prompt=prompt_score(trace,truth,ts),estimated_state=estimated,
                    detector=dict(alert_windows=int(alert.sum()),correct_alert_windows=int((alert&mismatch).sum()),mismatch_windows=int(mismatch.sum()),alerts_during_correct_state=int((alert&~mismatch).sum())),
                    latency_ms=dict(p50=float(np.median(latencies)),p95=float(np.percentile(latencies,95)),maximum=max(latencies)),
                    offline_truth_phase=ph_truth.tolist(), source_recording_id=e['source_recording_id'],dataset_root=e['dataset_root'],episode_index=e['episode_index'])
                full_state=StateRecovery(mode,session_id=eid,model_id=f'independent-h15-s{seed}')
                full_trace=[full_state.update(cp,fp,float(t)) for t,cp,fp in zip(z['times'],z['current_probabilities'],z['future_probabilities'])]
                full_truth=np.searchsorted([b['time'] for b in truth],z['times'],side='right')
                full_phases=[r['phase'] for r in full_trace]
                per[mode][eid]['full_current_timeline']=dict(prompt=prompt_score(full_trace,truth,z['times']),state=annotate_state(state_score(full_phases,full_truth,z['times']),truth,full_phases,z['times']))
                bench.setdefault(mode,[]).extend(latencies)
            # B2 root-cause timeline replay uses saved decisions only; no new threshold.
            b2=b2by[eid]['trace'];assert np.allclose([r['time'] for r in b2],ts)
            phase=np.array([r['phase_index'] for r in b2]);detector=StateInconsistencyDetector();b2alerts=[dict(time=float(t),**detector.update(cp,float(t),int(p))) for t,cp,p in zip(ts,c,phase)];bs=annotate_state(state_score(phase,ph_truth,ts),truth,phase,ts)
            arrivals={r['phase_index']:r['time'] for r in b2 if r['suggest_switch']}
            root=None
            for ordinal,b in enumerate(truth,1):
                target=CLASSES[ordinal];lo=0 if ordinal==1 else truth[ordinal-2]['time'];hi=b['end']
                # First available .8/K2 Current evidence while B2 is still waiting for this exact next phase.
                opportunities=[];count=0
                for i,t in enumerate(ts):
                    prior=0 if i==0 else phase[i-1]
                    q=lo<=t<hi and prior==ordinal-1 and c[i].argmax()==target and c[i,target]>=.8
                    count=count+1 if q else 0
                    if count==2:
                        window=b2[i].get('expected_transition_interval')
                        opportunities.append(dict(time_s=float(t),current=c[i].tolist(),future=f[i].tolist(),time_distribution=z['transition_probabilities'][valid][i].tolist(),online_phase_before=int(prior),truth_phase_offline=int(ph_truth[i]),predicted_window=window,switched=bool(b2[i]['suggest_switch']), reason='switch_allowed' if b2[i]['suggest_switch'] else 'no_time_window' if window is None else 'outside_time_window' if not window[0]<=t<=window[1]+.5 else 'switch_allowed'))
                baseevent=per['baseline'][eid]['prompt']['events'][ordinal-1]
                arrival=arrivals.get(ordinal);b2out='missing' if arrival is None else 'early' if arrival<b['time'] else 'on_time' if arrival<min(b['end'],b['time']+2) else 'late'
                if root is None and any(not q['switched'] for q in opportunities) and b2out in ('missing','late'):
                    root=dict(seed=seed,episode_id=eid,ordinal=ordinal,boundary_s=b['time'],transition=ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']],baseline_arrival_s=baseevent['arrival_s'],baseline_outcome=baseevent['outcome'],B2_arrival_s=arrival,B2_outcome=b2out,current_opportunities=opportunities,
                              cause='time_gate_blocks_available_current' if any(not q['switched'] for q in opportunities) else 'current_confirmation_or_already_wrong_phase',)
                root_rows.append(dict(seed=seed,episode_id=eid,ordinal=ordinal,baseline=baseevent,B2_arrival_s=arrival,B2_outcome=b2out,current_opportunities=opportunities))
            first=bs['runs'][0] if bs['runs'] else None;persist=bs['persistent'][0] if bs['persistent'] else None
            root_by[eid]=dict(inconsistency_alerts=[r for r in b2alerts if r['inconsistent']],root=root,first_disagreement=first,first_persistent_disagreement=persist,B2_state=bs,B2_phase=phase.tolist(),times=ts.tolist(),truth_phase_offline=ph_truth.tolist(),source_recording_id=e['source_recording_id'],dataset_root=e['dataset_root'],episode_index=e['episode_index'])
        for case in mechanisms[str(seed)]['B1_hits_lost_by_B2']:
            if case['mechanism']!='no_expected_online_phase_in_window':continue
            eid=case['episode_id'];root=root_by[eid];t=case['B1_event_time'];idx=int(np.argmin(abs(np.asarray(root['times'])-t)))
            comparisons={mode:dict(phase_at_lost_warning=per[mode][eid]['trace'][idx]['phase'],expected_phase_before=case['ordinal']-1,prompt_event=per[mode][eid]['prompt']['events'][case['ordinal']-1]) for mode in MODES}
            all_attribution.append(dict(**case,source_recording_id=root['source_recording_id'],dataset_root=root['dataset_root'],episode_index=root['episode_index'],root_failure=root['root'],first_disagreement=root['first_disagreement'],first_persistent_disagreement=root['first_persistent_disagreement'],B2_phase_at_warning=root['B2_phase'][idx],truth_phase_at_warning_offline=root['truth_phase_offline'][idx],current_at_warning=np.load(SOURCE/f'seed{seed}'/(eid+'.npz'))['current_probabilities'][idx].tolist(),future_at_warning=np.load(SOURCE/f'seed{seed}'/(eid+'.npz'))['future_probabilities'][idx].tolist(),B2_mismatch_runs=root['B2_state']['runs'],B2_report_only_alerts=root['inconsistency_alerts'],comparisons=comparisons))
        results[str(seed)]=dict(summary={mode:summarize(per[mode]) for mode in MODES},per_episode=per,B2_roots=root_by)
    assert len([r for r in all_attribution if r['seed']==42])==12
    receipt=dict(host=platform.platform(),python=platform.python_version(),process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,resource_note='CPU replay peak includes imports/historical provider and trace buffers; not incremental estimator RAM or V-JEPA GPU latency',git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),inputs={str(p):ds.digest(p) for p in frozen},source_files={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/tomato_state_recovery.py')]},test_arrays_read=False,encoder_calls=0,training=False,oracle_inputs=False,remote_push=False,labels='14 human boundary proposals; provisional, source annotations unchanged',baseline_warning_parity=True,historical_provider_seed42_phase_parity=True)
    ds.write_json(out/'results.json',results);ds.write_json(out/'attribution.json',all_attribution);ds.write_json(out/'confirmation_opportunities.json',root_rows);ds.write_json(out/'receipt.json',receipt)
    summary=dict(full_current_timeline={seed:{mode:dict(prompt_counts=dict(Counter(r['outcome'] for p in per.values() for r in p['full_current_timeline']['prompt']['events'])),mismatch_s=sum(p['full_current_timeline']['state']['mismatch_s'] for p in per.values()),persistent_mismatch_s=sum(p['full_current_timeline']['state']['persistent_mismatch_s'] for p in per.values()),resync_updates=sum(p['full_current_timeline']['prompt']['resync_updates'] for p in per.values())) for mode,per in v['per_episode'].items()} for seed,v in results.items()},seeds={seed:v['summary'] for seed,v in results.items()},latency_ms={k:dict(p50=float(np.median(v)),p95=float(np.percentile(v,95)),maximum=max(v),windows=len(v)) for k,v in bench.items()})
    ds.write_json(out/'summary.json',summary)
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);run(parser.parse_args().out)
