"""Locked validation transition audit and fresh-observation rate comparison.

Never trains, edits annotations, reads test features, or imports robot control.
Run extract then analyze then media; all outputs use a new dedicated directory.
"""
import argparse
import collections
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds, tomato_metrics as metrics, tomato_phase_model as model
from . import tomato_boundary_metrics as boundary
from . import tomato_independent_heads as independent

ROOT = Path('runs/tomato/transition_audit_20261109')
PRIOR = Path('runs/tomato/optimization_20261108')
MODELS = {
    'joint': ('runs/tomato/campaign94_20261107/experiments/joint_s42/best_head.pt', 1.),
    'independent_h10': (str(PRIOR/'bundles/independent_h10_s42.pt'), 1.),
    'independent_h15': (str(PRIOR/'bundles/independent_h15_s42.pt'), 1.5),
}


def manifest():
    protocol = ds.read_json(ROOT/'protocol.json')
    assert ds.digest(protocol['manifest']) == protocol['manifest_sha256']
    m = ds.read_json(protocol['manifest'])
    return m, [e for e in m['episodes'] if e['split'] == 'val']


def dist(values):
    a = np.asarray(values)
    return dict(samples=len(a), p50_ms=float(np.percentile(a, 50)),
                p95_ms=float(np.percentile(a, 95)), max_ms=float(a.max()))


def match_events(events, truth, head, horizon):
    """One prediction has at most one credit; half-open, exact from/to matching."""
    used = set(); rows = []
    for b in truth:
        lo = b['time'] if head == 'current' else b['time'] - horizon
        hi = min(b['end'], b['time'] + 2) if head == 'current' else b['time']
        hit = next((i for i, p in enumerate(events) if i not in used and
                    p['previous'] == b['previous'] and p['label'] == b['label']
                    and lo <= p['time'] < hi), None)
        if hit is not None:
            used.add(hit)
        t = events[hit]['time'] if hit is not None else None
        rows.append(dict(**b, detected=t is not None, event_time=t,
                         offset_s=t-b['time'] if t is not None else None))
    return rows, [p for i, p in enumerate(events) if i not in used]


def episode_eval(e, anchors, logits, horizon, hz, confirmations, common_horizon=None):
    ts = anchors/e['fps']; truth = boundary.boundaries(dict(episode=e))
    result = {}; duration = {}; probabilities = {}
    for head in ['current', 'future']:
        valid = np.ones(len(ts), bool) if head == 'current' else anchors+round(horizon*e['fps']) < e['length']
        p = metrics.probabilities(logits[head+'_logits']); probabilities[head] = p
        events = boundary.change_events(p[valid], ts[valid], .5, confirmations)
        rows, false = match_events(events, truth, head, common_horizon or horizon)
        y = np.array([ds.label_at_frame(e['annotation']['segments'], int(a)+(round(horizon*e['fps']) if head=='future' else 0)) for a in anchors[valid]])
        duration[head] = float(ts[valid][-1]-ts[valid][0]+1/hz)
        result[head] = dict(boundaries=rows, false_events=false, events=events,
                            duration_s=duration[head], classification=metrics.scores(logits[head+'_logits'][valid], y),
                            reversals_within_2s=sum(b['label']==a['previous'] and b['time']-a['time']<=2 for a,b in zip(events,events[1:])))
    return result, probabilities


def summarize(evaluations):
    result = {}
    for head in ['current', 'future']:
        r = [v[head] for v in evaluations.values()]; bs = [b for v in r for b in v['boundaries']]
        hits = [b for b in bs if b['detected']]; offsets = [b['offset_s'] for b in hits]
        false = sum(len(v['false_events']) for v in r); minutes = sum(v['duration_s'] for v in r)/60
        types = collections.defaultdict(lambda: dict(total=0, detected=0))
        for b in bs:
            x = types[ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']]]; x['total']+=1; x['detected']+=int(b['detected'])
        result[head] = dict(total=len(bs), detected=len(hits), recall=len(hits)/len(bs),
                            false_events=false, false_per_minute=false/minutes,
                            offset_p50_s=float(np.median(offsets)) if offsets else None,
                            offset_p95_s=float(np.percentile(offsets,95)) if offsets else None,
                            jitter_reversals=sum(v['reversals_within_2s'] for v in r), by_transition=dict(types))
    return result


def provider_replay(e, anchors, logits, horizon, version, confirmations=2):
    """Exercise the existing unchanged OpenPI provider and metadata bindings."""
    sys.path.insert(0, '/home/user/tomato-openpi/src')
    from openpi.subtasks.agent_bridge import PredictionPromptProvider, bind_chunk, chunk_matches_prompt
    p = {h:metrics.probabilities(logits[h+'_logits']) for h in ['current','future']}
    providers = [PredictionPromptProvider(confirmations=confirmations) for _ in range(2)]
    for x in providers: x.reset(e['episode_id'])
    rows=[]; old_binding=None; last_context=None; checks=0; changes=[]
    for i,a in enumerate(anchors):
        t=float(a/e['fps']); base=1e6+t
        response=dict(protocol='tomato-stage-prediction-v1',labels=list(ds.LABELS),valid=True,error=None,
                      observation_id=str(i),session_id=e['episode_id'],model_version=version,prediction_mode='joint',
                      future_horizon_s=horizon,input_timestamp=base,result_timestamp=base+.04,
                      **{h+'_logits':logits[h+'_logits'][i].astype(float).tolist() for h in p},
                      **{h+'_probabilities':p[h][i].tolist() for h in p})
        c=providers[0].accept_prediction(response,observation_id=str(i),now=base+.05)
        # Remove every future warning while preserving identical current observations.
        counter={**response,'future_logits':[0.,0.,0.,0.],'future_probabilities':[.25]*4}
        d=providers[1].accept_prediction(counter,observation_id=str(i),now=base+.05)
        assert (c.phase,c.prompt,c.version)==(d.phase,d.prompt,d.version)
        binding=bind_chunk(c,str(i)); assert chunk_matches_prompt(binding,c)
        if old_binding is not None and c.changed: assert not chunk_matches_prompt(old_binding,c)
        if c.changed: changes.append(dict(time=t,phase=c.phase_index,version=c.version))
        assert c.version==c.phase_index and not c.task_complete
        rows.append(dict(time=t,phase=c.phase_index,prompt_version=c.version,warning=c.future_warning,changed=c.changed))
        old_binding=binding; last_context=c; checks+=1
    return dict(rows=rows,changes=changes,counterfactual_future_does_not_change_prompt=True,
                chunk_checks=checks,final_phase=last_context.phase,model_version=version,
                timing='Replay clock: input+40ms result, consumed at+50ms; synthetic transport, not measured runtime')


def extract():
    import torch
    from .tomato_runtime import initialize_runtime, hardware_report
    initialize_runtime(dict(device='cuda',seed=42)); m,episodes=manifest()
    encoder,provenance=model.load_encoder('pretrained/vjepa2-vitl-fpc64-256','cuda',42)
    heads={n:independent.load_checkpoint(p,device='cuda') for n,(p,h) in MODELS.items()}
    out=ROOT/'fresh4hz'; out.mkdir(exist_ok=True)
    report=dict(hardware=hardware_report(),episodes=[],runtime={},scope='Fresh recorded RGB/state histories; CPU assembly/H2D/encoder/heads included; source decode separately. No camera acquisition/network/robot measured.')
    samples={2:[],4:[]}
    for en,e in enumerate(episodes):
        path=out/(e['episode_id']+'.npz')
        if path.exists():
            receipt=ds.read_json(path.with_suffix('.json')); assert ds.digest(path)==receipt['sha256'];report['episodes'].append(receipt)
            print('REUSE',e['episode_id'],flush=True);continue
        model.assert_source_unchanged(e)
        assert ds.digest(e['annotation']['path'])==e['annotation']['sha256']
        ts,q=ds.read_state(e,m['contract']['joint_names'],'arms-and-hands')
        a=ds.iter_phase_anchors(ts,4); histories=np.stack([ds.sample_causal_indices(ts,int(i)) for i in a])
        assert np.all(histories<=a[:,None]) and np.array_equal(histories[:,-1],a)
        began=time.perf_counter(); views,_=model.decode_selected(e,np.unique(histories)); decode_s=time.perf_counter()-began
        values=[]; began=time.perf_counter()
        with torch.inference_mode():
            for start in range(0,len(a),8):
                hs=histories[start:start+8]
                rgb=np.stack([np.stack([np.stack([v[int(i)] for i in h]) for v in views]) for h in hs])
                values.append(model.fuse_features(encoder,rgb,q[hs]))
        features=np.concatenate(values); torch.cuda.synchronize(); extraction_s=time.perf_counter()-began
        logits={}
        for name,(saved,head) in heads.items():
            z=model.predict_heads(head,features,saved['mean'],saved['std'])
            logits.update({name+'_'+key:value for key,value in z.items()})
        np.savez_compressed(path,anchors=a,features=features,histories=histories,**logits)
        identity=dict(source=e['input_sha256'],annotation=e['annotation']['sha256'],camera_keys=e['camera_keys'],joint_names=m['contract']['joint_names'],history_frames=16,history_hz=10,prediction_hz=4,sampler=ds.SAMPLER_VERSION,pretrained=provenance,anchors=a.tolist())
        receipt=dict(episode_id=e['episode_id'],sha256=ds.digest(path),identity=ds.fingerprint(identity),contract=identity,windows=len(a),decode_s=decode_s,extraction_s=extraction_s)
        ds.write_json(path.with_suffix('.json'),receipt); report['episodes'].append(receipt)
        # Paired fresh histories around the first boundary in every validation episode.
        b=boundary.boundaries(dict(episode=e))[0]['time']
        for hz in [2,4]:
            aa=ds.iter_phase_anchors(ts,hz)
            aa=aa[(ts[aa]>=b-1.5)&(ts[aa]<b+1.5)]
            for an in aa:
                h=ds.sample_causal_indices(ts,int(an)); rgb=np.stack([np.stack([v[int(i)] for i in h]) for v in views])
                samples[hz].append((e['episode_id'],int(an),rgb,q[h],float(ts[an])))
        print('EXTRACT',en+1,len(episodes),e['episode_id'],len(a),'decode',round(decode_s,2),'encode',round(extraction_s,2),flush=True)
        del views,values,features
    saved,head=heads['independent_h15']
    for hz,seq in samples.items():
        if not seq: continue
        latency=[];heads_ms=[];encode_ms=[];ages=[];late=[];observations=[]
        # Warm up on recorded data; no latency sample from these calls.
        for _,_,rgb,q,_ in seq[:5]: model.fuse_features(encoder,rgb[None],q[None])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
        for i,(eid,a,rgb,q,source_t) in enumerate(seq):
            scheduled=began+i/hz; time.sleep(max(0.,scheduled-time.perf_counter()));start=time.perf_counter()
            # Copy/stack inside measured path to include causal history assembly cost.
            rgb=np.stack([rgb]);q=np.stack([q]);f=model.fuse_features(encoder,rgb,q)
            torch.cuda.synchronize();mid=time.perf_counter()
            model.predict_heads(head,f,saved['mean'],saved['std']);torch.cuda.synchronize();end=time.perf_counter()
            latency.append((end-start)*1000);encode_ms.append((mid-start)*1000);heads_ms.append((end-mid)*1000)
            # Sample grid floor quantization plus actual queue and compute time.
            grid_age=max(0.,round(source_t*hz)/hz-source_t)
            ages.append((end-scheduled+grid_age)*1000);late.append(end>scheduled+1/hz);observations.append(eid+':'+str(a))
        assert len(set(observations))==len(observations)
        report['runtime'][str(hz)]=dict(latency=dist(latency),encoder_and_assembly=dist(encode_ms),head=dist(heads_ms),
            observation_age=dist(ages),deadline_misses=sum(late),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            unique_observations=len(observations),observation_ids=observations,all_fresh=True,
            age_scope='Rebased offline paced scheduler + floor-to-source-grid age; excludes capture and decode. Different episode intervals concatenated; source time gap is not wall-clock pause.')
        print('RUNTIME',hz,report['runtime'][str(hz)]['latency'],flush=True)
    if not report['runtime']:
        if not (ROOT/'extraction_runtime.json').exists():
            raise RuntimeError('Caches exist but runtime is absent; run a dedicated fresh runtime probe')
        previous=ds.read_json(ROOT/'extraction_runtime.json')
        assert [r['sha256'] for r in previous['episodes']]==[r['sha256'] for r in report['episodes']]
        print('Verified complete existing extraction/runtime; left report unchanged',flush=True)
        return
    ds.write_json(ROOT/'extraction_runtime.json',report)


def analyze():
    m,episodes=manifest(); all_results={}; catalogue=[]; providers={}
    old={n:np.load(PRIOR/'evaluations'/f'{n}_s42_val_logits.npz') for n in MODELS}
    rates=[(2,2,'2hz_k2')]
    if not all((ROOT/'fresh4hz'/(e['episode_id']+'.npz')).exists() for e in episodes):
        raise FileNotFoundError('Complete all validation4Hz caches before final paired analysis')
    rates += [(4,2,'4hz_k2'),(4,3,'4hz_k3')]
    for hz,k,suffix in rates:
        for name,(checkpoint,horizon) in MODELS.items():
            per={}; common={}; offset=0; prov={}; allz={h:[] for h in ['current','future']};ally={h:[] for h in allz}
            for e in episodes:
                if hz==2:
                    a=np.array(e['anchors']);n=len(a);z={h+'_logits':old[name][h+'_logits'][offset:offset+n] for h in allz};offset+=n
                else:
                    cache=np.load(ROOT/'fresh4hz'/(e['episode_id']+'.npz'));a=cache['anchors'];z={h+'_logits':cache[name+'_'+h+'_logits'] for h in allz}
                per[e['episode_id']],p=episode_eval(e,a,z,horizon,hz,k)
                common[e['episode_id']],_=episode_eval(e,a,z,horizon,hz,k,1.)
                prov[e['episode_id']]=provider_replay(e,a,z,horizon,ds.digest(checkpoint),k)
                for head in allz:
                    valid=np.ones(len(a),bool) if head=='current' else a+round(horizon*e['fps'])<e['length']
                    allz[head].append(z[head+'_logits'][valid]);ally[head].append(np.array([ds.label_at_frame(e['annotation']['segments'],int(i)+(round(horizon*e['fps']) if head=='future' else 0)) for i in a[valid]]))
                if name=='independent_h15' and hz==2:
                    ts=a/e['fps']; time_q,q=ds.read_state(e,m['contract']['joint_names'],'arms-and-hands')
                    vel=np.diff(q,axis=0)/np.diff(time_q)[:,None]
                    for j,b in enumerate(boundary.boundaries(dict(episode=e))):
                        r=per[e['episode_id']]['future']['boundaries'][j];target=b['label'];keep=(ts>=b['time']-horizon)&(ts<b['time'])
                        good=keep&(p['future'].argmax(1)==target)&(p['future'][:,target]>=.5)
                        stable=good & np.r_[False,good[:-1]]
                        first=float(ts[np.flatnonzero(good)[0]]) if good.any() else None
                        stable_t=float(ts[np.flatnonzero(stable)[0]]) if stable.any() else None
                        local=(ts>=b['time']-3)&(ts<=b['time']+3)
                        false=per[e['episode_id']]['future']['false_events'];near_false=[v for v in false if abs(v['time']-b['time'])<=3]
                        raw=[ev for ev in per[e['episode_id']]['future']['events'] if ev['previous']==b['previous'] and ev['label']==b['label'] and abs(ev['time']-b['time'])<=4]
                        tags=[]
                        if not r['detected']:
                            tags.append('miss_at_strict_operating_point')
                            if stable_t is not None:tags.append('stable_target_probability_but_no_exact_from_to_change')
                            elif first is not None:tags.append('warning_not_sustained_for_two_observations')
                            elif raw:tags.append('same_transition_outside_allowed_time_window')
                            else:tags.append('target_probability_or_class_confusion')
                        if near_false:tags.append('nearby_unmatched_changes')
                        fi=round(b['time']*e['fps']);lo=max(0,fi-30);hi=min(len(vel),fi+30)
                        catalogue.append(dict(event_id=e['episode_id']+f'_b{j+1}',episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],ordinal=j+1,
                            from_stage=ds.LABELS[b['previous']],to_stage=ds.LABELS[target],boundary_s=b['time'],boundary_frame=fi,annotation_sha256=e['annotation']['sha256'],
                            strict=r,first_valid_target_warning_s=first,stable_target_warning_s=stable_t,
                            current=per[e['episode_id']]['current']['boundaries'][j],nearby_false_events=near_false,attribution_evidence=tags,
                            motion=dict(arm_speed_rms=float(np.sqrt(np.mean(vel[lo:hi,:14]**2))),hand_speed_rms=float(np.sqrt(np.mean(vel[lo:hi,14:]**2))),max_state_gap_s=float(np.diff(time_q)[lo:hi].max())),
                            probability_curve=dict(times_s=ts[local].tolist(),current=p['current'][local].tolist(),future=p['future'][local].tolist()),
                            review=dict(status='unreviewed',original_boundary_frame=fi,proposed_boundary_frame=None,revision_history=[],reason='Visual occlusion/semantic ambiguity require human review; automated tags describe prediction evidence only.')))
            key=name+'_'+suffix;all_results[key]=dict(summary=summarize(per),common_1s_event_window=summarize(common),
                classification={h:metrics.scores(np.concatenate(allz[h]),np.concatenate(ally[h])) for h in allz},per_episode=per)
            providers[key]=prov
    # Preserve already generated media while refreshing paired rate metrics.
    if (ROOT/'events.json').exists():
        prior_events={v['event_id']:v for v in ds.read_json(ROOT/'events.json')['events']}
        for v in catalogue:
            if 'media' in prior_events.get(v['event_id'],{}): v['media']=prior_events[v['event_id']]['media']
    ds.write_json(ROOT/'comparison.json',all_results);ds.write_json(ROOT/'events.json',dict(label_order=list(ds.LABELS),events=catalogue))
    ds.write_json(ROOT/'openpi_replay.json',providers)
    ds.write_json(ROOT/'human_review.json',dict(original_labels_untouched=True,items=[v for v in catalogue if v['attribution_evidence']],review_policy='No automatic label revision. Human proposes semantic boundary with rationale/uncertainty interval and revision history. Independent test never relabeled for model selection.'))
    print({n:{h:dict(recall=r['summary'][h]['recall'],false_min=r['summary'][h]['false_per_minute']) for h in ['current','future']} for n,r in all_results.items()},flush=True)


def media():
    import av
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m,episodes=manifest();byid={e['episode_id']:e for e in episodes}
    events=ds.read_json(ROOT/'events.json')['events'];out=ROOT/'media';out.mkdir(exist_ok=True)
    contact_eid=None;contact_views=None
    for r in events:
        key=r['event_id']; e=byid[r['episode_id']];b=r['boundary_s'];lo=max(0,b-4);hi=min(e['length']/e['fps'],b+4)
        ts,q=ds.read_state(e,m['contract']['joint_names'],'arms-and-hands');v=np.diff(q,axis=0)/np.diff(ts)[:,None]
        fig,ax=plt.subplots(3,1,figsize=(10,7),sharex=True);curve=r['probability_curve'];t=np.array(curve['times_s'])-b
        for j,label in enumerate(ds.LABELS):
            ax[0].plot(t,np.array(curve['current'])[:,j],label=label);ax[1].plot(t,np.array(curve['future'])[:,j],label=label)
        ax[0].set_ylabel('Current p');ax[1].set_ylabel('Future 1.5s p');ax[0].legend(ncol=4);ax[0].set_title(f'{key}: {r["from_stage"]} -> {r["to_stage"]}, strict hit={r["strict"]["detected"]}')
        keep=(ts[1:]>=lo)&(ts[1:]<=hi)
        ax[2].plot(ts[1:][keep]-b,np.sqrt(np.mean(v[keep,:14]**2,axis=1)),label='arms14 RMS rad/s');ax[2].plot(ts[1:][keep]-b,np.sqrt(np.mean(v[keep,14:]**2,axis=1)),label='hands44 RMS rad/s');ax[2].legend();ax[2].set_xlabel('Seconds relative to annotation boundary')
        for a in ax:a.axvline(0,color='k',ls='--');a.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/(key+'.png'),dpi=110);plt.close(fig)
        # Preserve original camera geometry inside a 3-view letterboxed video.
        dest=out/(key+'.mp4')
        if not dest.exists():
            cmd=['ffmpeg','-v','error','-y']
            for path in e['videos']:cmd+=['-ss',str(lo),'-t',str(hi-lo),'-i',path]
            filters=';'.join(f'[{i}:v]scale=384:256:force_original_aspect_ratio=decrease,pad=384:256:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]' for i in range(3))+';[v0][v1][v2]hstack=inputs=3[v]'
            cmd+=['-filter_complex_threads','1','-filter_complex',filters,'-map','[v]','-an','-r','15','-c:v','libx264','-preset','veryfast','-crf','26',str(dest)]
            subprocess.run(cmd,check=True)
        # Three time points x three cameras, exact annotated frame included.
        selected=[max(0,r['boundary_frame']-15),r['boundary_frame'],min(e['length']-1,r['boundary_frame']+15)]
        if contact_eid!=e['episode_id']:
            all_frames=sorted({max(0,min(e['length']-1,v['boundary_frame']+delta)) for v in events if v['episode_id']==e['episode_id'] for delta in [-15,0,15]})
            contact_views,_=model.decode_selected(e,np.array(all_frames));contact_eid=e['episode_id']
        sheet=np.concatenate([np.concatenate([x[i] for x in contact_views],axis=1) for i in selected],axis=0)
        cv2.imwrite(str(out/(key+'_contact.jpg')),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
        r['media']=dict(video=str(dest.resolve()),curve=str((out/(key+'.png')).resolve()),contact=str((out/(key+'_contact.jpg')).resolve()),clip_start_s=lo,annotation_boundary_offset_s=b-lo)
        print('MEDIA',key,flush=True)
    ds.write_json(ROOT/'events.json',dict(label_order=list(ds.LABELS),events=events))
    lines=['# Validation transition catalogue','', 'Probability curves and three-view clips; all times relative to original episode. Annotation boundary is at +4s in most clips; inspect individual clip offset. No labels changed.','', '| Event | Source ID | Transition | Boundary s | Strict warning | First / stable warning s | Current delay s | False changes ±3s | Artifacts |','|---|---|---|---:|---|---|---|---:|---|']
    for r in events:
        md=r['media'];offset=r['current']['offset_s']
        lines.append(f'| {r["event_id"]} | {r["source_recording_id"]} | {r["from_stage"]}→{r["to_stage"]} | {r["boundary_s"]:.3f} | {r["strict"]["detected"]} | {r["first_valid_target_warning_s"]} / {r["stable_target_warning_s"]} | {offset} | {len(r["nearby_false_events"])} | [video]({md["video"]}) [curve]({md["curve"]}) [frames]({md["contact"]}) |')
    (ROOT/'EVENT_CATALOGUE.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['extract','analyze','media']);args=parser.parse_args()
    {'extract':extract,'analyze':analyze,'media':media}[args.stage]()
