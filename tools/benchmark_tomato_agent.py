"""Real validation-video benchmark and unchanged OpenPI bridge offline replay."""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds, tomato_phase_model as model, tomato_metrics as metrics
from . import tomato_independent_heads as dual
from .tomato_agent_inference import TomatoAgent
from .tomato_runtime import initialize_runtime, hardware_report

ROOT=Path('runs/tomato/optimization_20261108')


def distribution(values):
    a=np.array(values)
    return dict(count=len(a),mean_ms=float(a.mean()),p50_ms=float(np.percentile(a,50)),
                p95_ms=float(np.percentile(a,95)),p99_ms=float(np.percentile(a,99)),max_ms=float(a.max()),
                over_500ms=int((a>500).sum()))


def export_bundle(choice):
    path=Path(choice['checkpoint']);saved,head=dual.load_checkpoint(path)
    out=ROOT/'selected_bundle';out.mkdir(exist_ok=True)
    import shutil
    target=out/'agent.pt'
    if not target.exists(): shutil.copyfile(path,target)
    assert ds.digest(target)==choice['checkpoint_sha256']
    if saved.get('architecture')=='independent_dual_v1':
        for name, source in saved['sources'].items():
            destination=out/(name+'_head.pt')
            if not destination.exists(): shutil.copyfile(source['path'],destination)
            assert ds.digest(destination)==source['sha256']
    np.savez(out/'normalization.npz',mean=saved['mean'].numpy(),std=saved['std'].numpy())
    schema=dict(protocol='tomato-stage-prediction-v1',rgb=dict(shape=[3,16,256,256,3],dtype='uint8',color='RGB',cameras=saved['contract']['cameras']),
                joints=dict(shape=[16,58],dtype='float32',names=saved['joint_names']),
                history_sample_hz=10,prediction_hz=2,causal=True,feature_dimension=4000,
                label_order=saved['label_order'],joint_alignment='by exact measured .position names',
                architecture=saved.get('architecture','shared_joint'),pretrained=saved['pretrained'],
                feature_version=saved['feature_version'],split_sha256=saved['split_sha256'])
    ds.write_json(out/'input_schema.json',schema)
    config=dict(checkpoint=str(target.resolve()),model_version=ds.digest(target),
                pretrained_dir=str(Path('pretrained/vjepa2-vitl-fpc64-256').resolve()),
                encoder_files_not_duplicated=True,device='cuda',max_age_s=2.,future_horizon_s=saved['future_horizon_s'],
                research_candidate_only=not choice['reliable_transition_candidate_approved'],
                stable_default_checkpoint=str(Path(choice['stable_default_checkpoint']).resolve()),
                validation_analysis_operating_point=choice['selected'],
                openpi_interface_default=dict(threshold=.8,confirmations=2),
                note='Analysis supports separate Current/Future thresholds; existing OpenPI provider remains unchanged at .8/2. No robot deployment.',
                selection_rule_sha256=choice['rule_sha256'])
    ds.write_json(out/'inference_config.json',config)
    return target,saved,head


def replay(choice, saved):
    sys.path.insert(0,'/home/user/tomato-openpi/src')
    from openpi.subtasks.agent_bridge import PredictionPromptProvider, bind_chunk, chunk_matches_prompt, validate_prediction
    from openpi.subtasks.tomato import COARSE_SEQUENCE, PHASES
    manifest=ds.read_json(ROOT/'prepared_h10/manifest.json')
    rows=[e for e in manifest['episodes'] if e['split']=='val']
    name=choice['selected']['name'];logits=np.load(ROOT/'evaluations'/f'{name}_s42_val_logits.npz')
    pc=metrics.probabilities(logits['current_logits']);pf=metrics.probabilities(logits['future_logits'])
    provider=PredictionPromptProvider();offset=0;episodes={};last=None;last_context=None
    for e in rows:
        session='offline-val-'+e['episode_id'];provider.reset(session);contexts=[]
        segments=metrics.continuous_phases(e['annotation']['segments'])
        expected=[s['label'] for s in segments]
        prefix=expected==list(COARSE_SEQUENCE[:len(expected)])
        phase_errors=0;absolute_error=0
        for i,a in enumerate(e['anchors']):
            n=offset+i;t=1000000.+a/e['fps'];oid=e['episode_id']+':'+str(a)
            response=dict(protocol='tomato-stage-prediction-v1',model_version=choice['checkpoint_sha256'],
                observation_id=oid,session_id=session,input_timestamp=t,result_timestamp=t+.04,
                labels=list(ds.LABELS),future_horizon_s=saved['future_horizon_s'],prediction_mode='joint',valid=True,error=None,
                current_logits=logits['current_logits'][n].astype(float).tolist(),future_logits=logits['future_logits'][n].astype(float).tolist(),
                current_probabilities=pc[n].tolist(),future_probabilities=pf[n].tolist())
            context=provider.accept_prediction(response,observation_id=oid,now=t+.05)
            chunk=bind_chunk(context,'offline:'+oid);assert chunk_matches_prompt(chunk,context)
            assert context.action_class==COARSE_SEQUENCE[context.phase_index] and context.phase==PHASES[context.phase_index]
            assert context.model_version==choice['checkpoint_sha256'] and not context.task_complete
            truth_phase=next(j for j,s in enumerate(segments) if s['start_frame']<=a<s['end_frame'])
            phase_errors+=int(context.phase_index!=truth_phase);absolute_error+=abs(context.phase_index-truth_phase)
            if context.changed: contexts.append(dict(anchor=a,phase=context.phase,action_class=context.action_class,timestamp=t))
            last=response;last_context=context
        offset+=len(e['anchors']);episodes[e['episode_id']]=dict(windows=len(e['anchors']),mapped_changes=contexts,
            final_phase=last_context.phase,needs_review=last_context.needs_review,
            truth_sequence_is_supported_prefix=prefix,
            phase_position_accuracy=(1-phase_errors/len(e['anchors'])) if prefix else None,
            mean_absolute_phase_index_error=(absolute_error/len(e['anchors'])) if prefix else None,
            note='Provider initialized explicitly at pickup_1; four labels cannot recover arbitrary mid-task phase.')
    rejected={}
    for name,mut in [('stale',dict(input_timestamp=last['input_timestamp']-10)),('model_changed',dict(model_version='different')),
                     ('wrong_session',dict(session_id='wrong')),('wrong_observation',dict(observation_id='wrong'))]:
        bad={**last,**mut}
        if name=='model_changed': bad.update(input_timestamp=last['input_timestamp']+.5,result_timestamp=last['input_timestamp']+.54)
        try: provider.accept_prediction(bad,observation_id=last['observation_id'],now=last['input_timestamp']+.55)
        except ValueError as exc: rejected[name]=str(exc)
        else: raise AssertionError('Not rejected: '+name)
    result=dict(passed=True,windows=offset,episodes=episodes,rejections=rejected,model_version=choice['checkpoint_sha256'],
        timestamp_scope='Synthetic monotonic replay clock from recorded anchor/fps; result +40ms, consume +50ms. Live latency measured separately.',
        scope='Real validation cached predictions -> unchanged OpenPI provider and chunk metadata; no policy actions, robot or RTC.',
        openpi_configuration=dict(threshold=.8,confirmations=2))
    ds.write_json(ROOT/'offline_replay.json',result)


def benchmark(checkpoint,choice,saved):
    initialize_runtime(dict(device='cuda',seed=42))
    manifest=ds.read_json(ROOT/'prepared_h10/manifest.json');samples=[];decode=[]
    for e in [e for e in manifest['episodes'] if e['split']=='val']:
        ts,q=ds.read_state(e,saved['joint_names'],'arms-and-hands')
        # Two real, distinct causal windows per validation source; one near a boundary.
        a=e['anchors'][min(len(e['anchors'])-1,max(1,len(e['anchors'])//3))]
        b=min(e['length']-1,e['annotation']['segments'][1]['start_frame'])
        histories=[ds.sample_causal_indices(ts,int(v),sampler_version=ds.SAMPLER_VERSION) for v in (a,b)]
        start=time.perf_counter();views,prep=model.decode_selected(e,np.unique(histories));elapsed=time.perf_counter()-start
        decode.append(dict(episode_id=e['episode_id'],wall_s=elapsed,frames_decoded=3*e['length'],selected_rgb_frame_ms=distribution(prep)))
        for ids in histories:
            rgb=np.stack([np.stack([v[int(i)] for i in ids]) for v in views])
            samples.append((rgb,q[ids]))
    np.savez_compressed(ROOT/'validation_observation.npz',rgb=samples[0][0],joints=samples[0][1],joint_names=np.array(saved['joint_names']))
    t=time.perf_counter();agent=TomatoAgent(checkpoint,'pretrained/vjepa2-vitl-fpc64-256');load_s=time.perf_counter()-t
    encoder_calls=[]
    hook=agent.encoder.register_forward_hook(lambda *args:encoder_calls.append(1))
    for rgb,q in samples[:5]: agent.predict(rgb,q,saved['joint_names'],observation_id='warm',session_id='bench',timestamp=time.time())
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();total=[];extract=[];headtimes=[];currenttimes=[];futuretimes=[]
    for i in range(120):
        rgb,q=samples[i%len(samples)];began=time.perf_counter()
        features=model.fuse_features(agent.encoder,rgb[None],q[None]);torch.cuda.synchronize();middle=time.perf_counter()
        model.predict_heads(agent.head,features,saved['mean'],saved['std']);torch.cuda.synchronize();end=time.perf_counter()
        extract.append((middle-began)*1000);headtimes.append((end-middle)*1000);total.append((end-began)*1000)
        if isinstance(agent.head,dual.IndependentHeads):
            for h,values in [(agent.head.current,currenttimes),(agent.head.future,futuretimes)]:
                start=time.perf_counter();model.predict_heads(h,features,saved['mean'],saved['std']);torch.cuda.synchronize();values.append((time.perf_counter()-start)*1000)
    paced=[];start=time.perf_counter()
    for i in range(60):
        deadline=start+i*.5;time.sleep(max(0.,deadline-time.perf_counter()))
        rgb,q=samples[i%len(samples)];began=time.perf_counter()
        response=agent.predict(rgb,q,saved['joint_names'],observation_id=str(i),session_id='paced',timestamp=time.time())
        torch.cuda.synchronize();paced.append((time.perf_counter()-began)*1000)
        assert response['model_version']==choice['checkpoint_sha256']
    hook.remove();assert len(encoder_calls)==185,len(encoder_calls)
    result=dict(hardware=hardware_report(),model_version=choice['checkpoint_sha256'],model_load_s=load_s,
                samples=len(samples),unpaced=distribution(total),feature_extraction=distribution(extract),
                both_heads=distribution(headtimes),current_head=distribution(currenttimes) if currenttimes else None,
                future_head=distribution(futuretimes) if futuretimes else None,paced_2hz=distribution(paced),
                encoder_calls=len(encoder_calls),expected_encoder_calls=185,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                video_decode=decode,scope='Single request, batch1 containing 3 camera clips; real validation RGB and states; synchronized CUDA; includes H2D/preprocessing/fusion and logits D2H. Paced includes Agent schema/probability checks. Decode/storage/transport/ROS excluded from inference; offline full-video decode separately measured.',
                sustainable_2hz_in_this_benchmark=max(paced)<500)
    ds.write_json(ROOT/'benchmark.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('video_decode','hardware')}),flush=True)


def service_check():
    sys.path.insert(0,'/home/user/tomato-openpi/src')
    from openpi.subtasks.agent_bridge import PredictionPromptProvider
    cfg=ds.read_json(ROOT/'selected_bundle/inference_config.json')
    with (ROOT/'service_stderr.log').open('w') as err:
        process=subprocess.Popen([sys.executable,'-u','-m','tools.serve_tomato_agent',
            '--checkpoint',cfg['checkpoint'],'--pretrained-dir',cfg['pretrained_dir']],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True)
        try:
            ready=json.loads(process.stdout.readline());assert ready['status']=='ready'
            request=dict(protocol='tomato-stage-prediction-v1',observation_npz=str((ROOT/'validation_observation.npz').resolve()),
                         observation_id='new-validation-window',session_id='offline-new',timestamp=time.time())
            process.stdin.write(json.dumps(request)+'\n');process.stdin.flush()
            response=json.loads(process.stdout.readline());assert response['valid'],response
            assert response['model_version']==cfg['model_version']
            provider=PredictionPromptProvider();provider.reset(request['session_id'])
            context=provider.accept_prediction(response,observation_id=request['observation_id'],now=time.time())
            process.stdin.write(json.dumps({**request,'timestamp':time.time()-100})+'\n');process.stdin.flush()
            stale=json.loads(process.stdout.readline());assert not stale['valid']
            ds.write_json(ROOT/'service_bridge.json',dict(passed=True,ready=ready,response=response,
                prompt_context=context.__dict__,stale_rejected=True,
                source='Real validation video window; independent dual checkpoint; unchanged JSONL protocol and OpenPI bridge',
                robot_connected=False,rtc_modified=False))
        finally:
            process.stdin.close();process.wait(timeout=30)


def compare_head_latency():
    initialize_runtime(dict(device='cuda',seed=42))
    choice=ds.read_json(ROOT/'selection.json');parent=ds.read_json(ROOT/'prepared_h10/manifest.json')
    first=next(e for e in parent['episodes'] if e['split']=='val')
    idx=ds.read_json(ROOT/'prepared_h10/feature_index.json')
    path=next(e['path'] for e in idx['episodes'] if e['episode_id']==first['episode_id'])
    with np.load(path,allow_pickle=False) as data: x=data['features'][:100]
    results={}
    for name,path in [('joint_seed42','runs/tomato/campaign94_20261107/experiments/joint_s42/best_head.pt'),
                      ('independent_research_seed42',choice['checkpoint'])]:
        saved,head=dual.load_checkpoint(path,device='cuda');times=[]
        for i in range(210):
            torch.cuda.synchronize();start=time.perf_counter()
            model.predict_heads(head,x[i%len(x):i%len(x)+1],saved['mean'],saved['std'])
            torch.cuda.synchronize()
            if i>=10: times.append((time.perf_counter()-start)*1000)
        results[name]=dict(latency=distribution(times),parameters=sum(p.numel() for p in head.parameters()))
    ds.write_json(ROOT/'head_latency_comparison.json',dict(results=results,
        scope='Same cached real validation features, batch1, synchronized GPU, H2D normalization and D2H logits included; excludes common encoder.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-root',default=str(ROOT))
    args=parser.parse_args();ROOT=Path(args.output_root)
    choice=ds.read_json(ROOT/'selection.json');checkpoint,saved,head=export_bundle(choice)
    replay(choice,saved);benchmark(checkpoint,choice,saved);compare_head_latency();service_check()
