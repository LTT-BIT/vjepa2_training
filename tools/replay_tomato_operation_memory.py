"""Real cached stage probabilities -> outcome sidecar -> optional OpenPI memory.

State bins are an explicit tokenizer fixture. No action model is run or scored.
"""
from pathlib import Path
import argparse,sys,time
import numpy as np
from . import tomato_dataset as ds
from .tomato_operation_sidecar import OperationSidecar
from .tomato_operation_memory import oracle_memory_at
from .evaluate_tomato_reviewed_boundaries import revise_episode
from .evaluate_tomato_state_recovery import state_score,annotate_state,prompt_score,event_score,summarize
from .evaluate_tomato_task_memory import standard_trace


def run(out,openpi,tokenizer):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);sys.path.insert(0,str(openpi/'src'))
    import sentencepiece as spm
    from openpi.subtasks.agent_bridge import PredictionPromptProvider
    from openpi.subtasks.memory_bridge import MemoryProvider,bind_memory_chunk,memory_chunk_matches
    sp=spm.SentencePieceProcessor(model_file=str(tokenizer));source=Path('runs/tomato/transition_time_20261109')
    identity=ds.read_json(source/'data_identity.json');manifest=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');by={e['episode_id']:e for e in manifest['episodes']};props=ds.read_json('docs/validation/tomato_reviewed_boundaries_20261109/proposals.json')['truth_proposals']
    model=ds.read_json(source/'seed42/result.json')['identity']['original_classifiers'];per={};lat=[];count=0
    for rec in identity['episodes']:
        if rec['split']!='val':continue
        eid=rec['episode_id'];e=revise_episode(by[eid],props.get(eid,[]));z=np.load(source/'seed42'/(eid+'.npz'))
        side=OperationSidecar(eid,eid,model['sha256']);base=PredictionPromptProvider();base.reset(eid)
        providers={mode:MemoryProvider(mode,encode=sp.encode if mode!='none' else None) for mode in ('none','oracle','predicted')}
        for p in providers.values():p.reset(eid)
        phases=[];warnings=[];trace=[]
        for i,t in enumerate(z['times']):
            t=float(t);response=dict(protocol='tomato-stage-prediction-v1',labels=['pickup','harvest','return','adjust'],valid=True,error=None,session_id=eid,observation_id=str(i),model_version=model['sha256'],input_timestamp=t,result_timestamp=t+.005,future_horizon_s=1.5,prediction_mode='joint',**{h+'_probabilities':z[h+'_probabilities'][i].tolist() for h in ('current','future')},**{h+'_logits':np.log(np.maximum(z[h+'_probabilities'][i],1e-30)).tolist() for h in ('current','future')})
            ctx=base.accept_prediction(response,observation_id=str(i),now=t+.01);start=time.perf_counter_ns();packet=side.process(response,now=t+.01);lat.append((time.perf_counter_ns()-start)/1e6)
            assert packet['prediction']==response and side.b0.phase==ctx.phase_index
            oracle=oracle_memory_at([],session_id=eid,episode_id=eid,source_recording_id=e['source_recording_id'],observation_id=str(i),timestamp=t,fps=e['fps'],phase_index=ctx.phase_index)
            for mode,p in providers.items():
                value=None if mode=='none' else oracle if mode=='oracle' else packet['memory']
                current=p.accept(ctx,value,state_bins=[0]*58,now=t+.01)
                assert (current.prompt,current.phase_index,current.version)==(ctx.prompt,ctx.phase_index,ctx.version)
                assert memory_chunk_matches(bind_memory_chunk(current,str(i)),current,now=t+.01)
            assert not packet['memory']['verified_facts'] and packet['memory']['picked_count'] is None
            phases.append(ctx.phase_index)
            if packet['b0']['warning']:warnings.append(packet['b0']['warning'])
            trace.append(dict(time=t,phase=ctx.phase_index,prompt_version=ctx.version,physical_results='unknown',memory_version=packet['memory']['memory_version'],modes_identical=True));count+=1
        from . import tomato_boundary_metrics as bm
        truth=bm.boundaries(dict(episode=e));gt=np.searchsorted([b['time'] for b in truth],z['times'],side='right');valid=z['anchors']+45<e['length'];duration=float(z['times'][valid][-1]+.5)
        per[eid]=dict(warnings=event_score([w for w in warnings if w['time']<=z['times'][valid][-1]],truth,duration),prompt=prompt_score(standard_trace(z['times'],phases),truth,z['times']),state=annotate_state(state_score(phases,gt,z['times']),truth,phases,z['times']))
        ds.write_json(out/(eid+'.json'),dict(trace=trace,final_memory=side.memory.snapshot(now=float(z['times'][-1])+.01)))
    result=summarize(per);assert result['warnings']['hits']==39 and result['warnings']['false_events']==13 and result['prompt_counts']=={'on_time':46,'early':2,'late':2} and result['mismatch_s']==40.
    ds.write_json(out/'result.json',dict(summary=result,observations=count,sidecar_cpu_ms=dict(p50=float(np.median(lat)),p95=float(np.percentile(lat,95)),maximum=max(lat)),independent_operation_labels=0,oracle_vs_predicted='not_estimable_empty_oracle_no_incremental_information',action_mae=None,action_rmse=None,action_model_calls=0,state_bin_fixture='58 zeros, tokenizer/interface test only; not measured normalized Pi05 robot state',real_probabilities=True,openpi_checkout=str(openpi),tokenizer_sha256=ds.digest(tokenizer),original_stage_outputs_preserved=True))
    print('REPLAY',count,'real stage predictions; B0',result['warnings']['hits'],'/50; no qualified physical outcomes or action inference')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--openpi',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);a=p.parse_args();run(a.out,a.openpi,a.tokenizer)
