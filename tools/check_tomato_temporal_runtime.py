"""Real history5Hz video -> research Agent -> unchanged OpenPI interface; no robot."""
import sys,time
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_phase_model as model,tomato_temporal_heads as tm,tomato_time_sampling as sampler
from .tomato_causal_agent import CausalVideoAgent
from .tomato_agent_mixture import MixtureVideoAgent
from .tomato_runtime import initialize_runtime
ROOT=Path('runs/tomato/temporal_final_20261108');LONG=Path('runs/tomato/long_history_20261108_v2');BASE=Path('runs/tomato/temporal_20261108_v2')


def distribution(x):return dict(samples=len(x),mean_ms=float(np.mean(x)),p50_ms=float(np.percentile(x,50)),p95_ms=float(np.percentile(x,95)),p99_ms=float(np.percentile(x,99)),max_ms=float(max(x)),deadline_misses=int(np.sum(np.array(x)>500)))


def run():
 initialize_runtime(dict(device='cuda',seed=42));sys.path.insert(0,'/home/user/tomato-openpi/src')
 from openpi.subtasks.agent_bridge import PredictionPromptProvider,bind_chunk,chunk_matches_prompt
 m=ds.read_json(LONG/'prepared/manifest.json');e=next(e for e in m['episodes'] if e['split']=='val');names=m['contract']['joint_names'];times,q=ds.read_state(e,names,'arms-and-hands')
 anchors=np.array(e['anchors'][:60]);ids=np.stack([sampler.sample(times,int(a),16,5,False) for a in anchors]);views,_=model.decode_selected(e,np.unique(ids))
 checkpoint=LONG/'experiments/mlp_16x5_s42/best.pt';agent=CausalVideoAgent(checkpoint,'pretrained/vjepa2-vitl-fpc64-256');calls=[];handle=agent.encoder.register_forward_hook(lambda *args:calls.append(1))
 original=model.fuse_features;features=[]
 def capture(*args):
  value=original(*args);features.append(value[0]);return value
 model.fuse_features=capture
 latencies=[];responses=[];contexts=[]
 try:
  rgb=np.stack([np.stack([v[int(i)] for i in ids[0]]) for v in views])
  for i in range(5):
   agent.reset('warm'+str(i));now=time.time();hist=times[ids[0]]-times[anchors[0]]+now
   agent.predict(rgb,q[ids[0]],names,hist,timestamp=now,observation_id='warm',session_id='warm'+str(i))
  features.clear();agent.reset('live-replay');provider=PredictionPromptProvider();provider.reset('live-replay')
  torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
  for i,history in enumerate(ids):
   time.sleep(max(0.,started+i*.5-time.perf_counter()));rgb=np.stack([np.stack([v[int(j)] for j in history]) for v in views]);now=time.time();hist=times[history]-times[anchors[i]]+now
   t=time.perf_counter();response=agent.predict(rgb,q[history][:,::-1],names[::-1],hist,timestamp=now,observation_id=str(anchors[i]),session_id='live-replay')
   torch.cuda.synchronize();latencies.append((time.perf_counter()-t)*1000)
   context=provider.accept_prediction(response,observation_id=response['observation_id'],now=time.time());assert chunk_matches_prompt(bind_chunk(context,str(i)),context)
   responses.append(response);contexts.append(context.__dict__)
  assert len(calls)==65
 finally:
  handle.remove();model.fuse_features=original
 idx=ds.read_json(LONG/'prepared/feature_index.json');path=next(v['path'] for v in idx['episodes'] if v['episode_id']==e['episode_id'])
 with np.load(path,allow_pickle=False) as cached:expected=cached['features'][:len(anchors)].copy()
 actual=np.array(features);assert np.array_equal(expected[:,-928:],actual[:,-928:])
 saved,head=tm.load_checkpoint(checkpoint,device='cuda');pred=tm.predict(head,(expected-saved['mean'].numpy())/saved['std'].numpy(),tm.history_indices(len(expected),1))
 comparison={h:int(np.sum(pred[h+'_logits'].argmax(1)!=np.array([r[h+'_logits'] for r in responses]).argmax(1))) for h in ['current','future']}
 report=dict(passed=True,checkpoint=str(checkpoint),model_version=ds.digest(checkpoint),history= sampler.contract(16,5,False),
  source_episode=e['episode_id'],unique_observations=len(set(int(a) for a in anchors)),latency=distribution(latencies),encoder_calls=len(calls),expected_encoder_calls=65,
  encoder_parameters=sum(p.numel() for p in agent.encoder.parameters()),frozen_encoder=all(not p.requires_grad for p in agent.encoder.parameters()),
  peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
  cache_live_feature_max_abs_error=float(np.max(np.abs(actual-expected))),cache_live_visual_mean_abs_error=float(np.mean(np.abs(actual[:,:3072]-expected[:,:3072]))),
  measured_state_order_exact=True,classification_argmax_disagreements_vs_cache=comparison,first_response=responses[0],last_prompt_context=contexts[-1],
  scope='60 distinct recorded observations at2Hz, actual 16x5Hz RGB and named state histories; replay timestamps rebased to wall clock; synchronized real encoder and new checkpoint. Decode/capture/transport excluded. Existing OpenPI provider .8/K2, no action inference or robot.',robot_connected=False,rtc_modified=False)
 ds.write_json(ROOT/'real_runtime_check.json',report)
 # Real cached sequential histories must match batched causal indexing for the GRU.
 index=ds.read_json(BASE/'prepared/feature_index.json');source=next(v['path'] for v in index['episodes'] if v['episode_id']==e['episode_id'])
 with np.load(source,allow_pickle=False) as z:x=z['features'][:100].copy()
 gp=BASE/'experiments/gru_s42/best.pt'
 if not gp.exists():gp=Path('runs/tomato/temporal_20261108/experiments/gru_s42/best.pt')
 session=tm.CausalFeatureSession(gp,'cuda');s=session.saved;batch=tm.predict(session.head,(x-s['mean'].numpy())/s['std'].numpy(),tm.history_indices(len(x),4));stream=[session.predict(v,i*.5) for i,v in enumerate(x)]
 errors={h:float(np.max(np.abs(np.concatenate([v[h+'_logits'] for v in stream])-batch[h+'_logits']))) for h in ['current','future']}
 assert max(errors.values())<1e-3
 ds.write_json(ROOT/'causal_stream_parity.json',dict(passed=True,real_observations=len(x),max_abs_logit_error=errors,scope='Real cached Episode, strict context4 FIFO vs batched causal gather; no future windows.'))
 del agent,head,session;torch.cuda.empty_cache()
 # Independently exercise the mixture wrapper with one real, valid padded initial window.
 p=BASE/'mixtures/ensemble_half_s42/agent.pt';mix=MixtureVideoAgent(p,'pretrained/vjepa2-vitl-fpc64-256');rgb=np.stack([np.stack([v[int(j)] for j in ids[0]]) for v in views])
 response=mix.predict(rgb,q[ids[0]],names,timestamp=time.time(),observation_id='mixture0',session_id='mixture')
 provider=PredictionPromptProvider();provider.reset('mixture');context=provider.accept_prediction(response,observation_id='mixture0',now=time.time())
 ds.write_json(ROOT/'mixture_runtime_check.json',dict(passed=True,response=response,prompt_context=context.__dict__,robot_connected=False))
 print('REAL RUNTIME AND CAUSAL PARITY PASS',report['latency'],flush=True)

if __name__=='__main__':run()
