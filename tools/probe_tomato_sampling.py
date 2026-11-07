"""Real flexible-history compatibility/cost and fresh-video higher-rate feasibility probes."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_phase_model as model,tomato_time_sampling as sampler
from .tomato_runtime import initialize_runtime,hardware_report
ROOT=Path('runs/tomato/temporal_20261108')
DESIGNS={'8x10':(8,10,False),'16x10':(16,10,False),'16x5':(16,5,False),'16nonuniform':(16,10,True)}


@torch.inference_mode()
def extract(encoder,rgb,q):
 rgb=np.asarray(rgb);q=np.asarray(q,dtype=np.float32);h=rgb.shape[1]
 if rgb.shape!=(3,h,256,256,3) or rgb.dtype!=np.uint8 or h not in (8,16) or q.shape!=(h,58):raise ValueError('Flexible input schema mismatch')
 parameter=next(encoder.parameters());pixels=torch.from_numpy(np.ascontiguousarray(rgb)).to(device=parameter.device,dtype=parameter.dtype).permute(0,1,4,2,3)/255
 pixels=(pixels-pixels.new_tensor([.485,.456,.406])[None,None,:,None,None])/pixels.new_tensor([.229,.224,.225])[None,None,:,None,None]
 features=encoder(pixel_values_videos=pixels).last_hidden_state.mean(1).float().cpu().numpy()
 if features.shape!=(3,1024):raise ValueError('Official encoder incompatible with chosen frame count')
 return np.concatenate([features.reshape(-1),q.reshape(-1)]).astype(np.float32)


def dist(x):return dict(samples=len(x),mean_ms=float(np.mean(x)),p50_ms=float(np.percentile(x,50)),p95_ms=float(np.percentile(x,95)),max_ms=float(max(x)))


def run():
 initialize_runtime(dict(device='cuda',seed=42));m=ds.read_json(ROOT/'prepared/manifest.json');episodes=[e for e in m['episodes'] if e['split']=='val']
 names=m['contract']['joint_names'];encoder,provenance=model.load_encoder('pretrained/vjepa2-vitl-fpc64-256','cuda',42)
 measured={name:[] for name in DESIGNS};rows={name:[] for name in DESIGNS};decode=[];peak={name:0 for name in DESIGNS};frequency_samples={hz:[] for hz in [2,4,5]}
 cache=ROOT/'sampling_probe_cache';cache.mkdir(exist_ok=True)
 for episode_number,e in enumerate(episodes):
  times,q=ds.read_state(e,names,'arms-and-hands');boundary=e['annotation']['segments'][1]['start_frame'];a=np.array(e['anchors']);chosen=[int(a[len(a)//3]),int(a[max(0,np.searchsorted(a,boundary)-1)])]
  ids={name:[sampler.sample(times,a,*design) for a in chosen] for name,design in DESIGNS.items()}
  freq_ids={}
  if episode_number<2:
   for hz in [2,4,5]:
    anchors=ds.iter_phase_anchors(times,hz);anchors=anchors[(times[anchors]>=3)&(times[anchors]<9)]
    freq_ids[hz]=[(int(a),ds.sample_causal_indices(times,int(a))) for a in anchors]
  allids=np.unique(np.concatenate([v for seq in ids.values() for v in seq]+[v for seq in freq_ids.values() for _,v in seq]))
  began=time.perf_counter();views,_=model.decode_selected(e,allids);decode.append(dict(episode_id=e['episode_id'],seconds=time.perf_counter()-began))
  for name,design in DESIGNS.items():
   contract=sampler.contract(*design);key=ds.fingerprint(dict(source=e['input_sha256'],episode_id=e['episode_id'],pretrained=provenance,history=contract,anchors=chosen,preprocessing=model.FEATURE_VERSION))
   values=[];torch.cuda.reset_peak_memory_stats()
   for history in ids[name]:
    rgb=np.stack([np.stack([v[int(i)] for i in history]) for v in views]);torch.cuda.synchronize();start=time.perf_counter();features=extract(encoder,rgb,q[history]);torch.cuda.synchronize()
    measured[name].append((time.perf_counter()-start)*1000);values.append(features)
   peak[name]=max(peak[name],torch.cuda.max_memory_allocated())
   path=cache/(key+'.npz');np.savez_compressed(path,features=np.stack(values),input_indices=np.stack(ids[name]),anchors=np.array(chosen))
   ds.write_json(path.with_suffix('.json'),dict(fingerprint=key,sha256=ds.digest(path),history=contract,encoder=provenance,episode_id=e['episode_id'],kind='compatibility/cost sample, not training cache',joint_names=names))
   rows[name].append(dict(episode_id=e['episode_id'],cache_key=key,dimension=len(values[0]),frames=design[0]))
  for hz,seq in freq_ids.items():
   for a,history in seq:
    rgb=np.stack([np.stack([v[int(i)] for i in history]) for v in views]);frequency_samples[hz].append((e['episode_id'],a,rgb,q[history]))
 report=dict(runtime=hardware_report(),designs={name:dict(contract=sampler.contract(*design),latency=dist(measured[name][2:]),peak_allocated_bytes=peak[name],compatible=True,episodes=rows[name],
  estimated_train_val_encoder_minutes=sum(len(e['anchors']) for e in m['episodes'] if e['split']!='test')*np.mean(measured[name][2:])/60000) for name,design in DESIGNS.items()},decode=decode,
  caveat='20 sampled real windows per design; no accuracy selection, full alternative-history training cache not extracted. 8-frame encoder works but feature3536D differs from legacy4000D; old runtime remains16-only. New sampler is shared with input construction and identity includes offsets.',
  noninteger_horizons=[sampler.horizon_frames(h,30) for h in [.25,.5,.75,1.,1.5]])
 # Pace each fresh recorded observation; static old MLP avoids applying 2Hz GRU context at 5Hz.
 saved,head=model.load_checkpoint('runs/tomato/campaign94_20261107/experiments/joint_s42/best_head.pt',device='cuda');rates={}
 for hz,observations in frequency_samples.items():
  latencies=[];began=time.perf_counter();source_ids=[]
  for i,(eid,a,rgb,q) in enumerate(observations):
   time.sleep(max(0.,began+i/hz-time.perf_counter()));start=time.perf_counter();f=model.fuse_features(encoder,rgb[None],q[None]);model.predict_heads(head,f,saved['mean'],saved['std']);torch.cuda.synchronize();latencies.append((time.perf_counter()-start)*1000);source_ids.append(eid+':'+str(a))
  assert len(set(source_ids))==len(source_ids)
  rates[str(hz)]=dict(latency=dist(latencies),deadline_ms=1000/hz,deadline_misses=int(np.sum(np.array(latencies)>1000/hz)),unique_observations=len(set(source_ids)),all_observations_fresh=True)
 report['frequency_feasibility']=rates;report['rate_scope']='Real fresh RGB/state each call, synchronized encoder+static MLP, excludes capture/transport; no repeated stale features. Training/evaluation accuracy remains2Hz; temporal heads cannot silently change update cadence.'
 ds.write_json(ROOT/'sampling_probe.json',report);print('SAMPLING AND RATE PROBE COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-root',default=str(ROOT));args=parser.parse_args();ROOT=Path(args.output_root);run()
