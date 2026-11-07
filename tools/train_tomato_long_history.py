"""One predeclared 3-second sensory-history ablation; fresh cache, train/val only."""
import argparse,copy,time
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_phase_model as model,tomato_time_sampling as sampler
from . import train_tomato_temporal as campaign
from .tomato_runtime import initialize_runtime
ROOT=Path('runs/tomato/long_history_20261108');BASE=Path('runs/tomato/temporal_20261108_v2');REUSE_CACHE=None


def run():
 initialize_runtime(dict(device='cuda',seed=42));ROOT.mkdir(parents=True,exist_ok=True)
 contract=sampler.contract(16,5,False)
 protocol=dict(source_baseline='Same-horizon Joint MLP1.5s, sixteen frames10Hz',new_history=contract,
  input_dimension=4000,seeds=[42,43,44],horizon_s=1.5,architecture='mlp',lambda_future=.5,
  batch_windows=16,feature_splits=['train','val'],test_never_extracted=True,
  motivation='Temporal heads over old features did not improve; test longer raw sensory context before increasing classifier complexity.',
  selection='Use same fixed stage2 event and classification metrics; no gate relaxation; compare actual validation three-seed results.')
 campaign.lock(ROOT/'protocol.json',protocol)
 original=ds.read_json(BASE/'prepared/manifest.json');manifest=copy.deepcopy(original)
 manifest['contract']['history_sample_hz']=5;manifest['contract']['sampler_version']=contract['sampler_version'];manifest['sampler_version']=contract['sampler_version'];manifest['history_contract']=contract
 manifest['episodes']=[e for e in original['episodes'] if e['split']!='test'];manifest['omitted_test_sources']=original['splits']['test']
 manifest['prediction']=ds.prediction_contract(dict(prediction_mode='joint',future_horizon_s=1.5))
 manifest['summary']={s:copy.deepcopy(original['summary'][s]) for s in ['train','val']}
 for split in ['train','val']:
  labels=[];valids=[]
  for episode in manifest['episodes']:
   if episode['split']!=split:continue
   targets=ds.build_phase_targets(np.array(episode['anchors']),episode['annotation']['segments'],episode['length'],episode['fps'],1.5)
   mask=targets['future_valid_mask'];episode['future_offset_frames']=round(1.5*episode['fps'])
   episode['future_invalid_count']=int((~mask).sum());episode['future_class_counts']=np.bincount(targets['future_labels'][mask],minlength=4).tolist()
   labels.append(targets['future_labels'][mask]);valids.append(mask)
  keep=np.concatenate(valids);manifest['summary'][split].update(future_valid=int(keep.sum()),future_invalid=int((~keep).sum()),future_class_counts=np.bincount(np.concatenate(labels),minlength=4).tolist())
 prepared=ROOT/'prepared';prepared.mkdir(exist_ok=True);campaign.lock(prepared/'manifest.json',manifest)
 campaign.lock(ROOT/'frozen_source_baseline.json',{str(p):ds.digest(p) for p in sorted(BASE.rglob('*')) if p.is_file()})
 encoder=None;entries={s:[] for s in ['train','val']};index=[];started=time.perf_counter();stats=[]
 cache=ROOT/'feature_cache';cache.mkdir(exist_ok=True)
 for num,e in enumerate(manifest['episodes']):
  model.assert_source_unchanged(e);times,q=ds.read_state(e,manifest['contract']['joint_names'],'arms-and-hands');anchors=np.array(e['anchors']);ids=np.stack([sampler.sample(times,int(a),16,5,False) for a in anchors])
  key=ds.fingerprint(dict(episode_id=e['episode_id'],input_sha256=e['input_sha256'],anchors=anchors.tolist(),history=contract,pretrained=manifest['pretrained'],preprocessing=model.FEATURE_VERSION))
  path=cache/(key+'.npz');elapsed=0.;decode_s=0.;peak=0
  if REUSE_CACHE is not None and not path.exists():
   candidate=REUSE_CACHE/(key+'.npz')
   if candidate.is_file() and candidate.with_suffix('.json').is_file():
    receipt=ds.read_json(candidate.with_suffix('.json'))
    if receipt['sha256']==ds.digest(candidate) and receipt['fingerprint']==key:path=candidate
  if path.exists():
   receipt=ds.read_json(path.with_suffix('.json'));assert receipt['sha256']==ds.digest(path) and receipt['fingerprint']==key
   with np.load(path,allow_pickle=False) as data:
    assert np.array_equal(data['input_indices'],ids) and np.array_equal(data['anchors'],anchors);features=data['features'].copy()
   status='reused'
  else:
   if encoder is None:encoder,_=model.load_encoder('pretrained/vjepa2-vitl-fpc64-256','cuda',42)
   start=time.perf_counter();views,_=model.decode_selected(e,np.unique(ids));decode_s=time.perf_counter()-start
   values=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
   for b in range(0,len(anchors),16):
    history=ids[b:b+16];rgb=np.stack([[np.stack([v[int(i)] for i in h]) for v in views] for h in history])
    values.append(model.fuse_features(encoder,rgb,q[history]))
   torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();features=np.concatenate(values)
   assert features.shape==(len(anchors),4000) and np.isfinite(features).all()
   temporary=path.with_name(path.stem+'.tmp.npz')
   np.savez_compressed(temporary,features=features,input_indices=ids,anchors=anchors,times=times[anchors]);temporary.replace(path)
   ds.write_json(path.with_suffix('.json'),dict(fingerprint=key,sha256=ds.digest(path),history=contract,encoder=manifest['pretrained'],input_sha256=e['input_sha256']))
   status='extracted';del views
  target=ds.build_phase_targets(anchors,e['annotation']['segments'],e['length'],e['fps'],1.5)
  entries[e['split']].append(dict(episode=e,features=features,times=times[anchors],input_indices=ids,**target))
  index.append(dict(episode_id=e['episode_id'],path=str(path),fingerprint=key))
  stats.append(dict(episode_id=e['episode_id'],windows=len(anchors),decode_s=decode_s,extract_s=elapsed,peak_allocated_bytes=peak,status=status))
  ds.write_json(ROOT/'extraction_progress.json',dict(episodes_done=len(index),total=len(manifest['episodes']),elapsed_s=time.perf_counter()-started,episodes=stats))
  print('FEATURES',len(index),'/',len(manifest['episodes']),e['episode_id'],'windows',len(anchors),'seconds',round(elapsed,1),flush=True)
 ds.write_json(prepared/'feature_index.json',dict(manifest_sha256=ds.digest(prepared/'manifest.json'),history_contract=contract,episodes=index,pretrained=manifest['pretrained']))
 if encoder is not None:del encoder
 torch.cuda.empty_cache();campaign.ROOT=ROOT;campaign.REUSE_ROOT=None
 cfg=ds.load_config('configs/tomato_campaign94.yaml');cfg.update(history_sample_hz=5,feature_sampler_version=contract['sampler_version'],history_contract=contract)
 spec=dict(architecture='mlp',context=1,dimension=4000,horizon_s=1.5,lambda_future=.5,boundary_weight=0.,focal_gamma=0.)
 results=[campaign.train_one('mlp_16x5',spec,seed,entries,manifest,cfg) for seed in [42,43,44]]
 ds.write_json(ROOT/'results_index.json',dict(results=[dict(checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256']) for r in results],history_contract=contract))
 print('LONG HISTORY COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-root',default=str(ROOT));parser.add_argument('--reuse-cache-root');a=parser.parse_args();ROOT=Path(a.output_root);REUSE_CACHE=Path(a.reuse_cache_root) if a.reuse_cache_root else None;run()
