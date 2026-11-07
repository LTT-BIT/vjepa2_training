"""Read-only experiment consolidation and numerical train-only normalization audit."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_metrics as metrics,tomato_boundary_metrics as bm,tomato_temporal_heads as tm
ROOT=Path('runs/tomato/temporal_final_20261108');BASE=Path('runs/tomato/temporal_20261108_v2');PRIOR=Path('runs/tomato/optimization_20261108')


def read_results():
 index=ds.read_json(BASE/'results_index.json');paths=[Path(r['checkpoint']).parent/'result.json' for r in index['experiments']+index['cv']]
 paths+=list((BASE/'mixtures').glob('*/result.json'))
 for directory in ['runs/tomato/current_velocity_20261108','runs/tomato/long_history_20261108_v2','runs/tomato/long_history_cv_20261108']:
  idx=Path(directory)/'results_index.json'
  if idx.exists():paths +=[Path(r['checkpoint']).parent/'result.json' for r in ds.read_json(idx)['results']]
 return [ds.read_json(p) for p in dict.fromkeys(paths)]


def near_boundary(entries,logits,head):
 result={};offsets=np.cumsum([0]+[len(r['anchors']) for r in entries])
 for width in [.25,.5,1.]:
  for side in ['before','after']:
   zs=[];ys=[]
   for j,r in enumerate(entries):
    b=np.array([v['time'] for v in bm.boundaries(r)]);d=r['times'][:,None]-b[None,:];signed=d[np.arange(len(d)),np.abs(d).argmin(1)]
    keep=(signed>=-width)&(signed<0) if side=='before' else (signed>=0)&(signed<=width)
    if head=='future':keep&=r['future_valid_mask']
    zs.append(logits[offsets[j]:offsets[j+1]][keep]);ys.append(r[head+'_labels'][keep])
   z=np.concatenate(zs);y=np.concatenate(ys);result[f'{side}_{width}s']=metrics.scores(z,y) if len(y) else None
 return result


def run():
 ROOT.mkdir(parents=True,exist_ok=True);results=read_results();manifest=ds.read_json(BASE/'prepared/manifest.json');val=[]
 for e in manifest['episodes']:
  if e['split']=='val':
   a=np.array(e['anchors']);val.append(dict(episode=e,times=a/e['fps'],**ds.build_phase_targets(a,e['annotation']['segments'],e['length'],e['fps'],1.5)))
 extra={}
 for r in results:
  if '_fold' in r['name']:continue
  p=Path(r['checkpoint']).parent/'val_logits.npz'
  with np.load(p,allow_pickle=False) as logits:extra[r['name']+'_s'+str(r['seed'])]={h:near_boundary(val,logits[h+'_logits'],h) for h in ['current','future']}
 ds.write_json(ROOT/'boundary_neighborhoods.json',extra)
 # Original cache arrays loaded once; derivative velocities checked from real state timestamps.
 index=ds.read_json(BASE/'prepared/feature_index.json');paths={v['episode_id']:v['path'] for v in index['episodes']};by_source={};velocity={}
 for e in manifest['episodes']:
  if e['split']=='test':continue
  path=Path(paths[e['episode_id']]);receipt=ds.read_json(path.with_suffix('.json'));assert ds.digest(path)==receipt['content_sha256']
  with np.load(path,allow_pickle=False) as z:
   by_source[e['source_recording_id']]=dict(episode=e,features=z['features'].copy(),input_indices=z['input_indices'].copy())
 statistics={};checks=[]
 for r in results:
  if r['name'] in ['hybrid_new_future','ensemble_half']:continue
  saved=torch.load(r['checkpoint'],map_location='cpu',weights_only=True);identity=saved['provenance'];spec=saved['spec'];ids=identity['split']['train']
  if saved['input_schema']['feature_sampler_version']=='tomato-flexible-history-v1':continue # Audited separately from new cache below.
  key=ds.fingerprint(dict(train_sources=ids,dimension=spec['dimension']))
  if key not in statistics:
   xs=[]
   for source in ids:
    item=by_source[source]
    if spec['architecture']=='velocity':
     if source not in velocity:velocity[source]=tm.append_velocity(item)
     xs.append(velocity[source])
    else:xs.append(item['features'])
   x=np.concatenate(xs);statistics[key]=(x.mean(0,dtype=np.float64).astype(np.float32),x.std(0,dtype=np.float64).clip(.01).astype(np.float32))
  mean,std=statistics[key]
  assert np.array_equal(mean,saved['mean'].numpy()) and np.array_equal(std,saved['std'].numpy())
  assert not set(ids)&set(manifest['splits']['test']) and not set(ids)&set(identity['split']['val'])
  checks.append(dict(name=r['name'],seed=r['seed'],train_only_statistics_exact=True,train_sources=len(ids),val_sources=len(identity['split']['val']),split_sha256=r['split_sha256']))
 # Verify new3s caches also bind actual measured joint order, not just feature dimension.
 long=Path('runs/tomato/long_history_20261108_v2');long_checks=[]
 if (long/'results_index.json').exists():
  m=ds.read_json(long/'prepared/manifest.json');idx=ds.read_json(long/'prepared/feature_index.json');pp={r['episode_id']:r['path'] for r in idx['episodes']};xs=[]
  for e in m['episodes']:
   path=Path(pp[e['episode_id']]);receipt=ds.read_json(path.with_suffix('.json'));assert ds.digest(path)==receipt['sha256']
   times,q=ds.read_state(e,m['contract']['joint_names'],'arms-and-hands')
   with np.load(path,allow_pickle=False) as z:
    assert np.all(z['input_indices']<=z['anchors'][:,None]);assert np.array_equal(z['features'][:,-928:],q[z['input_indices']].reshape(-1,928))
    if e['split']=='train':xs.append(z['features'].copy())
   long_checks.append(dict(episode_id=e['episode_id'],position_order_exact=True,causal=True,cache_sha256=ds.digest(path)))
  x=np.concatenate(xs);mean=x.mean(0,dtype=np.float64).astype(np.float32);std=x.std(0,dtype=np.float64).clip(.01).astype(np.float32)
  long_refs=ds.read_json(long/'results_index.json')['results']
  cv_index=Path('runs/tomato/long_history_cv_20261108/results_index.json')
  if cv_index.exists():long_refs+=ds.read_json(cv_index)['results']
  norm_cache={}
  source_features={}
  for e in m['episodes']:
   with np.load(pp[e['episode_id']],allow_pickle=False) as data:source_features[e['source_recording_id']]=data['features'].copy()
  for ref in long_refs:
   s=torch.load(ref['checkpoint'],weights_only=True,map_location='cpu');sources=s['provenance']['split']['train'];key=ds.fingerprint(sources)
   if key not in norm_cache:
    xx=np.concatenate([source_features[v] for v in sources]);norm_cache[key]=(xx.mean(0,dtype=np.float64).astype(np.float32),xx.std(0,dtype=np.float64).clip(.01).astype(np.float32))
   mean,std=norm_cache[key]
   assert np.array_equal(mean,s['mean'].numpy()) and np.array_equal(std,s['std'].numpy())
   checks.append(dict(name=Path(ref['checkpoint']).parent.name,seed=s['training_seed'],train_only_statistics_exact=True,train_sources=len(sources),val_sources=len(s['provenance']['split']['val']),feature_history='16x5Hz'))
 ds.write_json(ROOT/'normalization_audit.json',dict(passed=True,checks=checks,long_history=long_checks,no_test_sources_in_training=True,mean_std_exact=True))
 # Summaries retain each seed, confusion matrix and episode scores; full event grids remain local.
 compact=[]
 for r in results:
  heads={}
  for h in ['current','future']:
   v=r['evaluation'][h];heads[h]=dict(overall=v['overall'],per_episode=v['distance']['per_episode'],events={p:{k:x for k,x in e.items() if k not in ['per_episode','definition']} for p,e in v['events'].items()})
  compact.append({**{k:v for k,v in r.items() if k!='evaluation'},'evaluation':{**heads,**{h:v for h,v in r['evaluation'].items() if h.startswith('boundary')}}})
 ds.write_json(ROOT/'results_summary.json',compact)
 grouped={}
 for name in sorted({r['name'] for r in results}):
  rows=[r for r in results if r['name']==name]
  grouped[name]=dict(seeds=[r['seed'] for r in rows],heads={h:dict(mean=float(np.mean([r['evaluation'][h]['overall']['macro_f1'] for r in rows])),sample_std=float(np.std([r['evaluation'][h]['overall']['macro_f1'] for r in rows],ddof=1)) if len(rows)>1 else None,
   sample_variance=float(np.var([r['evaluation'][h]['overall']['macro_f1'] for r in rows],ddof=1)) if len(rows)>1 else None) for h in ['current','future']})
 ds.write_json(ROOT/'family_summary.json',grouped);print('ANALYSIS COMPLETE',len(results),'results',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-root',default=str(ROOT));a=parser.parse_args();ROOT=Path(a.output_root);run()
