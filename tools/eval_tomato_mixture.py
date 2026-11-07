"""Predeclared two-candidate inference-only extension, validation only."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_metrics as metrics,tomato_temporal_heads as tm,tomato_agent_mixture as mix
from . import train_tomato_temporal as campaign
from .tomato_runtime import initialize_runtime
ROOT=Path('runs/tomato/temporal_20261108_v2');PRIOR=Path('runs/tomato/optimization_20261108')


def run():
 initialize_runtime(dict(device='cuda',seed=42))
 selection=ds.read_json(ROOT/'selection.json');name=selection['winner']
 rule=dict(component_selection='Use winner of previously locked second-stage recipe rule, never best seed.',
 candidates=dict(hybrid_new_future=1.,ensemble_half=.5),fixed_current_source='prior Current-only same seed',
 legacy_future='prior Future-only1.5s same seed',head_horizon_s=1.5,seeds=[42,43,44],
 operating_point=dict(current_threshold=.5,current_k=2,future_threshold=.5,future_k=2),
 selection='Mean same rank and reliability gates as stage2; compare old independent1.5 with identical thresholds; retain old Joint stable default if gates fail.',
 motivation='Preserve stronger Current-only model while testing new future representation and fixed probability averaging. No extra training or encoded frames.',
 cv_scope='Mixtures evaluated on original validation only; no claim of cross-validation because fixed Current-only would leak into regrouped folds.')
 campaign.lock(ROOT/'mixture_protocol.json',rule)
 # Reuse verified val caches; labels from original reviewed boundaries at1.5s.
 m=ds.read_json(ROOT/'prepared/manifest.json');index=ds.read_json(ROOT/'prepared/feature_index.json');paths={v['episode_id']:v['path'] for v in index['episodes']};entries=[]
 for e in m['episodes']:
  if e['split']!='val':continue
  with np.load(paths[e['episode_id']],allow_pickle=False) as z:
   a=z['anchors'];entries.append(dict(episode=e,features=z['features'],times=z['times'],input_indices=z['input_indices'],**ds.build_phase_targets(a,e['annotation']['segments'],e['length'],e['fps'],1.5)))
 x=np.concatenate([r['features'] for r in entries]);results=[]
 for seed in [42,43,44]:
  prior=torch.load(PRIOR/'bundles'/f'independent_h15_s{seed}.pt',weights_only=True,map_location='cpu')
  newpath=next(Path(v['checkpoint']) for v in ds.read_json(ROOT/'results_index.json')['experiments'] if v['name']==name and v['seed']==seed)
  new=torch.load(newpath,weights_only=True,map_location='cpu')
  for variant,weight in rule['candidates'].items():
   out=ROOT/'mixtures'/f'{variant}_s{seed}';out.mkdir(parents=True,exist_ok=True)
   path=out/'agent.pt'
   payload=dict(format_version=5,architecture='tomato-agent-mixture-v1',current=prior['current'],legacy_future=prior['future'],new_future=new,new_future_weight=weight,
    sources=dict(independent_checkpoint_sha256=ds.digest(PRIOR/'bundles'/f'independent_h15_s{seed}.pt'),new_checkpoint_sha256=ds.digest(newpath),new_checkpoint=str(newpath)),
    protocol_sha256=ds.digest(ROOT/'mixture_protocol.json'),training_seed=seed)
   if not path.exists():torch.save(payload,path)
   saved,head=mix.load_checkpoint(path,device='cuda')
   from . import tomato_phase_model as model
   prediction=model.predict_heads(head,x,saved['mean'],saved['std']);evaluation=campaign.evaluate(prediction,entries,1.5)
   result=dict(name=variant,seed=seed,checkpoint=str(path),checkpoint_sha256=ds.digest(path),evaluation=evaluation,
       parameters=sum(p.numel() for p in head.parameters()),training_s=0.,extra_encoder_calls=0)
   np.savez_compressed(out/'val_logits.npz',**prediction);ds.write_json(out/'result.json',result);results.append(result)
 rs={n:[r for r in results if r['name']==n] for n in rule['candidates']};rows=[]
 for n,v in rs.items():
  c=np.mean([r['evaluation']['current']['overall']['macro_f1'] for r in v]);cr=np.mean([r['evaluation']['current']['events']['0.5/2']['recall'] for r in v]);d=np.mean([r['evaluation']['current']['events']['0.5/2']['delay_p95_s'] for r in v]);fr=np.mean([r['evaluation']['future']['events']['0.5/2']['recall'] for r in v]);ff=np.mean([r['evaluation']['future']['events']['0.5/2']['false_events_per_minute'] for r in v]);rank=np.mean([campaign.rank(r) for r in v])
  rows.append(dict(name=n,current_f1=float(c),current_recall=float(cr),current_p95_delay=float(d),future_recall=float(fr),future_false_per_min=float(ff),mean_rank=float(rank),feasible=bool(cr>=.9 and d<=1.5 and fr>=.8 and ff<=1)))
 baseline=[]
 for seed in [42,43,44]:
  p=ds.read_json(PRIOR/'evaluations'/f'independent_h15_s{seed}.json');baseline.append(dict(evaluation=p['heads']))
 baseline_rank=float(np.mean([campaign.rank(r) for r in baseline]))
 rows.append(dict(name='prior_independent_h15',mean_rank=baseline_rank,feasible=False,
                  note='Prior locked candidate failed reliable forecast gate; unchanged'))
 rows.sort(key=lambda v:(v['feasible'],v['mean_rank']),reverse=True);winner=rows[0]
 choice=dict(winner=winner['name'],candidates=rows,prior_independent_rank=baseline_rank,
  representative_seed=42,reliable_approved=winner['feasible'],no_independent_test=True,stable_default='runs/tomato/campaign94_20261107/experiments/joint_s42/best_head.pt',protocol_sha256=ds.digest(ROOT/'mixture_protocol.json'))
 campaign.lock(ROOT/'mixture_selection.json',choice);print(choice,flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-root',default=str(ROOT));a=parser.parse_args();ROOT=Path(a.output_root);run()
