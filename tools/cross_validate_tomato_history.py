"""Conditional grouped CV for the longer sensory window, no original test sources."""
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds,train_tomato_temporal as campaign
ROOT=Path('runs/tomato/long_history_cv_20261108');LONG=Path('runs/tomato/long_history_20261108_v2');BASE=Path('runs/tomato/temporal_20261108_v2')
RULE=dict(folds=3,group='original source_recording_id',fold_seed=20261108,training_seed=42,
 criterion='Run if long-history mean Current or Future F1 exceeds matched10Hz MLP by .002, or mean online rank exceeds it by .02; otherwise skip.',
 scope='Diagnostic grouped CV on86 train+validation recordings, never original8 test sources; fixed-split final candidates already use3 seeds.')


def run():
 ROOT.mkdir(parents=True,exist_ok=True);campaign.lock(ROOT/'protocol.json',RULE)
 oldidx=ds.read_json(BASE/'results_index.json')['experiments'];old=[ds.read_json(Path(r['checkpoint']).parent/'result.json') for r in oldidx if r['name']=='mlp_h15']
 new=[ds.read_json(Path(r['checkpoint']).parent/'result.json') for r in ds.read_json(LONG/'results_index.json')['results']]
 avg=lambda rs,h:np.mean([r['evaluation'][h]['overall']['macro_f1'] for r in rs])
 improve=(avg(new,'current')>=avg(old,'current')+.002 or avg(new,'future')>=avg(old,'future')+.002 or np.mean([campaign.rank(r) for r in new])>=np.mean([campaign.rank(r) for r in old])+.02)
 if not improve:
  ds.write_json(ROOT/'results_index.json',dict(skipped=True,reason='Predeclared improvement screen not met',results=[]));print('LONG CV SKIPPED BY SCREEN');return
 m=ds.read_json(LONG/'prepared/manifest.json');idx=ds.read_json(LONG/'prepared/feature_index.json');paths={r['episode_id']:r['path'] for r in idx['episodes']};entries={s:[] for s in ['train','val']}
 for e in m['episodes']:
  with np.load(paths[e['episode_id']],allow_pickle=False) as data:
   entries[e['split']].append(dict(episode=e,features=data['features'].copy(),times=data['times'].copy(),input_indices=data['input_indices'].copy(),**ds.build_phase_targets(data['anchors'],e['annotation']['segments'],e['length'],e['fps'],1.5)))
 prepared=ROOT/'prepared';prepared.mkdir(exist_ok=True)
 for name in ['manifest.json','feature_index.json']:campaign.lock(prepared/name,ds.read_json(LONG/'prepared'/name))
 cfg=ds.load_config('configs/tomato_campaign94.yaml');cfg.update(history_sample_hz=5,feature_sampler_version=m['sampler_version'],history_contract=m['history_contract'])
 spec=dict(architecture='mlp',context=1,dimension=4000,horizon_s=1.5,lambda_future=.5,boundary_weight=0.,focal_gamma=0.)
 allrows=entries['train']+entries['val'];sources=sorted(r['episode']['source_recording_id'] for r in allrows);np.random.default_rng(20261108).shuffle(sources)
 campaign.ROOT=ROOT;campaign.REUSE_ROOT=None;results=[]
 for fold,group in enumerate(np.array_split(np.array(sources),3)):
  validation=set(group.tolist());split={s:[r for r in allrows if (r['episode']['source_recording_id'] in validation)==(s=='val')] for s in ['train','val']}
  results.append(campaign.train_one(f'mlp_16x5_fold{fold}',spec,42,split,m,cfg,'group_cv'))
 ds.write_json(ROOT/'results_index.json',dict(skipped=False,results=[dict(checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256']) for r in results]))
 print('LONG CV COMPLETE',flush=True)

if __name__=='__main__':run()
