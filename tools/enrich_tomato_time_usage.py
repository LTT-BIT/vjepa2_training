from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from tools import tomato_dataset as ds,tomato_boundary_metrics as bm
from tools.evaluate_tomato_reviewed_boundaries import revise_episode
from tools.tomato_transition_time import survival_targets
root=Path('runs/tomato/time_usage_analysis_20261109');src=Path('runs/tomato/transition_time_20261109');x=ds.read_json(root/'analysis.json');old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');by={e['episode_id']:e for e in old['episodes']};props=ds.read_json('docs/validation/tomato_reviewed_boundaries_20261109/proposals.json')['truth_proposals'];detail={}
for seed in [42,43,44]:
 r=ds.read_json(src/f'seed{seed}/result.json');run=r['external_validation_thresholds_are_ablation_not_selection'][str(r['selected_threshold'])];lost=[];pos=defaultdict(list);neg=defaultdict(list)
 for rec in ds.read_json(src/'data_identity.json')['episodes']:
  if rec['split']!='val':continue
  eid=rec['episode_id'];e=revise_episode(by[eid],props.get(eid,[]));f=np.load(src/f'seed{seed}'/(eid+'.npz'));t=survival_targets(e['annotation']['segments'],f['anchors'],e['fps'],e['length']);risk=1-f['transition_probabilities'][:,-1];bs=bm.boundaries(dict(episode=e))
  for i,ts in enumerate(f['times']):
   future=next((b for b in bs if b['time']>ts+1e-9),None)
   if future is None:continue
   typ=ds.LABELS[future['previous']]+'->'+ds.LABELS[future['label']]
   (pos if future['time']-ts<=1.5 else neg)[typ].append(float(risk[i]))
  trace=next(v['trace'] for v in run['decision_replay'] if v['episode_id']==eid)
  for er in x['seeds'][str(seed)]['B1_hit_B2_miss']:
   if er['episode_id']!=eid:continue
   b=bs[er['ordinal']-1];window=[t for t in trace if b['time']-1.5<=t['time']<b['time']];expected=er['ordinal']-1;prior=[p for p in run['per_episode']['B2'][eid]['events'] if p['previous']==b['previous'] and p['label']==b['label'] and p['time']<b['time']-1.5]
   reason='no_expected_online_phase_in_window' if not any(t.get('phase_index')==expected for t in window) else ('earlier_one_shot_warning_not_reissued' if prior else 'risk_future_confirmation_or_phase_timing')
   lost.append(dict(**er,mechanism=reason,window_online_phases=[t.get('phase_index') for t in window]))
 detail[str(seed)]=dict(B1_hits_lost_by_B2=lost,mechanism_counts=dict(Counter(z['mechanism'] for z in lost)),true_near_event_risk={k:dict(positive_windows=len(v),positive_mean=float(np.mean(v)),negative_mean=float(np.mean(neg[k]))) for k,v in pos.items()})
ds.write_json(root/'mechanisms.json',detail)
print('B2 losses',detail['42']['mechanism_counts'],'conditional risks',detail['42']['true_near_event_risk'])
