"""Apply the previously locked global selection rule; package research artifacts only."""
import shutil
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_phase_model as model,tomato_independent_heads as dual
from . import tomato_causal_agent as causal,tomato_agent_mixture as mixture,analyze_tomato_temporal as analysis
ROOT=Path('runs/tomato/temporal_final_20261108');PRIOR=Path('runs/tomato/optimization_20261108')


def collect():
 rows=[r for r in analysis.read_results() if '_fold' not in r['name']]
 for seed in [42,43,44]:
  r=ds.read_json(PRIOR/'evaluations'/f'independent_h15_s{seed}.json')
  rows.append(dict(name='prior_independent_h15',seed=seed,evaluation=r['heads'],checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],parameters=2050568))
 return rows


def summary(name,rows,reference):
 avg=lambda h,k:float(np.mean([r['evaluation'][h]['overall'][k] for r in rows]))
 event=lambda h,k:float(np.mean([r['evaluation'][h]['events']['0.5/2'][k] for r in rows]))
 c=avg('current','macro_f1');f=avg('future','macro_f1');cr=event('current','recall');delay=event('current','delay_p95_s');fr=event('future','recall');false=event('future','false_events_per_minute')
 return dict(name=name,current_f1=c,future_f1=f,current_recall=cr,current_delay_p95_s=delay,future_recall=fr,future_false_per_min=false,
  future_miss_rate=event('future','miss_rate'),future_lead_p50_seed_mean_s=event('future','lead_p50_s'),rank=fr-.1*false+.01*c,
  feasible=bool(c>=reference-.005 and cr>=.9 and delay<=1.5 and fr>=.8 and false<=1))


def availability(event,head,horizon,latency):
 totals=[];false=0;duration=0.;offsets=[]
 for eid,e in event['per_episode'].items():
  used=set();events=e['events']
  for b in e['boundaries']:
   eligible=[]
   for i,v in enumerate(events):
    if i in used or v['previous']!=b['previous'] or v['label']!=b['label']:continue
    delivered=v['time']+latency
    good=(v['time']>=b['time']-horizon and delivered<b['time']) if head=='future' else (v['time']>=b['time'] and delivered<min(b['end'],b['time']+2.))
    if good:eligible.append((i,v,delivered))
   if eligible:
    i,v,delivery=eligible[0];used.add(i);offsets.append(delivery-b['time']);totals.append(1)
   else:totals.append(0)
  false+=len(events)-len(used);duration+=e['duration_s']
 return dict(recall=float(np.mean(totals)),miss_rate=float(1-np.mean(totals)),false_events_per_minute=false/(duration/60),
  lead_p50_s=float(np.median(-np.array(offsets))) if head=='future' and offsets else None,
  delay_p95_s=float(np.percentile(offsets,95)) if head=='current' and offsets else None,
  assumed_result_available_latency_s=latency,scope='Sensitivity, not measured transport; input-time causality retained and availability deadline also required.')


def run():
 protocol=ds.read_json(ROOT/'global_selection_protocol.json');rows=collect();groups={name:[r for r in rows if r['name']==name] for name in protocol['candidates']}
 for name,v in groups.items():
  if sorted(r['seed'] for r in v)!=[42,43,44]:raise ValueError('Final family missing three unique seeds: '+name)
 candidates=[summary(name,v,protocol['current_reference_mean_f1']) for name,v in groups.items()]
 candidates.sort(key=lambda r:(-int(r['feasible']),-r['rank'],r['name']));winner=candidates[0];chosen=next(r for r in groups[winner['name']] if r['seed']==42)
 selection=dict(winner=winner,all_candidates=candidates,checkpoint=chosen['checkpoint'],checkpoint_sha256=chosen['checkpoint_sha256'],representative_seed=42,
  protocol_sha256=ds.digest(ROOT/'global_selection_protocol.json'),reliable_validation_gate_passed=winner['feasible'],test_used=False,
  stable_default='/home/user/vjepa2_training/runs/tomato/campaign94_20261107/experiments/joint_s42/best_head.pt',no_real_deployment=True)
 from .train_tomato_temporal import lock
 lock(ROOT/'global_selection.json',selection)
 # Retain all64 threshold combinations as secondary sensitivity, without changing global selection.
 grid=[];points=[f'{p}/{k}' for p in [.5,.65,.8,.9] for k in [1,2]]
 for name,rs in groups.items():
  for cp in points:
   for fp in points:
    c=winner['current_f1'] if name==winner['name'] else float(np.mean([r['evaluation']['current']['overall']['macro_f1'] for r in rs]))
    cr=np.mean([r['evaluation']['current']['events'][cp]['recall'] for r in rs]);delay=np.mean([r['evaluation']['current']['events'][cp]['delay_p95_s'] for r in rs]);fr=np.mean([r['evaluation']['future']['events'][fp]['recall'] for r in rs]);ff=np.mean([r['evaluation']['future']['events'][fp]['false_events_per_minute'] for r in rs])
    grid.append(dict(name=name,current_point=cp,future_point=fp,current_f1=c,current_recall=float(cr),current_delay_p95_s=float(delay),future_recall=float(fr),future_false_per_min=float(ff),
      feasible=bool(c>=protocol['current_reference_mean_f1']-.005 and cr>=.9 and delay<=1.5 and fr>=.8 and ff<=1)))
 ds.write_json(ROOT/'threshold_diagnostics.json',dict(points=grid,feasible_count=sum(r['feasible'] for r in grid),selection_unchanged=True))
 available={}
 for name,rs in groups.items():
  available[name]={str(delay):{h:[availability(r['evaluation'][h]['events']['0.5/2'],h,1.5,delay) for r in rs] for h in ['current','future']} for delay in [.05,.1]}
 ds.write_json(ROOT/'availability_diagnostics.json',available)
 out=ROOT/'selected_research_bundle';out.mkdir(exist_ok=True);path=Path(chosen['checkpoint']);target=out/'agent.pt'
 if not target.exists():shutil.copyfile(path,target)
 assert ds.digest(target)==chosen['checkpoint_sha256']
 payload=torch.load(target,weights_only=True,map_location='cpu');version=payload['format_version']
 if version==3:
  saved,head=dual.load_checkpoint(target);runtime='tools.tomato_agent_inference.TomatoAgent'
  schema=ds.read_json(PRIOR/'selected_bundle/input_schema.json')
 elif version==4:
  saved,head=causal.load_checked(target);runtime='tools.tomato_causal_agent.CausalVideoAgent';schema=saved['input_schema']
 elif version==5:
  saved,head=mixture.load_checkpoint(target);runtime='tools.tomato_agent_mixture.MixtureVideoAgent';schema=ds.read_json(PRIOR/'selected_bundle/input_schema.json');schema['architecture']='mixture'
 else:raise ValueError('Unsupported packaged architecture')
 schema.setdefault('rgb',dict(shape=[3,16,256,256,3],dtype='uint8',color='RGB',cameras=['cam0','cam1','cam2']))
 schema.setdefault('joint_dtype','float32')
 np.savez(out/'normalization.npz',mean=saved['mean'].numpy(),std=saved['std'].numpy())
 ds.write_json(out/'input_schema.json',schema)
 config=dict(runtime_class=runtime,checkpoint=str(target.resolve()),model_version=ds.digest(target),
  pretrained_dir='/home/user/vjepa2_training/pretrained/vjepa2-vitl-fpc64-256',pretrained=saved['pretrained'],device='cuda',prediction_hz=2,
  future_horizon_s=1.5,max_age_s=2.,analysis_operating_point=protocol['fixed_operating_point'],
  openpi_existing_provider_unchanged=dict(threshold=.8,confirmations=2,current_confirmations=2,future_warning_confirmations=1,future_warning_requires_expected_next_class=True),research_only=True,analysis_operating_point_scope='Classifier change-event evaluation only; not automatically applied to existing provider',
  stable_default=selection['stable_default'],selection_protocol_sha256=selection['protocol_sha256'])
 ds.write_json(out/'inference_config.json',config)
 ds.write_json(out/'file_manifest.json',{p.name:ds.digest(p) for p in sorted(out.iterdir()) if p.name!='file_manifest.json'})
 print('GLOBAL SELECT',selection,flush=True)

if __name__=='__main__':run()
