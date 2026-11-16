"""Fixed fault schedules depend on source anchor index or causal first commit, never GT."""
import copy
from .contract import *
from .replay import run
from .score import score,aggregate

SCENARIOS=['delay_010','delay_030','long_gap','duplicate','reorder','cross_session','wrong_model','held','future_single','future_run','current_flip','current_loss','blackout_to_hard']

def inject(rows,scenario):
 result=[]
 for i,original in enumerate(rows):
  r=copy.deepcopy(original);r['available_at']=round(r['observed_at']+.09,9)
  if scenario=='long_gap' and i%97 in range(30,36):continue
  if scenario.startswith('delay_'):r['available_at']+=.1 if scenario=='delay_010' else .3
  if scenario=='reorder' and i%41==10:r['available_at']+=.6
  if scenario=='cross_session' and i%41==10:r['session_id']='injected-other-session'
  if scenario=='wrong_model' and i%41==10:r['model_version']='injected-other-model'
  if scenario=='held' and i%41==10 and i>0:r['observed_at']=rows[i-1]['observed_at']
  if (scenario=='future_single' and i%37==10) or (scenario=='future_run' and i%97 in [30,31,32]) or (scenario=='current_flip' and i%41 in [10,11]):
   h='current' if scenario=='current_flip' else 'future';p=r['heads'][h]['probabilities'];stage=LABELS[(max(range(4),key=lambda j:p[LABELS[j]])+1)%4];r['heads'][h]['probabilities']={s:.97 if s==stage else .01 for s in LABELS}
  if scenario=='current_loss' and i%41 in [10,11]:r['valid']=False
  result.append(r)
  if scenario=='duplicate' and i%41==10:result.append(copy.deepcopy(r))
 return result


def execute(parts,policies,out,save):
 summary={}
 for scenario in SCENARIOS:
  summary[scenario]={}
  for name,policy in policies.items():
   reports=[];triggered=0
   for e in parts['val']:
    m=e['metadata'];r=run(inject(e['rows'],scenario),m['length']/m['fps'],policy,m['episode_id'],blackout_after_commit=scenario=='blackout_to_hard');triggered+=r['stress_blackout_first_commit'] is not None;reports.append(save(out,name,'stress_'+scenario,e,r,save_records=False))
   a=aggregate(reports);a.update(cross_session_commit=sum(r['cross_session_commit'] for r in reports),non_planning_commit=sum(r['non_planning_commit'] for r in reports),blackout_triggered_episodes=triggered);summary[scenario][name]=a
  print('STRESS',scenario,'PASS',flush=True)
 write(out/'stress/summary.json',summary);write(out/'stress/schedule.json',dict(scenarios=SCENARIOS,definition='see locked stress.py SHA: source anchor modular indices; blackout begins first actual causal commit for5.5s; no GT timing; available delay, never interpolate',seed=42,protected_clocks=H0_CONFIG,note='Faulty but syntactically valid predictions may change legal decisions; protocol guarantees do not imply semantic fault detection. All metrics descriptive, never selection inputs.'))
 return summary
