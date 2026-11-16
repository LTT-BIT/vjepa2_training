from dataclasses import dataclass
import yaml
from ..learned_handshake.contracts import REPO,LABELS,STATES,H0_CONFIG,TASK_POLICY,read,write,sha,json_sha,lock,Observation,DecisionProposal
from pathlib import Path

@dataclass(frozen=True)
class GateDecision:
 candidate_stage:str|None
 action:str
 score:float
 observation_id:str
 observed_at:float
 available_at:float
 decision_policy_sha:str


def config(path):
 c=yaml.safe_load(Path(path).read_text())
 for k in ['baseline_root','memory_experiment','reference_experiment','output_root']:
  p=Path(c[k]);c[k]=str(p if p.is_absolute() else REPO/p)
 if c['groups']!=['G0','G1','G2','G3'] or c['seeds']!=[42,43,44] or c['history']!=4:raise ValueError('Unexpected primary experiment matrix')
 return c
