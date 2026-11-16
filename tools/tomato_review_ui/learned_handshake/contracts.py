from dataclasses import dataclass
from pathlib import Path
import hashlib,json,subprocess
import yaml
from ..handshake import DEFAULT,LABELS

REPO=Path(__file__).resolve().parents[3]
H0_CONFIG={**DEFAULT,'late_ack_enabled':True,'late_ack_timeout_s':5.}
TASK_POLICY={'pickup':['harvest'],'harvest':['return'],'return':['adjust'],'adjust':['pickup']}
STATES=['READY','CANDIDATE_PENDING','SWITCH_REQUESTED','WAIT_CURRENT_ACK','WAIT_LATE_ACK','RECOVERY_REQUIRED']

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def json_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def lock(p,v):
 if Path(p).exists() and read(p)!=v:raise ValueError('Immutable protocol mismatch: '+str(p))
 if not Path(p).exists():write(p,v)
def config(p):
 c=yaml.safe_load(Path(p).read_text())
 for k in ['baseline_root','memory_experiment','output_root']:
  v=Path(c[k]);c[k]=str(v if v.is_absolute() else REPO/v)
 return c

@dataclass(frozen=True)
class DecisionProposal:
 observation_id:str
 observed_at:float
 available_at:float
 candidate_stage:str|None
 request_commit:bool
 proposal_score:float
 reason:str
 model_version:str
 policy_version:str

@dataclass(frozen=True)
class Observation:
 session_id:str
 observation_id:str
 model_version:str
 policy_version:str
 observed_at:float
 available_at:float
 current:tuple
 future:tuple
 valid:bool=True
