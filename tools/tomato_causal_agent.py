"""Research-only causal-head runtime. Existing OpenPI/RTC and legacy Agent stay unchanged."""
import time
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_temporal_heads as temporal,tomato_phase_model as model
from .tomato_agent_inference import PROTOCOL


def load_checked(path,device='cpu'):
 saved,head=temporal.load_checkpoint(path,device)
 schema=saved['input_schema'];spec=saved['spec'];names=schema['joint_names']
 if (len(names)!=58 or len(set(names))!=58 or any(not n.endswith('.position') for n in names)
     or schema['state_history_shape']!=[16,58] or schema['feature_hz']!=2 or schema['causal'] is not True
     or schema['feature_sampler_version'] not in (ds.SAMPLER_VERSION,'tomato-flexible-history-v1') or saved['pretrained']['sha256']!=model.WEIGHTS_SHA256
     or saved['pretrained']['config_sha256']!=model.CONFIG_SHA256 or saved['feature_version']!=model.FEATURE_VERSION
     or saved['normalization_split']!='train only' or saved['provenance']['spec']!=spec
     or saved['provenance']['seed']!=saved['training_seed']):raise ValueError('Temporal input/provenance schema mismatch')
 if schema['feature_sampler_version']=='tomato-flexible-history-v1':
  from .tomato_time_sampling import contract
  if schema.get('history_sample_hz')!=5 or schema.get('history_contract')!=contract(16,5,False):raise ValueError('Flexible history sampling contract changed')
 elif schema.get('history_sample_hz',10)!=10:raise ValueError('Legacy sample rate changed')
 valid_arch=('mlp','gru','tcn','gated','velocity','split','vision_only','state_only')
 if spec['architecture'] not in valid_arch or spec['dimension']!=(4058 if spec['architecture']=='velocity' else 4000):raise ValueError('Temporal architecture/dimension mismatch')
 if spec['context']!=(4 if spec['architecture'] in ('gru','tcn') else 1):raise ValueError('Temporal context changed')
 if schema['arms_indices']!=list(range(14)) or schema['hands_indices']!=list(range(14,58)):raise ValueError('Joint group order changed')
 expected=[f'{side}_fr3_joint{i}.position' for side in ['left','right'] for i in range(1,8)]
 if names[:14]!=expected:raise ValueError('Arm joint name/order changed')
 return saved,head


class CausalVideoAgent:
 def __init__(self,checkpoint,pretrained_dir,device='cuda'):
  self.saved,_=load_checked(checkpoint)
  self.encoder,actual=model.load_encoder(pretrained_dir,device,42)
  if actual!=self.saved['pretrained']:raise ValueError('Frozen encoder identity mismatch')
  self.session=temporal.CausalFeatureSession(checkpoint,device)
  self.version=ds.digest(checkpoint);self.names=self.saved['input_schema']['joint_names'];self.session_id=None
 def reset(self,session_id):
  if not isinstance(session_id,str) or not session_id:raise ValueError('Explicit session identity required')
  self.session.reset();self.session_id=session_id
 def predict(self,rgb,joints,joint_names,history_timestamps,*,timestamp,observation_id,session_id,max_age_s=2.):
  started=time.time()
  if session_id!=self.session_id or not observation_id or not np.isfinite(timestamp) or not 0<=started-timestamp<=max_age_s:raise ValueError('Invalid session/observation time')
  names=list(joint_names);q=np.asarray(joints,np.float32);ts=np.asarray(history_timestamps,float)
  if len(names)!=58 or len(set(names))!=58 or set(names)!=set(self.names) or q.shape!=(16,58):raise ValueError('58 named positions required')
  if ts.shape!=(16,) or not np.isfinite(ts).all() or np.any(np.diff(ts)<0) or np.any(ts>timestamp+1e-6) or abs(ts[-1]-timestamp)>1e-4:raise ValueError('Causal history timestamps required')
  if self.session.last_time is not None and timestamp<=self.session.last_time:raise ValueError('Fresh monotonic observation required')
  if self.saved['spec']['context']>1 and self.session.last_time is not None:
   gap=timestamp-self.session.last_time
   if 0<gap<.425:raise ValueError('Temporal head trained at2Hz; higher-rate context requires explicit resampling/retraining')
  q=q[:,[names.index(n) for n in self.names]]
  feature=model.fuse_features(self.encoder,np.asarray(rgb)[None],q[None])[0]
  if self.saved['spec']['architecture']=='velocity':
   dt=ts[-1]-ts[-2];velocity=(q[-1]-q[-2])/dt if dt>0 else np.zeros(58,np.float32)
   feature=np.r_[feature,velocity].astype(np.float32)
  result=self.session.predict(feature,timestamp)
  response=dict(protocol=PROTOCOL,model_version=self.version,observation_id=observation_id,session_id=session_id,
   input_timestamp=timestamp,result_timestamp=time.time(),labels=list(ds.LABELS),future_horizon_s=self.saved['spec']['horizon_s'],prediction_mode='joint',valid=True,error=None,
   architecture=self.saved['spec']['architecture'],research_only=True)
  for h in ['current','future']:
   z=result[h+'_logits'][0].astype(float);p=np.exp(z-z.max());p/=p.sum();response[h+'_logits']=z.tolist();response[h+'_probabilities']=p.tolist()
  if 'boundary_logits' in result:response['switch_within_horizon_probability']=float(1/(1+np.exp(-result['boundary_logits'][0])))
  response['latency_ms']=(response['result_timestamp']-started)*1000
  return response
