"""Versioned raw-four-view bridge for future Memory checkpoints; B0 unchanged."""
import time
import numpy as np
from . import tomato_dataset as ds,tomato_phase_model as model
from .tomato_rgb_memory_inference import RGBMemorySession
from .tomato_agent_inference import PROTOCOL


class TomatoRGBMemoryAgent:
    requires_local_roi=True
    def __init__(self,checkpoint,pretrained_dir,device='cuda',*,offline_replay=False):
        self.checkpoint=checkpoint;self.device=device;self.offline_replay=offline_replay
        self.session=RGBMemorySession(checkpoint,session_id='loading',episode_id='loading',device=device,offline_replay=offline_replay)
        self.saved=self.session.saved;self.model_version=ds.digest(checkpoint);self.joint_names=self.saved['identity']['joint_names']
        self.contract='offline_bidirectional_local_rgb_v6' if self.session.offline_only else 'live-causal-mixed-hand-v2'
        if not offline_replay and not self.saved['protocol'].get('online_roi_validated',False):raise ValueError('Memory checkpoint has no accepted live ROI contract; offline replay only')
        if self.saved['input_schema']['labels']!=list(ds.LABELS):raise ValueError('Four-class output schema mismatch')
        expected=self.saved['identity'].get('pretrained')
        if expected is None:raise ValueError('Versioned encoder identity required in Memory checkpoint')
        self.encoder,provenance=model.load_encoder(pretrained_dir,device)
        if provenance!=expected:raise ValueError('Memory checkpoint/encoder identity mismatch')
        self.session_key=None

    def predict(self,rgb,joints,joint_names,*,roi_metadata,roi_contract,observation_id,session_id,timestamp,max_age_s=2.):
        started=time.time();rgb=np.asarray(rgb);q=np.asarray(joints,np.float32);meta=np.asarray(roi_metadata,np.float32)
        if rgb.dtype!=np.uint8 or rgb.shape!=(4,16,256,256,3) or q.shape!=(16,58) or meta.shape!=(116,):raise ValueError('Memory Agent requires four real video histories, named measured58 and ROI metadata116')
        if len(joint_names)!=58 or len(set(joint_names))!=58 or set(joint_names)!=set(self.joint_names):raise ValueError('Joint schema mismatch')
        if not observation_id or not session_id or not np.isfinite(timestamp) or not 0<=started-timestamp<=max_age_s:raise ValueError('Invalid or stale observation')
        if roi_contract!=self.contract:raise ValueError('ROI contract mismatch; never replace offline r11 with live crops silently')
        if self.session_key!=session_id:
            self.session=RGBMemorySession(self.checkpoint,session_id=session_id,episode_id=session_id,device=self.device,offline_replay=self.offline_replay);self.session_key=session_id
        q=q[:,[joint_names.index(n) for n in self.joint_names]];visual=model.encode_clips(self.encoder,rgb);r=self.session.observe(visual,q,meta,timestamp=timestamp,arrival_timestamp=started,session_id=session_id,episode_id=session_id,cache_signature=self.saved['identity']['cache_signature'])
        if not r['valid']:raise ValueError(r['reason'])
        result=dict(protocol=PROTOCOL,model_version=self.model_version,observation_id=observation_id,session_id=session_id,input_timestamp=timestamp,result_timestamp=time.time(),labels=list(ds.LABELS),prediction_mode='joint',future_horizon_s=self.saved['protocol']['future_horizon_s'],valid=True,error=None,control_eligible=False,requires_local_roi=True,roi_contract=self.contract,history_windows=r['history_windows'],offline_replay_only=self.offline_replay)
        # New Memory adapters remain observation/shadow only until separate full
        # chain acceptance; a model reload does not authorize robot execution.
        for h in ['current','future']:
            p=np.asarray(r[h+'_probabilities'],np.float64);p=p/p.sum();result[h+'_probabilities']=p.tolist();result[h+'_logits']=np.log(np.maximum(p,1e-300)).tolist()
        result['latency_ms']=(result['result_timestamp']-started)*1000
        return result
