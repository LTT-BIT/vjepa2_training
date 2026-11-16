"""Experimental readout session; no Prompt, action or Oracle input API."""
from collections import deque
import numpy as np
import torch
from .tomato_rgb_memory_model import RGBMemoryReader
from .tomato_rgb_memory_training import history_length,memory_scope


class RGBMemorySession:
    def __init__(self,checkpoint,*,session_id,episode_id,device='cpu',offline_replay=False):
        if not session_id or not episode_id:raise ValueError('Session and episode identity required')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved.get('architecture') not in ('old100_causal_rgb_memory_reader_v1','old100_rgb_memory_reader_v2') or saved.get('format_version')!=1:raise ValueError('Experimental RGB-memory checkpoint required')
        self.offline_only=bool(saved['protocol'].get('roi_uses_future_coordinates',False))
        if self.offline_only and not offline_replay:raise ValueError('Recorded r11 future-assisted ROI checkpoint is offline-replay-only; live ROI acceptance required')
        self.saved=saved;self.device=device;self.net=RGBMemoryReader(saved['state_mean'],saved['state_std'],saved['protocol']['hidden'],memory_scope=memory_scope(saved['method'])).to(device).eval();self.net.load_state_dict(saved['head']);self.session_id=session_id;self.episode_id=episode_id;self.signature=saved['identity']['cache_signature'];self.method=saved['method'];self.rows=deque(maxlen=history_length(self.method));self.last=None;self.last_arrival=None

    @torch.inference_mode()
    def observe(self,visual,positions,metadata,*,timestamp,arrival_timestamp,session_id,episode_id,cache_signature):
        if (session_id,episode_id,cache_signature)!=(self.session_id,self.episode_id,self.signature):return dict(valid=False,reason='identity_mismatch',suggests_prompt_update=False)
        try:
            t=float(timestamp);arrival=float(arrival_timestamp);v=np.asarray(visual,np.float32);q=np.asarray(positions,np.float32);m=np.asarray(metadata,np.float32)
            valid=(q.shape==(16,58) and v.shape==(4,1024) and m.shape==(116,) and np.isfinite(v).all() and np.isfinite(q).all() and np.isfinite(m).all() and np.isfinite([t,arrival]).all() and 0<=arrival-t<=1 and (self.last is None or t>self.last) and (self.last_arrival is None or arrival>=self.last_arrival) and abs(m[112])<1e-6 and np.allclose(m[80:96],np.linspace(-1.5,0,16),atol=.040001) and np.all(m[80:96]<=1e-6) and np.all(m[80:96]+m[96:112]<=1e-6) and np.all(m[96:112]<=.040001) and np.all(m[96:112]>=-.050001) and np.all(np.isin(m[64:80],[0,1])))
        except (TypeError,ValueError):valid=False
        if not valid:return dict(valid=False,reason='invalid_stale_or_reordered_window',suggests_prompt_update=False)
        if self.last is not None and t-self.last>.75:self.rows.clear()
        if self.method=='G3_local_only_history8':
            # Retain no older global RGB or measured-state history.
            for _, old_v, old_q, old_m in self.rows:
                old_v[:3]=0
                old_q[:]=np.tile(self.saved['state_mean'].numpy(),16)
                old_m[96:112]=0
        self.rows.append((t,v.copy(),q.reshape(928).copy(),m.copy()));self.last=t;self.last_arrival=arrival
        visual=np.stack([r[1] for r in self.rows])[None];q=np.stack([r[2] for r in self.rows])[None];meta=np.stack([r[3] for r in self.rows])[None]
        for k,r in enumerate(self.rows):meta[0,k,80:96]+=r[0]-t;meta[0,k,112]=r[0]-t
        if self.method=='G0_global_current':visual[:,:,3]=0;meta[:,:,:80]=0
        result=self.net(torch.from_numpy(visual).to(self.device),torch.from_numpy(q).to(self.device),torch.from_numpy(meta).to(self.device));p={k:result[k+'_logits'][0].softmax(-1).cpu().numpy() for k in ['current','future']}
        return dict(valid=True,offline_replay_only=self.offline_only,strict_online_causality_validated=bool(self.saved["protocol"].get("online_roi_validated",False)),observation_timestamp=t,model_version=self.saved['model_version'],labels=list(self.saved['input_schema']['labels']),future_horizon_s=self.saved['protocol']['future_horizon_s'],current_probabilities=p['current'],future_probabilities=p['future'],history_windows=len(self.rows),suggests_prompt_update=False,physical_success='unknown')
