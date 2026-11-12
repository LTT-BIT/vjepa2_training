"""Bounded-history evidence shadow adapter. Never writes stage/Prompt/robot state."""
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_transition_evidence import change_features,EvidenceReadout
from .tomato_state_recovery import PHASES


class EvidenceShadow:
    def __init__(self,bundle,session_id):
        p=torch.load(bundle,weights_only=True,map_location='cpu')
        if p.get('scope')!='shadow_only_not_selected_for_deployment' or p.get('architecture')!='six_conditional_linear_transition_evidence':raise ValueError('Identified shadow bundle required')
        s=p['feature_schema']
        if s['feature_source_sha256']!=ds.digest(Path(__file__).with_name('tomato_transition_evidence.py')):raise ValueError('Feature implementation identity mismatch')
        self.schema=s;self.arm=p['arm'];self.model=EvidenceReadout(p['input_dimension']);self.model.load_state_dict(p['state_dict']);self.model.eval();self.mean=p['mean'].numpy();self.std=p['std'].numpy()
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std<=0).any() or any(not torch.isfinite(v).all() for v in self.model.parameters()):raise ValueError('Invalid evidence parameters')
        self.version=ds.digest(bundle);self.reset(session_id)

    def reset(self,session_id):
        if not session_id:raise ValueError('Explicit session identity required')
        self.session_id=session_id;self.history=[];self.last_vector=None

    @torch.inference_mode()
    def observe(self,features,current,future,timestamp,online_phase,*,session_id,now=None):
        try:
            x=np.asarray(features,np.float32);c=np.asarray(current,np.float32);f=np.asarray(future,np.float32);t=float(timestamp);arrival=t if now is None else float(now)
            valid=(session_id==self.session_id and type(online_phase) is int and 0<=online_phase<=6
                and x.shape==(4000,) and np.isfinite(x).all() and all(p.shape==(4,) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-4 for p in [c,f])
                and np.isfinite([t,arrival]).all() and 0<=arrival-t<=1 and (not self.history or t>self.history[-1][0]))
        except (ValueError,TypeError):valid=False
        if not valid:
            self.history=[];self.last_vector=None
            return dict(valid=False,reason='invalid_stale_reordered_or_session_mismatch',suggests_prompt_update=False)
        if self.history and t-self.history[-1][0]>.75:self.history=[]
        self.history.append((t,x,c,f));cut=t-1.5
        # Keep latest at/before max lag plus subsequent observations; no future frame.
        older=[i for i,r in enumerate(self.history) if r[0]<=cut]
        if older:self.history=self.history[older[-1]:]
        times=np.array([r[0] for r in self.history]);a=change_features(np.stack([r[1] for r in self.history]),np.stack([r[2] for r in self.history]),np.stack([r[3] for r in self.history]),times,self.schema['joint_names'])[self.arm][-1]
        self.last_vector=a
        scores=self.model.all_phases(torch.from_numpy(((a-self.mean)/self.std)[None,:])).numpy()[0]
        return dict(valid=True,observed_at=t,session_id=self.session_id,model_version=self.version,
            confirmed_phase=PHASES[online_phase],candidate_phase=PHASES[online_phase+1] if online_phase<6 else None,
            evidence_score=float(scores[online_phase]) if online_phase<6 else None,score_is_calibrated_probability=False,
            history_span_s=float(times[-1]-times[0]),history_complete=bool(times[-1]-times[0]>=1.5),evidence_usable=bool(online_phase<6 and times[-1]-times[0]>=1.5),
            estimated_transition_time=None,suggests_prompt_update=False,physical_success='unknown',scope='shadow_only')
