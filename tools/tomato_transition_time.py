"""Causal next-class-transition survival targets, hazard likelihood and additive schema.

Four logits are conditional risks. Five masses are mutually exclusive time bins;
the last is survival beyond1.5s, not a label assigned to unknown episode tails.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from . import tomato_dataset as ds

EDGES=np.array([.25,.5,1.,1.5])
PHASES=('pickup_1','harvest_1','return_truss','adjust','pickup_2','harvest_2','return_stem')
CLASSES=(0,1,2,3,0,1,2)


def survival_targets(segments,anchors,fps,length):
    if fps<=0 or length<=0:raise ValueError('Invalid recording')
    # Only actual changes of operation class, not metadata cuts with same class.
    boundaries=np.array([b['start_frame']/fps for a,b in zip(segments,segments[1:]) if a['label']!=b['label']])
    times=np.asarray(anchors)/fps;end=(length-1)/fps
    if np.any(times>end) or np.any(times<0):raise ValueError('Anchor outside episode')
    labels=np.zeros((len(times),4),np.float32);mask=np.zeros_like(labels);delta=np.full(len(times),np.nan);observed=np.zeros(len(times),bool)
    for i,t in enumerate(times):
        future=boundaries[boundaries>t+1e-9];follow=max(0.,end-t)
        if len(future):
            dt=float(future[0]-t);delta[i]=dt;observed[i]=True
            if dt<=1.5+1e-9:
                j=int(np.searchsorted(EDGES,dt-1e-9,side='left'));mask[i,:j+1]=1;labels[i,j]=1
            else:mask[i,:]=1
        else:
            # Only fully observed bins provide survival evidence; partial-bin censoring
            # is discarded conservatively, never assigned to the >1.5s class.
            mask[i]=EDGES<=follow+1e-9
    return dict(labels=labels,mask=mask,time_to_event=delta,event_observed=observed,followup=end-times)


def hazard_nll(logits,labels,mask):
    if logits.shape!=labels.shape or logits.shape!=mask.shape or logits.shape[-1]!=4:raise ValueError('Invalid hazard target shape')
    per=nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')*mask
    valid=mask.sum(1)>0
    return per.sum(1)[valid].mean() if valid.any() else logits.sum()*0


def event_distribution(logits):
    risk=torch.sigmoid(logits);survival=torch.cumprod(1-risk,dim=-1)
    prior=torch.cat([torch.ones_like(survival[...,:1]),survival[...,:-1]],-1)
    return torch.cat([risk*prior,survival[...,-1:]],-1)


class TransitionTimeHead(nn.Module):
    def __init__(self,dimension=4000):
        super().__init__();self.net=nn.Sequential(nn.Linear(dimension,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,4))
    def forward(self,x):return self.net(x)


def transition_fields(logits, timestamp,version):
    if not math.isfinite(timestamp) or not version:raise ValueError('Invalid identity')
    p=event_distribution(torch.as_tensor(logits,dtype=torch.float32))
    if p.shape!=(4+1,) or not torch.isfinite(p).all():raise ValueError('Invalid hazard logits')
    return dict(transition_time=dict(schema_version='tomato-transition-time-v1',model_version=version,input_timestamp=timestamp,bin_upper_s=[.25,.5,1.,1.5,None],probabilities=p.tolist(),semantics='next operation-class transition; final bin survival beyond1.5s'))


@dataclass
class PredictiveTransitionDecision:
    threshold:float=.5
    confirmations:int=2
    max_age:float=1.
    def __post_init__(self):
        if not 0<self.threshold<1 or self.confirmations<2 or not math.isfinite(self.max_age) or self.max_age<=0:raise ValueError('Invalid policy')
        self.phase=0;self.last=-float('inf');self.count=0;self.warn_count=0;self.warned=False;self.window=None;self.prompt_version=0;self.session_id=None;self.model_identity=None
    def update(self,current,future,masses,timestamp,now,transition_timestamp=None):
        invalid=dict(valid=False,confirmed_phase=PHASES[self.phase],suggest_switch=False,warning=False,reason='invalid_or_stale')
        try:
            c=np.asarray(current,dtype=float);f=np.asarray(future,dtype=float);p=np.asarray(masses,dtype=float);timestamp=float(timestamp);now=float(now)
        except (ValueError,TypeError):
            self.count=0;self.warn_count=0;self.window=None;return invalid
        if transition_timestamp is not None and transition_timestamp!=timestamp:
            self.count=0;self.warn_count=0;self.window=None;return invalid
        if not np.isfinite([timestamp,now]).all() or timestamp<=self.last or now<timestamp or now-timestamp>self.max_age:
            self.count=0;self.warn_count=0;self.window=None;return invalid
        if any(a.shape!=shape or not np.isfinite(a).all() or (a<0).any() or abs(a.sum()-1)>1e-4 for a,shape in [(c,(4,)),(f,(4,)),(p,(5,))]):
            self.count=0;self.warn_count=0;self.window=None;return invalid
        if timestamp-self.last>.75:self.count=0;self.warn_count=0;self.window=None
        self.last=timestamp;changed=False;warning=False;interval=None
        if self.phase<6:
            target=CLASSES[self.phase+1];risk=1-p[-1]
            evidence=int(f.argmax())==target and f[target]>=.5 and risk>=self.threshold
            self.warn_count=self.warn_count+1 if evidence else 0
            if self.warn_count>=self.confirmations:
                j=int(p[:4].argmax());lo=0. if j==0 else float(EDGES[j-1]);hi=float(EDGES[j]);interval=[timestamp+lo,timestamp+hi]
                # Freeze first forecast interval: do not slide an existing event forward
                # indefinitely. Expire before accepting another candidate forecast.
                if self.window is None or timestamp>self.window[1]+.5:self.window=interval
                if not self.warned:warning=True;self.warned=True
            self.count=self.count+1 if int(c.argmax())==target and c[target]>=.8 else 0
            # Forecast alone cannot change Prompt. Require currentK2 and arrival inside
            # a previously forecast interval (plus the2Hz quantization allowance).
            if self.count>=self.confirmations and self.window is not None and self.window[0]<=timestamp<=self.window[1]+.5:
                self.phase+=1;self.prompt_version+=1;changed=True;self.count=0;self.warn_count=0;self.warned=False;self.window=None
        return dict(valid=True,confirmed_phase=PHASES[self.phase],phase_index=self.phase,next_phase=PHASES[self.phase+1] if self.phase<6 else None,expected_transition_interval=interval or self.window,warning=warning,suggest_switch=changed,prompt_version=self.prompt_version,confidence=float(1-p[-1]),reason='current_k2_and_forecast_arrival' if changed else 'warning_separate_from_confirmation')


    def update_response(self,response,now):
        """Versioned inference adapter; no user prompts/chunks or robot actions executed."""
        field=response.get('transition_time',{})
        if not response.get('valid') or response.get('labels')!=list(ds.LABELS) or field.get('schema_version')!='tomato-transition-time-v1' or not field.get('model_version') or field.get('input_timestamp') is None:
            self.count=0;self.warn_count=0;self.window=None
            return dict(valid=False,confirmed_phase=PHASES[self.phase],suggest_switch=False,warning=False,reason='invalid_schema')
        session=response.get('session_id')
        if not session or not response.get('model_version'):
            self.count=0;self.warn_count=0;self.window=None
            return dict(valid=False,suggest_switch=False,warning=False,reason='missing_identity')
        if self.session_id not in (None,session):self.__post_init__()
        self.session_id=session
        identity=(response['model_version'],field['model_version'])
        if self.model_identity is not None and self.model_identity!=identity:
            self.count=0;self.warn_count=0;self.window=None
            return dict(valid=False,suggest_switch=False,warning=False,reason='model_changed_requires_reset')
        self.model_identity=identity
        return self.update(response.get('current_probabilities',[]),response.get('future_probabilities',[]),field.get('probabilities',[]),response.get('input_timestamp',float('nan')),now,transition_timestamp=field.get('input_timestamp'))
