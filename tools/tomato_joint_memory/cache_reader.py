"""One causal constructor for randomized training samples and streaming inference."""
from collections import deque
import numpy as np

class CausalBuffer:
    def __init__(self,length,gap=.75):
        if length not in [1,4,8]:raise ValueError('Unsupported history')
        self.length=length;self.gap=gap;self.identity=None;self.values=deque(maxlen=length);self.last_arrival=None
    def reset(self,episode,session,model):
        if not all(isinstance(v,str) and v for v in [episode,session,model]):raise ValueError('Explicit nonempty identity required')
        self.identity=(episode,session,model);self.values.clear();self.last_arrival=None
    def push(self,x,t,episode,session,model,valid=True,available_at=None):
        x=np.asarray(x,dtype=np.float32)
        if self.identity!=(episode,session,model):raise ValueError('Identity mismatch; explicit episode/session/model reset required')
        if not valid or x.shape!=(4000,) or not np.isfinite(x).all() or not np.isfinite(t):raise ValueError('Invalid evidence')
        if available_at is not None and (not np.isfinite(available_at) or available_at<t or available_at-t>self.gap):raise ValueError('Expired/noncausal evidence')
        arrival=t if available_at is None else available_at
        if self.last_arrival is not None and arrival<self.last_arrival:raise ValueError('Reversed arrival clock')
        if self.values and t<=self.values[-1][0]:raise ValueError('Duplicate/reversed observation')
        if self.values and t-self.values[-1][0]>self.gap+1e-9:self.values.clear()
        self.values.append((float(t),x.copy()));self.last_arrival=arrival
        v=list(self.values);n=len(v);a=np.zeros((self.length,4000),np.float32);mask=np.zeros(self.length,bool);dt=np.zeros(self.length,np.float32)
        a[-n:]=[x for _,x in v];mask[-n:]=True;dt[-n:]=[time-t for time,_ in v]
        return a,mask,dt

def history_indices(times,length,gap=.75):
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid causal timeline')
    ids=np.full((len(times),length),-1,np.int64);dt=np.zeros((len(times),length),np.float32);start=0
    for i,t in enumerate(times):
        if i and t-times[i-1]>gap+1e-9:start=i
        j=np.arange(max(start,i-length+1),i+1);ids[i,-len(j):]=j;dt[i,-len(j):]=times[j]-t
    return ids,ids>=0,dt

def causal_motion(features,times,gap=.75):
    """Named measured-position differences, no action or velocity columns."""
    q=np.asarray(features)[:,-58:];out=np.zeros_like(q);d=np.diff(times)
    good=(d>0)&(d<=gap+1e-9);ii=np.flatnonzero(good)+1
    out[ii]=(q[ii]-q[ii-1])/d[good,None]
    return out.astype(np.float32)

def past_beliefs(probabilities,times,gap=.75):
    p=np.asarray(probabilities,dtype=np.float32)
    if p.shape!=(len(times),8) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.reshape(-1,2,4).sum(-1),1,atol=1e-5):raise ValueError('Not Joint beliefs')
    out=np.zeros((len(p),9),np.float32);ii=np.flatnonzero(np.diff(times)<=gap+1e-9)+1
    out[ii,:8]=p[ii-1];out[ii,8]=1
    return out

def ema(probabilities,times,alpha=.6,gap=.75):
    p=np.asarray(probabilities);out=np.empty_like(p)
    for i in range(len(p)):
        out[i]=p[i] if i==0 or times[i]-times[i-1]>gap+1e-9 else alpha*p[i]+(1-alpha)*out[i-1]
    return out
