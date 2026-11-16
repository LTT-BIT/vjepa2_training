"""Causal cache history construction; no class/position/Prompt is an input."""
import numpy as np
import torch


def history_indices(times,length,gap=.75):
    times=np.asarray(times);out=np.full((len(times),length),-1,np.int64);first=0
    if len(times)>1 and np.any(np.diff(times)<=0):raise ValueError('Strict increasing history times required')
    for i in range(len(times)):
        if i and times[i]-times[i-1]>gap:first=i
        ids=np.arange(max(first,i-length+1),i+1);out[i,-len(ids):]=ids
    return out


def gather_inputs(visual,positions,metadata,times,indices,method):
    """Input banks have a zero sentinel at0; indices are logical zero-based rows."""
    valid=indices>=0;ix=indices+1;v=visual[ix];q=positions[ix];meta=metadata[ix].clone();delta=times[ix]-times[indices[:,-1]+1,None]
    meta[:,:,80:96]+=delta[:,:,None];meta[:,:,112]=delta
    if method=='G0_global_current':
        v=v.clone();v[:,:,3]=0;meta[:,:,:80]=0
    if (meta[:,:,112][valid]>1e-6).any() or (meta[:,:,80:96][valid]>1e-6).any():raise ValueError('Future timestamp entered causal history')
    return dict(visual=v,positions=q,metadata=meta,history_valid=valid)
