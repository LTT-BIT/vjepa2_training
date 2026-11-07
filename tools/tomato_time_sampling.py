"""Explicit causal sampler identity; no changes to legacy cache or runtime schema."""
import math
import numpy as np
from . import tomato_dataset as ds


def history_offsets(frames=16,sample_hz=10,nonuniform=False):
    if frames not in (8,16) or sample_hz not in (5,10):raise ValueError('Unsupported history design')
    if nonuniform:
        if frames!=16:raise ValueError('Nonuniform design requires16 frames')
        # Dense recent .1s, older .2/.4s; endpoint current frame retained.
        return np.array([3.2,2.8,2.4,2.,1.6,1.4,1.2,1.,.8,.6,.5,.4,.3,.2,.1,0.])
    return np.arange(frames-1,-1,-1)/sample_hz


def sample(times,anchor,frames=16,sample_hz=10,nonuniform=False):
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0) or not np.isfinite(times).all() or not 0<=anchor<len(times):raise ValueError('Invalid timeline')
    offsets=history_offsets(frames,sample_hz,nonuniform)
    ids=np.maximum(0,np.searchsorted(times[:anchor+1],times[anchor]-offsets+1e-6,side='right')-1)
    ids[-1]=anchor
    assert ids.max()<=anchor and np.all(np.diff(ids)>=0)
    return ids.astype(np.int64)


def contract(frames=16,sample_hz=10,nonuniform=False):
    return dict(sampler_version='tomato-flexible-history-v1',frames=frames,sample_hz=sample_hz,
                offsets_s=history_offsets(frames,sample_hz,nonuniform).tolist(),latest_frame_retained=True,causal=True)


def horizon_frames(requested_s,fps):
    if requested_s<=0 or fps<=0 or not np.isfinite([requested_s,fps]).all():raise ValueError('Invalid horizon')
    offset=int(math.ceil(requested_s*fps-1e-8))
    return dict(requested_s=requested_s,offset_frames=offset,effective_s=offset/fps,
                mapping='First recorded frame at or after requested future time (ceil); no future observations used')
