"""Short causal readouts; consistency compares the SAME observation in two contexts."""
import numpy as np
import torch
from torch import nn


def contexts(times, length=4, gap_reset_s=.6):
    times=np.asarray(times)
    if len(times)>1 and np.any(np.diff(times)<=0):
        raise ValueError('Observations must be strictly increasing')
    result=np.full((len(times),length),-1,np.int64)
    start=0
    for i in range(len(times)):
        if i and times[i]-times[i-1]>gap_reset_s: start=i
        ids=np.arange(max(start,i-length+1),i+1)
        result[i,-len(ids):]=ids
    return result


class CausalReadouts(nn.Module):
    def __init__(self,dim=4000,hidden=128,dropout=.3):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(dropout))
        self.cell=nn.GRUCell(hidden,hidden)
        self.current=nn.Linear(hidden,4)

    def forward(self,x,valid):
        h=x.new_zeros((len(x),self.cell.hidden_size));outputs=[]
        for k in range(x.shape[1]):
            proposed=self.cell(self.project(x[:,k]),h)
            h=torch.where(valid[:,k,None],proposed,h)
            outputs.append(self.current(h))
        return torch.stack(outputs,1)


def supervised_loss(logits,labels,valid,cw,dense=False,earlier_weight=.25):
    weights=valid.to(logits.dtype)*earlier_weight
    weights[:,-1]=valid[:,-1].to(logits.dtype)
    if not dense: weights[:,:-1]=0
    ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),labels,weight=cw,reduction='none')
    return (ce*weights).sum()/(cw[labels]*weights).sum().clamp_min(1e-12)


def overlap_loss(a,b,ids_a,ids_b):
    """Adjacent context end times; align shared absolute observation IDs only."""
    valid=(ids_a[:,:-1]>=0)&(ids_a[:,:-1]==ids_b[:,1:])
    la=a[:,:-1].log_softmax(-1);lb=b[:,1:].log_softmax(-1)
    sym=.5*((la.exp()-lb.exp())*(la-lb)).sum(-1)
    return (sym*valid).sum()/valid.sum().clamp_min(1)
