"""Causal frozen-feature classifiers and binary boundary supervision."""
from collections import deque
import numpy as np
import torch
from torch import nn
from . import tomato_dataset as ds, tomato_metrics as metrics


def history_indices(length, context):
    if length<1 or context<1: raise ValueError('Positive episode/context length required')
    return np.maximum(np.arange(length)[:,None]-np.arange(context-1,-1,-1)[None,:],0)


def boundary_targets(episode, anchors, horizon):
    target=ds.build_phase_targets(anchors,episode['annotation']['segments'],episode['length'],episode['fps'],horizon)
    seg=metrics.continuous_phases(episode['annotation']['segments'])
    boundaries=np.array([s['start_frame'] for s in seg[1:]],dtype=np.int64)
    future=anchors+round(horizon*episode['fps'])
    y=((boundaries[None,:]>anchors[:,None])&(boundaries[None,:]<=future[:,None])).any(1).astype(np.float32)
    return y,target['future_valid_mask']


def append_velocity(entry):
    e=entry['episode'];times,_=ds.read_state(e,[n for n in e['info']['features']['observation.state']['names'] if n.endswith('.position')],'arms-and-hands')
    q=entry['features'][:,-928:].reshape(-1,16,58)
    ids=entry['input_indices'];delta=times[ids[:,-1]]-times[ids[:,-2]]
    velocity=np.divide(q[:,-1]-q[:,-2],delta[:,None],out=np.zeros((len(q),58),np.float32),where=delta[:,None]>0)
    return np.concatenate([entry['features'],velocity],axis=1).astype(np.float32)


class CausalConv(nn.Module):
    def __init__(self,dimension,dilation):
        super().__init__();self.padding=dilation
        self.conv=nn.Conv1d(dimension,dimension,2,dilation=dilation)
    def forward(self,x):
        return self.conv(nn.functional.pad(x,(self.padding,0)))


class TemporalHead(nn.Module):
    def __init__(self,spec):
        super().__init__();self.spec=dict(spec);arch=spec['architecture'];dim=spec['dimension']
        if arch=='gated':
            self.gate=nn.Linear(3072,3);nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
        if arch in ('gru','tcn'):
            self.projection=nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.Dropout(.3))
            if arch=='gru':self.sequence=nn.GRU(128,128,batch_first=True)
            else:self.sequence=nn.Sequential(CausalConv(128,1),nn.GELU(),nn.Dropout(.3),CausalConv(128,2),nn.GELU())
            hidden=128
        elif arch=='split':
            self.vision=nn.Sequential(nn.Linear(3072,128),nn.GELU())
            self.arms=nn.Sequential(nn.Linear(16*14,64),nn.GELU())
            self.hands=nn.Sequential(nn.Linear(16*44,64),nn.GELU())
            self.trunk=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Dropout(.3));hidden=256
        else:
            self.trunk=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Dropout(.3));hidden=256
        self.current=nn.Linear(hidden,4);self.future=nn.Linear(hidden,4)
        self.boundary=nn.Linear(hidden,1) if spec.get('boundary_weight',0)>0 else None

    def forward(self,x):
        arch=self.spec['architecture']
        if arch in ('gru','tcn'):
            z=self.projection(x)
            if arch=='gru':z=self.sequence(z)[0][:,-1]
            else:z=self.sequence(z.transpose(1,2))[:,:,-1]
        else:
            z=x[:,-1]
            if arch=='gated':
                gates=torch.softmax(self.gate(z[:,:3072]),-1)*3
                z=torch.cat([(z[:,:3072].reshape(-1,3,1024)*gates[:,:,None]).flatten(1),z[:,3072:]],1)
            elif arch=='vision_only':z=torch.cat([z[:,:3072],torch.zeros_like(z[:,3072:])],1)
            elif arch=='state_only':z=torch.cat([torch.zeros_like(z[:,:3072]),z[:,3072:]],1)
            elif arch=='split':
                q=z[:,3072:].reshape(-1,16,58)
                z=torch.cat([self.vision(z[:,:3072]),self.arms(q[:,:,:14].flatten(1)),self.hands(q[:,:,14:].flatten(1))],1)
            z=self.trunk(z)
        result=dict(current_logits=self.current(z),future_logits=self.future(z))
        if self.boundary is not None:result['boundary_logits']=self.boundary(z).squeeze(-1)
        return result


def normalized_features(entries, mean, std, spec):
    rows=[];indices=[];offset=0
    for r in entries:
        x=append_velocity(r) if spec['architecture']=='velocity' else r['features']
        rows.append((x-mean)/std);indices.append(history_indices(len(x),spec['context'])+offset);offset+=len(x)
    return np.concatenate(rows).astype(np.float32),np.concatenate(indices)


@torch.inference_mode()
def predict(head,x,indices,batch_size=256):
    device=next(head.parameters()).device;head.eval();parts={}
    x=torch.as_tensor(x,device=device)
    for ids in np.array_split(indices,max(1,int(np.ceil(len(indices)/batch_size)))):
        out=head(x[torch.as_tensor(ids,device=device)])
        for k,v in out.items():parts.setdefault(k,[]).append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in parts.items()}


def load_checkpoint(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format_version')!=4 or saved.get('architecture_version')!='tomato-causal-head-v1':raise ValueError('Invalid temporal checkpoint')
    if saved['input_schema']['label_order']!=list(ds.LABELS):raise ValueError('Label schema changed')
    spec=saved['spec'];d=spec['dimension']
    if saved['mean'].shape!=(d,) or saved['std'].shape!=(d,) or not torch.isfinite(saved['mean']).all() or not torch.isfinite(saved['std']).all() or torch.any(saved['std']<=0):raise ValueError('Invalid normalization')
    head=TemporalHead(spec);head.load_state_dict(saved['head'])
    if any(not torch.isfinite(p).all() for p in head.parameters()):raise ValueError('Nonfinite weights')
    return saved,head.to(device).eval()


class CausalFeatureSession:
    """Offline/deployment-identical FIFO. Never treats an old feature as a new observation."""
    def __init__(self,checkpoint,device='cpu'):
        self.saved,self.head=load_checkpoint(checkpoint,device);self.reset()
    def reset(self):self.rows=deque(maxlen=self.saved['spec']['context']);self.last_time=None
    def predict(self,feature,timestamp):
        if not np.isfinite(timestamp) or (self.last_time is not None and timestamp<=self.last_time):raise ValueError('Fresh monotonic feature timestamp required')
        if self.last_time is not None and timestamp-self.last_time>.6:self.rows.clear()
        row=np.asarray(feature,np.float32)
        if row.shape!=(self.saved['spec']['dimension'],) or not np.isfinite(row).all():raise ValueError('Feature schema mismatch')
        self.rows.append(row);self.last_time=timestamp
        values=[self.rows[0]]*(self.saved['spec']['context']-len(self.rows))+list(self.rows)
        x=(np.array(values)-self.saved['mean'].numpy())/self.saved['std'].numpy()
        return predict(self.head,x,np.arange(len(x))[None,:])
