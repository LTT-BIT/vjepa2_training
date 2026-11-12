"""Held-out same-timestamp consistency, distinct from transition correctness."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_runtime import initialize_runtime
from .tomato_causal_objective import CausalReadouts,contexts


@torch.inference_mode()
def run(root):
    initialize_runtime(dict(device='cuda',seed=42));cfg=ds.read_json(root/'protocol.json');manifest=ds.read_json('runs/tomato/complete100_20261111/dataset_manifest.json');by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']};results={}
    for seed in cfg['seeds']:
        results[str(seed)]={}
        for method in cfg['methods']:
            saved=torch.load(root/f'seed{seed}'/f'{method}_final84.pt',map_location='cpu',weights_only=True);net=CausalReadouts(4000,cfg['hidden'],cfg['dropout']).cuda().eval();net.load_state_dict(saved['head']);symmetric=[];flips=[];per={}
            for e in manifest['episodes']:
                if e['split']!='val':continue
                z=np.load(by[e['episode_id']]['feature_path']);raw=(z['features']-saved['mean'].numpy())/saved['std'].numpy();x=torch.from_numpy(np.concatenate([np.zeros((1,4000),np.float32),raw])).cuda();indices=contexts(z['times']);prior=np.full_like(indices,-1)
                for i in range(1,len(indices)):
                    if z['times'][i]-z['times'][i-1]<=cfg['gap_reset_s']:prior[i]=indices[i-1]
                kl=[];change=[]
                for start in range(0,len(indices),cfg['batch_size']):
                    a=torch.from_numpy(indices[start:start+cfg['batch_size']]).cuda();b=torch.from_numpy(prior[start:start+cfg['batch_size']]).cuda();valid=(a[:,:-1]>=0)&(a[:,:-1]==b[:,1:]);pa=net(x[a+1],a>=0)[:,:-1].log_softmax(-1);pb=net(x[b+1],b>=0)[:,1:].log_softmax(-1)
                    kl.extend((.5*((pa.exp()-pb.exp())*(pa-pb)).sum(-1))[valid].cpu().tolist());change.extend((pa.argmax(-1)!=pb.argmax(-1))[valid].cpu().tolist())
                per[e['episode_id']]=dict(same_timestamp_kl_mean=float(np.mean(kl)),same_timestamp_argmax_disagreement=float(np.mean(change)),pairs=len(kl));symmetric+=kl;flips+=change
            results[str(seed)][method]=dict(same_timestamp_kl_mean=float(np.mean(symmetric)),same_timestamp_argmax_disagreement=float(np.mean(flips)),pairs=len(flips),per_episode=per)
    ds.write_json(root/'consistency_diagnostic.json',results);print(results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
