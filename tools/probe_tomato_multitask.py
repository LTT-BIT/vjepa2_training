"""Local gradient and recording-cluster diagnostics; no optimizer updates or test predictions."""
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_temporal_heads as tm,train_tomato_vjepa_phase as core
ROOT=Path('runs/tomato/temporal_final_20261108');BASE=Path('runs/tomato/temporal_20261108_v2')


def run():
 torch.set_num_threads(1);m=ds.read_json(BASE/'prepared/manifest.json');idx=ds.read_json(BASE/'prepared/feature_index.json');paths={x['episode_id']:x['path'] for x in idx['episodes']};xs=[];cs=[];fs=[];vs=[]
 for e in m['episodes']:
  if e['split']!='train':continue
  with np.load(paths[e['episode_id']],allow_pickle=False) as data:
   t=ds.build_phase_targets(data['anchors'],e['annotation']['segments'],e['length'],e['fps'],1.5);xs.append(data['features'].copy());cs.append(t['current_labels']);fs.append(t['future_labels']);vs.append(t['future_valid_mask'])
 x=np.concatenate(xs);c=np.concatenate(cs);f=np.concatenate(fs);valid=np.concatenate(vs);weights=[torch.from_numpy(core.fit_class_weights(y)[0]) for y in [c,f[valid]]]
 rng=np.random.default_rng(20261108);batches=[rng.choice(np.flatnonzero(valid),128,replace=False) for _ in range(12)];results={}
 index=ds.read_json(BASE/'results_index.json')['experiments']
 for name in ['mlp_h15','loss11']:
  for seed in [42,43,44]:
   path=next(r['checkpoint'] for r in index if r['name']==name and r['seed']==seed);s,h=tm.load_checkpoint(path);cos=[]
   for ids in batches:
    z=torch.from_numpy((x[ids]-s['mean'].numpy())/s['std'].numpy())[:,None,:];out=h(z);grads=[]
    for key,y,w in [('current',c,weights[0]),('future',f,weights[1])]:
     loss=torch.nn.functional.cross_entropy(out[key+'_logits'],torch.from_numpy(y[ids]),weight=w)
     grads.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss,tuple(h.trunk.parameters()),retain_graph=True)]))
    cos.append(float(torch.nn.functional.cosine_similarity(grads[0],grads[1],dim=0)))
   results[name+'_s'+str(seed)]=dict(mean_cosine=float(np.mean(cos)),negative_fraction=float(np.mean(np.array(cos)<0)),cosines=cos)
 ds.write_json(ROOT/'multitask_gradient_probe.json',dict(results=results,batches=12,batch_size=128,split='train',dropout='disabled',weights='same sqrt inverse class CE from training labels',
  limits='Selected checkpoint local diagnostic only, not a causal attribution of generalization or an estimate of full training-trajectory conflict.'))
 print('GRADIENT PROBE COMPLETE',flush=True)

if __name__=='__main__':run()
