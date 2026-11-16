import numpy as np
from .targets import label,sample_weights
from .contract import write,json_sha


def dataset(episodes,rollouts,shift=0.):
 seq=[];mask=[];labels=[];sources=[];keys=[];counts={}
 for e in episodes:
  eid=e['metadata']['episode_id'];count=0
  for i,r in enumerate(rollouts[eid]['records']):
   target=label(e['metadata'],r['feature'],shift)
   if not target['eligible']:continue
   seq.append(r['feature']['sequence']);mask.append(r['feature']['mask']);labels.append(target);sources.append(e['metadata']['source_recording_id']);keys.append((eid,i));count+=1
  counts[eid]=count
 if not seq:raise ValueError('No executable candidate-supervision ticks')
 return dict(sequence=np.asarray(seq,np.float32),mask=np.asarray(mask,bool),labels=labels,sources=sources,keys=keys,source_ticks=counts,weights=sample_weights(sources,labels))


def statistics(data_list):
 values=np.concatenate([d['sequence'][d['mask']] for d in data_list]);return dict(mean=values.mean(0,dtype=np.float64).astype(np.float32),std=values.std(0,dtype=np.float64).clip(.05).astype(np.float32))


def aggregate(base,own,seed):
 rng=np.random.default_rng(seed);n=len(base['labels']);ids=rng.choice(len(own['labels']),size=n,replace=len(own['labels'])<n)
 out={key:np.concatenate([base[key],own[key][ids]]) for key in ['sequence','mask']};out['labels']=base['labels']+[own['labels'][i] for i in ids];out['sources']=base['sources']+[own['sources'][i] for i in ids];out['keys']=base['keys']+[own['keys'][i] for i in ids];out['weights']=sample_weights(out['sources'],out['labels']);out['own_sampling']=dict(original=len(own['labels']),selected=n,replacement=len(own['labels'])<n,indices_sha256=json_sha(ids.tolist()));return out


def audit(data):
 labels=data['labels'];return dict(eligible_ticks=len(labels),source_count=len(set(data['sources'])),gate_censored_masked=int(sum(not r['gate_valid'] for r in labels)),positives=int(sum(r['gate'] for r in labels if r['gate_valid'])),negatives=int(sum(not r['gate'] for r in labels if r['gate_valid'])),observed_boundaries=int(sum(r['observed'] for r in labels)),censored=int(sum(r['censored'] for r in labels)),hazard_masked=int(sum(not r['hazard_valid'] for r in labels)),bins=[int(sum(r['hazard_bucket']==i for r in labels)) for i in range(6)],labels_sha256=json_sha([{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()} for r in labels]),sample_keys_sha256=json_sha(data['keys']))
