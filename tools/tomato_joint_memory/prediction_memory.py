"""Five-fold source cross-fitting with an inner teacher early-stopping split."""
import random
import numpy as np
from .contracts import write
from .train import fit,predict_entries
from tools.tomato_metrics import probabilities


def folds(train,c):
    groups=sorted(e['episode']['source_recording_id'] for e in train)
    if len(set(groups))!=len(groups):raise ValueError('Duplicate train source')
    random.Random(c['oof']['seed']).shuffle(groups);out=[]
    for i in range(c['oof']['folds']):
        held=sorted(groups[i::c['oof']['folds']]);inner=[s for s in groups if s not in held];random.Random(c['oof']['seed']+i+1).shuffle(inner);n=max(1,round(len(inner)*c['oof']['inner_val_fraction']))
        record=dict(fold=i,train=sorted(inner[n:]),val=sorted(inner[:n]),held=held,seed=c['oof']['seed']+i)
        if set(record['train'])&set(record['held']) or set(record['val'])&set(record['held']):raise ValueError('OOF leakage')
        out.append(record)
    if sorted(s for f in out for s in f['held'])!=sorted(groups):raise ValueError('OOF coverage')
    return out


def crossfit(train,c,out,bindings):
    beliefs={};members=folds(train,c);write(out/'folds.lock.json',members)
    for fold in members:
        select=lambda key:[e for e in train if e['episode']['source_recording_id'] in fold[key]]
        folder=out/f"fold_{fold['fold']}"
        model,saved,meta,_=fit(select('train'),select('val'),c,'teacher',fold['seed'],folder,bindings)
        zz=predict_entries(model,saved,select('held'),c,'teacher')
        write(folder/'oof_membership.json',fold)
        for eid,z in zz.items():
            pp=np.concatenate([probabilities(z[h]) for h in ['current','future']],axis=1).astype(np.float32);beliefs[eid]=pp
            np.savez_compressed(folder/(eid+'.npz'),probabilities=pp)
        del model
    if set(beliefs)!={e['episode']['episode_id'] for e in train}:raise ValueError('OOF missing episodes')
    return beliefs
