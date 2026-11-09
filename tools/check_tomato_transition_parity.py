"""Control batch-size numeric drift at all original2Hz timestamps."""
import numpy as np
from . import tomato_dataset as ds, tomato_metrics as metrics, tomato_boundary_metrics as b
from .audit_tomato_transitions import ROOT,PRIOR,MODELS,manifest,provider_replay


def run():
    _,episodes=manifest();old={n:np.load(PRIOR/'evaluations'/f'{n}_s42_val_logits.npz') for n in MODELS};offset=0;rows=[]
    for e in episodes:
        fresh=np.load(ROOT/'fresh4hz'/(e['episode_id']+'.npz'));a=np.array(e['anchors']);select=np.searchsorted(fresh['anchors'],a);n=len(a)
        for name,(cp,h) in MODELS.items():
            oldz={head+'_logits':old[name][head+'_logits'][offset:offset+n] for head in ['current','future']}
            newz={head+'_logits':fresh[name+'_'+head+'_logits'][select] for head in ['current','future']}
            row=dict(episode_id=e['episode_id'],model=name,gates={},events={})
            for head in ['current','future']:
                p=metrics.probabilities(oldz[head+'_logits']);q=metrics.probabilities(newz[head+'_logits'])
                for threshold in [.5,.8]:
                    row['gates'][head+str(threshold)]=int(np.sum((p.max(1)>=threshold)!=(q.max(1)>=threshold)))
                row['events'][head]=b.change_events(p,a/e['fps'],.5,2)==b.change_events(q,a/e['fps'],.5,2)
            oldp=provider_replay(e,a,oldz,h,ds.digest(cp));newp=provider_replay(e,a,newz,h,ds.digest(cp))
            row['provider_rows_identical']=oldp['rows']==newp['rows'];rows.append(row)
        offset+=n
    assert all(not any(r['gates'].values()) and all(r['events'].values()) and r['provider_rows_identical'] for r in rows)
    ds.write_json(ROOT/'operating_point_parity.json',dict(passed=True,rows=rows,scope='Fresh re-encoding at all2898 original2Hz anchors: confidence gates .5/.8, strict events and actual provider phase/warning/prompt outcomes identical despite BF16 numerical drift.'))
    print('ALL_ORIGINAL_OPERATING_POINTS_IDENTICAL')


if __name__=='__main__':run()
