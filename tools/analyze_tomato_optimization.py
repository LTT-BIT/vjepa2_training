"""Recording-level uncertainty and train-only gradient diagnostics; no test inference."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds, tomato_metrics as metrics, tomato_phase_model as model
from . import train_tomato_vjepa_phase as core
from .tomato_runtime import initialize_runtime
ROOT=Path('runs/tomato/optimization_20261108')


def audit_extra():
    parent=ds.read_json(ROOT/'prepared_h10/manifest.json');episodes=parent['episodes'];overlaps=[];duplicates=[]
    for i,a in enumerate(episodes):
        for b in episodes[i+1:]:
            if a['split']==b['split']: continue
            ae,be=a['entry'],b['entry']
            if max(ae['source_start_ns'],be['source_start_ns'])<min(ae['source_end_ns'],be['source_end_ns']):
                overlaps.append([a['episode_id'],b['episode_id']])
            ah={v for k,v in a['input_sha256'].items() if k.endswith(('.parquet','.mp4'))}
            bh={v for k,v in b['input_sha256'].items() if k.endswith(('.parquet','.mp4'))}
            if ah&bh: duplicates.append(dict(a=a['episode_id'],b=b['episode_id'],shared_content_sha256=sorted(ah&bh)))
    out=dict(recorded_wall_clock_interval_overlap_across_splits=overlaps,
             exact_video_or_state_duplicate_across_splits=duplicates,
             interpretation='Recorded acquisition timestamps and exact source content hashes; does not detect re-encoded duplicates or unknown clock errors.')
    ds.write_json(ROOT/'cross_split_content_audit.json',out)


def bootstrap():
    # Source-clustered (one episode per source), paired resampling. Exploratory intervals after selection.
    rng=np.random.default_rng(20261108);all_results={}
    for family in ('independent_h10','independent_h15','joint_transition'):
        comparisons={}
        for h in ('current','future'):
            differences=[];boundary_delta=[];episode_ids=None
            for seed in (42,43,44):
                new=ds.read_json(ROOT/'evaluations'/f'{family}_s{seed}.json')['heads'][h]
                old=ds.read_json(ROOT/'evaluations'/f'joint_s{seed}.json')['heads'][h]
                ids=sorted(new['distance']['per_episode']);episode_ids=ids
                differences.append([new['distance']['per_episode'][i]['macro_f1']-old['distance']['per_episode'][i]['macro_f1'] for i in ids])
                # Same fixed threshold .8/2. Horizon 1.5 vs 1s is an intervention bundle, not architecture-only.
                ne=new['events']['0.8/2']['per_episode'];oe=old['events']['0.8/2']['per_episode']
                boundary_delta.append([np.mean([x['detected'] for x in ne[i]['boundaries']])-np.mean([x['detected'] for x in oe[i]['boundaries']]) for i in ids])
            d=np.mean(differences,axis=0);b=np.mean(boundary_delta,axis=0);draws=rng.integers(0,len(d),size=(5000,len(d)))
            comparisons[h]=dict(equal_episode_f1_delta=float(d.mean()),episode_f1_delta_ci95=np.percentile(d[draws].mean(1),[2.5,97.5]).tolist(),
                                equal_episode_boundary_recall_delta=float(b.mean()),boundary_recall_delta_ci95=np.percentile(b[draws].mean(1),[2.5,97.5]).tolist(),
                                per_episode_delta=dict(zip(episode_ids,d.tolist())))
        all_results[family]=comparisons
    ds.write_json(ROOT/'paired_recording_bootstrap.json',dict(results=all_results,replicates=5000,seed=20261108,
        warning='Exploratory paired cluster bootstrap on only 10 reused validation recordings; not independent confirmatory evidence, no correction for candidate search. Absent classes count zero in fixed four-class episode Macro-F1. h15 changes both horizon and architecture.'))


def gradients():
    initialize_runtime(dict(device='cpu',seed=42));torch.set_num_threads(1)
    p=ds.read_json(ROOT/'prepared_h10/manifest.json');idx=ds.read_json(ROOT/'prepared_h10/feature_index.json')
    paths={e['episode_id']:e['path'] for e in idx['episodes']};xs=[];yc=[];yf=[];valid=[]
    for e in p['episodes']:
        if e['split']!='train':continue
        path=Path(paths[e['episode_id']]);receipt=ds.read_json(path.with_suffix('.json'))
        assert ds.digest(path)==receipt['content_sha256']
        with np.load(path,allow_pickle=False) as data:
            targets=ds.build_phase_targets(data['anchors'],e['annotation']['segments'],e['length'],e['fps'],1.)
            xs.append(data['features']);yc.append(targets['current_labels']);yf.append(targets['future_labels']);valid.append(targets['future_valid_mask'])
    x=np.concatenate(xs);c=np.concatenate(yc);f=np.concatenate(yf);v=np.concatenate(valid)
    weights=[torch.from_numpy(core.fit_class_weights(y)[0]) for y in (c,f[v])]
    batches=[a for a in np.array_split(np.random.default_rng(42).permutation(np.flatnonzero(v)),24)]
    out={}
    for seed in (42,43,44):
        saved,h=model.load_checkpoint(f'runs/tomato/campaign94_20261107/experiments/joint_s{seed}/best_head.pt');h.eval()
        cos=[]
        for ids in batches:
            ids=ids[:128];z=torch.from_numpy((x[ids]-saved['mean'].numpy())/saved['std'].numpy());logits=h(z)
            losses=[torch.nn.functional.cross_entropy(logits[name+'_logits'],torch.from_numpy(y[ids]),weight=w)
                    for name,y,w in [('current',c,weights[0]),('future',f,weights[1])]]
            gs=[]
            for loss in losses:
                gs.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss,tuple(h.trunk.parameters()),retain_graph=True)]))
            cos.append(float(torch.nn.functional.cosine_similarity(gs[0],gs[1],dim=0)))
        out[str(seed)]=dict(cosines=cos,mean=float(np.mean(cos)),negative_fraction=float(np.mean(np.array(cos)<0)))
    ds.write_json(ROOT/'gradient_probe.json',dict(results=out,batches=24,batch_size_max=128,split='train',dropout='eval disabled',
        interpretation='Gradient alignment at selected shared-trunk checkpoints only; negative values show local competing objectives, cannot establish cause of validation gap or compare training trajectories. No optimizer step or checkpoint mutation.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-root',default=str(ROOT))
    args=parser.parse_args();ROOT=Path(args.output_root)
    audit_extra();bootstrap();gradients();print('DIAGNOSTICS COMPLETE')
