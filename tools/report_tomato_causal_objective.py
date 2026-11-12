"""Recording-paired objective report, with unchanged frozen B0 reference."""
import argparse
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds,tomato_metrics as metrics
from .train_tomato_boundary_supervision import evaluate,BASE


def run(root):
    cfg=ds.read_json(root/'protocol.json');results=ds.read_json(root/'results.json');manifest=ds.read_json('runs/tomato/complete100_20261111/dataset_manifest.json')
    seen=set(ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json')['splits']['train']);rows=[]
    for e in manifest['episodes']:
        if e['split']=='test':continue
        y=ds.build_phase_targets(np.array(e['anchors']),e['annotation']['segments'],e['length'],e['fps'],1.5)['current_labels']
        rows.append(dict(episode=e,labels=y))
    cohorts={'oof84':[r for r in rows if r['episode']['split']=='train'],'oof10_base_unseen_exposed':[r for r in rows if r['episode']['split']=='train' and r['episode']['source_recording_id'] not in seen],'development8_exposed':[r for r in rows if r['episode']['split']=='val']}
    for seed in cfg['seeds']:
        pred={r['episode']['episode_id']:np.load(BASE/f'seed{seed}'/f'{r["episode"]["episode_id"]}_input.npz')['current'] for r in rows}
        results[str(seed)]['B0_frozen']=dict(evaluation={co:evaluate(rr,pred,seed) for co,rr in cohorts.items()})
    methods=['B0_frozen']+cfg['methods'];tables={};differences={};boundary={}
    for co,rr in cohorts.items():
        tables[co]={}
        for method in methods:
            ss=[results[str(s)][method]['evaluation'][co]['summary'] for s in cfg['seeds']];f1=np.array([s['classification']['macro_f1'] for s in ss]);events=[x for seed in cfg['seeds'] for x in results[str(seed)][method]['evaluation'][co]['events']];offset=np.array([x['offset_s'] for x in events if x['offset_s'] is not None])
            tables[co][method]=dict(current_f1_mean=float(f1.mean()),current_f1_seed_std=float(f1.std(ddof=1)),per_seed=[dict(seed=seed,summary=s) for seed,s in zip(cfg['seeds'],ss)],prompt_counts={key:sum(s['prompt_counts'].get(key,0) for s in ss) for key in ['on_time','early','late','missing']},within_delay={str(t):sum(s['within_delay'][str(t)] for s in ss) for t in [1.,1.5,2.]},mismatch_s=sum(s['mismatch_s'] for s in ss),severe_early_gt_1s=int((offset < -1).sum()),late_gt_1p5_s=int((offset>1.5).sum()),late_gt_2_s=int((offset>2).sum()),absolute_error_median_s=float(np.median(abs(offset))),prompt_by_transition={tr:{key:sum(s['prompt_by_transition'].get(tr,{}).get(key,0) for s in ss) for key in ['on_time','early','late','missing']} for tr in ['pickup->harvest','harvest->return','return->adjust','adjust->pickup']})
        # Paired recording bootstrap averages seeds BEFORE resampling recordings.
        differences[co]={}
        for method in cfg['methods']:
            for ref in ['B0_frozen','T0_last']:
                diffs=[];recording=[]
                for r in rr:
                    eid=r['episode']['episode_id'];delta=[]
                    for seed in cfg['seeds']:
                        a=results[str(seed)][method]['evaluation'][co]['per_episode'][eid]['prompt']['events'];b=results[str(seed)][ref]['evaluation'][co]['per_episode'][eid]['prompt']['events']
                        delta.append((sum(x['outcome']=='on_time' for x in a)-sum(x['outcome']=='on_time' for x in b))/6*100)
                    diffs.append(np.mean(delta));recording.append(dict(episode_id=eid,source_recording_id=r['episode']['source_recording_id'],delta_on_time_percentage_points=float(np.mean(delta))))
                v=np.array(diffs);rng=np.random.default_rng(20261112);boot=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
                differences[co][method+'_minus_'+ref]=dict(mean_percentage_points=float(v.mean()),recording_bootstrap_95_percent_interval=np.percentile(boot,[2.5,97.5]).tolist(),recordings=recording,scope='exposed diagnostic, not independent generalization evidence')
    for seed in cfg['seeds']:
        boundary[str(seed)]={}
        for method in methods:
            zall=[];yall=[];distall=[]
            for r in cohorts['development8_exposed']:
                e=r['episode'];eid=e['episode_id'];base=np.load(BASE/f'seed{seed}'/f'{eid}_input.npz');c=base['current'] if method=='B0_frozen' else np.load(root/f'seed{seed}'/f'{method}_{eid}.npz')['current']
                bounds=np.array([s['start_frame']/e['fps'] for s in e['annotation']['segments'][1:]])
                zall.append(np.log(np.maximum(c,1e-30)));yall.append(r['labels']);distall.append(np.min(abs(base['times'][:,None]-bounds),axis=1))
            z=np.concatenate(zall);y=np.concatenate(yall);dist=np.concatenate(distall)
            boundary[str(seed)][method]={str(t):metrics.scores(z[dist<=t],y[dist<=t]) for t in [.25,.5,1.]}
    ds.write_json(root/'comparison.json',dict(tables=tables,recording_paired_differences=differences,development_boundary_quality=boundary,warning='OOF is Current-head-only: fixed B0 Future has seen 74 training sources; unseen10 also previously explored. No blind evaluation. Future and its strict warnings remain unchanged.'))
    ds.write_json(root/'evaluations_with_b0.json',results)
    print({co:{m:dict(f1=round(v['current_f1_mean'],5),prompt=v['prompt_counts'],mismatch=v['mismatch_s'],within1p5=v['within_delay']['1.5']) for m,v in table.items()} for co,table in tables.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
