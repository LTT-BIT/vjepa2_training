"""Re-evaluate exposed dev8 across the complete Agent decision chain; no tuning."""
from pathlib import Path
import argparse,json
import numpy as np
from . import tomato_dataset as ds
from .tomato_frozen_protocol import verify
from .tomato_full_chain import evaluate_episode,summarize_chain,episode_report
from .evaluate_tomato_complete100_history import SOURCE


def run(lock_path,out):
    if out.exists():raise FileExistsError('Preserve previous replay')
    lock=verify(lock_path);m=ds.read_json(lock['manifest']);out.mkdir(parents=True);summary={};inputs={}
    prior=ds.read_json('runs/tomato/complete100_history_20261111_final/receipt.json')
    for seed,model in lock['models'].items():
        folder=out/f'seed{seed}';folder.mkdir();per={}
        for e in m['episodes']:
            if e['split']!='val':continue
            p=SOURCE/f'seed{seed}'/(e['episode_id']+'.npz');sha=ds.digest(p);assert sha==prior['inputs'][str(p)];inputs[str(p)]=sha
            result,timeline,journal=evaluate_episode(e,np.load(p),model['model_version']);per[e['episode_id']]=result
            ds.write_json(folder/(e['episode_id']+'_analysis.json'),result)
            (folder/(e['episode_id']+'_report.md')).write_text(episode_report(e,result))
            for name,rows in [('timeline',timeline),('memory',journal)]:
                with (folder/(e['episode_id']+'_'+name+'.jsonl')).open('x') as f:
                    for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        summary[seed]=summarize_chain(per);ds.write_json(folder/'episodes.json',per)
    ds.write_json(out/'summary.json',summary);ds.write_json(out/'receipt.json',dict(lock_sha256=ds.digest(lock_path),inputs=inputs,training=False,new_independent_data_evaluated=False,
                 source_sha256={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/tomato_full_chain.py')]}))
    print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,default=Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.lock,a.out)
