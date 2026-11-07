"""Current-specific finalist extension: verify the best Current pilot over three seeds."""
import argparse
from pathlib import Path
from . import tomato_dataset as ds,train_tomato_vjepa_phase as core,train_tomato_temporal as campaign
ROOT=Path('runs/tomato/current_velocity_20261108');BASE=Path('runs/tomato/temporal_20261108_v2')


def run():
 ROOT.mkdir(parents=True,exist_ok=True)
 protocol=dict(selection='Best Current-F1 fusion pilot (velocity seed42) receives seeds43/44; seed42 reused read-only.',
 objective='Current-specific confirmation, separate from previously locked forecast-oriented fusion selection.',
 fixed_training_recipe='Same Joint .5 Future weight,1.5s target,100 epochs/patience15; does not retune selected epoch for Current.',
 seeds=[42,43,44],test_used=False)
 campaign.lock(ROOT/'protocol.json',protocol)
 cfg=ds.load_config('configs/tomato_campaign94.yaml');manifest=core.load_locked_manifest(cfg);prepared=ROOT/'prepared';prepared.mkdir(exist_ok=True)
 for name in ['manifest.json','split.json','validation_receipt.json','validation_lock.json','feature_index.json']:campaign.lock(prepared/name,ds.read_json(BASE/'prepared'/name))
 entries=core.load_training_arrays({**cfg,'output_dir':str(prepared)},manifest);entries={s:entries[s] for s in ['train','val']}
 campaign.ROOT=ROOT;campaign.REUSE_ROOT=BASE
 spec=ds.read_json(BASE/'recipes.json')['velocity']
 results=[campaign.train_one('velocity',spec,seed,entries,manifest,cfg) for seed in [42,43,44]]
 ds.write_json(ROOT/'results_index.json',dict(results=[dict(checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256']) for r in results]))
 print('CURRENT VELOCITY CONFIRMATION COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-root',default=str(ROOT));a=parser.parse_args();ROOT=Path(a.output_root);run()
