"""Correct inherited descriptive manifest fields without changing trained tensors.

Executed manifests/checkpoints are immutable. Finalized inference exports get fresh
paths and SHA identities, with explicit exact-tensor equivalence receipts.
"""
from pathlib import Path
import copy
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_independent_heads import pack,load_checkpoint
ROOT=Path('runs/tomato/refresh101_20261109_v2')


def corrected_manifest(source,split_path):
    m=copy.deepcopy(source);m['parent_training_manifest_sha256']=ds.fingerprint(source)
    m['external_split_sha256']=ds.digest(split_path)
    m['annotation_index_sha256']=ds.fingerprint([dict(episode_id=e['episode_id'],source_annotation_sha256=e['annotation']['sha256'],effective_segments=e['annotation']['segments']) for e in m['episodes']])
    m['prediction']=ds.prediction_contract(dict(prediction_mode='joint',future_horizon_s=1.5));m['summary']={}
    for split,ids in m['splits'].items():
        episodes=[e for e in m['episodes'] if e['source_recording_id'] in ids];cur=np.zeros(4,dtype=int);future=np.zeros(4,dtype=int);invalid=0
        for e in episodes:
            t=ds.build_phase_targets(np.asarray(e['anchors']),e['annotation']['segments'],e['length'],e['fps'],1.5);cur+=np.bincount(t['current_labels'],minlength=4);future+=np.bincount(t['future_labels'][t['future_valid_mask']],minlength=4);invalid+=int((~t['future_valid_mask']).sum())
        m['summary'][split]=dict(sources=len(ids),episodes=len(episodes),anchors=int(cur.sum()),class_counts=cur.tolist(),future_class_counts=future.tolist(),future_invalid=invalid)
    m['manifest_metadata_note']='Descriptive fields regenerated for refreshed cohort; actual training memberships/targets/normalization remain exactly those in executed data_identity.'
    return m


def run():
    output=ROOT/'finalized_exports';output.mkdir(exist_ok=False);receipts=[]
    for cohort in ['clean91_control','refresh101']:
        group=output/cohort;group.mkdir();mf=corrected_manifest(ds.read_json(ROOT/cohort/'manifest.json'),Path('configs/tomato_refresh101_20261109')/(cohort+'_split.json'));ds.write_json(group/'dataset_manifest.json',mf)
        for seed in [42,43,44]:
            heads={}
            for mode in ['current_only','future_only']:
                original=ROOT/cohort/f'{mode}_s{seed}'/'best_head.pt';payload=torch.load(original,map_location='cpu',weights_only=True);folder=group/f'{mode}_s{seed}';folder.mkdir();m=copy.deepcopy(mf);m['contract']['seed']=seed;m['prediction']=payload['prediction'];ds.write_json(folder/'manifest.json',m)
                revised=copy.deepcopy(payload);revised['manifest_sha256']=ds.digest(folder/'manifest.json');revised['original_training_artifact']=dict(path=str(original),sha256=ds.digest(original),change='manifest linkage only; weights/normalization/selection epoch unchanged');new=folder/'head.pt';torch.save(revised,new);loaded=torch.load(new,map_location='cpu',weights_only=True)
                assert all(torch.equal(payload['head'][k],loaded['head'][k]) for k in payload['head'])
                assert torch.equal(payload['mean'],loaded['mean']) and torch.equal(payload['std'],loaded['std']);heads[mode]=new
                receipts.append(dict(original=str(original),original_sha256=ds.digest(original),finalized=str(new),finalized_sha256=ds.digest(new),tensor_identical=True,normalization_identical=True,manifest_sha256=revised['manifest_sha256']))
            bundle=group/f'independent_h15_s{seed}.pt';pack(heads['current_only'],heads['future_only'],bundle);saved,_=load_checkpoint(bundle,'cpu');assert saved['prediction_mode']=='joint'
    ds.write_json(output/'equivalence_receipt.json',dict(reason='Executed custom-run manifests inherited old94 descriptive summary/index/prediction fields; these were not used for actual targets or membership. Training manifest copies/checkpoints remain untouched. Finalized metadata regenerated and every tensor compared.',receipts=receipts,training_repeated=False))
    print('Finalized12 heads and6 bundles; all weights and normalization exactly identical')

if __name__=='__main__':run()
