"""One-command import -> frozen encoder/heads -> full Agent offline evaluation."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_phase_model as pm,tomato_metrics as metrics
from .tomato_frozen_protocol import verify
from .tomato_independent_import import validate_index,claim_collection
from .tomato_independent_heads import load_checkpoint
from .tomato_full_chain import evaluate_episode,summarize_chain,episode_report

DEFAULT_REGISTRY=Path('runs/tomato/frozen_independent_registry_v1')


def run(index,out,lock_path,registry,validate_only=False,synthetic_smoke=False):
    if out.exists():raise FileExistsError('New output required; never overwrite an evaluation')
    lock=verify(lock_path)
    # Validate destination before creating any folder near raw inputs.
    catalog=ds.read_json(index)
    if synthetic_smoke and catalog.get('is_synthetic_test_fixture') is not True:raise ValueError('Smoke override requires explicitly synthetic catalog')
    if not synthetic_smoke and registry.resolve()!=DEFAULT_REGISTRY.resolve():raise ValueError('Real independent collection must use the fixed shared evaluation registry')
    roots=[r['dataset_root'] for r in catalog.get('records',[])]+lock['historical_registry']['dataset_roots']+[str(Path(v['path']).parent) for v in lock['models'].values()]
    annotations=[r['annotation_path'] for r in catalog.get('records',[])]
    ds.ensure_output_outside_sources(out,roots,annotations);ds.ensure_output_outside_sources(registry,roots,annotations)
    out.mkdir(parents=True)
    try:manifest=validate_index(index,lock,out,allow_synthetic_fixture=synthetic_smoke)
    except Exception as exc:
        ds.write_json(out/'validation.json',dict(passed=False,error=str(exc),evaluation_started=False));raise
    ds.write_json(out/'validation.json',dict(passed=True,records=len(manifest['episodes']),full_media_decode=True,scope='Input admission, no performance result'))
    manifest['lock_path']=str(lock_path.resolve());manifest['lock_sha256']=ds.digest(lock_path);ds.write_json(out/'manifest.json',manifest)
    if validate_only:return
    manifest['registry_receipt_sha256']=claim_collection(manifest,registry,ds.digest(lock_path));ds.write_json(out/'manifest.json',manifest)
    try:
        encoder,provenance=pm.load_encoder('pretrained/vjepa2-vitl-fpc64-256','cuda',42)
        if provenance!=lock['pretrained']:raise ValueError('Encoder identity differs from lock')
        features={}
        for e in manifest['episodes']:
            path,perf=pm.extract_episode(encoder,e,lock['contract'],provenance,out/'features',batch_windows=4)
            data=pm.load_cache(path,e,lock['contract'],provenance);features[e['episode_id']]=data
        del encoder;torch.cuda.empty_cache();summary={}
        for seed,model in lock['models'].items():
            saved,head=load_checkpoint(model['path'],'cuda');per={};folder=out/f'seed{seed}';folder.mkdir()
            for e in manifest['episodes']:
                data=features[e['episode_id']];z=pm.predict_heads(head,data['features'],saved['mean'],saved['std'])
                bundle=dict(times=data['times'],anchors=data['anchors'],**{h+'_probabilities':metrics.probabilities(z[h+'_logits']) for h in ['current','future']})
                score,timeline,journal=evaluate_episode(e,bundle,model['model_version']);per[e['episode_id']]=score
                np.savez_compressed(folder/(e['episode_id']+'.npz'),**bundle)
                ds.write_json(folder/(e['episode_id']+'_analysis.json'),score)
                (folder/(e['episode_id']+'_report.md')).write_text(episode_report(e,score))
                for name,rows in [('timeline',timeline),('memory',journal)]:
                    with (folder/(e['episode_id']+'_'+name+'.jsonl')).open('x') as f:
                        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
            summary[seed]=summarize_chain(per);ds.write_json(folder/'episodes.json',per);del head;torch.cuda.empty_cache()
        verify(lock_path)
        for e in manifest['episodes']:pm.assert_source_unchanged(e)
        ds.write_json(out/'summary.json',summary);ds.write_json(out/'receipt.json',dict(lock_sha256=ds.digest(lock_path),training=False,normalization_refit=False,
                     threshold_or_model_selected=False,robot_connected=False,physical_success_evaluated=False,synthetic_fixture=manifest['role']=='synthetic_smoke_only',
                     source_sha256={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/tomato_full_chain.py'),Path('tools/tomato_independent_import.py')]}))
    except Exception as exc:
        ds.write_json(out/'failure.json',dict(error=str(exc),registry_claim_retained=True,reason='Failed evaluation remains exposed; no automatic release or reclassification as blind test'));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--lock',type=Path,default=Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));p.add_argument('--registry',type=Path,default=Path('runs/tomato/frozen_independent_registry_v1'));p.add_argument('--validate-only',action='store_true');p.add_argument('--synthetic-smoke',action='store_true');a=p.parse_args();run(a.index,a.out,a.lock,a.registry,a.validate_only,a.synthetic_smoke)
