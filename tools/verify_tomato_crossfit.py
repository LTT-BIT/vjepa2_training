"""Verify own local S2 checkpoints, refit normalization and streaming replay."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_probability_residual import ResidualCalibration
from .train_tomato_probability_residual import history_features
from .tomato_residual_event_audit import accepted_events


def run(root,output=None):
    output=output or root/'verification'
    if output.exists():raise FileExistsError('Verification output already exists')
    output.mkdir();receipt=ds.read_json(root/'receipt.json');folds=ds.read_json(root/'folds.json')
    # Originals were produced by our current local run, with recorded SHA identities.
    # Fold metadata used numpy.str_: export plain-string metadata to NEW files.
    m=ds.read_json(receipt['dataset_lock']['manifest']);by={e['source_recording_id']:e for e in m['episodes']};checks=[];exports=[]
    for seed in (42,43,44):
        folder=root/f'seed{seed}';events=ds.read_json(folder/'events.json');hist={};normalization_hist={};arrays={}
        for sid in receipt['pool_source_ids']:
            e=by[sid];z=np.load(folder/('oof_'+e['episode_id']+'.npz'));arrays[e['episode_id']]=z;hist[sid]=history_features(z['current'],z['original_future'],z['times'])
            normalization_hist[sid]=hist[sid][np.array(e['anchors'])+45<e['length']]
        nets={}
        for k,f in enumerate(folds):
            original=folder/f'fold{k}.pt';original_sha=ds.digest(original)
            # Trusted own-run pickle, never a downloaded or user-supplied model.
            c=torch.load(original,weights_only=False,map_location='cpu');state=c['state_dict']
            export=output/f'seed{seed}_fold{k}.pt';torch.save(dict(state_dict=state,fold=f,selected_epoch=int(c['selected_epoch']),seed=seed,base_model=c['base_model']),export)
            clean=torch.load(export,weights_only=True,map_location='cpu');assert all(torch.equal(v,clean['state_dict'][key]) for key,v in state.items())
            x=np.concatenate([normalization_hist[s] for s in receipt['pool_source_ids'] if s in f['refit']]);expected=x.mean(0,dtype=np.float64).astype('float32');assert np.allclose(expected,state['mean'].numpy(),atol=1e-6)
            std=x.std(0,dtype=np.float64).clip(.01).astype('float32');assert np.allclose(std,state['std'].numpy(),atol=1e-6)
            net=ResidualCalibration(state['mean'],state['std']);net.load_state_dict(state);net.eval()
            for sid in f['outer']:nets[by[sid]['episode_id']]=net
            assert ds.digest(original)==original_sha
            exports.append(dict(seed=seed,fold=k,original=str(original),original_sha256=original_sha,export=str(export),export_sha256=ds.digest(export),tensors_identical=True,refit_normalization_verified=True))
        ck=torch.load(folder/'residual.pt',weights_only=True,map_location='cpu');state=ck['state_dict'];final=ResidualCalibration(state['mean'],state['std']);final.load_state_dict(state);final.eval()
        x=np.concatenate([normalization_hist[s] for s in receipt['pool_source_ids']]);assert np.allclose(x.mean(0,dtype=np.float64).astype('float32'),state['mean'].numpy(),atol=1e-6)
        for scope,part in [('oof',events['oof']['S2']),('development',events['development']['S2'])]:
            for eid,score in part.items():
                z=arrays[eid] if scope=='oof' else np.load(folder/(eid+'.npz'));x=history_features(z['current'],z['original_future'],z['times']);net=nets[eid] if scope=='oof' else final
                with torch.inference_mode():p=net(torch.from_numpy(x),torch.from_numpy(np.log(np.maximum(z['original_future'],1e-6)).astype('float32')))[0].softmax(-1).numpy()
                err=float(abs(p-z['calibrated_future']).max());assert err<2e-6
                valid=z['times']<=z['times'][0]+score['duration_s']-.5+1e-7
                assert accepted_events(p[valid],z['times'][valid])==score['events']
                checks.append(dict(seed=seed,scope=scope,episode_id=eid,maximum_probability_error=err,events_identical=True))
    for p,sha in receipt['input_sha256'].items():assert ds.digest(p)==sha,p
    ds.write_json(output/'result.json',dict(checks=checks,exports=exports,input_hashes_unchanged=True,original_checkpoints_unchanged=True,normalization_train_only_verified=True,
                   metadata_fix='Numpy string fold IDs converted to plain strings in separate safe-load exports. No weight changes or repeated training. Nested fold creator fixed for future runs.'))
    print('VERIFIED',len(checks),'replays',len(exports),'safe fold exports; maximum error',max(c['maximum_probability_error'] for c in checks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args();run(a.run,a.out)
