"""UI-only 5 Hz observation grid; original 2 Hz training contract stays immutable."""
from pathlib import Path
import gc
import numpy as np
from .catalog import read,sha
from .prediction_store import sha_bytes


def validate_hz(hz):
    if type(hz) not in (int,float) or hz not in (2,5):
        raise ValueError('prediction_hz must be 2 or 5')
    return int(hz)


def episode_grid(e,hz):
    hz=validate_hz(hz)
    if hz==2:return e
    stride=e['fps']/hz
    if not np.isclose(stride,round(stride),rtol=0,atol=1e-8):raise ValueError('Prediction rate must map to exact source frames')
    return {**e,'anchors':list(range(0,e['length'],round(stride))),
            'ui_prediction_hz':5,'ui_timestamp_tolerance_s':max(1e-6,2**-22*e['length']/e['fps'])}


def source_grid(e,times):
    virtual=episode_grid(e,5)
    if (len(times)!=e['length'] or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or
        not np.allclose(times,np.arange(len(times))/e['fps'],atol=virtual['ui_timestamp_tolerance_s'],rtol=0)):
        raise ValueError('Source timestamps violate the registered uniform camera frame contract')
    return np.asarray(virtual['anchors'],dtype=np.int64)


def load_features(catalog,eid,mode,output,device='cuda',progress=lambda v:None):
    from .. import tomato_dataset as ds,tomato_phase_model as model
    from ..tomato_runtime import initialize_runtime
    import torch
    e=catalog.get(eid);m=catalog.manifests[mode];contract=m['contract'];run=catalog.root/mode
    model.assert_source_unchanged(e)
    times,q=ds.read_state(e,contract['joint_names'],contract['joint_mode'])
    anchors=source_grid(e,times)
    histories=np.stack([ds.sample_causal_indices(times,int(i),sampler_version=contract['sampler_version']) for i in anchors])
    key=sha_bytes(dict(schema='ui-five-hz-v1',training_feature_key=model.feature_fingerprint(e,contract,m['pretrained']),
        inference_hz=5,training_hz=2,anchors=anchors.tolist()))
    path=Path(output)/'features_5hz'/(key+'.npz')
    def checked():
        receipt=read(path.with_suffix('.json'))
        if receipt!=dict(fingerprint=key,content_sha256=sha(path)):raise ValueError('5 Hz feature receipt changed')
        with np.load(path,allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
        if (str(d['fingerprint'])!=key or not np.array_equal(d['anchors'],anchors) or
            not np.array_equal(d['input_indices'],histories) or not np.array_equal(d['times'],times[anchors]) or
            d['features'].shape!=(len(anchors),3072+16*q.shape[1]) or d['features'].dtype!=np.float32 or
            not np.isfinite(d['features']).all() or not np.array_equal(d['features'][:,3072:],q[histories].reshape(len(anchors),-1))):
            raise ValueError('5 Hz feature dimensions/history/state mismatch')
        return d
    if path.exists():
        d=checked()
        return d,dict(feature_sha256=sha(path),fingerprint=key,feature_cache_hit=True,encoder_executed=False,
                      training_prediction_hz=2,inference_prediction_hz=5)
    features=np.empty((len(anchors),3072+16*q.shape[1]),np.float32);missing=list(range(len(anchors)))
    indexpath=run/'feature_index.json';reused=0;original_sha=None
    if indexpath.exists():
        index=read(indexpath)
        if index['manifest_sha256']!=sha(run/'manifest.json') or index['pretrained']!=m['pretrained']:raise ValueError('Original feature index binding mismatch')
        item=next((r for r in index['episodes'] if r['episode_id']==eid),None)
        if item and Path(item['path']).exists():
            old=model.load_cache(item['path'],e,contract,index['pretrained']);original_sha=sha(item['path'])
            positions={int(a):j for j,a in enumerate(old['anchors'])};missing=[]
            for k,a in enumerate(anchors):
                j=positions.get(int(a))
                if j is not None and np.array_equal(old['input_indices'][j],histories[k]):features[k]=old['features'][j];reused+=1
                else:missing.append(k)
    if missing:
        config=read(run/'training_config.json');initialize_runtime(dict(device=device,seed=config['seed']))
        progress(dict(stage='encoder_loading',completed=0,total=len(missing)))
        encoder,provenance=model.load_encoder(config['pretrained_dir'],device,config['seed'])
        if provenance!=m['pretrained']:raise ValueError('Pinned encoder differs from trained features')
        try:
            views,_=model.decode_selected(e,np.unique(histories[missing]))
            for n,k in enumerate(missing):
                rgb=np.stack([[np.stack([v[int(j)] for j in histories[k]]) for v in views]])
                features[k]=model.fuse_features(encoder,rgb,q[histories[k]][None])[0]
                if n%10==0 or n+1==len(missing):progress(dict(stage='features_5hz',completed=n+1,total=len(missing)))
        finally:
            del encoder
            gc.collect()
            if device.startswith('cuda'):torch.cuda.empty_cache()
    model.assert_source_unchanged(e)
    model.write_cache(path,features,anchors,times[anchors],histories,key)
    d=checked()
    return d,dict(feature_sha256=sha(path),fingerprint=key,feature_cache_hit=False,encoder_executed=bool(missing),
        encoded_anchors=len(missing),reused_2hz_anchors=reused,original_feature_sha256=original_sha,
        training_prediction_hz=2,inference_prediction_hz=5,encoder_device=device)


def infer(catalog,eid,mode,output,device='cuda',progress=lambda v:None):
    import torch
    torch.set_num_threads(4)
    from .. import tomato_phase_model as model,tomato_metrics as metrics
    from ..train_tomato_vjepa_phase import prediction_rows
    e=catalog.get(eid);m=catalog.manifests[mode];run=catalog.root/mode
    # Load the original unmodified checkpoint, including its original 2 Hz contract.
    saved,head=model.load_checkpoint(run/'best_head.pt',sha(run/'manifest.json'),'cpu',m['split_sha256'])
    if saved['prediction_mode']!=mode:raise ValueError('Checkpoint mode mismatch')
    d,provenance=load_features(catalog,eid,mode,output,device,progress)
    z=model.predict_heads(head,d['features'],saved['mean'],saved['std'])
    probs={h:metrics.probabilities(z[h+'_logits']) if z[h+'_logits'] is not None else None for h in ['current','future']}
    rows=prediction_rows(dict(episode=e,**d),probs,e['split'],m['prediction'],{h:z[h+'_logits'] for h in ['current','future']})
    return rows,{**provenance,'device':'cpu','head_checkpoint_contract_preserved':True}
