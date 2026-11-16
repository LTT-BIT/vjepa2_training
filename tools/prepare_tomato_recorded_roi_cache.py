"""Exact old100 policy ROI videos, offline replay ONLY (future-assisted boxes).

Never claims online causality merely because selected pixel timestamps are past.
No ROI detector, right-hand selector, new119 data or existing pixels are modified.
"""
from collections import deque
import gzip
from pathlib import Path
import time
import numpy as np
from . import tomato_dataset as ds
from .tomato_frozen_protocol import verify


def validate_binding(manifest,bundle,info):
    pairs={(r['source_dataset'],r['source_episode_index']) for r in bundle['mapping']}
    if len(pairs)!=100 or pairs!={(e['dataset_root'],e['episode_index']) for e in manifest['episodes']}:raise ValueError('Old100 recording identity mismatch')
    cams={k for k,v in info['features'].items() if v.get('dtype')=='video'}
    if cams!={f'observation.images.cam{i}' for i in range(3)}|{'observation.images.interaction'} or info['total_frames']!=425556 or info['fps']!=30:raise ValueError('Old policy four-view schema mismatch')


def prepare_recorded(out,config,reuse_partial=None):
    import av,cv2,json
    from .tomato_phase_model import load_encoder,encode_clips
    if out.exists():raise FileExistsError('Preserve historical cache')
    cfg=ds.read_json(config)
    if cfg.get('roi_source_mode')!='recorded_offline_r11' or not cfg.get('roi_uses_future_coordinates'):raise ValueError('Explicit offline r11 source identity required')
    lock=verify(Path(cfg['frozen_lock']));manifest=ds.read_json(lock['manifest']);root=Path(cfg['roi_bundle']);bundle=ds.read_json(root/'bundle.json');info=ds.read_json(root/'b_coarse/meta/info.json');validate_binding(manifest,bundle,info)
    mapping={(r['source_dataset'],r['source_episode_index']):r for r in bundle['mapping']};base={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']};episodes=[e for e in manifest['episodes'] if e['split']!='test'];positions={};sums=np.zeros(58);squares=np.zeros(58);n=0
    inputs={str(p):ds.digest(p) for p in [config,Path(lock['manifest']),root/'bundle.json',root/'crop_manifest.json',root/'b_coarse/meta/info.json']}
    for e in episodes:
        _,q=ds.read_state(e,manifest['contract']['joint_names'],manifest['contract']['joint_mode']);positions[e['episode_id']]=q
        if e['split']=='train':sums+=q.sum(0,dtype=np.float64);squares+=np.square(q.astype(np.float64)).sum(0);n+=len(q)
    mean=(sums/n).astype('float32');std=np.sqrt(np.maximum(squares/n-(sums/n)**2,0)).clip(.01).astype('float32');out.mkdir(parents=True);ds.write_json(out/'protocol.json',cfg);np.savez(out/'train_state_statistics.npz',mean=mean,std=std)
    signature=ds.fingerprint(dict(protocol=cfg,inputs=inputs,mean=mean.tolist(),std=std.tolist(),encoder=lock['pretrained'],source=ds.digest(__file__)))
    parents={}
    if reuse_partial is not None:
        reuse_partial=Path(reuse_partial)
        if ds.read_json(reuse_partial/'protocol.json')!=cfg:raise ValueError('Partial protocol mismatch')
        legacy=np.load(reuse_partial/'train_state_statistics.npz');np.testing.assert_array_equal(legacy['mean'],mean);np.testing.assert_array_equal(legacy['std'],std)
        legacy_signature=ds.fingerprint(dict(protocol=cfg,inputs=inputs,mean=mean.tolist(),std=std.tolist(),encoder=lock['pretrained'],source='ba276df21b15429a34e78ae1a702e704f1dcf29c1aa604a49e2d71aa6110ca06'))
        parents={r['episode_id']:r for r in ds.read_json(reuse_partial/'progress.json')['episodes']}
    encoder,_=load_encoder('pretrained/vjepa2-vitl-fpc64-256','cuda');rows=[];start=time.perf_counter()
    for e in episodes:
        eid=e['episode_id'];row=mapping[(e['dataset_root'],e['episode_index'])];ep=row['episode_index'];path=Path(base[eid]['feature_path']);assert ds.digest(path)==base[eid]['feature_sha256'];z=np.load(path);q=positions[eid];assert np.array_equal(z['anchors'],e['anchors']);np.testing.assert_array_equal(z['features'][:,-928:],q[z['input_indices']].reshape(len(z['anchors']),928));keep=np.flatnonzero(z['anchors']>=45);request={int(z['anchors'][k]):k for k in keep}
        for cam in range(3):
            rel=info['video_path'].format(episode_chunk=0,episode_index=ep,video_key=f'observation.images.cam{cam}');vp=root/'b_coarse'/rel;orig=str(Path(e['videos'][cam]).relative_to(e['dataset_root']));assert ds.digest(vp)==e['input_sha256'][orig];inputs[str(vp)]=ds.digest(vp)
        video=root/'b_coarse'/info['video_path'].format(episode_chunk=0,episode_index=ep,video_key='observation.images.interaction');coords=root/f'roi_metadata/episode_{ep:06d}.jsonl.gz';inputs[str(video)]=ds.digest(video);inputs[str(coords)]=ds.digest(coords);inputs[str(path)]=ds.digest(path);recent=deque();queue=[];visual=[];meta=[];valids=[];anchors=[];timings=[];seen=0
        if eid in parents:
            parent=parents[eid];pp=Path(parent['path'])
            if ds.digest(pp)!=parent['sha256'] or parent['roi_video_sha256']!=inputs[str(video)]:raise ValueError('Partial cached file/source changed')
            with np.load(pp,allow_pickle=False) as old:
                if str(old['cache_signature'])!=legacy_signature:raise ValueError('Unrecognized partial extraction signature')
                np.testing.assert_array_equal(old['anchors'],z['anchors'][keep]);np.testing.assert_array_equal(old['input_indices'],z['input_indices'][keep]);np.testing.assert_array_equal(old['positions'],q[z['input_indices'][keep]]);np.testing.assert_array_equal(old['visual'][:,:3].reshape(-1,3072),z['features'][keep,:3072])
                payload={k:old[k] for k in old.files};payload['cache_signature']=np.array(signature)
            p=out/f'{eid}.npz';np.savez_compressed(p,**payload);inputs[str(pp)]=parent['sha256'];new=dict(parent,path=str(p),sha256=ds.digest(p),reused_from=str(pp),legacy_algorithm_sha256='ba276df21b15429a34e78ae1a702e704f1dcf29c1aa604a49e2d71aa6110ca06');rows.append(new);ds.write_json(out/'progress.json',dict(episodes=rows));print('R11_REUSED',len(rows),eid,flush=True);continue

        def flush():
            if not queue:return
            t=time.perf_counter();features=encode_clips(encoder,np.stack([r[0] for r in queue]));timings.append((time.perf_counter()-t)*1000)
            for (_,k,m,v),f in zip(queue,features):visual.append(np.r_[z['features'][k,:3072],f].reshape(4,1024));meta.append(m);valids.append(v);anchors.append(int(z['anchors'][k]))
            queue.clear()
        with av.open(video) as stream,gzip.open(coords,'rt') as side:
            for i,frame in enumerate(stream.decode(video=0)):
                r=json.loads(next(side));assert r['source_frame']==i and abs(r['timestamp']-i/30)<1e-5
                if frame.time is None or abs(float(frame.time)-i/30)>1e-4:raise ValueError('Recorded ROI timestamp mismatch')
                if i%3==0:
                    rgb=frame.to_ndarray(format='rgb24');assert rgb.shape==(224,224,3);rgb=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA);w,h=r['source_size'];box=np.asarray(r['crop_box'],float)/[w,h,w,h];recent.append((i,rgb,box,bool(r['valid'])))
                    while recent and recent[0][0]<i-45:recent.popleft()
                if i in request:
                    k=request[i];assert len(recent)==16 and [r[0] for r in recent]==z['input_indices'][k].tolist();v=np.asarray([r[3] for r in recent]);offset=z['input_indices'][k]/30-i/30;m=np.r_[np.stack([r[2] for r in recent]).reshape(-1),v.astype(float),offset,np.zeros(16),0.,1.,0.,0.].astype('float32');queue.append((np.stack([r[1] for r in recent]),k,m,v))
                    if len(queue)>=cfg['cache_encoder_batch']:flush()
                seen+=1
            assert next(side,None) is None
        flush();assert seen==e['length'] and np.array_equal(anchors,z['anchors'][keep]);p=out/f'{eid}.npz';np.savez_compressed(p,visual=np.asarray(visual,np.float32),positions=q[z['input_indices'][keep]],metadata=np.asarray(meta,np.float32),anchors=np.asarray(anchors),times=np.asarray(anchors)/30,roi_valid=np.asarray(valids),input_indices=z['input_indices'][keep],cache_signature=np.array(signature));rows.append(dict(episode_id=eid,split=e['split'],source_recording_id=e['source_recording_id'],path=str(p),sha256=ds.digest(p),windows=len(anchors),valid_roi_fraction=float(np.asarray(valids).mean()),encoder_call_ms=timings,policy_bundle_episode=ep,roi_video_sha256=inputs[str(video)]));ds.write_json(out/'progress.json',dict(episodes=rows));print('R11_CACHED',len(rows),eid,e['split'],len(anchors),flush=True)
    ds.write_json(out/'index.json',dict(episodes=rows,signature=signature,manifest_sha256=lock['manifest_sha256'],joint_names=manifest['contract']['joint_names'],statistics_path=str(out/'train_state_statistics.npz'),statistics_sha256=ds.digest(out/'train_state_statistics.npz'),statistics_training_sources=[e['source_recording_id'] for e in episodes if e['split']=='train'],statistics_raw_frames=n,causal_roi=False,roi_stream_contract=cfg['roi_stream_contract'],historical_test_encoded=False))
    ds.write_json(out/'receipt.json',dict(inputs=inputs,source_sha256={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/prepare_tomato_rgb_memory_cache.py')]},wall_s=time.perf_counter()-start,local_windows=sum(r['windows'] for r in rows),global_encoder_calls=0,roi_uses_future_observations_for_coordinates=True,roi_uses_future_pixels=False,strict_online_causality_validated=False,source_data_modified=False));verify(Path(cfg['frozen_lock']))
