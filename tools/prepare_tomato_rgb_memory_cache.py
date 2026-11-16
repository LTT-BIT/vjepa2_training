"""Native cam0 local256 encoding; old100 train/dev only; source data read-only."""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path
import time
import numpy as np
from . import tomato_dataset as ds
from .tomato_phase_model import load_encoder,encode_clips
from .tomato_causal_local_roi import CausalLocalROI
from .tomato_interaction_rgb import crop_rgb
from .tomato_frozen_protocol import verify

CONFIG=Path('configs/tomato_rgb_memory_train_20261116/protocol.json')
DETECTOR=Path('/home/user/datasets/derived/tomato_roi_locator_experiment_r2')
DETECTION_CACHE=Path('/home/user/datasets/derived/tomato_complete100_state_guarded_candidates_r1')


def prepare(out,reuse_partial=None):
    if ds.read_json(CONFIG).get('roi_source_mode')=='recorded_offline_r11':
        from .prepare_tomato_recorded_roi_cache import prepare_recorded
        return prepare_recorded(out,CONFIG,reuse_partial=reuse_partial)
    import av
    if out.exists():raise FileExistsError('Preserve previous cache')
    cfg=ds.read_json(CONFIG);lock=verify(Path(cfg['frozen_lock']));manifest=ds.read_json(lock['manifest']);assert lock['manifest_sha256']==cfg['manifest_sha256'];episodes=[e for e in manifest['episodes'] if e['split']!='test'];by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']}
    detector_receipt=ds.read_json(DETECTOR/'receipt.json');modelsha=ds.digest(DETECTOR/'model.pt');supervised=[e for e in manifest['episodes'] if detector_receipt['source_sha256'] in e['input_sha256'].values()];assert len(supervised)==1 and supervised[0]['split']=='train'
    sources={};qall={};cachepaths={};sums=np.zeros(58);squares=np.zeros(58);count=0
    for e in episodes:
        _,q=ds.read_state(e,manifest['contract']['joint_names'],manifest['contract']['joint_mode']);qall[e['episode_id']]=q
        if e['split']=='train':sums+=q.sum(0,dtype=np.float64);squares+=np.square(q.astype(np.float64)).sum(0);count+=len(q)
        path=Path(by[e['episode_id']]['feature_path']);assert ds.digest(path)==by[e['episode_id']]['feature_sha256'];cachepaths[e['episode_id']]=path
    mean=(sums/count).astype('float32');std=np.sqrt(np.maximum(squares/count-(sums/count)**2,0)).clip(.01).astype('float32')
    for path in DETECTION_CACHE.glob('shard*/episode_*/receipt.json'):
        row=ds.read_json(path)
        if row['contract']['locator_sha256']!=modelsha:raise ValueError('Detector identity mismatch')
        key=(row['source_video_sha256'],row['source_parquet_sha256'])
        if key in sources:raise ValueError('Ambiguous source detection cache')
        sources[key]=(path,row)
    inputs={str(CONFIG):ds.digest(CONFIG),lock['manifest']:lock['manifest_sha256'],str(DETECTOR/'receipt.json'):ds.digest(DETECTOR/'receipt.json'),str(DETECTOR/'model.pt'):modelsha};out.mkdir(parents=True);np.savez(out/'train_state_statistics.npz',mean=mean,std=std);ds.write_json(out/'protocol.json',cfg)
    signature=ds.fingerprint(dict(protocol=cfg,encoder=lock['pretrained'],roi_source_sha=ds.digest('tools/tomato_causal_local_roi.py'),detector=modelsha,state_mean=mean.tolist(),state_std=std.tolist(),manifest=lock['manifest_sha256']))
    encoder,_=load_encoder('pretrained/vjepa2-vitl-fpc64-256','cuda');rows=[];start=time.perf_counter()
    for e in episodes:
        eid=e['episode_id'];z=np.load(cachepaths[eid]);assert np.array_equal(z['anchors'],e['anchors']);keep=np.flatnonzero(z['anchors']>=45);assert len(keep)>0 and e['annotation']['segments'][0]['end_frame']>45
        assert np.all(np.diff(z['input_indices'][keep],axis=1)==3) and np.array_equal(z['input_indices'][keep,-1],z['anchors'][keep]);q=qall[eid];np.testing.assert_array_equal(z['features'][:,-928:],q[z['input_indices']].reshape(len(z['anchors']),928))
        key=(e['input_sha256'][str(Path(e['videos'][0]).relative_to(e['dataset_root']))],e['input_sha256'][str(Path(e['parquet']).relative_to(e['dataset_root']))]);receipt_path,drec=sources[key];dp=receipt_path.parent/'detections.jsonl.gz';assert ds.digest(dp)==drec['detections_sha256'];inputs[str(receipt_path)]=ds.digest(receipt_path);inputs[str(dp)]=ds.digest(dp);inputs[str(cachepaths[eid])]=ds.digest(cachepaths[eid]);assert ds.digest(e['videos'][0])==key[0]
        selector=CausalLocalROI(manifest['contract']['joint_names'],std);recent=deque();queue=[];visual=[];metadata=[];anchors=[];positions=[];valids=[];timings=[];request={int(z['anchors'][k]):k for k in keep};seen=0
        def flush():
            if not queue:return
            t=time.perf_counter();local=encode_clips(encoder,np.stack([item[0] for item in queue]));timings.append((time.perf_counter()-t)*1000)
            for (clip,k,meta,valid),v in zip(queue,local):
                visual.append(np.r_[z['features'][k,:3072],v].reshape(4,1024));metadata.append(meta);anchors.append(int(z['anchors'][k]));positions.append(q[z['input_indices'][k]]);valids.append(valid)
            queue.clear()
        with av.open(e['videos'][0]) as video,gzip.open(dp,'rt') as detections:
            for i,frame in enumerate(video.decode(video=0)):
                row=json.loads(next(detections));assert row['source_frame']==i;image=frame.to_ndarray(format='rgb24');height,width=image.shape[:2]
                if frame.time is None or abs(float(frame.time)-i/e['fps'])>1e-4:raise ValueError('Video timestamp mismatch')
                result=selector.update(row['detections'],q[i],width,height,rgb=image)
                # Anchors and selected history use multiples of3; other current
                # frames do not allocate RGB crops. ROI never switches to left transport.
                if i%3==0:
                    crop=crop_rgb(image,result['interaction_box'],valid=bool(result['valid']),size=256);recent.append((i,crop['rgb'],crop['crop_box'],bool(result['valid'])))
                    while recent and recent[0][0]<i-45:recent.popleft()
                if i in request:
                    k=request[i];assert len(recent)==16 and [r[0] for r in recent]==z['input_indices'][k].tolist();boxes=np.asarray([r[2] for r in recent],float)/np.array([width,height,width,height]);valid=np.asarray([r[3] for r in recent],bool);offsets=z['input_indices'][k]/e['fps']-i/e['fps'];meta=np.r_[boxes.reshape(-1),valid.astype(float),offsets,np.zeros(16),0.,1.,0.,0.].astype('float32');assert len(meta)==116
                    queue.append((np.stack([r[1] for r in recent]),k,meta,valid))
                    if len(queue)>=cfg['cache_encoder_batch']:flush()
                seen+=1
            assert next(detections,None) is None
        flush();assert seen==e['length'] and np.array_equal(np.array(anchors),z['anchors'][keep]);path=out/f'{eid}.npz';np.savez_compressed(path,visual=np.asarray(visual,np.float32),positions=np.asarray(positions,np.float32),metadata=np.asarray(metadata,np.float32),anchors=np.asarray(anchors),times=np.asarray(anchors)/e['fps'],roi_valid=np.asarray(valids),input_indices=z['input_indices'][keep],cache_signature=np.array(signature));rows.append(dict(episode_id=eid,split=e['split'],source_recording_id=e['source_recording_id'],path=str(path),sha256=ds.digest(path),windows=len(anchors),valid_roi_fraction=float(np.asarray(valids).mean()),encoder_call_ms=timings,detector_cache_sha256=drec['detections_sha256']));ds.write_json(out/'progress.json',dict(episodes=rows));print('CACHED',len(rows),eid,e['split'],len(anchors),'roi_valid',round(rows[-1]['valid_roi_fraction'],3),flush=True)
    ds.write_json(out/'index.json',dict(episodes=rows,signature=signature,manifest_sha256=lock['manifest_sha256'],joint_names=manifest['contract']['joint_names'],statistics_path=str(out/'train_state_statistics.npz'),statistics_sha256=ds.digest(out/'train_state_statistics.npz'),statistics_training_sources=[e['source_recording_id'] for e in episodes if e['split']=='train'],statistics_raw_frames=count,detector_supervised_training_episode=supervised[0]['episode_id'],detector_quality_accepted=False,causal_roi=True,historical_test_encoded=False));ds.write_json(out/'receipt.json',dict(inputs=inputs,source_sha256={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/tomato_causal_local_roi.py'),Path('tools/tomato_interaction_rgb.py')]},wall_s=time.perf_counter()-start,local_windows=sum(r['windows'] for r in rows),global_encoder_calls=0,roi_uses_phase_labels=False,roi_uses_future_observations=False,source_data_modified=False));verify(Path(cfg['frozen_lock']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--reuse-partial',type=Path);a=p.parse_args();prepare(a.out,reuse_partial=a.reuse_partial)
