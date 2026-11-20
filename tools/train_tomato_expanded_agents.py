"""Expanded in-hand cohort: frozen encoder, independent heads, sealed new holdout.

Historical119 checkpoints are never loaded as model initialization. Only visual
cache files with matching raw-video hashes and sampling are eligible for reuse.
"""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import gzip
import json
import shutil
import subprocess
import time
import numpy as np
import torch
from . import tomato_dataset as ds
from . import train_tomato_rgb_memory as base
from .train_tomato_inhand119_agents import SEQUENCE,evaluate
from .train_tomato_independent_rgb_memory import fit_single,METHODS
from .tomato_independent_rgb_memory import IndependentRGBReader,restore_reader
from .tomato_phase_model import load_encoder,encode_clips,verify_pretrained
from .tomato_boundary_metrics import change_events,boundaries
from .evaluate_tomato_reviewed_boundaries import match_events

ARCH='tomato_inhand_independent_rgb_memory_v2'
CAMERAS=['cam0','cam1','cam2','interaction']


def freeze_code(out):
    paths=[Path(__file__),Path('tools/train_tomato_independent_rgb_memory.py'),
        Path('tools/train_tomato_inhand119_agents.py'),Path('tools/train_tomato_rgb_memory.py'),
        Path('tools/tomato_independent_rgb_memory.py'),Path('tools/tomato_rgb_memory_model.py'),
        Path('tools/tomato_rgb_memory_training.py'),Path('tools/tomato_dataset.py'),
        Path('tools/tomato_phase_model.py')]
    hashes={str(p):ds.digest(p) for p in paths};out.mkdir(parents=True,exist_ok=True)
    lock=out/'source_code.lock.json'
    if lock.exists() and ds.read_json(lock)['source_sha256']!=hashes:raise ValueError('Code changed during frozen experiment')
    if not lock.exists():
        ds.write_json(lock,dict(source_sha256=hashes,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        snapshot=out/'code_snapshot';snapshot.mkdir(exist_ok=True)
        for p in paths:shutil.copyfile(p,snapshot/p.name)


def evaluate_expanded(rows,pred):
    result=evaluate(rows,pred);per={}
    for r in rows:
        e=r['episode'];valid=r['target']['future_valid_mask'];ts=r['times'][valid]
        events=change_events(pred[e['episode_id']]['future'][valid],ts,.5,2)
        matched,false=match_events(events,boundaries(dict(episode=e)),'future',1.5)
        per[e['episode_id']]=dict(boundaries=matched,false_events=false,duration_s=float(ts[-1]-ts[0]+.5))
    rows_b=[b for p in per.values() for b in p['boundaries']];hit=sum(b['detected'] for b in rows_b)
    false=sum(len(p['false_events']) for p in per.values());minutes=sum(p['duration_s'] for p in per.values())/60
    result['strict_future']=dict(hits=hit,total=len(rows_b),recall=hit/len(rows_b),
        precision=hit/max(hit+false,1),false_events=false,false_per_minute=false/minutes,per_episode=per)
    return result


def allocate(sizes,total,reserve):
    if total<len(sizes) or total>sum(n-reserve for n in sizes.values()):
        raise ValueError('Cannot allocate stratified holdout with training reserve')
    counts={k:1 for k in sizes};n=sum(sizes.values())
    while sum(counts.values())<total:
        eligible=[k for k in sizes if counts[k]<sizes[k]-reserve]
        k=max(sorted(eligible),key=lambda k:total*sizes[k]/n-counts[k])
        counts[k]+=1
    return counts


def split_sources(rows,old_manifest):
    ids=[e['source_recording_id'] for e in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate original recording')
    old={e['source_recording_id']:e['split'] for e in old_manifest['episodes']}
    if len(old)!=119 or not set(old)<=set(ids):raise ValueError('Original119 identity missing')
    by=defaultdict(list)
    for e in rows:
        if e['source_recording_id'] not in old:by[e['package']].append(e['source_recording_id'])
    if sum(map(len,by.values()))!=85:raise ValueError('Expected frozen85 new recordings')
    sizes={k:len(v) for k,v in by.items()};blind=allocate(sizes,20,2)
    remaining={k:n-blind[k] for k,n in sizes.items()};dev=allocate(remaining,9,1)
    splits={k:('historical_test' if v=='test' else v) for k,v in old.items()}
    rng=np.random.default_rng(20261120)
    for k in sorted(by):
        ordered=rng.permutation(sorted(by[k])).tolist()
        for i,g in enumerate(ordered):
            splits[g]='blind_test' if i<blind[k] else 'val' if i<blind[k]+dev[k] else 'train'
    return splits


def prepare(root,out,oldrun):
    entries=[];names=None
    for side in ['A','B']:
        for pack in sorted((root/side).iterdir()):
            if not (pack/'meta/info.json').is_file():continue
            info=ds.read_json(pack/'meta/info.json')
            ns=[n for n in info['features']['observation.state']['names'] if n.endswith('.position')]
            if len(ns)!=58 or info['fps']!=30 or (names is not None and names!=ns):raise ValueError('Joint/FPS mismatch')
            names=ns
            metadata={e['episode_index']:e for e in ds.read_rows(pack/'meta/episodes.jsonl')}
            for p in sorted((pack/'annotations/subtasks/coarse').glob('*.json')):
                ann=ds.read_json(p);i=ann['source']['episode_index'];entry=metadata[i];length=entry['length']
                ds.validate_segments(ann['segments'],length)
                if [s['label'] for s in ann['segments']]!=SEQUENCE or ann.get('inherited') or ann['prompt_profile']!='tomato-in-hand-five-phases-v3':raise ValueError('Reviewed five-stage labels required')
                group=entry['source_recording_id']
                if ann['source']['source_recording_id']!=group:raise ValueError('Recording/annotation mismatch')
                paths=dict(parquet=pack/f'data/chunk-000/episode_{i:06d}.parquet',
                    roi=pack/f'roi/chunk-000/episode_{i:06d}.jsonl.gz')
                videos={c:pack/f'videos/chunk-000/observation.images.{c}/episode_{i:06d}.mp4' for c in CAMERAS}
                hashes={str(f):ds.digest(f) for f in [*paths.values(),p]}
                # Video bytes are verified before NEW feature extraction. Old
                # features bind their embedded video SHA to the frozen119 receipt.
                # Sealed-test RGB is neither downloaded nor encoded in this run.
                hashes.update({str(f):ann['source']['sha256'][str(f.relative_to(pack))] for f in videos.values()})
                for f in [paths['parquet']]:
                    if hashes[str(f)]!=ann['source']['sha256'][str(f.relative_to(pack))]:raise ValueError(f'Source bytes changed: {f}')
                entries.append(dict(episode_id='inhand204_'+group,episode_index=i,source_recording_id=group,
                    package=str(pack.relative_to(root)),length=length,fps=30,info=info,entry=entry,
                    parquet=str(paths['parquet']),roi=str(paths['roi']),videos={c:str(p) for c,p in videos.items()},
                    annotation=ann,annotation_path=str(p),annotation_sha256=hashes[str(p)],source_hashes=hashes))
    if len(entries)!=204:raise ValueError(f'Expected frozen204, got{len(entries)}')
    old=ds.read_json(oldrun/'manifest.json');splits=split_sources(entries,old)
    sums=np.zeros(58);squares=np.zeros(58);frames=0
    for e in entries:
        e['split']=splits[e['source_recording_id']]
        t,q=ds.read_state(e,names)
        if len(q)!=e['length'] or not np.allclose(t,np.arange(len(t))/30,atol=1e-4):raise ValueError('State time mismatch')
        if e['split']=='train':sums+=q.sum(0,dtype=np.float64);squares+=np.square(q.astype(np.float64)).sum(0);frames+=len(q)
    mean=(sums/frames).astype('float32');std=np.sqrt(np.maximum(squares/frames-(sums/frames)**2,0)).clip(.01).astype('float32')
    cfg=dict(version='inhand204-independent-v1',task_profile='tomato-in-hand-five-phases-v3',
        ordered_classes=[0,1,3,1,2],labels=list(ds.LABELS),methods=METHODS,seeds=[42,43,44],
        epochs=30,batch_size=128,hidden=128,learning_rate=.0003,weight_decay=.01,
        future_horizon_s=1.5,observation_hz=2,video_frames=16,video_sample_hz=10,
        raw_fps=30,cache_encoder_batch=32,joint_names=names,
        head_training='independent full readers; new initialization; frozen shared encoder',
        single_architecture='tomato_inhand_single_rgb_memory_v2',roi_uses_future_coordinates=True,
        roi_uses_future_pixels=False,online_roi_validated=False,control_eligible=False,
        selection='fixed30epochs; no threshold or holdout selection',
        normalization='151training recordings only',test_policy='historical12 and new blind20 sealed, not scored in this campaign')
    counts=dict(Counter(splits.values()))
    if counts!=dict(train=151,val=21,historical_test=12,blind_test=20):raise ValueError(counts)
    manifest=dict(episodes=entries,protocol=cfg,split=splits,counts=counts,training_statistics_frames=frames)
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('manifest.json',manifest),('protocol.json',cfg),('split.json',dict(counts=counts,sources=splits))]:
        if (out/name).exists() and ds.read_json(out/name)!=value:raise ValueError('Frozen source/protocol changed')
        ds.write_json(out/name,value)
    if (out/'statistics.npz').exists():
        z=np.load(out/'statistics.npz');np.testing.assert_array_equal(z['mean'],mean);np.testing.assert_array_equal(z['std'],std)
    else:np.savez(out/'statistics.npz',mean=mean,std=std)
    ds.status(out,'prepared',counts=counts)
    return manifest,mean,std


def extract(out,manifest,oldrun):
    import av,cv2
    cache=out/'features';cache.mkdir(exist_ok=True)
    active=[e for e in manifest['episodes'] if e['split'] in ('train','val')]
    old=ds.read_json(oldrun/'manifest.json');oldby={e['source_recording_id']:e for e in old['episodes']}
    oldreceipt=ds.read_json(oldrun/'encoder_receipt.json')
    if ds.digest('tools/train_tomato_inhand119_agents.py')!=oldreceipt['implementation_sha256']:raise ValueError('Original feature implementation changed')
    provenance=verify_pretrained('pretrained/vjepa2-vitl-fpc64-256')
    if provenance!=oldreceipt['pretrained']:raise ValueError('Encoder identity mismatch')
    signature=ds.fingerprint(dict(encoder=provenance,sampling='16x10Hz-query2Hz',resize='INTER_AREA256',manifest=ds.digest(out/'manifest.json')))
    encoder=None;receipts=[];start=time.monotonic()
    for e in active:
        anchors=np.arange(45,e['length'],15);indices=anchors[:,None]+np.arange(-45,1,3)[None]
        for camera in CAMERAS:
            path=cache/f'{e["episode_id"]}_{camera}.npz';video=Path(e['videos'][camera]);sha=e['source_hashes'][str(video)]
            if path.exists():
                z=np.load(path)
                if str(z['signature'])!=signature or str(z['video_sha256'])!=sha:raise ValueError('Resume cache identity mismatch')
                np.testing.assert_array_equal(z['input_indices'],indices)
                receipts.append(dict(path=str(path),sha256=ds.digest(path),reused=bool(z['reused'])))
                continue
            previous=oldby.get(e['source_recording_id']);reused=False
            if previous is not None:
                p=oldrun/'features'/f'{previous["episode_id"]}_{camera}.npz';z=np.load(p)
                if str(z['signature'])!=oldreceipt['signature'] or str(z['video_sha256'])!=sha:raise ValueError('Previous cache source changed')
                np.testing.assert_array_equal(z['input_indices'],indices)
                values=z['visual'];reused=True
            else:
                start_wait=time.monotonic()
                while not video.is_file():
                    if time.monotonic()-start_wait>3600:raise TimeoutError(f'Video synchronization incomplete: {video}')
                    ds.status(out,'waiting_for_video',path=str(video),completed=len(receipts),total=len(active)*4)
                    time.sleep(1)
                if ds.digest(video)!=sha:raise ValueError('Video bytes changed before encoding')
                if encoder is None:encoder,_=load_encoder('pretrained/vjepa2-vitl-fpc64-256','cuda')
                frames=[];seen=0
                with av.open(str(video)) as stream:
                    for frame in stream.decode(video=0):
                        if frame.time is None or abs(float(frame.time)-seen/30)>1e-4:raise ValueError('Video PTS mismatch')
                        if seen%3==0:frames.append(cv2.resize(frame.to_ndarray(format='rgb24'),(256,256),interpolation=cv2.INTER_AREA))
                        seen+=1
                if seen!=e['length']:raise ValueError('Video/state length mismatch')
                frames=np.asarray(frames);values=[]
                for ix in np.array_split(np.arange(len(anchors)),int(np.ceil(len(anchors)/manifest['protocol']['cache_encoder_batch']))):values.append(encode_clips(encoder,frames[indices[ix]//3]))
                values=np.concatenate(values)
            if values.shape!=(len(anchors),1024) or not np.isfinite(values).all():raise ValueError('Feature shape/value mismatch')
            np.savez_compressed(path,visual=values,anchors=anchors,input_indices=indices,signature=np.array(signature),video_sha256=np.array(sha),reused=np.array(reused))
            receipts.append(dict(path=str(path),sha256=ds.digest(path),reused=reused))
            ds.status(out,'extracting',completed=len(receipts),total=len(active)*4,elapsed_s=time.monotonic()-start,last=e['episode_id']+'/'+camera)
            print('FEATURE',len(receipts),len(active)*4,e['episode_id'],camera,'reused' if reused else 'new',flush=True)
    del encoder;torch.cuda.empty_cache()
    ds.write_json(out/'encoder_receipt.json',dict(signature=signature,pretrained=provenance,files=receipts,
        reused=sum(r['reused'] for r in receipts),new=sum(not r['reused'] for r in receipts),sealed_holdout_features_read=False))
    rows=[]
    for e in active:
        zs=[np.load(cache/f'{e["episode_id"]}_{c}.npz') for c in CAMERAS];indices=zs[0]['input_indices'];anchors=zs[0]['anchors']
        with gzip.open(e['roi'],'rt') as f:coords=[json.loads(line) for line in f]
        if len(coords)!=e['length'] or any(r['frame_index']!=i or abs(r['timestamp']-i/30)>1e-5 for i,r in enumerate(coords)):raise ValueError('ROI timestamp alignment mismatch')
        h,w=e['info']['features']['observation.images.cam0']['shape'][:2]
        boxes=np.array([r['crop_box'] for r in coords],np.float32)/[w,h,w,h]
        valid=np.array([r['image_available'] for r in coords],np.float32)
        meta=np.concatenate([boxes[indices].reshape(-1,64),valid[indices],(indices-anchors[:,None])/30,
            np.zeros((len(anchors),16)),np.tile([0,1,0,0],(len(anchors),1))],axis=1).astype('float32')
        _,q=ds.read_state(e,manifest['protocol']['joint_names'])
        rows.append(dict(episode=e,visual=np.stack([z['visual'] for z in zs],axis=1),positions=q[indices].reshape(-1,928),
            metadata=meta,times=anchors/30,anchors=anchors,target=ds.build_phase_targets(anchors,e['annotation']['segments'],e['length'],30,1.5)))
    return rows


def train(out,rows,manifest,mean,std):
    cfg=manifest['protocol'];training=[r for r in rows if r['episode']['split']=='train'];dev=[r for r in rows if r['episode']['split']=='val']
    assert len(training)==151 and len(dev)==21
    identity=dict(manifest_sha256=ds.digest(out/'manifest.json'),cache_signature=ds.read_json(out/'encoder_receipt.json')['signature'],
        statistics_sha256=ds.digest(out/'statistics.npz'),joint_names=cfg['joint_names'],pretrained=ds.read_json(out/'encoder_receipt.json')['pretrained'],
        source_code_lock_sha256=ds.digest(out/'source_code.lock.json'))
    results={}
    for seed in cfg['seeds']:
        folder=out/f'seed{seed}';folder.mkdir(exist_ok=True)
        for method in METHODS:
            cp=folder/f'{method}.pt';parts={};infos={}
            if cp.exists():
                saved=torch.load(cp,map_location='cpu',weights_only=True)
                if saved['identity']!=identity or saved['protocol']!=cfg:raise ValueError('Resume model mismatch')
                net=restore_reader(saved).cuda().eval();infos=saved['training']
            else:
                for target in ['current','future']:
                    parts[target],infos[target]=fit_single(training,method,target,seed,cfg,mean,std,folder/f'{method}_{target}.pt',identity)
                    print('TRAINED',seed,method,target,infos[target]['training_s'],flush=True)
                net=IndependentRGBReader(parts['current'],parts['future']).cuda().eval()
                torch.save(dict(format_version=1,architecture=ARCH,method=method,seed=seed,
                    model_version=f'inhand204-{method}-s{seed}-{identity["manifest_sha256"][:12]}',head={k:v.detach().cpu() for k,v in net.state_dict().items()},
                    state_mean=torch.from_numpy(mean),state_std=torch.from_numpy(std),protocol=cfg,identity=identity,training=infos,
                    training_sources=[r['episode']['source_recording_id'] for r in training],control_eligible=False,
                    input_schema=dict(labels=list(ds.LABELS),task_profile=cfg['task_profile'],joint_names=cfg['joint_names'])),cp)
            pred=base.predict(net,dev,method,cfg);ev=evaluate_expanded(dev,pred)
            for r in dev:
                eid=r['episode']['episode_id'];np.savez_compressed(folder/f'{method}_{eid}.npz',**pred[eid],times=r['times'],anchors=r['anchors'])
            restored=restore_reader(torch.load(cp,map_location='cpu',weights_only=True)).cuda().eval()
            checked=base.predict(restored,dev,method,cfg);error=0.
            for eid,p in pred.items():
                for h in ['current','future']:
                    np.testing.assert_allclose(checked[eid][h],p[h],atol=5e-6,rtol=1e-4);error=max(error,float(np.max(np.abs(checked[eid][h]-p[h]))))
            results[f'{seed}/{method}']=dict(checkpoint=str(cp.resolve()),sha256=ds.digest(cp),training=infos,validation=ev,reload_max_error=error)
            ds.write_json(out/'results.json',results);ds.status(out,'training',completed_bundles=len(results),total=9)
            print('SCORED',seed,method,ev['current']['macro_f1'],ev['future']['macro_f1'],flush=True)
            del net,restored,parts;torch.cuda.empty_cache()
    ds.write_json(out/'all_models_frozen.json',{k:v['sha256'] for k,v in results.items()})
    ds.write_json(out/'receipt.json',dict(status='complete',paired_models=9,branches=18,
        test_scored=False,historical_test_scored=False,default_changed=False,encoder_frozen=True,robot_calls=0,identity=identity))
    ds.status(out,'complete',paired_models=9,sealed_new_test_recordings=20)


def run(root,out,oldrun,stage):
    freeze_code(out)
    manifest,mean,std=prepare(root,out,oldrun)
    if stage=='prepare':return
    rows=extract(out,manifest,oldrun)
    if stage!='extract':train(out,rows,manifest,mean,std)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--old-cache-run',type=Path,default=Path('runs/tomato/inhand119_independent_20261118'))
    p.add_argument('--stage',choices=['prepare','extract','all'],default='all');a=p.parse_args()
    try:run(a.root,a.out,a.old_cache_run,a.stage)
    except Exception as exc:ds.status(a.out,'failed',error=repr(exc));raise
