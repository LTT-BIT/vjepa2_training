"""Train every new119 independent RGB/Memory readout on a recording-level split.

Source bundle stays read-only. Frozen visual features are shared by all seeds and
methods. Offline ROI coordinates are future-assisted; these are research weights,
not authorization for automatic robot control.
"""
from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import numpy as np
import torch

from . import tomato_dataset as ds, tomato_metrics as metrics
from . import train_tomato_rgb_memory as base
from .tomato_independent_rgb_memory import IndependentRGBReader, restore_reader
from .tomato_phase_model import load_encoder, encode_clips
from .train_tomato_independent_rgb_memory import fit_single, METHODS

SEQUENCE = ['pickup', 'harvest', 'adjust', 'harvest', 'return']
ARCHITECTURE = 'inhand119_independent_rgb_memory_v1'


def split_recordings(selection):
    groups = sorted({e['recording_group'] for e in selection['episodes']})
    if len(groups) != 119 or len(selection['episodes']) != 119:
        raise ValueError('Expected exactly119 distinct normal source recordings')
    order = np.random.default_rng(42).permutation(groups)
    return {str(g): ('test' if i < 12 else 'val' if i < 24 else 'train')
            for i, g in enumerate(order)}


def wait_file(path, timeout=7200):
    start = time.monotonic()
    while not path.is_file():
        if time.monotonic()-start > timeout:
            raise TimeoutError(f'Source synchronization incomplete: {path}')
        time.sleep(1)
    return path


def prepare(root, out):
    """Lock labels, split and train-only normalization before encoder training."""
    source = root/'source'; data = source/'b_coarse'
    selection = ds.read_json(source/'selection.json')
    bundle = ds.read_json(source/'bundle.json')
    if (ds.digest(source/'selection.json') != 'e9e56482e8b6e819487d6c3b5583e22b090ee0023e81490e849ea7b07b446e86'
            or ds.digest(source/'bundle.json') != 'e0f0554ae45bd32e4327534726ff0a7bb1bded22a85d9bd34f72c5a744b3bb31'
            or bundle['sequence'] != SEQUENCE):
        raise ValueError('New119 policy source identity mismatch')
    splits = split_recordings(selection)
    info = ds.read_json(data/'meta/info.json')
    names = [n for n in info['features']['observation.state']['names'] if n.endswith('.position')]
    if len(names) != 58 or info['fps'] != 30:
        raise ValueError('Expected30Hz58 measured positions')
    entries = {r['episode_index']: r for r in ds.read_rows(data/'meta/episodes.jsonl')}
    episodes = []; sums = np.zeros(58); squares = np.zeros(58); count = 0
    for row in selection['episodes']:
        i = row['episode_index']; annpath = data/f'annotations/episode_{i:06d}.json'
        ann = ds.read_json(annpath); ds.validate_segments(ann['segments'], row['length'])
        if ([s['label'] for s in ann['segments']] != SEQUENCE
                or ann['prompt_profile'] != 'tomato-in-hand-five-phases-v3'
                or ann['source']['source_recording_id'] != row['recording_group']
                or entries[i]['source_recording_id'] != row['recording_group']):
            raise ValueError('Annotation/source/five-phase mismatch')
        parquet = wait_file(data/f'data/chunk-000/episode_{i:06d}.parquet')
        rel = str(parquet.relative_to(data))
        if ds.digest(parquet) != ann['source']['sha256'][rel]:
            raise ValueError('Annotated parquet identity mismatch')
        e = dict(episode_id=f'inhand119_{i:06d}', episode_index=i,
                 source_recording_id=row['recording_group'], split=splits[row['recording_group']],
                 length=row['length'], fps=30, info=info, entry=entries[i],
                 parquet=str(parquet), annotation=ann,
                 annotation_sha256=ds.digest(annpath), source_mapping=row)
        e['roi_sidecar_sha256']=ds.digest(wait_file(root/f'roi/episode_{i:06d}.jsonl.gz'))
        _, q = ds.read_state(e, names)
        if e['split'] == 'train':
            sums += q.sum(0, dtype=np.float64)
            squares += np.square(q.astype(np.float64)).sum(0); count += len(q)
        episodes.append(e)
    mean = (sums/count).astype(np.float32)
    std = np.sqrt(np.maximum(squares/count-(sums/count)**2, 0)).clip(.01).astype(np.float32)
    cfg = dict(version='inhand119-independent-readouts-v1', labels=list(ds.LABELS),
               single_architecture='inhand119_single_rgb_memory_v1',
               task_profile='tomato-in-hand-five-phases-v3', ordered_classes=[0,1,3,1,2],
               methods=METHODS, seeds=[42,43,44], epochs=30, batch_size=128,
               hidden=128, learning_rate=.0003, weight_decay=.01,
               future_horizon_s=1.5, raw_fps=30, video_frames=16,
               video_sample_hz=10, observation_hz=2,
               selection='fixed30epochs; no threshold or test tuning',
               head_training='independent Current/Future readouts; frozen shared V-JEPA2',
               roi_source_mode='recorded_inhand119_right_truss_v1',
               roi_uses_future_coordinates=True, roi_uses_future_pixels=False,
               online_roi_validated=False, control_eligible=False,
               language_prompt_input=False, oracle_stage_input=False,
               normalization='95training recordings only',
               selection_sha256=ds.digest(source/'selection.json'),
               bundle_sha256=ds.digest(source/'bundle.json'),
               joint_names=names, cache_encoder_batch=16)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(episodes=episodes, split=splits, protocol=cfg,
                    training_statistics_frames=count)
    for name, value in [('manifest.json', manifest), ('protocol.json', cfg)]:
        path=out/name
        if path.exists() and ds.read_json(path) != value:
            raise ValueError('Refuse changed protocol/manifest on resume')
        ds.write_json(path,value)
    stat = out/'train_state_statistics.npz'
    if stat.exists():
        old=np.load(stat); np.testing.assert_array_equal(old['mean'],mean);np.testing.assert_array_equal(old['std'],std)
    else:
        np.savez(stat,mean=mean,std=std)
    return manifest,mean,std


def extract(root, out, manifest):
    import av, cv2, json
    cache=out/'features';cache.mkdir(exist_ok=True)
    signature=ds.fingerprint(dict(manifest=ds.digest(out/'manifest.json'),
        encoder='25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b',
        implementation=ds.digest(__file__)))
    encoder,provenance=load_encoder('pretrained/vjepa2-vitl-fpc64-256','cuda')
    rows=[]; start=time.perf_counter()
    # Decode cameras independently, so synchronization can overlap encoding.
    for camera in ['cam0','cam1','cam2','interaction']:
        for e in manifest['episodes']:
            eid=e['episode_id'];i=e['episode_index'];anchors=np.arange(45,e['length'],15)
            indices=anchors[:,None]+np.arange(-45,1,3)[None]
            rel=f'videos/chunk-000/observation.images.{camera}/episode_{i:06d}.mp4'
            video=wait_file(root/'source/b_coarse'/rel)
            expected=e['annotation']['source']['sha256'][rel]
            if ds.digest(video)!=expected:
                raise ValueError('Video bytes differ from reviewed annotation')
            path=cache/f'{eid}_{camera}.npz'
            if path.exists():
                z=np.load(path)
                if str(z['signature'])!=signature or str(z['video_sha256'])!=expected:
                    raise ValueError('Feature resume identity mismatch')
                np.testing.assert_array_equal(z['anchors'],anchors)
                continue
            frames=[];seen=0
            with av.open(str(video)) as stream:
                for f in stream.decode(video=0):
                    if f.time is None or abs(float(f.time)-seen/30)>1e-4:
                        raise ValueError('Camera timestamps inconsistent')
                    if seen%3==0:
                        frames.append(cv2.resize(f.to_ndarray(format='rgb24'),(256,256),interpolation=cv2.INTER_AREA))
                    seen+=1
            if seen!=e['length']:raise ValueError('Video frame count mismatch')
            frames=np.asarray(frames);values=[];t=time.perf_counter()
            for ix in np.array_split(np.arange(len(anchors)), int(np.ceil(len(anchors)/16))):
                values.append(encode_clips(encoder,frames[indices[ix]//3]))
            values=np.concatenate(values)
            np.savez_compressed(path,visual=values,anchors=anchors,input_indices=indices,
                                signature=np.array(signature),video_sha256=np.array(expected))
            rows.append(dict(episode_id=eid,camera=camera,windows=len(anchors),
                             encoder_s=time.perf_counter()-t,path=str(path),sha256=ds.digest(path)))
            ds.write_json(out/'extraction_progress.json',dict(camera=camera,episode=i,
                completed_files=len(list(cache.glob('*.npz'))),total_files=476,
                elapsed_s=time.perf_counter()-start,last=rows[-1]))
            print('ENCODED',camera,i,len(anchors),round(rows[-1]['encoder_s'],2),flush=True)
    del encoder;torch.cuda.empty_cache()
    ds.write_json(out/'encoder_receipt.json',dict(signature=signature,pretrained=provenance,
        frozen=True,wall_s=time.perf_counter()-start,files=476,
        implementation_sha256=ds.digest(__file__)))
    allrows=[]
    for e in manifest['episodes']:
        zs=[np.load(cache/f'{e["episode_id"]}_{c}.npz') for c in ['cam0','cam1','cam2','interaction']]
        for z in zs[1:]:np.testing.assert_array_equal(z['input_indices'],zs[0]['input_indices'])
        indices=zs[0]['input_indices'];anchors=zs[0]['anchors'];_,q=ds.read_state(e,manifest['protocol']['joint_names'])
        side=wait_file(root/f'roi/episode_{e["episode_index"]:06d}.jsonl.gz')
        if ds.digest(side)!=e['roi_sidecar_sha256']:raise ValueError('ROI metadata changed')
        with gzip.open(side,'rt') as f: coords=[json.loads(line) for line in f]
        if len(coords)!=e['length'] or any(r['frame_index']!=j or abs(r['timestamp']-j/30)>1e-5 for j,r in enumerate(coords)):
            raise ValueError('ROI metadata alignment mismatch')
        boxes=np.array([r['crop_box'] for r in coords],np.float32)/[640,400,640,400]
        valid=np.array([r['image_available'] for r in coords],np.float32)
        metadata=np.concatenate([boxes[indices].reshape(-1,64),valid[indices],
            (indices-anchors[:,None])/30,np.zeros((len(anchors),16)),
            np.tile([0.,1.,0.,0.],(len(anchors),1))],axis=1).astype(np.float32)
        allrows.append(dict(episode=e,visual=np.stack([z['visual'] for z in zs],axis=1),
            positions=q[indices].reshape(-1,928),metadata=metadata,times=anchors/30,
            anchors=anchors,target=ds.build_phase_targets(anchors,e['annotation']['segments'],e['length'],30,1.5)))
    return allrows


def phase_replay(times, probabilities, segments):
    """Fixed Current>=.8, two fresh votes; adapt legal sequence to new five phases."""
    sequence=[0,1,3,1,2]; phase=0;votes=0;transitions=[];pred=[]
    for t,p in zip(times,probabilities):
        nxt=sequence[phase+1] if phase<4 else None
        votes=votes+1 if nxt is not None and int(p.argmax())==nxt and p[nxt]>=.8 else 0
        if votes>=2:
            phase+=1;votes=0;transitions.append(float(t))
        pred.append(phase)
    truth=np.searchsorted([s['end_frame']/30 for s in segments[:-1]],times,side='right')
    offsets=[transitions[i]-segments[i]['end_frame']/30 if i<len(transitions) else None for i in range(4)]
    return dict(position_accuracy=float(np.mean(np.array(pred)==truth)),
        transitions_s=transitions,offsets_s=offsets,missed=sum(x is None for x in offsets),
        early_gt1s=sum(x is not None and x< -1 for x in offsets),
        late_gt1p5s=sum(x is not None and x>1.5 for x in offsets),ordered_complete=len(transitions)==4)


def evaluate(rows,pred):
    result={}
    for target in ['current','future']:
        ys=[];ps=[]
        for r in rows:
            y=r['target'][target+'_labels'];valid=y>=0
            ys.append(y[valid]);ps.append(pred[r['episode']['episode_id']][target][valid])
        result[target]=metrics.scores(np.log(np.maximum(np.concatenate(ps),1e-30)),np.concatenate(ys))
    result['five_phase_replay']={r['episode']['episode_id']:phase_replay(r['times'],
        pred[r['episode']['episode_id']]['current'],r['episode']['annotation']['segments']) for r in rows}
    return result


def train(out,rows,manifest,mean,std):
    cfg=manifest['protocol'];trainrows=[r for r in rows if r['episode']['split']=='train']
    val=[r for r in rows if r['episode']['split']=='val'];test=[r for r in rows if r['episode']['split']=='test']
    assert [len(x) for x in [trainrows,val,test]]==[95,12,12]
    identity=dict(manifest_sha256=ds.digest(out/'manifest.json'),
        cache_signature=ds.read_json(out/'encoder_receipt.json')['signature'],
        joint_names=cfg['joint_names'],statistics_sha256=ds.digest(out/'train_state_statistics.npz'),
        source_sha256={str(p):ds.digest(p) for p in [Path(__file__),
            Path('tools/train_tomato_independent_rgb_memory.py'),
            Path('tools/tomato_rgb_memory_model.py'),Path('tools/tomato_rgb_memory_training.py')]})
    results={}
    for seed in cfg['seeds']:
        folder=out/f'seed{seed}';folder.mkdir(exist_ok=True)
        for method in METHODS:
            cp=folder/f'{method}.pt';branches={};infos={}
            if cp.exists():
                saved=torch.load(cp,map_location='cpu',weights_only=False)
                if saved['identity']!=identity or saved['protocol']!=cfg:raise ValueError('Resume weights identity mismatch')
                net=restore_reader(saved).cuda().eval()
            else:
                for target in ['current','future']:
                    branches[target],infos[target]=fit_single(trainrows,method,target,seed,cfg,mean,std,
                        folder/f'{method}_{target}.pt',identity)
                    print('TRAINED',seed,method,target,infos[target]['training_s'],flush=True)
                net=IndependentRGBReader(branches['current'],branches['future']).cuda().eval()
                torch.save(dict(format_version=1,architecture=ARCHITECTURE,method=method,seed=seed,
                    model_version=f'inhand119-{method}-s{seed}-{identity["cache_signature"][:12]}',
                    head={k:v.detach().cpu() for k,v in net.state_dict().items()},
                    state_mean=torch.from_numpy(mean),state_std=torch.from_numpy(std),
                    protocol=cfg,identity=identity,training=infos,
                    training_sources=[r['episode']['source_recording_id'] for r in trainrows],
                    input_schema=dict(visual=[4,1024],positions=[16,58],metadata=116,
                        joint_names=cfg['joint_names'],labels=list(ds.LABELS),
                        history=8 if method=='G3_local_only_history8' else 1,
                        task_profile=cfg['task_profile']),control_eligible=False),cp)
            pred=base.predict(net,val,method,cfg);evaluation=evaluate(val,pred)
            results[f'{seed}/{method}']=dict(checkpoint=str(cp.resolve()),sha256=ds.digest(cp),validation=evaluation)
            ds.write_json(out/'results.json',results)
            for r in val:
                np.savez_compressed(folder/f'{method}_val_{r["episode"]["episode_id"]}.npz',**pred[r['episode']['episode_id']])
            # Round-trip the on-disk bundle before accepting training.
            restored=restore_reader(torch.load(cp,map_location='cpu',weights_only=False)).cuda().eval()
            again=base.predict(restored,val[:1],method,cfg)
            for target in ['current','future']:
                np.testing.assert_allclose(again[val[0]['episode']['episode_id']][target],
                    pred[val[0]['episode']['episode_id']][target],rtol=1e-5,atol=1e-6)
            del net,restored,branches;torch.cuda.empty_cache()
            print('SCORED',seed,method,evaluation['current']['macro_f1'],evaluation['future']['macro_f1'],flush=True)
    # Every fixed configuration is now frozen. Held-out test is evaluated once;
    # no candidate selection, retraining or threshold changes based on this set.
    ds.write_json(out/'all_models_frozen.json',{k:v['sha256'] for k,v in results.items()})
    for key,row in results.items():
        saved=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        net=restore_reader(saved).cuda().eval();pred=base.predict(net,test,saved['method'],cfg)
        row['test']=evaluate(test,pred);del net;torch.cuda.empty_cache()
        ds.write_json(out/'results.json',results)
    ds.write_json(out/'receipt.json',dict(status='complete',paired_models=9,trained_branches=18,
        split_counts=dict(train=95,val=12,test=12),encoder_frozen=True,
        roi_uses_future_coordinates=True,online_roi_validated=False,control_eligible=False,
        old100_data_used=False,robot_calls=0,identity=identity,
        implementation_sha256=ds.digest(__file__)))


def run(root,out):
    manifest,mean,std=prepare(root,out)
    ds.status(out,'extracting',paired_models_requested=9,branches_requested=18)
    rows=extract(root,out,manifest)
    ds.status(out,'training',paired_models_requested=9,branches_requested=18)
    train(out,rows,manifest,mean,std)
    ds.status(out,'complete',paired_models=9,trained_branches=18)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:run(a.root,a.out)
    except Exception as exc:
        ds.status(a.out,'failed',error=repr(exc));raise
