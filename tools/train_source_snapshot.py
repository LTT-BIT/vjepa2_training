#!/usr/bin/env python3
"""Train a causal three-camera V-JEPA 2 phase classifier; no robot commands."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

DATASET = Path('/home/descfly/Convert_data/outputs/lemon_move_wrist_reset_20261008')
ANNOTATIONS = Path('/home/descfly/Downloads/phase_annotations_v3_clean.json')
RUN = Path('/home/descfly/hmj/training/experiments/lemon_vjepa2_phase_20261027')
MODEL_ID = 'facebook/vjepa2-vitl-fpc64-256'
REVISION = 'b3c1679b7c34d3255ef3547f27c7b226aefab26f'
BOUNDARIES = ('left_support_ready_s', 'scoop_ready_s', 'lemon_supported_s', 'pour_ready_s', 'task_complete_s')
LABELS = ('establish_left_support', 'align_scoop', 'scoop_lemon', 'move_to_cup', 'pour', 'task_complete')
CHINESE = ('建立左手支撑', '对准铲子', '铲取柠檬片', '移向杯口', '倒入杯子', '任务完成')
CAMERAS = ('cam0', 'cam1', 'cam2')
JOINTS = tuple(f'{side}_{j}_joint.position' for side in ('left', 'right') for j in
               ('shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3'))
CLIP, SIZE, STRIDE, SEED = 16, 256, 5, 42


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def progress(run, stage, **values):
    data = dict(stage=stage, updated_at=datetime.now().astimezone().isoformat(), **values)
    write_json(run/'status.json', data)
    print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)


def paths(root, i):
    stem = f'episode_{i:06d}'
    return ([root/f'videos/chunk-{i//1000:03d}/observation.images.{c}/{stem}.mp4' for c in CAMERAS],
            root/f'data/chunk-{i//1000:03d}/{stem}.parquet')


def history_indices(end, count=CLIP):
    if end < 0 or count <= 0:
        raise ValueError('Invalid causal history')
    return np.maximum(0, np.arange(end-count+1, end+1))


def labels_at(times, boundaries):
    b = np.asarray(boundaries, dtype=float)
    if b.shape != (5,) or not np.isfinite(b).all() or b[0] < 0 or np.any(np.diff(b) <= 0):
        raise ValueError('Expected five increasing finite timestamps')
    return np.searchsorted(b, times, side='right').astype(np.int64)


def grouped_split(episodes):
    groups = sorted({e['source'] for e in episodes})
    if len(groups) < 10:
        raise ValueError('Insufficient independent recordings')
    random.Random(SEED).shuffle(groups)
    held = max(1, round(.1*len(groups)))
    splits = {g: 'test' if i < held else 'val' if i < 2*held else 'train' for i,g in enumerate(groups)}
    return [{**e, 'split': splits[e['source']]} for e in episodes]


def read_state(root, i, columns):
    table = pq.read_table(paths(root, i)[1], columns=['timestamp', 'observation.state'])
    times = table['timestamp'].to_numpy().astype(np.float64)
    q = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)[:, columns]
    if (len(times) < CLIP or not np.isfinite(times).all() or not np.isfinite(q).all()
            or not np.allclose(np.diff(times), .1, atol=1e-5)):
        raise ValueError(f'Invalid contiguous 10Hz robot timeline: {i}')
    return times, q


def prepare(run, root, annotations):
    doc = json.loads(annotations.read_text())
    if doc.get('schema_version') != 3 or doc.get('boundary_fields') != list(BOUNDARIES):
        raise ValueError('Incompatible annotation schema')
    info = json.loads((root/'meta/info.json').read_text())
    if info['fps'] != 10:
        raise ValueError('Expected 10Hz data')
    columns = [info['features']['observation.state']['names'].index(j) for j in JOINTS]
    sources = {}
    for p in sorted((root/'conversion/records').glob('*.json')):
        r = json.loads(p.read_text())
        for e in r['episodes']:
            i = e['episode_index']
            if i in sources and sources[i] != r['source']:
                raise ValueError(f'Conflicting recording source: {i}')
            sources[i] = r['source']
    episodes, seen = [], set()
    for r in sorted(doc['episodes'], key=lambda x:x['episode_index']):
        i = r['episode_index']
        if i in seen or r.get('status') != 'complete' or r.get('excluded_from_training'):
            raise ValueError(f'Duplicate, incomplete or excluded annotation: {i}')
        seen.add(i)
        videos, parquet = paths(root, i)
        hashes = [digest(v) for v in videos]
        if hashes[0] != r['video_sha256']:
            raise ValueError(f'Annotation/video SHA256 mismatch: {i}')
        times, _ = read_state(root, i, columns)
        b = [r[k] for k in BOUNDARIES]
        labels_at(times, b)
        if len(times) != r['frame_count'] or b[-1] > times[-1]+1e-5:
            raise ValueError(f'Annotation timeline changed: {i}')
        for key in BOUNDARIES:
            ix = r['frame_indices'][key]
            if type(ix) is not int or not 0 <= ix < len(times) or abs(times[ix]-r[key]) > 1e-5:
                raise ValueError(f'Boundary/frame mismatch: {i}/{key}')
        for v in videos:
            cap = cv2.VideoCapture(str(v))
            try:
                if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(times) or abs(cap.get(cv2.CAP_PROP_FPS)-10) > .01:
                    raise ValueError(f'Video length/FPS mismatch: {v}')
            finally:
                cap.release()
        if i not in sources:
            raise ValueError(f'Missing recording provenance: {i}')
        episodes.append(dict(episode=i, source=sources[i], frames=len(times), video_sha256=hashes,
                             boundaries=b, parquet_sha256=digest(parquet)))
    episodes = grouped_split(episodes)
    # Choose the demonstration before training, independently of prediction quality.
    demo = next(e['episode'] for e in episodes if e['split'] == 'test')
    manifest = dict(schema_version=1, dataset=str(root), annotations=str(annotations), annotations_sha256=digest(annotations),
                    model_id=MODEL_ID, revision=REVISION, clip_frames=CLIP, image_size=SIZE, stride_frames=STRIDE,
                    joint_columns=columns, cameras=CAMERAS, joints=JOINTS, labels=LABELS, boundaries=BOUNDARIES,
                    episodes=episodes, demo_episode=demo, seed=SEED, causal=True,
                    window_padding='repeat first observed frame', normalization='train split only',
                    split_method='original recording grouped 80/10/10')
    write_json(run/'manifest.json', manifest)
    progress(run, 'prepared', episodes=len(episodes), splits=dict(Counter(e['split'] for e in episodes)), demo_episode=demo)
    return manifest


def load_encoder(run):
    from transformers import VJEPA2Model
    model = VJEPA2Model.from_pretrained(run/'pretrained', local_files_only=True,
                                       torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    return model.encoder.eval().requires_grad_(False).cuda()


def decode_videos(root, episode, count):
    views = []
    for path in paths(root, episode)[0]:
        cap, frames = cv2.VideoCapture(str(path)), []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE,SIZE), interpolation=cv2.INTER_AREA))
        finally:
            cap.release()
        if len(frames) != count:
            raise ValueError(f'Full video decode/state mismatch: {path}: {len(frames)} != {count}')
        views.append(np.stack(frames))
    return views


@torch.inference_mode()
def encode_clips(encoder, clips):
    if clips.ndim!=5 or clips.shape[1:]!=(CLIP,SIZE,SIZE,3) or clips.dtype!=np.uint8:
        raise ValueError('Expected uint8 RGB clips shaped B,16,256,256,3')
    pixels = torch.from_numpy(clips).cuda().permute(0,1,4,2,3).to(torch.bfloat16).div_(255)
    mean = pixels.new_tensor([.485,.456,.406])[None,None,:,None,None]
    std = pixels.new_tensor([.229,.224,.225])[None,None,:,None,None]
    features = encoder(pixel_values_videos=(pixels-mean)/std).last_hidden_state.mean(dim=1).float().cpu().numpy()
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite pretrained encoder features')
    return features


def extract(run, manifest, encoder, batch_windows):
    root = Path(manifest['dataset'])
    directory = run/'features'
    directory.mkdir(exist_ok=True)
    started = time.monotonic()
    for n,e in enumerate(manifest['episodes'],1):
        key = hashlib.sha256(json.dumps(dict(episode=e, model=REVISION, clip=CLIP, size=SIZE, stride=STRIDE,
                                            pooling='mean', robot='16x12', preprocess='RGB full resize ImageNet'),sort_keys=True).encode()).hexdigest()
        out = directory/f'{e["episode"]:06d}.npz'
        if out.exists():
            with np.load(out,allow_pickle=False) as cache:
                if str(cache['fingerprint']) == key:
                    continue
            raise ValueError(f'Feature cache mismatch: {out}')
        progress(run,'extracting',completed=n-1,total=len(manifest['episodes']),episode=e['episode'],elapsed_s=time.monotonic()-started)
        times,q = read_state(root,e['episode'],manifest['joint_columns'])
        views = decode_videos(root,e['episode'],len(times))
        # Sampling times must not depend on ground-truth phase boundaries.
        indices = np.arange(0,len(times),STRIDE,dtype=int)
        features = []
        for start in range(0,len(indices),batch_windows):
            ends = indices[start:start+batch_windows]
            clips = np.stack([v[history_indices(int(i))] for i in ends for v in views])
            visual = encode_clips(encoder,clips).reshape(len(ends),-1)
            robot = np.stack([q[history_indices(int(i))].reshape(-1) for i in ends])
            features.append(np.concatenate([visual,robot],axis=1))
        tmp = out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,features=np.concatenate(features),labels=labels_at(times[indices],e['boundaries']),
                            indices=indices,times=times[indices],fingerprint=np.array(key))
        tmp.replace(out)
        del views
    progress(run,'features_complete',episodes=len(manifest['episodes']),elapsed_s=time.monotonic()-started)


def scores(logits,y):
    logits = np.asarray(logits,dtype=np.float64)
    if logits.shape != (len(y),6) or not len(y) or not np.isfinite(logits).all():
        raise ValueError('Invalid phase scores')
    probs = np.exp(logits-logits.max(axis=1,keepdims=True))
    probs /= probs.sum(axis=1,keepdims=True)
    pred = probs.argmax(axis=1)
    confusion = np.zeros((6,6),dtype=int)
    np.add.at(confusion,(y,pred),1)
    denominator = confusion.sum(axis=0)+confusion.sum(axis=1)
    f1 = np.divide(2*confusion.diagonal(),denominator,out=np.zeros(6),where=denominator>0)
    return dict(frames=len(y),accuracy=float((pred==y).mean()),macro_f1=float(f1.mean()),per_class_f1=f1.tolist(),
                confusion_true_rows=confusion.tolist(),nll=float(-np.log(np.maximum(probs[np.arange(len(y)),y],1e-12)).mean()))


def head_model(dim):
    return nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,6))


@torch.inference_mode()
def predict(head,x,mean,std):
    return np.concatenate([head(torch.from_numpy((p-mean)/std).cuda()).cpu().numpy()
                           for p in np.array_split(x,max(1,math.ceil(len(x)/512)))])


def load_head(run):
    saved = torch.load(run/'best_head.pt',map_location='cpu',weights_only=False)
    if saved['manifest_sha256'] != digest(run/'manifest.json'):
        raise ValueError('Checkpoint/manifest mismatch')
    head = head_model(saved['dimension']).cuda().eval()
    head.load_state_dict(saved['head'])
    return saved,head


def train(run,manifest):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    arrays = {s:[] for s in ('train','val','test')}
    for e in manifest['episodes']:
        with np.load(run/'features'/f'{e["episode"]:06d}.npz',allow_pickle=False) as data:
            arrays[e['split']].append((data['features'],data['labels']))
    arrays = {s:(np.concatenate([p[0] for p in a]),np.concatenate([p[1] for p in a])) for s,a in arrays.items()}
    x,y = arrays['train']
    mean = x.mean(axis=0,dtype=np.float64).astype(np.float32)
    std = x.std(axis=0,dtype=np.float64).clip(.01).astype(np.float32)
    head = head_model(x.shape[1]).cuda()
    optimizer = torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.01)
    counts = np.bincount(y,minlength=6)
    if np.any(counts==0):
        raise ValueError('A training phase has no samples')
    weight = torch.tensor(np.sqrt(counts.sum()/counts),dtype=torch.float32,device='cuda')
    weight /= weight.mean()
    best, best_epoch, stale, history = (-1.,-float('inf')),0,0,[]
    for epoch in range(1,101):
        head.train()
        losses = []
        for ix in np.array_split(rng.permutation(len(y)),max(1,math.ceil(len(y)/128))):
            batch,target = torch.from_numpy((x[ix]-mean)/std).cuda(),torch.from_numpy(y[ix]).cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(batch),target,weight=weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        val = scores(predict(head,arrays['val'][0],mean,std),arrays['val'][1])
        row = dict(epoch=epoch,train_loss=float(np.mean(losses)),val=val)
        history.append(row)
        with (run/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
        key = (val['macro_f1'],-val['nll'])
        if key>best:
            best,best_epoch,stale = key,epoch,0
            torch.save(dict(head={k:v.detach().cpu() for k,v in head.state_dict().items()},mean=mean,std=std,
                            dimension=x.shape[1],epoch=epoch,manifest_sha256=digest(run/'manifest.json'),labels=LABELS,
                            model_id=MODEL_ID,revision=REVISION,clip=CLIP,size=SIZE,stride=STRIDE),run/'best_head.tmp.pt')
            (run/'best_head.tmp.pt').replace(run/'best_head.pt')
        else:
            stale += 1
        progress(run,'training',epoch=epoch,best_epoch=best_epoch,train_loss=row['train_loss'],
                 val_loss=val['nll'],val_macro_f1=val['macro_f1'],val_accuracy=val['accuracy'])
        if stale>=15:
            break
    saved,head = load_head(run)
    report = dict(best_epoch=best_epoch,selection='val macro F1, tie-break val NLL',history=history,
                  train_class_counts=counts.tolist(),head_parameters=sum(p.numel() for p in head.parameters()),
                  splits={s:scores(predict(head,xx,saved['mean'],saved['std']),yy) for s,(xx,yy) in arrays.items()})
    write_json(run/'evaluation.json',report)
    return report


def benchmark(run,manifest,encoder,saved,head):
    e = next(e for e in manifest['episodes'] if e['episode']==manifest['demo_episode'])
    root = Path(manifest['dataset'])
    views = decode_videos(root,e['episode'],e['frames'])
    _,q = read_state(root,e['episode'],manifest['joint_columns'])
    measured = []
    for k,i in enumerate(np.arange(CLIP-1,e['frames'],STRIDE)[:45]):
        torch.cuda.synchronize()
        start = time.perf_counter()
        clips = np.stack([v[history_indices(int(i))] for v in views])
        x = np.concatenate([encode_clips(encoder,clips).reshape(1,-1),q[history_indices(int(i))].reshape(1,-1)],axis=1)
        predict(head,x,saved['mean'],saved['std'])
        torch.cuda.synchronize()
        if k>=5:
            measured.append((time.perf_counter()-start)*1000)
    if len(measured)<10:
        raise ValueError('Insufficient latency samples')
    caps = [cv2.VideoCapture(str(p)) for p in paths(root,e['episode'])[0]]
    prep = []
    try:
        for _ in range(35):
            raw = [cap.read() for cap in caps]
            if not all(ok for ok,_ in raw):
                raise ValueError('Source video ended during benchmark')
            start = time.perf_counter()
            for _,bgr in raw:
                cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(SIZE,SIZE),interpolation=cv2.INTER_AREA)
            prep.append((time.perf_counter()-start)*1000)
    finally:
        for cap in caps:
            cap.release()
    a,p = np.array(measured),np.array(prep)
    report = dict(device=torch.cuda.get_device_name(0),views=3,clip_frames=CLIP,input_fps=10,target_prediction_hz=2,
                  measured_iterations=len(a),inference_mean_ms=float(a.mean()),inference_p95_ms=float(np.percentile(a,95)),
                  inference_max_ms=float(a.max()),three_view_preprocess_mean_ms=float(p.mean()),
                  end_to_end_mean_ms=float(a.mean()+p.mean()),
                  end_to_end_p95_conservative_ms=float(np.percentile(a,95)+np.percentile(p,95)),
                  sustained_2hz_budget_p95_ms=float(np.percentile(a,95)+5*np.percentile(p,95)),
                  sustainable_estimated_hz=float(1000/a.mean()),
                  meets_2hz_compute_budget=bool(np.percentile(a,95)+5*np.percentile(p,95)<500),
                  gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,startup_history_s=1.5,
                  scope='Offline prerecorded input; synchronized CUDA; clip assembly, transfer, normalization, 3-view encoder, robot fusion and head. RGB resize separately measured. Excludes ROS transport and concurrent Pi05 GPU contention.')
    write_json(run/'latency.json',report)
    progress(run,'benchmark_complete',**report)
    return report


def render_demo(run,manifest,saved,head):
    from PIL import Image,ImageDraw,ImageFont
    e = next(e for e in manifest['episodes'] if e['episode']==manifest['demo_episode'])
    if e['split']!='test' or any(r['source']==e['source'] and r['split']=='train' for r in manifest['episodes']):
        raise ValueError('Demo source leaked into training')
    with np.load(run/'features'/f'{e["episode"]:06d}.npz',allow_pickle=False) as d:
        x,indices = d['features'],d['indices']
    probs = torch.softmax(torch.from_numpy(predict(head,x,saved['mean'],saved['std'])),dim=1).numpy()
    root = Path(manifest['dataset'])
    times,_ = read_state(root,e['episode'],manifest['joint_columns'])
    cap = cv2.VideoCapture(str(paths(root,e['episode'])[0][0]))
    w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = run/f'test_episode_{e["episode"]:06d}_phases_raw.mp4'
    final = run/f'test_episode_{e["episode"]:06d}_phases.mp4'
    writer = cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),10.,(w,h+300))
    if not writer.isOpened():
        raise RuntimeError('Video writer failed')
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',24)
    truth = labels_at(times,e['boundaries'])
    rows = []
    try:
        for i,t in enumerate(times):
            ok,bgr = cap.read()
            if not ok:
                raise ValueError('Demo video ended early')
            pos = int(np.searchsorted(indices,i,side='right')-1)
            if pos<0:
                raise ValueError('Future prediction in replay')
            pred = int(probs[pos].argmax())
            image = Image.new('RGB',(w,h+300),(22,24,29))
            image.paste(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),(0,0))
            draw = ImageDraw.Draw(image)
            draw.text((15,h+8),f'保留测试集 Episode {e["episode"]} | {t:.1f}s | 三路图像+机器人状态 | 仅使用过去帧',font=font,fill='white')
            draw.text((15,h+44),f'预测：{CHINESE[pred]} {probs[pos,pred]:.1%}    人工标注：{CHINESE[truth[i]]}',font=font,
                      fill=(100,230,120) if pred==truth[i] else (255,160,90))
            for k,label in enumerate(CHINESE):
                y = h+88+k*33
                draw.text((18,y),f'{k} {label}',font=font,fill='white')
                draw.rectangle((235,y+7,235+int(probs[pos,k]*(w-350)),y+23),fill=(85,160,240))
                draw.text((w-100,y),f'{probs[pos,k]:.1%}',font=font,fill='white')
            writer.write(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR))
            rows.append(dict(frame=i,time_s=float(t),prediction=pred,truth=int(truth[i]),
                             last_input_frame=int(indices[pos]),probabilities=probs[pos].tolist()))
    finally:
        cap.release()
        writer.release()
    write_json(run/'demo_predictions.json',dict(episode=e['episode'],split='test',source=e['source'],rows=rows))
    if shutil.which('ffmpeg'):
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(raw),'-c:v','libx264','-preset','fast',
                        '-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(final)],check=True)
        raw.unlink()
    else:
        raw.replace(final)
    cap = cv2.VideoCapture(str(final))
    try:
        for j,t in enumerate(np.linspace(0,times[-1],6)):
            cap.set(cv2.CAP_PROP_POS_MSEC,float(t*1000))
            ok,image = cap.read()
            if ok and not cv2.imwrite(str(run/f'demo_preview_{j}.jpg'),image):
                raise RuntimeError('Preview save failed')
    finally:
        cap.release()
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--dataset',type=Path,default=DATASET)
    parser.add_argument('--annotations',type=Path,default=ANNOTATIONS)
    parser.add_argument('--batch-windows',type=int,default=4)
    a = parser.parse_args(argv)
    if not 1<=a.batch_windows<=8:
        parser.error('batch-windows must be in 1..8')
    a.run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    try:
        if (a.run/'manifest.json').exists():
            manifest = json.loads((a.run/'manifest.json').read_text())
            if manifest['annotations_sha256']!=digest(a.annotations) or manifest['dataset']!=str(a.dataset):
                raise ValueError('Manifest inputs changed; use a new run directory')
        else:
            manifest = prepare(a.run,a.dataset,a.annotations)
        if a.mode=='prepare':
            return 0
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        progress(a.run,'loading_encoder')
        encoder = load_encoder(a.run)
        progress(a.run,'encoder_loaded',parameters=sum(p.numel() for p in encoder.parameters()),frozen=True)
        extract(a.run,manifest,encoder,a.batch_windows)
        evaluation = train(a.run,manifest)
        saved,head = load_head(a.run)
        benchmark(a.run,manifest,encoder,saved,head)
        video = render_demo(a.run,manifest,saved,head)
        progress(a.run,'complete',test=evaluation['splits']['test'],video=str(video))
        return 0
    except Exception as exc:
        progress(a.run,'failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    raise SystemExit(main())
