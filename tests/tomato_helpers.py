"""Synthetic fixtures only. These small videos/tensors carry no accuracy evidence."""
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools import tomato_dataset as ds


def make_episode(root, index=0, source=None, length=120):
    root = Path(root)
    names = [f'{s}_fr3_joint{i}.position' for s in ('left', 'right') for i in range(1, 8)]
    names += [f'{s}_thumb_IP.position' for s in ('left', 'right')]
    info = dict(codebase_version='v2.1', robot_type='dual_fr3_sharpa', fps=30, chunks_size=1000,
                data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
                features={'observation.state': dict(shape=[len(names)], names=names, dtype='float32')})
    for c in ds.CAMERAS:
        info['features'][f'observation.images.{c}'] = dict(dtype='video', shape=[32, 32, 3])
    ds.write_json(root / 'meta/info.json', info)
    entry = dict(episode_index=index, length=length, source_recording_id=source or f'source-{index}',
                 source_start_ns=100, source_end_ns=100 + int(length / 30 * 1e9))
    (root / 'meta/episodes.jsonl').write_text(json.dumps(entry) + '\n')
    (root / 'meta/tasks.jsonl').write_text('{"task_index":0,"task":"deliberately ignored"}\n')
    parquet = root / f'data/chunk-000/episode_{index:06d}.parquet'
    parquet.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({'timestamp': np.arange(length) / 30, 'frame_index': np.arange(length),
                      'episode_index': np.full(length, index),
                      'observation.state': np.arange(length * len(names), dtype=np.float32).reshape(length, -1).tolist(),
                      'action': np.full((length, 2), np.nan).tolist(),
                      'task_index': np.full(length, 99), 'coarse_stage_index': np.full(length, 99)})
    pq.write_table(table, parquet)
    for cam, c in enumerate(ds.CAMERAS):
        p = root / f'videos/chunk-000/observation.images.{c}/episode_{index:06d}.mp4'
        p.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'mp4v'), 30., (32, 32))
        assert writer.isOpened()
        for i in range(length):
            writer.write(np.full((32, 32, 3), i % 200 + cam, dtype=np.uint8))
        writer.release()
    e = ds.load_episode_info(root, index)
    files = ['meta/info.json', 'meta/episodes.jsonl', 'meta/tasks.jsonl',
             str(parquet.relative_to(root)), *[str(Path(p).relative_to(root)) for p in e['videos']]]
    cuts = np.linspace(0, length, 5, dtype=int)
    doc = dict(schema_version=2, stage='coarse', revision=1, saved_at=100., inherited=False,
               source={k: e[k] for k in ('episode_index', 'length', 'fps', 'source_recording_id')},
               segments=[dict(start_frame=int(a), end_frame=int(b), label=label, reviewed=True)
                         for a, b, label in zip(cuts[:-1], cuts[1:], ds.LABELS)])
    doc['source']['sha256'] = {p: ds.digest(root / p) for p in files}
    ann = root / f'annotations/subtasks/coarse/episode_{index:06d}.json'
    ds.write_json(ann, doc)
    return e, ann, names


def make_config(tmp_path, count=3, length=120):
    rows = []
    for i in range(count):
        e, ann, names = make_episode(tmp_path / f'data-{i}', i, length=length)
        rows.append(dict(dataset_root=e['dataset_root'], episode_index=i,
                         coarse_annotation=str(ann), variant='normal'))
    index = tmp_path / 'index.json'
    ds.write_json(index, rows)
    return dict(project='tomato_sharpa_four_stage', robot='sharpa', variant='normal',
                cameras=list(ds.CAMERAS), labels=list(ds.LABELS), model_id=ds.MODEL_ID,
                model_revision=ds.REVISION, history_frames=16, history_sample_hz=10,
                prediction_hz=2, image_size=256, joint_mode='arms-and-hands', joint_names=names,
                annotation_index=str(index), output_dir=str(tmp_path / 'run'),
                pretrained_dir=str(tmp_path / 'pretrained'), seed=42, device='cpu', batch_windows=2,
                epochs=2, patience=15, batch_size=128, learning_rate=3e-4, weight_decay=.01)
