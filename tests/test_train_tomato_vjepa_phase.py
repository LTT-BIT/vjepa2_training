import copy
from pathlib import Path

import pytest

from tools import tomato_dataset as ds
from tools import train_tomato_vjepa_phase as tr
from tomato_helpers import make_config


def test_group_split_reproducible_and_source_locked():
    episodes = [dict(source_recording_id=f'group-{i // 2}') for i in range(30)]
    split = tr.grouped_split(episodes)
    assert split == tr.grouped_split(list(reversed(episodes)))
    assert len({g for rows in split.values() for g in rows}) == 15
    with pytest.raises(ValueError):
        tr.grouped_split([dict(source_recording_id='same')] * 5)
    bad = copy.deepcopy(split); bad['test'].append(bad['train'][0])
    with pytest.raises(ValueError, match='leakage'):
        tr.grouped_split(episodes, explicit=bad)


def test_prepare_locks_full_manifest_no_gpu_and_rejects_changes(tmp_path):
    cfg = make_config(tmp_path)
    m = tr.prepare(cfg)
    assert m == tr.prepare(cfg)
    assert set(m['summary']) == {'train', 'val', 'test'}
    assert all(s['class_counts'] == [2, 2, 2, 2] for s in m['summary'].values())
    assert len({g for rows in m['splits'].values() for g in rows}) == 3
    old = ds.digest(Path(cfg['output_dir']) / 'manifest.json')
    ann = Path(m['episodes'][0]['annotation']['path'])
    doc = ds.read_json(ann); doc['revision'] += 1; ds.write_json(ann, doc)
    with pytest.raises(ValueError, match='Locked'):
        tr.prepare(cfg)
    assert ds.digest(Path(cfg['output_dir']) / 'manifest.json') == old


def test_prepare_rejects_source_hash_and_full_schema_change(tmp_path):
    cfg = make_config(tmp_path)
    p = Path(ds.read_json(cfg['annotation_index'])[0]['dataset_root']) / 'meta/info.json'
    doc = ds.read_json(p); doc['unrelated_export_note'] = True; ds.write_json(p, doc)
    with pytest.raises(ValueError, match='invalid selected'):
        tr.prepare(cfg)
    rejected = ds.read_json(Path(cfg['output_dir']) / 'prepare_rejections.json')
    assert 'SHA256 mismatch' in rejected['rejected'][0]['error']
    assert not (Path(cfg['output_dir']) / 'manifest.json').exists()
