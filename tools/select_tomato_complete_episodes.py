"""New complete-task selection; never edit historical manifests or raw episodes."""
from pathlib import Path
import argparse
import copy
from collections import Counter
import numpy as np
from . import tomato_dataset as ds

PARENT = Path('runs/tomato/refresh101_20261109_v2/finalized_exports/refresh101/dataset_manifest.json')
ACTIVE = Path('runs/tomato/complete100_20261111/dataset_manifest.json')
ORDER = ['pickup', 'harvest', 'return', 'adjust', 'pickup', 'harvest', 'return']


def select_complete(source):
    m = copy.deepcopy(source)
    kept, excluded = [], []
    for e in m['episodes']:
        if [s['label'] for s in e['annotation']['segments']] == ORDER:
            kept.append(e)
        else:
            excluded.append(dict(episode_id=e['episode_id'], source_recording_id=e['source_recording_id'],
                                 split=e['split'], dataset_root=e['dataset_root'], episode_index=e['episode_index'],
                                 segment_count=len(e['annotation']['segments']), reason='incomplete_seven_phase_task'))
    m['episodes'] = kept
    m['splits'] = {s: [e['source_recording_id'] for e in kept if e['split'] == s] for s in source['splits']}
    m['split_sha256'] = ds.fingerprint(m['splits'])
    m['summary'] = {}
    for split, ids in m['splits'].items():
        cur = np.zeros(4, dtype=int); future = cur.copy(); invalid = 0
        for e in kept:
            if e['split'] != split: continue
            t = ds.build_phase_targets(np.asarray(e['anchors']), e['annotation']['segments'], e['length'], e['fps'], 1.5)
            cur += np.bincount(t['current_labels'], minlength=4)
            future += np.bincount(t['future_labels'][t['future_valid_mask']], minlength=4)
            invalid += int((~t['future_valid_mask']).sum())
        m['summary'][split] = dict(sources=len(ids), episodes=len(ids), anchors=int(cur.sum()),
                                  class_counts=cur.tolist(), future_class_counts=future.tolist(), future_invalid=invalid)
    m['annotation_index_sha256'] = ds.fingerprint([dict(episode_id=e['episode_id'], source_annotation_sha256=e['annotation']['sha256'], effective_segments=e['annotation']['segments']) for e in kept])
    m['selection_note'] = 'Complete seven-phase task only; original split assignments unchanged; no retraining or new metric claim.'
    return m, excluded


def run(out):
    if out.exists(): raise FileExistsError('New selection output required')
    parent = ds.read_json(PARENT); m, excluded = select_complete(parent)
    assert len(m['episodes']) == 100 and len(excluded) == 1
    identity = ds.read_json('runs/tomato/refresh101_20261109_v2/data_identity.json')
    selected = {e['episode_id'] for e in m['episodes']}
    identity = {**identity, 'episodes': [r for r in identity['episodes'] if r['episode_id'] in selected],
                'parent_identity_sha256': ds.digest('runs/tomato/refresh101_20261109_v2/data_identity.json')}
    out.mkdir(parents=True)
    ds.write_json(out/'split.json', m['splits']); ds.write_json(out/'data_identity.json', identity)
    m['external_split_sha256'] = ds.digest(out/'split.json'); m['data_identity_sha256'] = ds.digest(out/'data_identity.json')
    m['parent_selection_manifest_sha256'] = ds.digest(PARENT)
    ds.write_json(out/'dataset_manifest.json', m)
    receipt = dict(parent_manifest=str(PARENT), parent_sha256=ds.digest(PARENT), manifest=str(out/'dataset_manifest.json'),
                   manifest_sha256=ds.digest(out/'dataset_manifest.json'), recordings=len(m['episodes']),
                   split_counts=dict(Counter(e['split'] for e in m['episodes'])), excluded=excluded,
                   raw_data_deleted=False, historical_results_modified=False, training_repeated=False)
    ds.write_json(out/'selection_receipt.json', receipt)
    print(receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=ACTIVE.parent); a = p.parse_args(); run(a.out)
