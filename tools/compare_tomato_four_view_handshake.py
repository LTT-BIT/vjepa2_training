"""Replay existing four-view predictions under pinned policies; never retrain."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import numpy as np
from . import tomato_dataset as ds
from .compare_tomato_paired_decisions import (
    b0_replay, colleague_replay, colleague_module, score, summarize, COMMIT,
)
from .tomato_frozen_protocol import verify


def run(source, out):
    if out.exists():
        raise FileExistsError('Preserve prior results; use a new directory')
    cfg = ds.read_json(source/'protocol.json')
    lock = verify(Path(cfg['frozen_lock']))
    manifest = ds.read_json(lock['manifest'])
    episodes = [e for e in manifest['episodes'] if e['split'] == 'val']
    assert len(episodes) == 8
    prior = ds.read_json(source/'results.json')
    out.mkdir(parents=True)
    _, raw = colleague_module()
    (out/'pinned_handshake.py').write_bytes(raw)
    inputs = {}
    summaries = {}
    for seed in cfg['seeds']:
        summaries[str(seed)] = {}
        for method in ['G1_local_current', 'G2_local_history8']:
            per = {a: {} for a in ('current_native', 'current_same_clock', 'future_commit_current_ack_k2')}
            traces = {}
            for e in episodes:
                eid = e['episode_id']
                path = source/f'seed{seed}'/f'{method}_{eid}.npz'
                inputs[str(path)] = ds.digest(path)
                with np.load(path) as z:
                    c, f, ts = z['current'].astype(float), z['future'].astype(float), z['times']
                    assert np.allclose(ts, z['anchors']/e['fps'])
                duration = min(e['length']/e['fps'], float(ts[-1]+.5))
                for arm in per:
                    if arm == 'future_commit_current_ack_k2':
                        changes, meta = colleague_replay(c, f, ts, duration, 2, True)
                    else:
                        changes, meta = b0_replay(c, f, ts, duration, arm == 'current_same_clock')
                    per[arm][eid] = score(e, ts, changes, meta)
                    traces[f'{eid}/{arm}'] = dict(changes=changes, metadata=meta)
            summaries[str(seed)][method] = {a: summarize(p) for a, p in per.items()}
            assert summaries[str(seed)][method]['current_native']['prompt_counts'] == prior[str(seed)][method]['evaluation']['summary']['prompt_counts']
            ds.write_json(out/f'{method}_s{seed}.json', dict(scores=per, traces=traces))
    aggregate = {}
    for method in summaries['42']:
        aggregate[method] = {}
        for arm in summaries['42'][method]:
            rows = [summaries[str(s)][method][arm] for s in cfg['seeds']]
            counts = Counter()
            for r in rows:
                counts.update(r['prompt_counts'])
            aggregate[method][arm] = dict(prompt_counts=dict(counts),
                repeated_events=sum(r['events'] for r in rows),
                non_early_within_1p5s=sum(r['within_delay']['1.5'] for r in rows),
                mismatch_exact_s=sum(r['mismatch_exact_s'] for r in rows),
                complete=sum(r['ordered_complete'] for r in rows),
                timeouts=sum(r['timeouts'] for r in rows))
    ds.write_json(out/'results.json', dict(per_seed=summaries, aggregate=aggregate))
    ds.write_json(out/'receipt.json', dict(inputs=inputs, manifest_sha256=lock['manifest_sha256'],
        git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        handshake_commit=COMMIT, runner_sha256=ds.digest(__file__),
        source_results_sha256=ds.digest(source/'results.json'),
        scope='Exposed development8,48 boundaries repeated across3 seeds; not independent144 events',
        clocks='simulated .09s availability/.5s planning; not measured deployment latency',
        interpretation='Future commit then Current ACK; ACK is NOT a pre-commit safety check',
        threshold_search=False, training=False, historical_test_scored=False,
        default_changed=False, robot_calls=0))
    print(aggregate)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.out)
