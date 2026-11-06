"""List candidate coarse versions; lock ONLY an explicit user selection, read-only inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .tomato_dataset import (digest, ensure_output_outside_sources, load_annotation_index,
                                 load_episode_info, load_reviewed_coarse, read_json, write_json)
except ImportError:
    from tomato_dataset import (digest, ensure_output_outside_sources, load_annotation_index,
                                load_episode_info, load_reviewed_coarse, read_json, write_json)


def scan(roots):
    result = []
    for root in roots:
        for p in sorted((Path(root) / 'annotations/subtasks/coarse').glob('*.json')):
            doc = read_json(p)
            row = dict(dataset_root=str(Path(root).resolve()), coarse_annotation=str(p.resolve()),
                       episode_index=doc.get('source', {}).get('episode_index'), revision=doc.get('revision'),
                       saved_at=doc.get('saved_at'), sha256=digest(p),
                       reviewed_segments=sum(s.get('reviewed') is True for s in doc.get('segments', [])),
                       segments=len(doc.get('segments', [])))
            try:
                e = load_episode_info(root, row['episode_index'])
                load_reviewed_coarse(p, e)
                row['eligible'] = True
            except (ValueError, KeyError, OSError) as exc:
                row.update(eligible=False, error=str(exc))
            result.append(row)
    for r in result:
        r['candidate_count'] = sum((s['dataset_root'], s['episode_index']) ==
                                   (r['dataset_root'], r['episode_index']) for s in result)
    return result


def lock_selection(selection, output):
    rows = load_annotation_index(selection)
    output = Path(output).resolve()
    ensure_output_outside_sources(output.parent, [r['dataset_root'] for r in rows],
                                  [r['coarse_annotation'] for r in rows])
    if output.exists():
        raise ValueError('Index exists; use a new output path, never overwrite a locked selection')
    locked = []
    for row in rows:
        e = load_episode_info(row['dataset_root'], row['episode_index'])
        ann = load_reviewed_coarse(row['coarse_annotation'], e)
        if row.get('annotation_sha256', ann['sha256']) != ann['sha256']:
            raise ValueError('Selected annotation changed')
        locked.append({**row, 'annotation_sha256': ann['sha256'], 'revision': ann['revision']})
    write_json(output, locked)
    return locked


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', action='append', default=[])
    p.add_argument('--report')
    p.add_argument('--selection', help='Explicit JSON/JSONL paths/episode/version selection')
    p.add_argument('--output')
    a = p.parse_args(argv)
    if a.selection:
        if not a.output:
            p.error('--selection requires --output')
        print(json.dumps(lock_selection(a.selection, a.output), ensure_ascii=False, indent=2))
    else:
        if not a.dataset:
            p.error('Provide --dataset for read-only inventory')
        report = scan(a.dataset)
        if a.report:
            ensure_output_outside_sources(Path(a.report).resolve().parent, a.dataset)
            write_json(a.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
