"""Training-only representative pilot; stage intervals locate, never label outcomes."""
from pathlib import Path
import argparse
import numpy as np
from . import tomato_dataset as ds
from .select_tomato_complete_episodes import ACTIVE
from .tomato_operation_review_quality import enrich_cases


def select_pilot(episodes, count=16):
    rows=sorted([e for e in episodes if e['split']=='train'],key=lambda e:e['episode_id'])
    if not 12<=count<=18 or len(rows)<count:raise ValueError('Pilot requires12-18 training recordings')
    duration=np.array([[s['end_frame']-s['start_frame'] for s in e['annotation']['segments']] for e in rows],float)
    x=np.c_[np.log(duration.sum(1)),duration/duration.sum(1)[:,None]]
    x=(x-x.mean(0))/np.maximum(x.std(0),.01)
    groups={e['dataset_root'] for e in rows};chosen=[]
    # Cover source packages first, then duration/phase-balance extremes.
    for group in sorted(groups):
        ix=[i for i,e in enumerate(rows) if e['dataset_root']==group]
        pick=min(ix,key=lambda i:(float(np.sum(x[i]**2)),rows[i]['episode_id']))
        if len(chosen)<count:chosen.append(pick)
    while len(chosen)<count:
        candidates=[i for i in range(len(rows)) if i not in chosen]
        pick=max(candidates,key=lambda i:(float(np.min(((x[chosen]-x[i])**2).sum(1))),rows[i]['episode_id']))
        chosen.append(pick)
    return [rows[i] for i in chosen]


def run(out):
    if out.exists():raise FileExistsError('Fresh pilot output required')
    m=ds.read_json(ACTIVE);selected=select_pilot(m['episodes']);ids={e['episode_id'] for e in selected}
    parent=Path('runs/tomato/operation_memory_20261111_complete100');catalog=[r for r in ds.read_json(parent/'catalog.json') if r['episode_id'] in ids]
    cases=[r for r in ds.read_json(parent/'cases.json') if r['episode_id'] in ids]
    cases=enrich_cases(cases,m['episodes'])
    rank={e['episode_id']:i for i,e in enumerate(selected)};priority={'truss_stably_lifted':0,'tomato_detached':1,'truss_returned':2}
    cases.sort(key=lambda r:(rank[r['episode_id']],priority.get(r['event_type'],3),r['event_type']))
    assert len(catalog)==16 and len(cases)==128 and all(r['split']=='train' for r in catalog)
    out.mkdir(parents=True);ds.write_json(out/'catalog.json',catalog);ds.write_json(out/'cases.json',cases)
    old=ds.read_json(parent/'identity.json')
    identity=dict(manifest=str(ACTIVE),manifest_sha256=ds.digest(ACTIVE),catalog_sha256=ds.digest(out/'catalog.json'),
                  previous_catalog_sha256=[old['catalog_sha256']]+old.get('previous_catalog_sha256',[]),
                  selection_note='训练集操作结果试点：16条录制、128项。先看稳定拿起、实际脱离、稳定放回。不是重标Subtask；遮挡请选Unknown。非试点的原保存意见仍保留在原目录/浏览器存储。',
                  original_annotations_modified=False,review_label_status='awaiting_independent_review',pilot=True)
    identity['review_schema']='tomato-operation-review-v2'
    ds.write_json(out/'identity.json',identity)
    for name in ('reviews.json','approved.json'):
        source=ds.read_json(parent/name)
        case_ids={r['case_id'] for r in cases}
        for key in ('reviews','approved','pending'):
            if key in source:source[key]=[r for r in source[key] if r.get('case_id') in case_ids]
        ds.write_json(out/name,source)
    (out/'index.html').write_text(Path('tools/tomato_operation_review.html').read_text())
    receipt=dict(version='tomato-operation-pilot16-v1',manifest_sha256=ds.digest(ACTIVE),selection_rule='Training only; one central recording from each source package, then farthest standardized log-duration and7 phase-fractions; no validation/test/model scores used',
                 source_packages=len({e['dataset_root'] for e in selected}),count=len(selected),cases=len(cases),
                 independent_outcome_truth=False,success_failure_balance='Unknown until human review; representative timing does not prove representative outcome difficulty',
                 recordings=[dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],dataset_root=e['dataset_root'],episode_index=e['episode_index'],split=e['split'],duration_s=e['length']/e['fps'],phase_duration_s=[(s['end_frame']-s['start_frame'])/e['fps'] for s in e['annotation']['segments']]) for e in selected],
                 catalog_sha256=ds.digest(out/'catalog.json'),cases_sha256=ds.digest(out/'cases.json'),labels_changed=False,training=False)
    ds.write_json(out/'selection.json',receipt);print('PILOT',len(selected),'train recordings;',len(cases),'unlabelled cases;',receipt['source_packages'],'source packages')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
