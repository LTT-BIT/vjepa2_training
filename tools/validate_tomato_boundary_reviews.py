"""Validate exported human opinions without ever writing annotation files."""
import argparse
from . import tomato_dataset as ds

STATUSES={'pending_human_review','retain_original','propose_revision','ambiguous','unobservable'}


def validate(payload,case_data):
    if payload.get('convention')!=case_data['convention']:raise ValueError('Convention mismatch')
    cases={c['case_id']:c for c in case_data['cases']};used=set()
    for r in payload['reviews']:
        key=(r['case_id'],r.get('reviewer'))
        if key in used:raise ValueError('Duplicate case/reviewer; retain history instead')
        used.add(key);c=cases[r['case_id']]
        for field,value in [('source_annotation_sha256',c['annotation_sha256']),('original_boundary_frame',c['original_boundary_frame']),('target_event_id',c['target_event_id'])]:
            if r.get(field)!=value:raise ValueError('Original evidence identity changed: '+field)
        if r.get('status') not in STATUSES:raise ValueError('Invalid status')
        if type(r.get('model_seen')) is not bool:raise ValueError('Exposure must be explicit bool')
        p,lo,hi=[r.get(k) for k in ['proposed_boundary_frame','earliest_plausible_frame','latest_plausible_frame']]
        for v in [p,lo,hi]:
            if v is not None and (type(v) is not int or not 0<=v<c['episode_length']):raise ValueError('Invalid source frame')
        if (lo is None)!=(hi is None) or lo is not None and lo>hi:raise ValueError('Invalid uncertainty interval')
        if p is not None and lo is not None and not lo<=p<=hi:raise ValueError('Proposed frame outside interval')
        if r['status']=='propose_revision' and (p is None or lo is None):raise ValueError('Revision needs point and uncertainty interval')
        if r['status']=='retain_original' and p not in (None,c['original_boundary_frame']):raise ValueError('Retention contradicts proposed frame')
        if r['status']!='pending_human_review' and (not r.get('reviewer') or not str(r.get('reason','')).strip()):raise ValueError('Completed opinion needs reviewer and motion evidence')
        if not isinstance(r.get('revision_history'),list):raise ValueError('Missing revision history')
    return dict(valid=True,opinions=len(payload['reviews']),labels_written=0,scope='Identity and structural validation only; not consensus or scientific approval')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('opinions');p.add_argument('--cases',default='runs/tomato/boundary_review_20261109/review_package/cases.json');a=p.parse_args();print(validate(ds.read_json(a.opinions),ds.read_json(a.cases)))
