"""Validate real exports against frozen pilot; never invent reviews or labels."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import itertools
import numpy as np
from . import tomato_dataset as ds

SCHEMA='tomato-operation-review-v2'
OUTCOME={'positive':'success','negative':'failure','unknown':'unknown'}
REFERENCE={'right_grasp_started':0,'truss_stably_lifted':1,'left_pick_started':1,'tomato_detached':2,
           'truss_returned':3,'adjustment_ended':4,'right_regrasp_started':4,'second_grasp_verified':5}


def enrich_cases(cases, episodes):
    by={e['episode_id']:e for e in episodes};result=[]
    for c in cases:
        e=by[c['episode_id']];i=REFERENCE[c['event_type']];s=e['annotation']['segments'][i]
        result.append({**c,'reference_transition_frame':s['start_frame'],
                       'reference_kind':'stage_transition' if i else 'episode_start',
                       'reference_transition_description':f"{i}: {s['label']} begins (comparison only)"})
    return result


def validate_review(row,case,episode):
    if row.get('review_schema')!=SCHEMA:raise ValueError('Legacy opinion requires explicit first-evidence review')
    if not isinstance(row.get('reviewer'),str) or not row['reviewer'].strip():raise ValueError('Reviewer required')
    for k in ('case_id','episode_id','source_recording_id','annotation_sha256','event_type'):
        if row.get(k)!=case[k]:raise ValueError('Frozen case identity mismatch: '+k)
    if row.get('outcome')!=OUTCOME.get(row.get('verdict')):raise ValueError('Success/Failure/Unknown mapping mismatch')
    a,b,ev=row.get('start_frame'),row.get('end_frame'),row.get('evidence_end_frame')
    if any(type(v) is not int for v in (a,b,ev)) or not 0<=a<=ev<b<=episode['length']:raise ValueError('Reviewed interval/evidence frame invalid')
    first=row.get('first_confirmable_frame')
    if row['verdict']=='unknown':
        if first is not None:raise ValueError('Unknown has no invented first-confirmable result time')
    elif type(first) is not int or not a<=first<b or ev!=first:raise ValueError('Explicit first-confirmable frame required')
    cameras=row.get('visible_cameras',[])
    if not cameras or any(c not in ('cam0','cam1','cam2') for c in cameras) or not row.get('evidence_note','').strip():raise ValueError('Direct evidence and camera sources required')
    if row.get('reference_transition_frame')!=case['reference_transition_frame']:raise ValueError('Reference boundary changed')
    delta=(first-case['reference_transition_frame'])/episode['fps'] if first is not None and case['reference_kind']=='stage_transition' else None
    return {**row,'evidence_time_s':first/episode['fps'] if first is not None else None,
            'relative_to_transition':'not_applicable' if delta is None else 'before' if delta<0 else 'after' if delta>0 else 'at',
            'evidence_minus_transition_s':delta}


def quality_report(rows,cases,catalog):
    bycase={r['case_id']:r for r in cases};byep={r['episode_id']:r for r in catalog};valid=[];invalid=[];seen=set()
    for row in rows:
        try:
            if (row.get('case_id'),row.get('reviewer')) in seen:raise ValueError('Duplicate reviewer revision: select an explicit revision')
            r=validate_review(row,bycase[row['case_id']],byep[row['episode_id']]);seen.add((r['case_id'],r['reviewer']));valid.append(r)
        except (KeyError,ValueError,TypeError) as error:invalid.append(dict(case_id=row.get('case_id'),reviewer=row.get('reviewer'),reason=str(error)))
    grouped=defaultdict(list)
    for r in valid:grouped[r['case_id']].append(r)
    paired=[];times=[]
    for key,rs in grouped.items():
        for a,b in itertools.combinations(rs,2):
            fields=[k for k in ('verdict','start_frame','end_frame','first_confirmable_frame') if a[k]!=b[k]]
            dt=abs(a['first_confirmable_frame']-b['first_confirmable_frame'])/byep[a['episode_id']]['fps'] if all(r['first_confirmable_frame'] is not None for r in (a,b)) else None
            if dt is not None:times.append(dt)
            paired.append(dict(case_id=key,reviewers=[a['reviewer'],b['reviewer']],disagreement_fields=fields,
                               outcome_agrees=a['verdict']==b['verdict'],first_evidence_difference_s=dt,
                               camera_sources=[a['visible_cameras'],b['visible_cameras']]))
    return dict(schema=SCHEMA,candidate_count=len(cases),review_rows=len(rows),valid_rows=len(valid),invalid=invalid,
                distinct_reviewers=sorted({r['reviewer'] for r in valid}),cases_reviewed=len(grouped),paired_comparisons=paired,
                outcome_agreement=float(np.mean([r['outcome_agrees'] for r in paired])) if paired else None,
                first_evidence_difference_p50_s=float(np.median(times)) if times else None,
                first_evidence_difference_p95_s=float(np.percentile(times,95)) if times else None,
                unknown_rate=sum(r['verdict']=='unknown' for r in valid)/len(valid) if valid else None,
                paired_case_count=len({r['case_id'] for r in paired}),
                camera_sources=dict(Counter(c for r in valid for c in r['visible_cameras'])),
                by_event={e:dict(Counter(r['outcome'] for r in valid if r['event_type']==e)) for e in sorted({c['event_type'] for c in cases})},
                timing_vs_subtask=dict(Counter(r['relative_to_transition'] for r in valid)),
                can_assess_label_reliability=bool(paired),human_independence='Separate people/browser profiles required; reviewer names alone are not identity authentication',
                physical_accuracy=None,training=False,normalized_reviews=valid)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--reviews',type=Path,nargs='*',default=[]);p.add_argument('--out',type=Path,required=True);p.add_argument('--consensus-out',type=Path);a=p.parse_args()
    if a.out.exists():raise FileExistsError('New quality output required')
    if a.consensus_out and a.consensus_out.exists():raise FileExistsError('New consensus output required')
    rows=[r for path in a.reviews for r in ds.read_json(path)['reviews']]
    report=quality_report(rows,ds.read_json(a.pilot/'cases.json'),ds.read_json(a.pilot/'catalog.json'));ds.write_json(a.out,report)
    if a.consensus_out:
        from .prepare_tomato_operation_review import merge_reviews
        ds.write_json(a.consensus_out,{**merge_reviews(report['normalized_reviews']),'invalid_reviews':report['invalid']})
