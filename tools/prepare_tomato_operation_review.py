"""Audit101 recorded schemas and prepare independent outcome review, never labels."""
from pathlib import Path
import argparse
from collections import Counter
import pyarrow.parquet as pq
from . import tomato_dataset as ds
from .tomato_operation_memory import EVENTS
from .tomato_outcome_verifier import training_readiness, validate_label

MANIFEST=Path('runs/tomato/refresh101_20261109_v2/finalized_exports/refresh101/dataset_manifest.json')


def merge_reviews(rows):
    grouped={}
    for r in rows:
        if not r.get('reviewer') or r.get('verdict') not in ('positive','negative','unknown') or not r.get('evidence_note'):raise ValueError('Incomplete human review')
        group=grouped.setdefault(r['case_id'],{})
        if r['reviewer'] in group:raise ValueError('Duplicate reviewer export; select one explicit revision')
        group[r['reviewer']]=r
    approved=[];pending=[]
    for key,by in grouped.items():
        rs=list(by.values());fields=('episode_id','source_recording_id','event_type','start_frame','end_frame','evidence_end_frame','verdict','annotation_sha256')
        if len(rs)<2 or any(any(r.get(k)!=rs[0].get(k) for k in fields) for r in rs[1:]):pending.append(dict(case_id=key,status='needs_independent_consensus',reviews=rs));continue
        first=rs[0];a={k:first[k] for k in fields};a.update(case_id=key,status='approved',reviewers=sorted(by),label_source='independent_operation_review',stage_labels_used_as_truth=False,visible_cameras=first.get('visible_cameras',[]),evidence_note=' | '.join(r['evidence_note'] for r in rs),coverage='candidate_interval_only_not_exhaustive')
        validate_label(a);approved.append(a)
    return dict(approved=approved,pending=pending)


def run(out):
    if out.exists():raise FileExistsError('Fresh review version required')
    out.mkdir(parents=True);m=ds.read_json(MANIFEST);episodes=m['episodes'];assert len(episodes)==101 and len({e['source_recording_id'] for e in episodes})==101
    audit=[];catalog=[];cases=[];independent=[]
    for e in episodes:
        info=e['info']['features'];schema=pq.read_schema(e['parquet']);meta=pq.read_metadata(e['parquet'])
        assert meta.num_rows==e['length'] and all(k in schema.names for k in ('observation.state','action','timestamp','source_timestamp_ns'))
        assert info['action']['shape']==[58] and len(m['contract']['joint_names'])==58
        ann=ds.read_json(e['annotation']['path']);assert ann.get('stage')=='coarse'
        # Inspect annotation namespaces; independent results must be explicit, never phase labels.
        root=Path(e['dataset_root'])/'annotations'
        for p in root.rglob('*.json'):
            data=ds.read_json(p)
            if data.get('schema_version')=='tomato-operation-review-v1':independent.append(str(p))
        sensor_keys=[k for k in info if any(x in k.lower() for x in ('tactile','force','depth','grasp','torque','contact'))]
        cams=e['camera_keys'];audit.append(dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],split=e['split'],root=e['dataset_root'],episode_index=e['episode_index'],frames=meta.num_rows,fps=e['fps'],state_shape=info['observation.state']['shape'],state_positions=sum(n.endswith('.position') for n in info['observation.state']['names']),state_velocities=sum(n.endswith('.velocity') for n in info['observation.state']['names']),action_shape=info['action']['shape'],action_names=info['action']['names'],camera_keys=cams,rgb_shapes=[info[k]['shape'] for k in cams],sensor_keys=sensor_keys,arm_engaged='observation.arm_engaged' in schema.names,has_real_source_ns=True,independent_outcome_annotation=False,parquet_fields=schema.names))
        if e['split']=='test':continue
        seg=e['annotation']['segments'];anchors=[(0,'start'),(0,'end'),(1,'start'),(1,'middle'),(2,'end'),(3,'end'),(4,'start'),(4,'end')]
        catalog.append(dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],split=e['split'],fps=e['fps'],length=e['length'],videos=e['videos'],camera_keys=cams,camera_roles='unverified_assign_by_human',annotation_sha256=ds.digest(e['annotation']['path'])))
        for event,(i,edge) in zip(EVENTS,anchors):
            if i>=len(seg):continue
            s=seg[i];frame=s['start_frame'] if edge=='start' else max(s['start_frame'],s['end_frame']-15) if edge=='end' else (s['start_frame']+s['end_frame'])//2
            cases.append(dict(case_id=e['episode_id']+':'+event,episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],split=e['split'],event_type=event,suggested_frame=frame,annotation_sha256=ds.digest(e['annotation']['path']),phase_used_only_to_locate_review=True,verdict=None))
    # Deliberately do not silently load new labels from raw folders without identity review.
    report=dict(recordings=101,split_counts=dict(Counter(e['split'] for e in episodes)),phase_segment_counts=dict(Counter(len(e['annotation']['segments']) for e in episodes)),records=audit,independent_annotation_candidates=sorted(set(independent)),approved_outcome_labels=0,training_readiness=training_readiness([]),camera_role_evidence='cam0/1/2 roles unproven by archived metadata',deployment_signals=dict(rgb_and_position='supported_by_existing_input_interfaces',velocity='causal finite differences available',tactile_depth_grasp_feedback='not recorded here; live hardware availability not verified'),test_values_read=False)
    ds.write_json(out/'audit101.json',report);ds.write_json(out/'catalog.json',catalog);ds.write_json(out/'cases.json',cases);ds.write_json(out/'reviews.json',dict(schema_version='tomato-operation-review-v1',reviews=[],note='Empty, awaiting real human reviews; no pseudo labels'))
    (out/'index.html').write_text((Path(__file__).with_name('tomato_operation_review.html')).read_text())
    ds.write_json(out/'identity.json',dict(manifest=str(MANIFEST),manifest_sha256=ds.digest(MANIFEST),catalog_sha256=ds.digest(out/'catalog.json'),original_annotations_modified=False,review_label_status='all_unreviewed'))
    print('AUDIT',report['split_counts'],'review cases',len(cases),'qualified operation labels',0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--merge',type=Path,nargs='*');a=p.parse_args()
    if a.merge:
        rows=[r for file in a.merge for r in ds.read_json(file)['reviews']]
        if a.out.exists():raise FileExistsError(a.out)
        ds.write_json(a.out,merge_reviews(rows))
    else:run(a.out)
