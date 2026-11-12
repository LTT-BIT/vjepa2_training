"""Strict evaluation-only ingestion. Never edits source data or fits statistics."""
from pathlib import Path
import fcntl
import numpy as np
from . import tomato_dataset as ds


def check_identity(source_id,content_hashes,seen_ids,seen_hashes):
    if not isinstance(source_id,str) or not source_id.strip():raise ValueError('Reliable source_recording_id required')
    if source_id in seen_ids:raise ValueError('Previously exposed source recording')
    if not content_hashes or set(content_hashes)&set(seen_hashes):raise ValueError('Previously exposed/repeated media content')


def validate_index(index_path,lock,output,*,allow_synthetic_fixture=False):
    index=ds.read_json(index_path)
    if index.get('is_synthetic_test_fixture') and not allow_synthetic_fixture:raise ValueError('Synthetic fixture is not an independent real recording')
    if index.get('purpose')!='independent_evaluation_only' or not index.get('records'):raise ValueError('Nonempty independent_evaluation_only catalog required')
    if not isinstance(index.get('collection_id'),str) or not index['collection_id'].strip():raise ValueError('Declared collection_id required')
    if index.get('used_for_training_or_selection') is not False:raise ValueError('Explicit unused-for-training/selection provenance required')
    roots=[r['dataset_root'] for r in index['records']]
    ds.ensure_output_outside_sources(output,roots,[r['annotation_path'] for r in index['records']])
    old=lock['historical_registry'];seen=set(old['source_ids']);hashes=set(old['content_sha256']);episodes=[]
    oldroots=[Path(p).resolve() for p in old['dataset_roots']]
    for rec in index['records']:
        root=Path(rec['dataset_root']).resolve(strict=True)
        if any(root==p or root.is_relative_to(p) or p.is_relative_to(root) for p in oldroots):raise ValueError('New collection must live outside historical dataset roots')
        if not isinstance(rec.get('conditions'),dict) or not rec['conditions']:raise ValueError('Recording conditions required for coverage audit')
        e=ds.load_episode_info(root,rec['episode_index'])
        if e['fps']!=lock['rules']['raw_fps']:raise ValueError('Raw FPS differs from frozen30Hz contract')
        if rec.get('source_recording_id')!=e['source_recording_id']:raise ValueError('Catalog/source identity mismatch')
        ann=ds.load_reviewed_coarse(rec['annotation_path'],e,False)
        if [s['label'] for s in ann['segments']]!=lock['rules']['class_sequence']:raise ValueError('Require complete reviewed seven-position Subtask sequence')
        content={str(Path(p).relative_to(root)):ds.digest(p) for p in [e['parquet'],*e['videos']]}
        if len(set(content.values()))!=len(content):raise ValueError('Repeated identical camera/data content within recording; requires source review')
        check_identity(e['source_recording_id'],content.values(),seen,hashes)
        report=ds.validate_source_and_media(e,lock['contract']['joint_names'],lock['contract']['joint_mode'])
        times,_=ds.read_state(e,lock['contract']['joint_names'],lock['contract']['joint_mode']);anchors=ds.iter_phase_anchors(times,lock['rules']['prediction_hz'])
        if len(anchors)<2 or not np.any(anchors+45<e['length']):raise ValueError('Episode too short for evaluable Future1.5s')
        durations={phase:(s['end_frame']-s['start_frame'])/e['fps'] for phase,s in zip(lock['rules']['stage_sequence'],ann['segments'])}
        e.update(annotation=ann,input_sha256=content,anchors=anchors.tolist(),split='independent_evaluation_only',media_validation=report,collection_conditions=rec['conditions'],
                 phase_duration_s=durations,duration_scope='Descriptive evaluation coverage only; never a fitted duration prior')
        episodes.append(e);seen.add(e['source_recording_id']);hashes.update(content.values())
    return dict(episodes=episodes,contract=lock['contract'],pretrained=lock['pretrained'],role='synthetic_smoke_only' if index.get('is_synthetic_test_fixture') else 'independent_evaluation_only',
                collection_id=index['collection_id'],catalog_path=str(Path(index_path).resolve()),catalog_sha256=ds.digest(index_path),
                forbidden_uses=['training','residual_fitting','threshold_selection','duration_prior_fitting','model_selection'],
                provenance_scope='IDs/content and published labels verified. Human collection provenance and re-encoded derivatives cannot be proven from hashes alone.')


def claim_collection(manifest,registry_dir,lock_sha):
    registry=Path(registry_dir);registry.mkdir(parents=True,exist_ok=True)
    with (registry/'registry.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        file=registry/'registry.json';old=ds.read_json(file) if file.exists() else dict(records=[])
        seen={r['source_recording_id'] for r in old['records']};hashes={v for r in old['records'] for v in r['content_sha256']}
        for e in manifest['episodes']:
            check_identity(e['source_recording_id'],e['input_sha256'].values(),seen,hashes)
            seen.add(e['source_recording_id']);hashes.update(e['input_sha256'].values())
        for e in manifest['episodes']:
            old['records'].append(dict(source_recording_id=e['source_recording_id'],content_sha256=list(e['input_sha256'].values()),
                                       collection_id=manifest['collection_id'],lock_sha256=lock_sha,role='evaluation_only_claimed',
                                       catalog_sha256=manifest['catalog_sha256']))
        ds.write_json(file,old)
        return ds.digest(file)
