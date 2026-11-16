"""Publish a portable merged four-view dataset without changing source artifacts.

Stdlib-only, so it can run on server28. Original labels are retained byte-for-byte
in provenance; active labels only receive new dataset/hash bindings.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

CAMERAS=[f'observation.images.cam{i}' for i in range(3)]+['observation.images.interaction']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def validate_info(info,episodes=100,frames=425556):
    keys=[k for k,v in info['features'].items() if v.get('dtype')=='video']
    if set(keys)!=set(CAMERAS) or info['total_episodes']!=episodes or info['total_frames']!=frames or info['fps']!=30:
        raise ValueError('Wrong source data identity or four-view schema')


def safe(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe relative path')
    result=(root/p).resolve(strict=True)
    if not result.is_relative_to(root.resolve()) or not result.is_file():raise ValueError('File outside source')
    return result


def publish(source,target,agent_identity):
    source=source.resolve(strict=True);target=target.absolute()
    if target.is_symlink() or (target.exists() and any(target.iterdir())):raise FileExistsError('Target must be absent or empty')
    if target.is_relative_to(source) or source.is_relative_to(target):raise ValueError('Source/target overlap')
    data=source/'b_coarse';bundle=json.loads((source/'bundle.json').read_text());info=json.loads((data/'meta/info.json').read_text());validate_info(info)
    identity=json.loads(agent_identity.read_text());episodes=identity['episodes'];mapping=bundle['mapping']
    pair=lambda r:(r['source_dataset'],r['source_episode_index'])
    if len(mapping)!=100 or {pair(r) for r in mapping}!={(r['dataset_root'],r['episode_index']) for r in episodes}:raise ValueError('Agent100 recording identity mismatch')
    split_by={(r['dataset_root'],r['episode_index']):r for r in episodes}
    from collections import Counter
    if Counter(r['split'] for r in episodes)!=Counter(train=84,val=8,test=8):raise ValueError('Agent split mismatch')
    stage=target.with_name(target.name+'.publishing')
    if stage.exists():raise FileExistsError('Preserve existing incomplete publication')
    stage.mkdir(parents=True);source_hashes={};copied=[]
    # Independent copies, no hardlinks/symlinks; data/videos are never rewritten.
    for p in sorted(data.rglob('*')):
        if not p.is_file():continue
        if p.is_symlink():raise ValueError('Unexpected source symlink')
        rel=p.relative_to(data);expected=sha(p);dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        if sha(dst)!=expected or dst.stat().st_ino==p.stat().st_ino and dst.stat().st_dev==p.stat().st_dev:raise ValueError('Copy verification failed')
        source_hashes[str(p)]=expected;copied.append(dict(path=str(rel),sha256=expected,bytes=p.stat().st_size))
    for p in sorted((source/'roi_metadata').glob('*')):
        if not p.is_file():continue
        dst=stage/'roi/chunk-000'/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);source_hashes[str(p)]=sha(p)
        if sha(dst)!=source_hashes[str(p)]:raise ValueError('ROI metadata copy mismatch')
    entries=[json.loads(line) for line in (stage/'meta/episodes.jsonl').read_text().splitlines() if line.strip()];by={r['episode_index']:r for r in entries}
    if len(by)!=100 or sum(r['length'] for r in entries)!=425556:raise ValueError('Episode counts mismatch')
    remapped=[]
    for r in mapping:
        ep=r['episode_index'];entry=by[ep];original=data/f'annotations/episode_{ep:06d}.json';doc=json.loads(original.read_text());a=split_by[pair(r)]
        if entry.get('source_recording_id')!=a['source_recording_id'] or entry['length']!=a['length']:raise ValueError('Recording mismatch')
        seg=doc['segments'];expected=['pickup','harvest','return','adjust','pickup','harvest','return']
        if [x['label'] for x in seg]!=expected or seg[0]['start_frame']!=0 or seg[-1]['end_frame']!=entry['length'] or any(x['end_frame']!=y['start_frame'] for x,y in zip(seg,seg[1:])):raise ValueError('Invalid seven-stage labels')
        snapshot=f'provenance/publication_source_annotations/episode_{ep:06d}.json';(stage/snapshot).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,stage/snapshot)
        paths=['meta/info.json','meta/episodes.jsonl','meta/tasks.jsonl',info['data_path'].format(episode_chunk=ep//1000,episode_index=ep)]
        paths += [info['video_path'].format(episode_chunk=ep//1000,episode_index=ep,video_key=c) for c in CAMERAS]
        bound=copy.deepcopy(doc);bound['source']=dict(dataset=str(target),episode_index=ep,length=entry['length'],fps=30,source_recording_id=entry.get('source_recording_id'),sha256={p:sha(safe(stage,p)) for p in paths})
        bound['source_migration']=dict(profile='merged100-publication-binding-v1',upstream_source=doc['source'],upstream_annotation=snapshot,upstream_annotation_sha256=sha(stage/snapshot))
        assert bound['segments']==doc['segments']
        (stage/f'annotations/episode_{ep:06d}.json').unlink();write(stage/f'annotations/subtasks/coarse/episode_{ep:06d}.json',bound)
        remapped.append(dict(episode_index=ep,source_dataset=r['source_dataset'],source_episode_index=r['source_episode_index'],source_recording_id=a['source_recording_id']))
    write(stage/'processing/source_mapping.json',dict(episodes=remapped,grouping='merged; original A/B retained only in provenance'))
    # Experimental splits and training bundle snapshots belong outside the data publication.
    for p,h in source_hashes.items():
        if sha(p)!=h:raise ValueError('Source changed during publication')
    files={str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    manifest=dict(version='complete100-merged-offline-roi-publication-v1',source_bundle=str(source),episodes=100,frames=425556,image_keys=CAMERAS,grouping='merged_no_AB_directories',files=files,source_files=source_hashes,state_action_unchanged=True,subtask_segments_unchanged=True,annotation_binding_updated_only=True,raw_source_unchanged=True,hardlinks_used=False,future_coordinates_used=True,current_rgb_pixels_only=True,roi_algorithm_unchanged=True,roi_target='existing r11 mixed-hand interaction; NOT newly generated right-truss ROI',independent_semantic_accuracy_measured=False,training_ready=False,scope='Data publication only; no training or default configuration change')
    write(stage/'roi_manifest.json',manifest)
    write(stage/'manifest.json',dict(version=manifest['version'],packages=[dict(path=str(target),episodes=100,frames=425556,manifest_sha256=sha(stage/'roi_manifest.json'))],episodes=100,frames=425556,merged=True,raw_unchanged=True,subtask_labels_generated=False))
    if target.exists():
        if any(target.iterdir()):raise FileExistsError('Target changed during publication')
        target.rmdir() # only the prechecked empty destination; never remove contents
    os.rename(stage,target)
    # Check final published bytes independently from the staging file handles.
    for rel,h in files.items():
        if sha(safe(target,rel))!=h:raise ValueError('Final publication hash mismatch')
    print(json.dumps(dict(status='published_and_hash_verified',target=str(target),episodes=100,frames=425556,files_verified=len(files),roi_manifest_sha256=sha(target/'roi_manifest.json')),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--agent-identity',type=Path,required=True);a=p.parse_args();publish(a.source,a.target,a.agent_identity)
