"""Immutable B0 evaluation identities; no model selection or threshold fitting."""
from pathlib import Path
import argparse
import subprocess
from datetime import datetime,timezone
import torch
from . import tomato_dataset as ds
from .evaluate_tomato_complete100_history import freeze_dataset,ACTIVE
from .tomato_independent_heads import load_checkpoint

OPENPI=Path('/home/user/tomato-openpi')
BRIDGE_COMMIT='e5036843ac26e612e78bf8a289e2c6145b15d433'
FILES=['tools/tomato_state_recovery.py','tools/tomato_stage_history.py','tools/tomato_boundary_metrics.py',
       'tools/tomato_metrics.py','tools/tomato_dataset.py','tools/tomato_phase_model.py','tools/tomato_independent_heads.py',
       'tools/evaluate_tomato_reviewed_boundaries.py','tools/evaluate_tomato_state_recovery.py','tools/tomato_task_memory.py',
       'tools/tomato_frozen_protocol.py','tools/tomato_full_chain.py','tools/tomato_independent_import.py','tools/run_tomato_frozen_evaluation.py',
       'tools/tomato_offline_contract.py','tools/replay_tomato_frozen_history.py','tools/check_tomato_frozen_openpi.py']


def git(*args):return subprocess.check_output(['git',*args],text=True).strip()


def build(path):
    if path.exists():raise FileExistsError('Do not overwrite a frozen protocol')
    m,identity=freeze_dataset();models={};schema=None;pretrained=None
    for seed in (42,43,44):
        p=Path(f'runs/tomato/optimization_20261108/bundles/independent_h15_s{seed}.pt');s,_=load_checkpoint(p)
        assert s['future_horizon_s']==1.5 and s['contract']['prediction_hz']==2
        normalized={k:v for k,v in s['contract'].items() if k!='seed'}
        if schema is None:schema=normalized;pretrained=s['pretrained']
        assert normalized==schema and pretrained==s['pretrained']
        models[str(seed)]=dict(path=str(p.resolve()),sha256=ds.digest(p),model_version=f"b0-independent-h15-s{seed}-{ds.digest(p)[:12]}",
                              normalization_sha256=ds.fingerprint(dict(mean=s['mean'].tolist(),std=s['std'].tolist())),
                              original_training_split_sha256=s['split_sha256'])
    registry={};seen=set();hashes=set();roots=set()
    for source in [ACTIVE,Path('runs/tomato/campaign94_20261107/prepared/manifest.json'),Path('runs/tomato/refresh101_20261109_v2/finalized_exports/refresh101/dataset_manifest.json')]:
        mm=ds.read_json(source);registry[str(source.resolve())]=ds.digest(source)
        for e in mm['episodes']:
            seen.add(e['source_recording_id']);hashes.update(e['input_sha256'].values());roots.add(e['dataset_root'])
    bridge={};archive=path.parent/'openpi_snapshot'
    for rel in ['subtasks/agent_bridge.py','subtasks/tomato.py','policies/fr3_sharpa_protocol.py']:
        import hashlib
        raw=subprocess.check_output(['git','-C',str(OPENPI),'show',BRIDGE_COMMIT+':src/openpi/'+rel])
        bridge[rel]=hashlib.sha256(raw).hexdigest()
        target=archive/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=raw:raise ValueError('Archived bridge fixture differs; never overwrite')
        if not target.exists():target.write_bytes(raw)
    value=dict(version='tomato-b0-frozen-independent-evaluation-v1',created_utc=datetime.now(timezone.utc).isoformat(),
               base_git_sha=git('rev-parse','HEAD'),branch=git('branch','--show-current'),models=models,
               dataset=identity,manifest=str(ACTIVE.resolve()),manifest_sha256=ds.digest(ACTIVE),contract=schema,pretrained=pretrained,
               rules=dict(representative_seed=42,secondary_seeds=[43,44],raw_fps=30.,future_horizon_s=1.5,prediction_hz=2.,history_frames=16,history_sample_hz=10,
                          future_threshold=.5,future_confirmations=2,current_prompt_threshold=.8,current_prompt_confirmations=2,
                          future_match='exact from/to, one-to-one [b-1.5,b); no backdating; valid future tail only',
                          prompt_match='ordinal1..6; early<t_boundary; on_time in[b,min(b+2,end)); late>=min(b+2,end); missing separate',
                          duration='Future valid timestamp range plus0.5s; state uses full current timeline sample-and-hold0.5s',
                          known_start='Explicit session starts pickup_1; no arbitrary mid-task restart recovery',
                          stage_sequence=['pickup_1','harvest_1','return_truss','adjust','pickup_2','harvest_2','return_stem'],
                          class_sequence=['pickup','harvest','return','adjust','pickup','harvest','return'],
                          outcome='Reaching return_stem is ordered decision completion only, never physical success',
                          threshold_selection_allowed=False,new_set_role='independent_evaluation_only',max_prediction_age_s=2.,
                          memory_mode='M1 shadow; no Prompt writes; physical outcomes unknown',
                          bridge_display_advisory='Single-window0.8 next-class hint; separate from strict Future0.5/K2 event scoring',
                          cascade='Persistent mismatch>2s; distinguish crossing its own inferred root boundary from later boundary overlap; temporal overlap is not proof of causality'),
               code_sha256={str(Path(p).resolve()):ds.digest(p) for p in FILES},
               historical_registry=dict(manifests=registry,source_ids=sorted(seen),content_sha256=sorted(hashes),dataset_roots=sorted(roots)),
               openpi=dict(repository=str(OPENPI),snapshot_commit=BRIDGE_COMMIT,files=bridge,archived_snapshot_dir=str(archive.resolve()),
                           current_branch=git('-C',str(OPENPI),'branch','--show-current'),current_sha=git('-C',str(OPENPI),'rev-parse','HEAD'),
                           current_bridge_present=(OPENPI/'src/openpi/subtasks/agent_bridge.py').exists(),
                           scope='Pinned historical interface snapshot; does not change or certify current checkout/deployment'))
    ds.write_json(path,value);print('LOCKED',path,ds.digest(path));return value


def verify(path):
    lock=ds.read_json(path)
    if lock.get('version')!='tomato-b0-frozen-independent-evaluation-v1':raise ValueError('Unknown locked protocol')
    if ds.digest(lock['manifest'])!=lock['manifest_sha256']:raise ValueError('Frozen100 changed')
    for row in lock['dataset']['recordings']:
        if ds.digest(row['annotation_path'])!=row['annotation_sha256']:raise ValueError('Historical annotation changed: '+row['annotation_path'])
    for p,sha in lock['code_sha256'].items():
        if ds.digest(p)!=sha:raise ValueError('Locked processing code changed: '+p)
    for source in lock['models'].values():
        if ds.digest(source['path'])!=source['sha256']:raise ValueError('Checkpoint changed')
    for p,sha in lock['historical_registry']['manifests'].items():
        if ds.digest(p)!=sha:raise ValueError('Historical registry changed')
    return lock


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);build(p.parse_args().out)
