import json
from pathlib import Path
from fractions import Fraction
import numpy as np
import pytest
from tools import tomato_dataset as ds
from tools.tomato_independent_import import validate_index,check_identity,claim_collection
from tools.tomato_full_chain import evaluate_episode,reconstruct_shadow,cascade_diagnostics


@pytest.fixture
def lock():
    return ds.read_json('configs/tomato_frozen_independent_20261112/b0.lock.v3.json')


@pytest.fixture
def synthetic_collection(tmp_path,lock):
    """Actual tiny three-camera/parquet fixture, never a real blind evaluation."""
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=tmp_path/'source';(root/'meta').mkdir(parents=True);n=120;sid='synthetic-independent-recording'
    features={'observation.state':dict(dtype='float32',shape=[58],names=lock['contract']['joint_names'])}
    for cam in ['cam0','cam1','cam2']:features['observation.images.'+cam]=dict(dtype='video',shape=[16,16,3])
    info=dict(codebase_version='v2.1',robot_type='dual_fr3_sharpa',fps=30,chunks_size=1000,features=features,
              data_path='data/episode_{episode_index:06d}.parquet',video_path='videos/{video_key}/episode_{episode_index:06d}.mp4')
    ds.write_json(root/'meta/info.json',info);(root/'meta/episodes.jsonl').write_text(json.dumps(dict(episode_index=0,length=n,source_recording_id=sid))+'\n')
    (root/'data').mkdir();pq.write_table(pa.table({'timestamp':np.arange(n)/30,'frame_index':np.arange(n),'episode_index':np.zeros(n,dtype=int),'observation.state':[[0.]*58 for _ in range(n)]}),root/'data/episode_000000.parquet')
    for k,cam in enumerate(['cam0','cam1','cam2']):
        p=root/f'videos/observation.images.{cam}/episode_000000.mp4';p.parent.mkdir(parents=True)
        with av.open(str(p),'w') as container:
            stream=container.add_stream('libx264',rate=30);stream.width=stream.height=16;stream.pix_fmt='yuv420p'
            for i in range(n):
                frame=av.VideoFrame.from_ndarray(np.full((16,16,3),40+k*50+i%3,dtype=np.uint8),format='rgb24');frame.pts=i;frame.time_base=Fraction(1,30)
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
    e=ds.load_episode_info(root,0);ends=[0,15,30,45,60,75,90,120]
    annotation=tmp_path/'labels.json';ds.write_json(annotation,dict(stage='coarse',schema_version=2,revision=1,saved_at=1,inherited=False,
        source=dict(episode_index=0,length=n,fps=30.,source_recording_id=sid,sha256={str(Path(p).relative_to(root)):ds.digest(p) for p in [e['parquet'],*e['videos']]}),
        segments=[dict(label=label,start_frame=ends[i],end_frame=ends[i+1],reviewed=True) for i,label in enumerate(lock['rules']['class_sequence'])]))
    index=tmp_path/'index.json';ds.write_json(index,dict(purpose='independent_evaluation_only',is_synthetic_test_fixture=True,collection_id='synthetic-test',used_for_training_or_selection=False,
        records=[dict(dataset_root=str(root),episode_index=0,source_recording_id=sid,annotation_path=str(annotation),conditions=dict(speed='synthetic'))]))
    return index,annotation,root


def test_full_media_admission_and_source_bytes_preserved(synthetic_collection,tmp_path,lock):
    index,annotation,root=synthetic_collection;before={str(p):ds.digest(p) for p in root.rglob('*') if p.is_file()}
    m=validate_index(index,lock,tmp_path/'output',allow_synthetic_fixture=True)
    assert len(m['episodes'])==1 and m['role']=='synthetic_smoke_only'
    assert [r['decoded_frames'] for r in m['episodes'][0]['media_validation']['media']]==[120]*3
    assert all(ds.digest(p)==sha for p,sha in before.items())


@pytest.mark.parametrize('problem',['draft','three_stages','changed_source'])
def test_invalid_annotations_and_changed_raw_data_fail_closed(synthetic_collection,tmp_path,lock,problem):
    index,ann,root=synthetic_collection;doc=ds.read_json(ann)
    if problem=='draft':doc['draft']=True;ds.write_json(ann,doc)
    elif problem=='three_stages':doc['segments']=doc['segments'][:3];doc['segments'][-1]['end_frame']=120;ds.write_json(ann,doc)
    else:
        with next(root.glob('data/*.parquet')).open('ab') as f:f.write(b'changed')
    with pytest.raises((ValueError,OSError)):validate_index(index,lock,tmp_path/'output',allow_synthetic_fixture=True)


def test_synthetic_not_admitted_as_real_independent_recording(synthetic_collection,tmp_path,lock):
    with pytest.raises(ValueError,match='Synthetic fixture'):validate_index(synthetic_collection[0],lock,tmp_path/'out')


def test_outputs_cannot_write_inside_raw_source(synthetic_collection,lock):
    from tools.run_tomato_frozen_evaluation import run
    index,_,root=synthetic_collection
    with pytest.raises(ValueError,match='protected input'):run(index,root/'out',Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'),root/'registry',synthetic_smoke=True)


@pytest.mark.parametrize('kind',['code','model'])
def test_lock_detects_changed_processing_or_model_bytes(tmp_path,lock,kind):
    from tools.tomato_frozen_protocol import verify
    import copy
    changed=copy.deepcopy(lock);file=tmp_path/'tampered';file.write_text('different bytes')
    if kind=='code':changed['code_sha256'][str(file)]='0'*64
    else:changed['models']['42']['path']=str(file)
    config=tmp_path/'lock.json';ds.write_json(config,changed)
    with pytest.raises(ValueError,match='changed'):verify(config)


def test_real_evaluation_cannot_bypass_shared_registry(synthetic_collection,tmp_path):
    from tools.run_tomato_frozen_evaluation import run
    with pytest.raises(ValueError,match='fixed shared evaluation registry'):
        run(synthetic_collection[0],tmp_path/'out',Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'),tmp_path/'empty_registry')


@pytest.mark.skipif(__import__('os').environ.get('TOMATO_FROZEN_GPU_SMOKE')!='1',reason='Opt-in real frozen encoder GPU smoke on synthetic media')
def test_real_encoder_one_command_smoke(synthetic_collection,tmp_path):
    from tools.run_tomato_frozen_evaluation import run
    index,_,_=synthetic_collection;out=tmp_path/'evaluation'
    run(index,out,Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'),tmp_path/'registry',synthetic_smoke=True)
    assert ds.read_json(out/'receipt.json')['synthetic_fixture'] is True
    assert set(ds.read_json(out/'summary.json'))=={'42','43','44'}


def test_renamed_old_source_or_duplicated_content_is_rejected():
    with pytest.raises(ValueError):check_identity('old',['new-hash'],{'old'},set())
    with pytest.raises(ValueError):check_identity('renamed',['old-hash'],set(),{'old-hash'})


def test_cross_run_registry_cannot_readmit_same_collection(tmp_path):
    m=dict(collection_id='one',catalog_sha256='catalog',episodes=[dict(source_recording_id='s',input_sha256={'parquet':'p','video':'v'})])
    claim_collection(m,tmp_path/'registry','lock')
    with pytest.raises(ValueError):claim_collection(m,tmp_path/'registry','another-lock')
    m['episodes'][0]['source_recording_id']='renamed'
    with pytest.raises(ValueError):claim_collection(m,tmp_path/'registry','lock')


def toy_episode():
    labels=['pickup','harvest','return','adjust','pickup','harvest','return'];anchors=np.arange(0,420,15)
    e=dict(episode_id='toy',length=420,fps=30,annotation=dict(segments=[dict(start_frame=i*60,end_frame=(i+1)*60,label=k,reviewed=True) for i,k in enumerate(labels)]))
    p=np.full((28,4),.001)
    for i in range(28):p[i,ds.LABELS.index(labels[i//4])]=.997
    return e,dict(times=anchors/30,anchors=anchors,current_probabilities=p,future_probabilities=p.copy())


def test_shadow_is_faithful_not_physical_success_and_future_cannot_drive_prompt():
    e,z=toy_episode();a,ta,logs=evaluate_episode(e,z,'mock-b0');z['future_probabilities']=np.full((28,4),.25);b,tb,_=evaluate_episode(e,z,'mock-b0')
    assert a['decision_completion']['ordered_sequence_complete'] and a['decision_completion']['physical_episode_success'] is None
    assert a['prompt']==b['prompt'] and a['state']==b['state']
    assert reconstruct_shadow(logs)==list(range(7))
    assert a['memory']['snapshot']['progress']['task_complete'] is False
    with pytest.raises(ValueError):reconstruct_shadow(logs+logs[-1:])


def test_evaluation_rejects_cache_time_shift():
    e,z=toy_episode();z['times']=z['times']+.02
    with pytest.raises(ValueError):evaluate_episode(e,z,'mock')


def test_early_confirmation_crossing_own_boundary_is_not_downstream_cascade():
    runs=[dict(first_index=0,start_s=1.,end_s=4.)];bounds=[dict(time=.5),dict(time=2.),dict(time=5.)]
    r=cascade_diagnostics(runs,[2],[1],bounds)[0]
    assert r['inferred_root_ordinal']==2 and r['boundaries_crossed']==[2] and r['potential_downstream_ordinals']==[]
    runs[0]['end_s']=6.
    assert cascade_diagnostics(runs,[2],[1],bounds)[0]['potential_downstream_ordinals']==[3]
