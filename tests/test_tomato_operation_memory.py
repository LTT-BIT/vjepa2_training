from dataclasses import replace
import numpy as np
import pytest
from tools.tomato_operation_memory import *
from tools.tomato_outcome_verifier import *
from tools.prepare_tomato_operation_review import merge_reviews


def ev(**kw):
    return EvidenceRecord(**{**dict(evidence_id='x',session_id='s',episode_id='e',observation_id='s:1',timestamp=1.,event_type='truss_stably_lifted',kind='verified_operation_result',source='independent_operation_review',probability=None,verdict='positive',reviewers=('a','b'),independent_label=True,support_ids=('s:1',)),**kw})


def test_physical_evidence_separate_from_stage_and_command():
    for kind in ('stage_prediction','executed_command','observation'):
        m=OperationMemory('s','e','b0');m.append(ev(kind=kind),now=1.)
        with pytest.raises(ValueError):m.confirm('x',now=1.)
        assert not m.snapshot(now=1.)['verified_results']
    m=OperationMemory('s','e','b0');m.append(ev(),now=1.);assert m.confirm('x',now=1.)
    assert m.snapshot(now=1.)['picked_count'] is None
    assert not m.snapshot(now=1.)['suggests_prompt_update']
    with pytest.raises(ValueError):m.confirm('x',now=1.)


def test_unqualified_verifier_cannot_self_declare_physical_truth():
    m=OperationMemory('s','e','b0');m.append(ev(source='outcome_verifier:fake',qualified_verifier=True,probability=.999,support_ids=('a','b')),now=1.)
    with pytest.raises(ValueError):m.confirm('x',now=1.)


def test_identity_time_expiry_revision_and_reset():
    m=OperationMemory('s','e','b0');e=ev(valid_until=2.);assert m.append(e,now=1.)
    assert not m.append(e,now=1.)
    with pytest.raises(ValueError):m.append(ev(evidence_id='b',session_id='other'),now=1.)
    m.expire(3.);assert m.records['x'].status=='expired'
    with pytest.raises(ValueError):m.confirm('x',now=3.)
    m.reset('next','new','b0');assert not m.records and not m.confirmed
    with pytest.raises(ValueError):m.append(e,now=4.)


def test_conflicting_independent_evidence_keeps_unknown_without_erasing():
    m=OperationMemory('s','e','b0');m.append(ev(),now=1.);m.append(ev(evidence_id='negative',verdict='negative'),now=1.)
    assert not m.confirm('x',now=1.) and m.conflicts and not m.confirmed
    assert m.records['x'].status=='uncertain'


def test_empty_verifier_truth_is_unknown_and_no_pseudo_labels():
    assert all(r['event_probability'] is None and not r['evidence_validity'] for r in OutcomeVerifier().predict(observation_id='s:1',timestamp=1.))
    assert not training_readiness([])['can_train']


@pytest.mark.parametrize('mode,dim',[('A',928),('B',3072),('C',4000),('D',8000)])
def test_causal_feature_views_and_model_backward(mode,dim):
    import torch
    x=causal_features(np.ones(4000),np.zeros(4000),mode=mode,timestamp=1.,previous_timestamp=.5)
    assert x.shape==(dim,)
    model=build_model(mode);logits=model(torch.tensor(x)[None]);assert logits.shape==(1,8)
    known=torch.zeros((1,8),dtype=torch.bool);known[:,0]=True
    loss=masked_bce(logits,torch.zeros_like(logits),known);loss.backward()
    with pytest.raises(ValueError):masked_bce(logits,torch.zeros_like(logits),known&False)


def test_future_feature_and_copied_stage_label_rejected():
    with pytest.raises(ValueError):causal_features(np.zeros(4000),np.zeros(4000),mode='D',timestamp=1.,previous_timestamp=2.)
    with pytest.raises(ValueError):validate_label(dict(label_source='stage_labels',event_type='tomato_detached',verdict='positive'))


def test_two_human_reviews_required_and_disagreement_not_averaged():
    row=dict(case_id='c',episode_id='e',source_recording_id='r',event_type='truss_stably_lifted',start_frame=0,end_frame=20,evidence_end_frame=10,verdict='positive',annotation_sha256='sha',evidence_note='Visible object moving with hand',visible_cameras=['cam0'],reviewer='a')
    assert not merge_reviews([row])['approved']
    assert len(merge_reviews([row,{**row,'reviewer':'b'}])['approved'])==1
    assert not merge_reviews([row,{**row,'reviewer':'b','verdict':'negative'}])['approved']


def test_late_contradiction_preserves_history_but_withholds_disputed_fact():
    m=OperationMemory('s','e','b0');m.append(ev(),now=1.);m.confirm('x',now=1.)
    m.append(ev(evidence_id='n',verdict='negative'),now=1.)
    assert m.confirmed['x'].status=='confirmed'
    snap=m.snapshot(now=1.);assert snap['verified_results'][0]['disputed']
    assert not public_memory(m,observation_id='s:1',timestamp=1.)['verified_facts']


def test_oracle_only_releases_independent_past_results():
    row=dict(case_id='independent',episode_id='e',source_recording_id='r',event_type='truss_stably_lifted',start_frame=0,end_frame=60,evidence_end_frame=30,verdict='positive',annotation_sha256='sha',evidence_note='Visible object lifted',visible_cameras=['cam0'],reviewers=['a','b'],status='approved',label_source='independent_operation_review',stage_labels_used_as_truth=False)
    kwargs=dict(session_id='s',episode_id='e',source_recording_id='r',observation_id='s:1',fps=30.,phase_index=0)
    assert not oracle_memory_at([row],timestamp=.9,**kwargs)['verified_facts']
    assert len(oracle_memory_at([row],timestamp=1.,**kwargs)['verified_facts'])==1


def test_research_verifier_checkpoint_is_loadable_but_never_qualified(tmp_path):
    import torch
    model=build_model('A');path=tmp_path/'research.pt'
    torch.save(dict(format='tomato-outcome-verifier-v1',mode='A',state_dict=model.state_dict(),mean=torch.zeros(928),std=torch.ones(928),independent_labels=True,qualified=False),path)
    result=OutcomeVerifier(path).predict(observation_id='s:1',timestamp=1.,features=np.zeros(928,np.float32))
    assert all(r['event_probability'] is not None and not r['evidence_validity'] for r in result)


def test_sidecar_preserves_stage_packet_and_never_predicts_fake_outcome():
    from tools.tomato_operation_sidecar import OperationSidecar
    import copy
    side=OperationSidecar('s','e','b0')
    p=[.97,.01,.01,.01]
    r=dict(valid=True,session_id='s',model_version='b0',labels=['pickup','harvest','return','adjust'],prediction_mode='joint',input_timestamp=0.,observation_id='frame0',current_probabilities=p,future_probabilities=p)
    old=copy.deepcopy(r);out=side.process(r,now=.01)
    assert out['prediction']==r==old
    assert not out['memory']['verified_facts'] and out['memory']['picked_count'] is None
    assert all(e['status']=='insufficient_evidence' for e in out['operation_evidence'])
    with pytest.raises(ValueError):side.process(r,now=.02)


def test_current_object_state_expires_but_past_result_is_preserved():
    m=OperationMemory('s','e','b0');m.append(ev(valid_until=2.),now=1.);m.confirm('x',now=1.)
    assert m.snapshot(now=1.)['object_state']['truss_support']=='right_hand'
    assert m.snapshot(now=3.)['object_state']=='unknown'
    assert m.snapshot(now=3.)['verified_results'][0]['status']=='confirmed'


def test_video_review_range_server_serves_only_catalogued_media(tmp_path):
    from tools.serve_tomato_operation_review import review_handler
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request,urlopen
    from urllib.error import HTTPError
    import json
    video=tmp_path/'known.mp4';video.write_bytes(bytes(range(64)))
    (tmp_path/'catalog.json').write_text(json.dumps([dict(episode_id='e',videos=[str(video)]*3)]))
    server=ThreadingHTTPServer(('127.0.0.1',0),review_handler(tmp_path));thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        root=f'http://127.0.0.1:{server.server_port}'
        with urlopen(Request(root+'/media/e/0',headers={'Range':'bytes=4-9'})) as r:assert r.status==206 and r.read()==bytes(range(4,10))
        with pytest.raises(HTTPError):urlopen(root+'/media/unknown/0')
        with pytest.raises(HTTPError):urlopen(root+'/media/e/-1')
    finally:server.shutdown();server.server_close();thread.join()


def test_same_session_restart_and_capacity_do_not_silently_drop_facts():
    m=OperationMemory('s','e','b0',limit=1);m.append(ev(),now=1.);m.confirm('x',now=1.)
    version=m.version
    with pytest.raises(ValueError):m.append(ev(evidence_id='next'),now=1.)
    assert list(m.confirmed)==['x'] and m.version==version
    with pytest.raises(ValueError):m.reset('s','e','b0')


def test_duplicate_reviewer_does_not_form_consensus():
    row=dict(case_id='c',reviewer='a',verdict='unknown',evidence_note='occluded')
    with pytest.raises(ValueError):merge_reviews([row,row])


def test_export_of_earlier_observation_never_includes_later_object_evidence():
    m=OperationMemory('s','e','b0');m.append(ev(timestamp=2.,valid_until=3.),now=2.);m.confirm('x',now=2.)
    r=public_memory(m,observation_id='old',timestamp=1.,now=2.)
    assert not r['verified_facts'] and r['object_state']=='unknown'
    assert r['uncertain']==['operation_results_unknown']


def test_fractional_frame_labels_are_invalid_not_silently_rounded():
    row=dict(event_type='truss_stably_lifted',verdict='unknown',label_source='independent_operation_review',stage_labels_used_as_truth=False,status='approved',reviewers=['a','b'],start_frame=.5,end_frame=10,evidence_end_frame=5)
    with pytest.raises(ValueError,match='integers'):validate_label(row)
