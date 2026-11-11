import numpy as np
import pytest
from tools.tomato_stage_history import HistoryReplay, oracle_anchor, oracle_prefix
from tools.tomato_operation_review_quality import validate_review, quality_report, SCHEMA
from tools.prepare_tomato_operation_review import merge_reviews
from tools.tomato_outcome_verifier import training_readiness


def test_single_anchor_equals_full_history_under_identical_correction_rule():
    segments=[dict(start_frame=i*30) for i in range(7)]
    a,b=HistoryReplay('anchor'),HistoryReplay('oracle')
    rng=np.random.default_rng(42)
    for i in range(20):
        t=i*.5;p=rng.dirichlet(np.ones(4));q=rng.dirichlet(np.ones(4));past=t-.5 if i else None
        assert a.update(p,q,t,oracle_position=oracle_anchor(segments,30,past))==b.update(p,q,t,oracle_history=oracle_prefix(segments,30,past))
    assert not a.history


def test_anchor_rejects_future_and_history_payload():
    m=HistoryReplay('anchor');p=[1.,0,0,0];m.update(p,p,0.)
    with pytest.raises(ValueError):m.update(p,p,.5,oracle_position=dict(phase=1,available_at=.5))
    with pytest.raises(ValueError):m.update(p,p,.5,oracle_position=dict(phase=1,available_at=0.,events=[]))
    with pytest.raises(ValueError):HistoryReplay('predicted').update(p,p,0.,oracle_position=dict(phase=0,available_at=-.5))


def fixture():
    c=dict(case_id='c',episode_id='e',source_recording_id='s',annotation_sha256='sha',event_type='truss_stably_lifted',reference_transition_frame=30,reference_kind='stage_transition')
    e=dict(episode_id='e',length=90,fps=30.)
    r={**{k:c[k] for k in ('case_id','episode_id','source_recording_id','annotation_sha256','event_type','reference_transition_frame')},'review_schema':SCHEMA,'reviewer':'human-fixture-a','verdict':'positive','outcome':'success','start_frame':0,'end_frame':60,'first_confirmable_frame':20,'evidence_end_frame':20,'visible_cameras':['cam0'],'evidence_note':'Synthetic unit fixture, never a human review'}
    return c,e,r


def test_evidence_timing_is_derived_not_copied_from_claimed_relation():
    c,e,r=fixture();v=validate_review({**r,'relative_to_transition':'after'},c,e)
    assert v['relative_to_transition']=='before' and v['evidence_minus_transition_s']==pytest.approx(-1/3)
    with pytest.raises(ValueError):validate_review({**r,'first_confirmable_frame':None},c,e)
    with pytest.raises(ValueError):validate_review({**r,'reference_transition_frame':0},c,e)


def test_unknown_has_no_invented_confirmed_outcome_time():
    c,e,r=fixture();r.update(verdict='unknown',outcome='unknown',first_confirmable_frame=None,evidence_end_frame=59)
    assert validate_review(r,c,e)['evidence_time_s'] is None
    with pytest.raises(ValueError):validate_review({**r,'first_confirmable_frame':20},c,e)


def test_two_reviewers_disagreement_and_camera_provenance_are_preserved():
    c,e,r=fixture();b={**r,'reviewer':'human-fixture-b','first_confirmable_frame':40,'evidence_end_frame':40,'visible_cameras':['cam1']}
    q=quality_report([r,b],[c],[e]);assert q['outcome_agreement']==1.
    assert q['paired_comparisons'][0]['disagreement_fields']==['first_confirmable_frame']
    assert q['first_evidence_difference_p50_s']==pytest.approx(2/3)
    pending=merge_reviews([r,b]);assert not pending['approved'] and pending['pending'][0]['reviews']==[r,b]
    same={**r,'reviewer':'human-fixture-b','visible_cameras':['cam1']};a=merge_reviews([r,same])['approved'][0]
    assert a['visible_cameras']==['cam0','cam1'] and len(a['review_opinions'])==2


def test_quality_does_not_count_duplicate_or_legacy_as_independent_labels():
    c,e,r=fixture();q=quality_report([r,r,{**r,'review_schema':'legacy'}],[c],[e])
    assert q['valid_rows']==1 and len(q['invalid'])==2 and q['outcome_agreement'] is None
    assert not training_readiness([])['can_train']


def test_readiness_requires_timed_v2_labels_not_legacy_count_alone():
    c,e,r=fixture();rows=[]
    for i in range(20):
        verdict='positive' if i<10 else 'negative'
        rows.append({**r,'case_id':str(i),'source_recording_id':str(i%6),'verdict':verdict,
                     'outcome':'success' if verdict=='positive' else 'failure',
                     'reviewers':['fixture-a','fixture-b'],'status':'approved',
                     'label_source':'independent_operation_review','stage_labels_used_as_truth':False})
    assert training_readiness(rows)['can_train']
    assert not training_readiness([{k:v for k,v in r.items() if k!='review_schema'} for r in rows])['can_train']
