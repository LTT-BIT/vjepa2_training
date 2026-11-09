from copy import deepcopy
import pytest
from tools.validate_tomato_boundary_reviews import validate


def sample():
    c=dict(case_id='a',target_event_id='b',annotation_sha256='sha',original_boundary_frame=20,episode_length=100)
    r=dict(case_id='a',target_event_id='b',source_annotation_sha256='sha',original_boundary_frame=20,reviewer='reviewer1',model_seen=False,status='propose_revision',proposed_boundary_frame=22,earliest_plausible_frame=21,latest_plausible_frame=24,reason='Object begins following gripper',revision_history=[])
    return dict(convention='v1',reviews=[r]),dict(convention='v1',cases=[c])


def test_opinion_validation_does_not_mutate_source_or_apply_suggestion():
    p,c=sample();original=deepcopy(c)
    assert validate(p,c)['labels_written']==0 and c==original


@pytest.mark.parametrize('field,value',[('source_annotation_sha256','other'),('original_boundary_frame',22),('proposed_boundary_frame',25),('earliest_plausible_frame',30),('reviewer',''),('reason',''),('model_seen',None)])
def test_rejects_identity_invalid_interval_or_missing_human_evidence(field,value):
    p,c=sample();p['reviews'][0][field]=value
    with pytest.raises(ValueError):validate(p,c)


def test_retention_and_revision_cannot_be_confused_and_duplicate_is_not_consensus():
    p,c=sample();p['reviews'][0]['status']='retain_original'
    with pytest.raises(ValueError):validate(p,c)
    p,c=sample();p['reviews'].append(deepcopy(p['reviews'][0]))
    with pytest.raises(ValueError):validate(p,c)


def test_explicit_no_transition_opinion_has_no_boundary_time():
    p,c=sample();r=p['reviews'][0];r['status']='no_transition'
    for field in ['proposed_boundary_frame','earliest_plausible_frame','latest_plausible_frame']:r[field]=None
    assert validate(p,c)['valid']
    r['proposed_boundary_frame']=22
    with pytest.raises(ValueError):validate(p,c)
