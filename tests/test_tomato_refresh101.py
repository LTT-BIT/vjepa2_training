import pytest
from tools.train_tomato_refresh101 import dataset_membership
from tools.analyze_tomato_time_usage import rank_metrics,calibration


def test_refresh_adds_new_sources_only_to_train_without_changing_archived_test():
    original=dict(train=['a'],val=['b'],test=['c'])
    new=dataset_membership(original,{'d','e'})
    assert new==dict(train=['a','d','e'],val=['b'],test=['c'])
    assert original['train']==['a']
    with pytest.raises(ValueError):dataset_membership(original,{'c'})


def test_candidate_ranking_ties_and_calibration_do_not_imply_gating():
    assert rank_metrics([.5,.5],[True,False])['auroc']==.5
    assert rank_metrics([.9,.1],[True,False])['auroc']==1.
    x=calibration([.1,.9],[0,1])
    assert x['brier']==pytest.approx(.01)


def test_time_metadata_never_vetoes_or_reorders_original_events():
    import numpy as np
    from tools.analyze_tomato_time_usage import attach_time_metadata
    events=[dict(time=.5,previous=3,label=0),dict(time=1.,previous=0,label=1)]
    z=attach_time_metadata(events,np.array([.5,1.]),np.array([[.001,.001,.001,.001,.996],[.2]*5]))
    assert len(z)==len(events)
    assert [{k:r[k] for k in ('time','previous','label')} for r in z]==events
    assert z[0]['time_risk']<.01


def test_manifest_refresh_does_not_retain_old94_summary_or_time_horizon(tmp_path):
    from copy import deepcopy
    from tools import tomato_dataset as ds
    from tools.finalize_tomato_refresh_manifest import corrected_manifest
    splits=dict(train=['a'],val=['b'],test=['c'])
    e=[dict(episode_id=s,source_recording_id=s,length=120,fps=30.,anchors=list(range(0,120,15)),annotation=dict(sha256='0'*64,segments=[dict(label='pickup',start_frame=0,end_frame=60,reviewed=True),dict(label='harvest',start_frame=60,end_frame=120,reviewed=True)])) for s in ['a','b','c']]
    source=dict(episodes=e,splits=splits,summary=dict(train=dict(episodes=76)),prediction=dict(future_horizon_s=1.))
    before=deepcopy(source);p=tmp_path/'split.json';ds.write_json(p,splits);m=corrected_manifest(source,p)
    assert source==before
    assert m['summary']['train']['episodes']==1
    assert m['prediction']['future_horizon_s']==1.5
    assert m['external_split_sha256']==ds.digest(p)
