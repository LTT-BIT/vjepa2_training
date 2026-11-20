from collections import Counter
import pytest
from tools.train_tomato_expanded_agents import split_sources,allocate


def fixture():
    old=dict(episodes=[dict(source_recording_id=f'old{i}',split='train' if i<95 else 'val' if i<107 else 'test') for i in range(119)])
    rows=[dict(source_recording_id=e['source_recording_id'],package='old') for e in old['episodes']]
    for pack,size in enumerate([38,6,14,12,3,9,3]):
        rows.extend(dict(source_recording_id=f'new{pack}-{i}',package=f'pack{pack}') for i in range(size))
    return old,rows


def test_expand_preserves_historical_identity_and_seals_new_holdout():
    old,rows=fixture();splits=split_sources(rows,old)
    assert Counter(splits.values())==dict(train=151,val=21,historical_test=12,blind_test=20)
    for e in old['episodes']:
        assert splits[e['source_recording_id']]==('historical_test' if e['split']=='test' else e['split'])
    for pack in range(7):
        assert {splits[r['source_recording_id']] for r in rows if r['package']==f'pack{pack}'}=={'train','val','blind_test'}
    assert splits==split_sources(list(reversed(rows)),old)


def test_duplicate_recording_cannot_leak_across_split():
    old,rows=fixture()
    with pytest.raises(ValueError,match='Duplicate'):split_sources(rows+[rows[0]],old)
    with pytest.raises(ValueError,match='identity missing'):split_sources(rows[1:],old)


def test_impossible_stratification_fails_instead_of_empty_training():
    with pytest.raises(ValueError):allocate({'small':2},1,2)


def test_five_phase_event_evaluation_uses_four_real_transition_types():
    import numpy as np
    from tools import tomato_dataset as ds
    from tools.train_tomato_expanded_agents import evaluate_expanded,SEQUENCE
    e=dict(episode_id='synthetic',fps=30,length=600,annotation=dict(segments=[
        dict(start_frame=i*120,end_frame=(i+1)*120,label=label,reviewed=True) for i,label in enumerate(SEQUENCE)]))
    anchors=np.arange(45,600,15);target=ds.build_phase_targets(anchors,e['annotation']['segments'],600,30,1.5)
    current=np.eye(4)[target['current_labels']]
    future=np.eye(4)[np.maximum(target['future_labels'],0)]
    row=dict(episode=e,anchors=anchors,times=anchors/30,target=target)
    result=evaluate_expanded([row],{'synthetic':dict(current=current,future=future)})
    assert result['strict_future']['hits']==result['strict_future']['total']==4
    assert result['strict_future']['false_events']==0
