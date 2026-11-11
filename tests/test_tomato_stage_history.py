import numpy as np
import pytest
from tools.tomato_stage_history import HistoryReplay, oracle_prefix
from tools.prepare_tomato_operation_pilot import select_pilot


def probs(label):
    p=np.full(4,.01/3);p[label]=.99;return p


def test_oracle_releases_only_past_prefix_and_no_future_ends():
    seg=[dict(start_frame=i*30,end_frame=(i+1)*30,label='unused') for i in range(7)]
    assert oracle_prefix(seg,30,None)==[]
    assert oracle_prefix(seg,30,.5)==[]
    p=oracle_prefix(seg,30,1.)
    assert p==[dict(phase=1,event_time=1.,available_at=1.,source='oracle_past_subtask')]
    seg[2]['start_frame']=99999;seg[1]['end_frame']=8888
    assert oracle_prefix(seg,30,1.)==p


def test_predicted_history_is_b0_state_not_new_evidence():
    a,b=HistoryReplay('none'),HistoryReplay('predicted')
    for i,k in enumerate([0]*3+[1]*3+[2]*3+[3]*3+[0]*3+[1]*3+[2]*3):
        assert a.update(probs(k),probs(k),i*.5)==b.update(probs(k),probs(k),i*.5)
    assert a.phase==b.phase==6


def test_oracle_cannot_enter_predicted_arm_or_use_current_time():
    m=HistoryReplay('predicted')
    with pytest.raises(ValueError):m.update(probs(0),probs(0),0.,oracle_history=[])
    o=HistoryReplay('oracle');o.update(probs(0),probs(0),0.,oracle_history=[])
    with pytest.raises(ValueError):o.update(probs(1),probs(1),.5,oracle_history=[dict(phase=1,event_time=.5,available_at=.5)])


def test_oracle_history_does_not_roll_back_an_early_confirmation():
    o=HistoryReplay('oracle')
    o.update(probs(1),probs(1),0.,oracle_history=[])
    assert o.update(probs(1),probs(1),.5,oracle_history=[])['phase']==1
    assert o.update(probs(0),probs(0),1.,oracle_history=[])['phase']==1


def test_pilot_uses_training_only_is_deterministic_and_covers_packages():
    rows=[dict(episode_id=str(i),split='train',dataset_root='package'+str(i%4),annotation=dict(segments=[dict(start_frame=j*10,end_frame=(j+1)*10+i) for j in range(7)])) for i in range(20)]
    rows.append(dict(episode_id='heldout',split='val'))
    selected=select_pilot(rows)
    assert len(selected)==16 and all(e['split']=='train' for e in selected)
    assert len({e['dataset_root'] for e in selected})==4
    assert [e['episode_id'] for e in selected]==[e['episode_id'] for e in select_pilot(list(reversed(rows)))]
