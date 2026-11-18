import numpy as np
import pytest

from tools.train_tomato_inhand119_agents import split_recordings, phase_replay, SEQUENCE
from tools import tomato_dataset as ds


def test_recording_split_is_deterministic_and_disjoint():
    rows=[dict(recording_group=f'recording{i:03d}') for i in range(119)]
    a=split_recordings(dict(episodes=rows))
    b=split_recordings(dict(episodes=rows[::-1]))
    assert a==b
    assert {k:list(a.values()).count(k) for k in ['train','val','test']}==dict(train=95,val=12,test=12)
    rows[-1]=rows[0]
    with pytest.raises(ValueError,match='distinct'):
        split_recordings(dict(episodes=rows))


def segments():
    return [dict(start_frame=i*90,end_frame=(i+1)*90,label=label,reviewed=True)
            for i,label in enumerate(SEQUENCE)]


def test_new_five_phase_replay_handles_repeated_harvest_without_old_return():
    times=np.arange(1.5,15,.5)
    labels=np.array([ds.label_at_frame(segments(),int(t*30)) for t in times])
    p=np.eye(4)[labels]
    r=phase_replay(times,p,segments())
    assert r['ordered_complete']
    assert r['offsets_s']==[.5,.5,.5,.5]
    assert r['transitions_s']==[3.5,6.5,9.5,12.5]


def test_replay_requires_two_confident_fresh_votes_and_legal_next_class():
    p=np.eye(4)[[2,1,0,1,0,1,0]]
    r=phase_replay(np.arange(7)*.5+1.5,p,segments())
    assert r['missed']==4 and not r['ordered_complete']
    p=np.tile([.1,.7,.1,.1],(8,1))
    assert phase_replay(np.arange(8)*.5+1.5,p,segments())['missed']==4


def test_future_target_follows_new_adjust_and_masks_end():
    anchors=np.array([45,135,225,315,405])
    y=ds.build_phase_targets(anchors,segments(),450,30,1.5)
    np.testing.assert_array_equal(y['current_labels'],[0,1,3,1,2])
    np.testing.assert_array_equal(y['future_labels'],[1,3,1,2,-100])
