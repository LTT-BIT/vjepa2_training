from copy import deepcopy
import pytest
from tools.evaluate_tomato_reviewed_boundaries import match_events, revise_episode


def test_half_open_one_to_one_exact_class_matching():
    truth=[dict(time=2.,previous=0,label=1,end=4.),dict(time=3.,previous=0,label=1,end=5.)]
    events=[dict(time=1.,previous=0,label=1),dict(time=2.,previous=0,label=2)]
    rows,fp=match_events(events,truth,'future',2.)
    assert [r['detected'] for r in rows]==[True,False]
    assert fp==[events[1]]
    assert not match_events([dict(time=2.,previous=0,label=1)],truth[:1],'future',1.)[0][0]['detected']
    assert not match_events([dict(time=4.,previous=0,label=1)],truth[:1],'current',1.)[0][0]['detected']


def episode():
    return dict(fps=30.,annotation=dict(segments=[dict(label='pickup',start_frame=0,end_frame=30),dict(label='harvest',start_frame=30,end_frame=60),dict(label='return',start_frame=60,end_frame=90)]))


def proposal(old,new):
    return dict(original_boundary_frame=old,boundary_frame_reconstructed_from_visible_time=new)


def test_revisions_preserve_source_and_adjacent_intervals():
    e=episode();original=deepcopy(e)
    revised=revise_episode(e,[proposal(30,33),proposal(60,62)])
    assert e==original
    assert revised['annotation']['segments'][0]['end_frame']==33
    assert revised['annotation']['segments'][1]['start_frame']==33
    assert revised['annotation']['segments'][1]['end_frame']==62


@pytest.mark.parametrize('rows',[[proposal(30,61)],[proposal(30,33),proposal(30,34)],[proposal(31,33)]])
def test_invalid_and_conflicting_revisions_rejected(rows):
    with pytest.raises(ValueError):revise_episode(episode(),rows)
