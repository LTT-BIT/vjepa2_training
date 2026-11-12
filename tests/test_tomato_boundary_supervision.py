import numpy as np
from tools.train_tomato_boundary_supervision import boundary_weights


def test_emphasis_is_symmetric_around_annotation_boundary():
    segments=[dict(start_frame=0),dict(start_frame=90)]
    times=np.array([0,1.9,2,2.5,3,3.5,4,4.1])
    np.testing.assert_array_equal(boundary_weights(times,segments,30),[1,1,3,3,3,3,3,1])


def test_no_implicit_episode_time_progress_or_label_change():
    segments=[dict(start_frame=0),dict(start_frame=300)]
    times=np.array([1,5,9,10,11,15])
    old=times.copy();w=boundary_weights(times,segments,30)
    np.testing.assert_array_equal(times,old)
    np.testing.assert_array_equal(w,[1,1,3,3,3,1])
