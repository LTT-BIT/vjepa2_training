import numpy as np
import pytest

from tools import tomato_metrics as metrics
from tools import tomato_dataset as ds


def test_four_class_scores_and_real_nll_including_major_mistakes():
    y = np.arange(4)
    good = metrics.scores(np.eye(4) * 10, y)
    assert good['accuracy'] == good['macro_f1'] == 1
    assert [v['support'] for v in good['per_class'].values()] == [1, 1, 1, 1]
    bad = metrics.scores(np.tile([1000., 0, 0, 0], (4, 1)), y)
    assert bad['accuracy'] == .25 and bad['nll'] == pytest.approx(750)
    for z, yy in ((np.ones((1, 6)), [0]), (np.ones((1, 4)), [4]), (np.full((1, 4), np.nan), [0])):
        with pytest.raises(ValueError):
            metrics.scores(z, yy)


def test_boundaries_misses_false_switches_and_error_intervals():
    e = dict(length=120, fps=30, annotation=dict(segments=[
        dict(start_frame=i * 30, end_frame=(i + 1) * 30, label=label) for i, label in enumerate(ds.LABELS)]))
    anchors = np.arange(0, 120, 15); times = anchors / 30
    pred = np.array([0, 1, 0, 1, 2, 2, 2, 2])
    result = metrics.temporal_report(e, anchors, times, pred)
    assert result['transitions'][0]['delay_s'] == .5
    assert result['transitions'][2]['missed']
    assert result['missed_transition_ratio'] == pytest.approx(1 / 3)
    assert result['false_switches'] == 2
    assert len(result['error_intervals']) == 2
    assert result['time_resolution_s'] == .5
