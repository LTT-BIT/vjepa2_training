import numpy as np
import pytest

from tools import tomato_dataset as ds, tomato_metrics as metrics


def entry(cuts=(0, 60, 120, 150, 180)):
    segments = [dict(start_frame=a, end_frame=b, label=label, reviewed=True)
                for a, b, label in zip(cuts[:-1], cuts[1:], ds.LABELS)]
    e = dict(length=cuts[-1], fps=30, episode_id='test', source_recording_id='recording', annotation=dict(segments=segments))
    anchors = np.arange(0, e['length'], 15)
    targets = ds.build_phase_targets(anchors, segments, e['length'], 30)
    return dict(episode=e, labels=targets['current_labels'], times=anchors / 30,
                input_indices=np.stack([np.maximum(0, i - np.arange(15, -1, -1) * 3) for i in anchors]), **targets)


def test_stable_confirmation_uses_kth_time_misses_short_stage_and_raw_f1_unchanged():
    r = entry(); p = r['current_labels'].copy(); p[3] = 1; p[5] = 0
    k2 = metrics.temporal_report(r['episode'], r['anchors'], r['times'], p, 2)
    k3 = metrics.temporal_report(r['episode'], r['anchors'], r['times'], p, 3)
    event = k2['transitions'][0]
    assert event['first_correct_latency_s'] == 0
    assert event['stable_confirmation_frame'] == 105
    assert event['stable_transition_latency_s'] == 1.5
    assert k3['transitions'][1]['stable_transition_miss']
    assert k3['transitions'][1]['stable_transition_latency_s'] is None
    assert k2['early_false_switch']['count'] == 1 and k2['early_false_switch']['duration_s'] == .5
    assert k2['wrong_stage_duration_s'] == 1.
    assert k2['false_switches'] == 3
    logits = np.eye(4)[p] * 10
    a = metrics.evaluate_entries([r], logits, 2); b = metrics.evaluate_entries([r], logits, 3)
    assert a['macro_f1'] == b['macro_f1'] and a['accuracy'] == b['accuracy']


def test_future_events_lead_false_alert_duration_and_tail_are_separate():
    r = entry(); pred = np.where(r['future_valid_mask'], r['future_labels'], 0)
    anticipation = metrics.future_anticipation(r, pred, 1.)
    assert len(anticipation['events']) == 3
    assert all(e['detected'] and e['lead_time_s'] == 1. for e in anticipation['events'])
    assert anticipation['out_of_window_false_alarm_count'] == 0
    pred[0] = 2  # Unsupported return warning four seconds before the true return boundary.
    anticipation = metrics.future_anticipation(r, pred, 1.)
    assert anticipation['out_of_window_false_alarm_count'] == 1
    assert anticipation['out_of_window_false_alarm_duration_s'] == .5
    assert anticipation['false_alarm_intervals'][0]['start_frame'] == 0


def test_multiple_boundaries_inside_horizon_are_explicitly_ambiguous():
    r = entry((0, 30, 45, 75, 90))
    pred = np.where(r['future_valid_mask'], r['future_labels'], 0)
    events = metrics.future_anticipation(r, pred, 1.)['events']
    assert events[0]['ambiguous'] and events[1]['ambiguous']
    assert events[0]['detected'] is None and events[0]['missed'] is None
    assert events[2]['detected']


def test_adjacent_identical_annotation_labels_do_not_create_or_truncate_a_phase():
    r = entry(); original = r['episode']['annotation']['segments']
    harvest = original[1]
    original[1:2] = [{**harvest, 'end_frame': 75}, {**harvest, 'start_frame': 75}]
    pred = r['current_labels'].copy(); pred[4] = 0
    report = metrics.temporal_report(r['episode'], r['anchors'], r['times'], pred, 2)
    assert len(report['transitions']) == 3
    assert report['transitions'][0]['first_correct_latency_s'] == .5
    assert report['transitions'][0]['stable_transition_latency_s'] == 1.
    assert original[1]['end_frame'] == 75 and len(original) == 5  # No source annotation mutation.


def test_oracle_and_model_persistence_diagnostics_and_masked_future_metrics():
    r = entry()
    future = np.where(r['future_valid_mask'], r['future_labels'], 0)
    logits = dict(current_logits=np.eye(4)[r['current_labels']] * 10, future_logits=np.eye(4)[future] * 10)
    report = metrics.evaluate_heads([r], logits, dict(prediction_mode='joint'))
    assert report['current']['accuracy'] == report['future']['accuracy'] == 1
    assert report['future']['samples'] == 10 and report['future']['masked_count'] == 2
    assert report['future']['forecast_error_duration_s'] == 0
    baseline = report['future']['persistence_baselines']['oracle_current_persistence']
    assert baseline['accuracy'] == .4 and baseline['deployable'] is False and baseline['nll'] is None
    assert report['future']['macro_f1_gain_over_oracle_persistence'] > 0
    assert report['future']['anticipation']['event_recall'] == 1
    assert report['future']['anticipation']['lead_time_p50_s'] == 1
    assert 'macro_f1' not in report  # Combined head selection score is not classification accuracy/F1.
    logits['future_logits'][~r['future_valid_mask']] = [-100, 100, -100, -100]
    again = metrics.evaluate_heads([r], logits, dict(prediction_mode='joint'))
    assert again['future']['confusion_true_rows'] == report['future']['confusion_true_rows']
    logits['future_logits'][0] = [0, 0, 10, 0]
    wrong = metrics.evaluate_heads([r], logits, dict(prediction_mode='joint'))
    assert wrong['future']['forecast_error_duration_s'] == .5
    interval = wrong['future']['per_episode']['test']['forecast_error_intervals'][0]
    assert interval['start_frame'] == 0 and interval['target_start_frame'] == 30
