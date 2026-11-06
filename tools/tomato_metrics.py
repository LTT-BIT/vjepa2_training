"""Four-class raw classifier metrics; no sequence constraints or label smoothing."""
from __future__ import annotations

import numpy as np

try:
    from .tomato_dataset import LABELS, label_at_frame
except ImportError:
    from tomato_dataset import LABELS, label_at_frame


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 4 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError('Expected nonempty finite [N,4] logits')
    shifted = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(shifted)
    return p / p.sum(axis=1, keepdims=True)


def scores(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    p = probabilities(logits)
    y = np.asarray(labels)
    if y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y > 3)):
        raise ValueError('Expected integer labels in fixed four-class order')
    pred = p.argmax(axis=1)
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    tp = confusion.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(4), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(4), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(4), where=(support + predicted) > 0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    nll = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]
    return dict(samples=len(y), accuracy=float((pred == y).mean()), macro_f1=float(f1.mean()),
                nll=float(nll.mean()), label_order=list(LABELS),
                per_class={label: dict(precision=float(precision[i]), recall=float(recall[i]),
                                       f1=float(f1[i]), support=int(support[i])) for i, label in enumerate(LABELS)},
                confusion_true_rows=confusion.tolist(),
                return_adjust_confusions=int(confusion[2, 3] + confusion[3, 2]),
                missing_classes=[LABELS[i] for i in range(4) if not support[i]],
                macro_definition='unweighted four-class mean, zero_division=0')


def temporal_report(episode, anchors, times, predictions, stable_transition_k=2):
    anchors, times, pred = np.asarray(anchors), np.asarray(times), np.asarray(predictions)
    if (anchors.ndim != 1 or times.shape != anchors.shape or pred.shape != anchors.shape
            or not len(anchors) or np.any(np.diff(anchors) <= 0) or np.any(np.diff(times) <= 0)
            or np.any(anchors < 0) or anchors[-1] >= episode['length']
            or np.any((pred < 0) | (pred > 3))):
        raise ValueError('Invalid temporal prediction timeline')
    segments = continuous_phases(episode['annotation']['segments'])
    truth = np.array([label_at_frame(segments, int(i)) for i in anchors])
    transitions = []
    for previous, seg in zip(segments[:-1], segments[1:]):
        if previous['label'] == seg['label']:
            continue
        boundary = seg['start_frame'] / episode['fps']
        eligible = np.flatnonzero((anchors >= seg['start_frame']) & (anchors < seg['end_frame'])
                                  & (pred == LABELS.index(seg['label'])))
        first = int(eligible[0]) if len(eligible) else None
        transitions.append(dict(frame_index=seg['start_frame'], boundary_timestamp_s=boundary,
                                from_stage=previous['label'], to_stage=seg['label'],
                                first_correct_timestamp_s=float(times[first]) if first is not None else None,
                                delay_s=max(0., float(times[first]) - boundary) if first is not None else None,
                                missed=first is None))
    wrong = pred != truth
    starts = np.flatnonzero(wrong & ~np.r_[False, wrong[:-1]])
    stops = np.flatnonzero(wrong & ~np.r_[wrong[1:], False]) + 1
    intervals = [dict(start_frame=int(anchors[a]),
                      end_frame_exclusive=int(anchors[b]) if b < len(anchors) else episode['length'],
                      start_timestamp_s=float(times[a]),
                      end_timestamp_s=float(times[b]) if b < len(times) else episode['length'] / episode['fps'],
                      wrong_anchors=int(b - a),
                      pairs=[dict(frame_index=int(anchors[k]), truth=LABELS[int(truth[k])], prediction=LABELS[int(pred[k])])
                             for k in range(a, b)]) for a, b in zip(starts, stops)]
    changes = pred[1:] != pred[:-1]
    false = changes & (truth[1:] == truth[:-1])
    delays = [r['delay_s'] for r in transitions if not r['missed']]
    result = dict(transitions=transitions, error_intervals=intervals, prediction_changes=int(changes.sum()),
                false_switches=int(false.sum()), false_switches_per_minute=float(false.sum() / (episode['length'] / episode['fps'] / 60)),
                delay_p50_s=float(np.percentile(delays, 50)) if delays else None,
                delay_p95_s=float(np.percentile(delays, 95)) if delays else None,
                missed_transition_ratio=float(np.mean([r['missed'] for r in transitions])) if transitions else None,
                prediction_hz=2, time_resolution_s=.5,
                definition='First correct anchor after true boundary, before end of that segment. False switch: prediction change between anchors with same true label. No sequence correction.')
    result.update(current_stability(episode, anchors, times, pred, transitions, stable_transition_k))
    return result


def evaluate_entries(entries, logits, stable_transition_k=2):
    y = np.concatenate([r['labels'] for r in entries])
    result = scores(logits, y)
    per_episode, per_source, offset = {}, {}, 0
    all_transitions, false_switches, duration = [], 0, 0.
    for r in entries:
        e = r['episode']; n = len(r['labels']); z = logits[offset:offset + n]
        temporal = temporal_report(e, r['anchors'], r['times'], z.argmax(axis=1), stable_transition_k)
        per_episode[e['episode_id']] = dict(metrics=scores(z, r['labels']), temporal=temporal,
                                           source_recording_id=e['source_recording_id'])
        per_source.setdefault(e['source_recording_id'], []).append((z, r['labels']))
        all_transitions += temporal['transitions']
        false_switches += temporal['false_switches']
        duration += e['length'] / e['fps']
        offset += n
    delays = [t['delay_s'] for t in all_transitions if not t['missed']]
    result.update(per_episode=per_episode,
                  per_source={g: scores(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])) for g, parts in per_source.items()},
                  temporal=dict(transitions=len(all_transitions),
                                delay_p50_s=float(np.percentile(delays, 50)) if delays else None,
                                delay_p95_s=float(np.percentile(delays, 95)) if delays else None,
                                missed_transition_ratio=float(np.mean([t['missed'] for t in all_transitions])) if all_transitions else None,
                                false_switches=false_switches, false_switches_per_minute=false_switches / (duration / 60),
                                prediction_hz=2, time_resolution_s=.5))
    stable = [t['stable_transition_latency_s'] for t in all_transitions if not t['stable_transition_miss']]
    temporal_rows = [r['temporal'] for r in per_episode.values()]
    result['temporal'].update(stable_transition_k=stable_transition_k,
        first_correct_latency_p50_s=result['temporal']['delay_p50_s'],
        first_correct_latency_p95_s=result['temporal']['delay_p95_s'],
        stable_transition_latency_p50_s=percentile_or_none(stable, 50),
        stable_transition_latency_p95_s=percentile_or_none(stable, 95),
        stable_transition_miss_ratio=float(np.mean([t['stable_transition_miss'] for t in all_transitions])) if all_transitions else None,
        early_false_switch_count=sum(t['early_false_switch']['count'] for t in temporal_rows),
        early_false_switch_duration_s=sum(t['early_false_switch']['duration_s'] for t in temporal_rows),
        wrong_stage_duration_s=sum(t['wrong_stage_duration_s'] for t in temporal_rows))
    return result


def evaluate_heads(entries, logits, config):
    try:
        from .tomato_dataset import prediction_contract, active_heads
    except ImportError:
        from tomato_dataset import prediction_contract, active_heads
    prediction = prediction_contract(config)
    report = dict(**prediction, current=None, future=None,
                  total_anchors=sum(len(r['anchors']) for r in entries))
    for name in active_heads(prediction['prediction_mode']):
        z = logits[name + '_logits']
        labels = np.concatenate([r[name + '_labels'] for r in entries])
        mask = np.ones(len(labels), dtype=bool) if name == 'current' else np.concatenate([r['future_valid_mask'] for r in entries])
        summary = scores(z[mask], labels[mask])
        summary.update(valid_samples=int(mask.sum()), masked_count=int((~mask).sum()), valid_ratio=float(mask.mean()))
        if name == 'current':
            summary = {**evaluate_entries(entries, z, prediction['stable_transition_k']), **summary}
        else:
            per_episode, sources, offset = {}, {}, 0
            for r in entries:
                n = len(r['anchors']); keep = r['future_valid_mask']; part = z[offset:offset + n][keep]; y = r['future_labels'][keep]
                e = r['episode']
                raw_pred = z[offset:offset + n].argmax(axis=1)
                forecast_errors = alarm_intervals(e, r['anchors'], r['times'], keep & (raw_pred != r['future_labels']), raw_pred)
                offset_frames = int(round(prediction['future_horizon_s'] * e['fps']))
                for interval in forecast_errors:
                    interval.update(time_axis='forecast issue anchor',
                                    target_start_frame=interval['start_frame'] + offset_frames,
                                    target_end_frame_exclusive=min(e['length'], interval['end_frame_exclusive'] + offset_frames))
                per_episode[e['episode_id']] = dict(metrics=scores(part, y) if len(y) else None,
                                                    masked_count=int((~keep).sum()), source_recording_id=e['source_recording_id'],
                                                    forecast_error_intervals=forecast_errors,
                                                    forecast_error_duration_s=sum(a['duration_s'] for a in forecast_errors),
                                                    anticipation=future_anticipation(r, z[offset:offset + n].argmax(axis=1), prediction['future_horizon_s']))
                if len(y): sources.setdefault(e['source_recording_id'], []).append((part, y))
                offset += n
            summary.update(per_episode=per_episode, per_source={g: scores(np.concatenate([p[0] for p in a]), np.concatenate([p[1] for p in a])) for g, a in sources.items()})
            current_truth = np.concatenate([r['current_labels'] for r in entries])
            baseline = scores(np.eye(4)[current_truth[mask]], labels[mask])
            baseline.update(nll=None, diagnostic_only=True, deployable=False,
                            definition='Oracle true current stage held as future prediction; hard labels, no probabilistic NLL')
            baselines = dict(oracle_current_persistence=baseline)
            if logits['current_logits'] is not None:
                baselines['model_current_persistence'] = scores(logits['current_logits'][mask], labels[mask])
            anticipation = [r['anticipation'] for r in per_episode.values()]
            events = [e for a in anticipation for e in a['events']]
            evaluable = [e for e in events if not e['ambiguous']]
            leads = [e['lead_time_s'] for e in evaluable if e['detected']]
            summary.update(persistence_baselines=baselines,
                forecast_error_duration_s=sum(r['forecast_error_duration_s'] for r in per_episode.values()),
                macro_f1_gain_over_oracle_persistence=summary['macro_f1'] - baseline['macro_f1'],
                macro_f1_gain_over_model_current_persistence=summary['macro_f1'] - baselines['model_current_persistence']['macro_f1'] if 'model_current_persistence' in baselines else None,
                anticipation=dict(total_events=len(events), ambiguous_events=len(events) - len(evaluable),
                    evaluable_events=len(evaluable), detected_events=sum(e['detected'] for e in evaluable),
                    event_recall=float(np.mean([e['detected'] for e in evaluable])) if evaluable else None,
                    missed_events=sum(e['missed'] for e in evaluable), lead_time_p50_s=percentile_or_none(leads, 50),
                    lead_time_p95_s=percentile_or_none(leads, 95),
                    out_of_window_false_alarm_count=sum(a['out_of_window_false_alarm_count'] for a in anticipation),
                    out_of_window_false_alarm_duration_s=sum(a['out_of_window_false_alarm_duration_s'] for a in anticipation),
                    out_of_window_false_alarms_per_minute=sum(a['out_of_window_false_alarm_count'] for a in anticipation) / (sum(a['valid_anchor_duration_s'] for a in anticipation) / 60) if any(a['valid_anchor_duration_s'] for a in anticipation) else None,
                    ambiguity_rule='Events with another boundary less than horizon away are excluded from recall, retained explicitly'))
        summary['phase_duration_strata'] = duration_strata(entries, z, name)
        report[name] = summary
    if prediction['prediction_mode'] == 'current_only':
        report.update(report['current'])  # Existing current-only report readers retain their fields.
    return report


def percentile_or_none(values, percentile):
    return float(np.percentile(values, percentile)) if len(values) else None


def continuous_phases(segments):
    """Metric-only copies: an adjacent duplicate label is not a true transition.

    Preserve reviewed source segments byte-for-byte; never change training targets.
    """
    phases = []
    for segment in segments:
        if phases and phases[-1]['label'] == segment['label']:
            phases[-1]['end_frame'] = segment['end_frame']
        else:
            phases.append(dict(segment))
    return phases


def anchor_widths(episode, times):
    times = np.asarray(times, dtype=float)
    return np.maximum(0., np.diff(np.r_[times, episode['length'] / episode['fps']]))


def alarm_intervals(episode, anchors, times, flags, predictions):
    """Runs of wrong/early alerts, splitting a run when its predicted class changes."""
    flags, pred = np.asarray(flags), np.asarray(predictions)
    widths = anchor_widths(episode, times)
    starts = np.flatnonzero(flags & np.r_[True, ~flags[:-1] | (pred[1:] != pred[:-1])])
    intervals = []
    for a in starts:
        b = int(a) + 1
        while b < len(flags) and flags[b] and pred[b] == pred[a]: b += 1
        intervals.append(dict(start_frame=int(anchors[a]),
                              end_frame_exclusive=int(anchors[b]) if b < len(anchors) else episode['length'],
                              start_timestamp_s=float(times[a]), duration_s=float(widths[a:b].sum()),
                              predicted_stage=LABELS[int(pred[a])], anchors=int(b - a)))
    return intervals


def current_stability(episode, anchors, times, pred, transitions, k):
    if type(k) is not int or k < 1: raise ValueError('stable_transition_k must be a positive integer')
    segments = continuous_phases(episode['annotation']['segments'])
    truth = np.array([label_at_frame(segments, int(i)) for i in anchors])
    widths = anchor_widths(episode, times)
    stable_delays, early = [], []
    for event in transitions:
        index = next(i for i, s in enumerate(segments) if s['start_frame'] == event['frame_index'])
        segment, previous = segments[index], segments[index - 1]
        positions = np.flatnonzero((anchors >= segment['start_frame']) & (anchors < segment['end_frame']))
        correct = pred[positions] == LABELS.index(segment['label'])
        confirmation = None
        for j in range(k - 1, len(positions)):
            if np.all(correct[j - k + 1:j + 1]):
                confirmation = int(positions[j]); break
        delay = max(0., float(times[confirmation]) - event['boundary_timestamp_s']) if confirmation is not None else None
        event.update(first_correct_latency_s=event['delay_s'], stable_transition_k=k,
                     first_correct_frame=int(anchors[positions[np.flatnonzero(correct)[0]]]) if np.any(correct) else None,
                     stable_confirmation_frame=int(anchors[confirmation]) if confirmation is not None else None,
                     stable_confirmation_timestamp_s=float(times[confirmation]) if confirmation is not None else None,
                     stable_transition_latency_s=delay, stable_transition_miss=confirmation is None)
        if delay is not None: stable_delays.append(delay)
        flags = ((anchors >= previous['start_frame']) & (anchors < segment['start_frame'])
                 & (pred == LABELS.index(segment['label'])) & (truth != pred))
        intervals = alarm_intervals(episode, anchors, times, flags, pred)
        # Clip approximation to the actual transition when it falls between two anchors.
        for interval in intervals:
            end = min(episode['length'], interval['end_frame_exclusive']) / episode['fps']
            interval['duration_s'] = max(0., min(end, event['boundary_timestamp_s']) - interval['start_timestamp_s'])
        early.append(dict(boundary_frame=event['frame_index'], to_stage=event['to_stage'],
                          count=len(intervals), duration_s=sum(r['duration_s'] for r in intervals), intervals=intervals))
    return dict(stable_transition_k=k, stable_transition_latency_p50_s=percentile_or_none(stable_delays, 50),
                stable_transition_latency_p95_s=percentile_or_none(stable_delays, 95),
                stable_transition_miss_ratio=float(np.mean([t['stable_transition_miss'] for t in transitions])) if transitions else None,
                early_false_switch=dict(count=sum(e['count'] for e in early), duration_s=sum(e['duration_s'] for e in early),
                                        events=early, definition='Successor-stage prediction bursts during the immediately preceding true segment, including transient and sustained errors'),
                wrong_stage_duration_s=float(widths[pred != truth].sum()),
                wrong_duration_definition='Anchor-label rectangle estimate: each raw argmax held until next anchor/end; 2 Hz resolution, no smoothing')


def future_anticipation(entry, predictions, horizon):
    e = entry['episode']; anchors, times = entry['anchors'], entry['times']
    pred = np.asarray(predictions); valid = entry['future_valid_mask']
    segments = continuous_phases(e['annotation']['segments'])
    boundaries = [s for previous, s in zip(segments[:-1], segments[1:]) if previous['label'] != s['label']]
    boundary_times = [s['start_frame'] / e['fps'] for s in boundaries]
    events = []
    for n, segment in enumerate(boundaries):
        b = boundary_times[n]
        ambiguous = any(0 < abs(b - other) < horizon - 1e-9 for other in boundary_times)
        offset_frames = int(round(horizon * e['fps']))
        eligible = valid & (anchors >= segment['start_frame'] - offset_frames) & (anchors < segment['start_frame'])
        correct = np.flatnonzero(eligible & (pred == LABELS.index(segment['label'])))
        first = int(correct[0]) if len(correct) else None
        events.append(dict(boundary_frame=segment['start_frame'], boundary_timestamp_s=b, to_stage=segment['label'],
                           eligible_anchors=int(eligible.sum()), new_stage_prediction_count=int(len(correct)),
                           ambiguous=ambiguous, detected=bool(len(correct)) if not ambiguous else None,
                           missed=not bool(len(correct)) if not ambiguous else None,
                           first_warning_timestamp_s=float(times[first]) if first is not None and not ambiguous else None,
                           lead_time_s=b - float(times[first]) if first is not None and not ambiguous else None))
    false, ambiguous_alerts = np.zeros(len(anchors), dtype=bool), 0
    for i in np.flatnonzero(valid & (pred != entry['current_labels'])):
        # Target frame, not rounded t+delta, owns the exact half-open boundary semantics.
        within = [s for s in boundaries if anchors[i] < s['start_frame'] <= entry['future_target_frames'][i]]
        if len(within) > 1:
            ambiguous_alerts += 1
        elif not within or pred[i] != LABELS.index(within[0]['label']):
            false[i] = True
    alarms = alarm_intervals(e, anchors, times, false, pred)
    duration = float(anchor_widths(e, times)[valid].sum())
    return dict(events=events, ambiguous_alert_anchors=ambiguous_alerts,
                out_of_window_false_alarm_count=len(alarms), out_of_window_false_alarm_duration_s=sum(a['duration_s'] for a in alarms),
                out_of_window_false_alarms_per_minute=len(alarms) / (duration / 60) if duration else None,
                false_alarm_intervals=alarms, valid_anchor_duration_s=duration,
                definition='Alerts differ from oracle true current stage; supported only by a matching true transition within the fixed horizon. Multiple-boundary events are explicitly ambiguous; no online decision uses truth.')


def duration_strata(entries, logits, name, threshold_s=1.5):
    short, labels, valid = [], [], []
    for r in entries:
        seg = continuous_phases(r['episode']['annotation']['segments']); fps = r['episode']['fps']
        frames = r['anchors'] if name == 'current' else r['future_target_frames']
        mask = np.ones(len(frames), dtype=bool) if name == 'current' else r['future_valid_mask']
        lengths = []
        for frame, keep in zip(frames, mask):
            pos = int(np.searchsorted([s['end_frame'] for s in seg], frame, side='right')) if keep else 0
            lengths.append((seg[pos]['end_frame'] - seg[pos]['start_frame']) / fps <= threshold_s)
        short.extend(lengths); labels.extend(r[name + '_labels']); valid.extend(mask)
    short, labels, valid = np.asarray(short), np.asarray(labels), np.asarray(valid)
    return dict(short_threshold_s=threshold_s,
                short=dict(samples=int((valid & short).sum()), metrics=scores(logits[valid & short], labels[valid & short]) if np.any(valid & short) else None),
                long=dict(samples=int((valid & ~short).sum()), metrics=scores(logits[valid & ~short], labels[valid & ~short]) if np.any(valid & ~short) else None))
