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


def temporal_report(episode, anchors, times, predictions):
    anchors, times, pred = np.asarray(anchors), np.asarray(times), np.asarray(predictions)
    if (anchors.ndim != 1 or times.shape != anchors.shape or pred.shape != anchors.shape
            or not len(anchors) or np.any(np.diff(anchors) <= 0) or np.any(np.diff(times) <= 0)
            or np.any(anchors < 0) or anchors[-1] >= episode['length']
            or np.any((pred < 0) | (pred > 3))):
        raise ValueError('Invalid temporal prediction timeline')
    segments = episode['annotation']['segments']
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
    return dict(transitions=transitions, error_intervals=intervals, prediction_changes=int(changes.sum()),
                false_switches=int(false.sum()), false_switches_per_minute=float(false.sum() / (episode['length'] / episode['fps'] / 60)),
                delay_p50_s=float(np.percentile(delays, 50)) if delays else None,
                delay_p95_s=float(np.percentile(delays, 95)) if delays else None,
                missed_transition_ratio=float(np.mean([r['missed'] for r in transitions])) if transitions else None,
                prediction_hz=2, time_resolution_s=.5,
                definition='First correct anchor after true boundary, before end of that segment. False switch: prediction change between anchors with same true label. No sequence correction.')


def evaluate_entries(entries, logits):
    y = np.concatenate([r['labels'] for r in entries])
    result = scores(logits, y)
    per_episode, per_source, offset = {}, {}, 0
    all_transitions, false_switches, duration = [], 0, 0.
    for r in entries:
        e = r['episode']; n = len(r['labels']); z = logits[offset:offset + n]
        temporal = temporal_report(e, r['anchors'], r['times'], z.argmax(axis=1))
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
            summary = {**evaluate_entries(entries, z), **summary}
        else:
            per_episode, sources, offset = {}, {}, 0
            for r in entries:
                n = len(r['anchors']); keep = r['future_valid_mask']; part = z[offset:offset + n][keep]; y = r['future_labels'][keep]
                e = r['episode']
                per_episode[e['episode_id']] = dict(metrics=scores(part, y) if len(y) else None,
                                                    masked_count=int((~keep).sum()), source_recording_id=e['source_recording_id'])
                if len(y): sources.setdefault(e['source_recording_id'], []).append((part, y))
                offset += n
            summary.update(per_episode=per_episode, per_source={g: scores(np.concatenate([p[0] for p in a]), np.concatenate([p[1] for p in a])) for g, a in sources.items()})
        report[name] = summary
    if prediction['prediction_mode'] == 'current_only':
        report.update(report['current'])  # Existing current-only report readers retain their fields.
    return report
