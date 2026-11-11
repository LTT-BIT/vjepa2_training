"""Explicit offline history ablation. Oracle annotations are never an online API.

The oracle arm is a diagnostic with human stage truth released at the previous
observation only. It cannot establish physical outcomes or deployable recovery.
"""
import math
from .tomato_state_recovery import B0State


def oracle_prefix(segments, fps, as_of):
    """Only already-started transitions; no future ends, durations or next labels."""
    if as_of is None: return []
    return [dict(phase=i, event_time=s['start_frame']/fps, available_at=as_of,
                 source='oracle_past_subtask') for i,s in enumerate(segments)
            if i and s['start_frame']/fps <= as_of]


def oracle_anchor(segments, fps, as_of):
    """H1 exposes one position and its observation time, no event history."""
    if as_of is None:return None
    return dict(phase=sum(s['start_frame']/fps<=as_of for s in segments)-1, available_at=as_of)


class HistoryReplay:
    def __init__(self, mode):
        if mode not in ('none', 'predicted', 'oracle', 'anchor'): raise ValueError('Explicit history arm required')
        self.mode=mode; self.base=B0State(); self.phase=0; self.version=0; self.last=None
        self.history=[]; self.corrections=0

    def update(self, current, future, timestamp, *, oracle_history=None, oracle_position=None):
        if not math.isfinite(timestamp) or self.last is not None and timestamp<=self.last: raise ValueError('Strict observation order required')
        if self.mode!='oracle' and oracle_history is not None: raise ValueError('Ground truth cannot enter predicted/none arm')
        if self.mode!='anchor' and oracle_position is not None:raise ValueError('Oracle position is an offline H1 diagnostic only')
        prefix=self.history if self.mode=='predicted' else (oracle_history or []) if self.mode=='oracle' else None
        corrected=False
        if self.mode=='anchor' and oracle_position is not None:
            if (set(oracle_position)!={'phase','available_at'} or type(oracle_position['phase']) is not int
                or not 0<=oracle_position['phase']<=6 or self.last is None
                or oracle_position['available_at']!=self.last or oracle_position['available_at']>=timestamp):raise ValueError('Only previous-observation anchor is permitted')
            prior=min(oracle_position['phase'],self.base.phase+1)
            if prior>self.base.phase:
                self.base.phase=prior;self.base.count=0;corrected=True;self.corrections+=1
        if prefix is not None:
            if len(prefix)>6:raise ValueError('Only six ordered transitions exist')
            for i,row in enumerate(prefix,1):
                if any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) for k in ('event_time','available_at')):raise ValueError('Finite history times required')
                if (row['phase']!=i or row['event_time']>row['available_at'] or row['available_at']>=timestamp
                    or self.last is None or row['available_at']>self.last): raise ValueError('Future/noncontiguous stage history rejected')
            prior=prefix[-1]['phase'] if prefix else 0
            # Even the diagnostic may only advance one ordered position. A
            # previously early confirmation cannot be undone by truth rollback.
            prior=min(prior,self.base.phase+1)
            if prior>self.base.phase:
                self.base.phase=prior; self.base.count=0; corrected=True; self.corrections+=1
        result=self.base.update(current,future,timestamp)
        before=self.phase; self.phase=result['phase']; changed=self.phase!=before
        if changed:
            self.version+=1
            if self.mode=='predicted':
                self.history.append(dict(phase=self.phase,event_time=timestamp,available_at=timestamp,source='predicted_confirmation'))
        self.last=timestamp
        return dict(time=float(timestamp),phase=self.phase,previous_phase=before,changed=changed,
                    prompt_version=self.version,resynchronized=corrected,history_corrected=corrected,warning=result['warning'])
