import bisect
import math
LABELS=('pickup','harvest','return','adjust')
MODES={'current_only':('current',),'future_only':('future',),'joint':('current','future')}

def validate_rows(rows,e,mode):
    if mode not in MODES:raise ValueError('Unknown mode')
    if [r['frame_index'] for r in rows]!=e['anchors']:raise ValueError('Missing/duplicate/out-of-order anchors')
    result=[]
    for r in rows:
        i=r['frame_index'];t=r['timestamp_s']
        if (r['episode_id']!=e['episode_id'] or r['source_recording_id']!=e['source_recording_id'] or r['split']!=e['split']
            or r['prediction_mode']!=mode or not math.isfinite(t) or abs(t-i/e['fps'])>e.get('ui_timestamp_tolerance_s',1e-6)
            or r['last_input_frame']>i or max(r['input_indices'])>i):raise ValueError('Prediction source/time/mode mismatch')
        row=dict(frame_index=i,observed_at=t,source_recording_id=e['source_recording_id'],heads={})
        for h in MODES[mode]:
            p=r[h+'_probabilities']
            if set(p)!=set(LABELS) or any(not math.isfinite(v) or v<0 for v in p.values()) or abs(sum(p.values())-1)>1e-6:raise ValueError('Invalid probabilities')
            stage=max(LABELS,key=lambda x:p[x]);target=t+(1 if h=='future' else 0)
            row['heads'][h]=dict(stage=stage,probabilities=p,confidence=p[stage],observed_at=t,target_at=target,
                target_scorable=target<e['length']/e['fps'])
        result.append(row)
    return result

def held(rows,frame):
    i=bisect.bisect_right([r['frame_index'] for r in rows],frame)-1
    return rows[i] if i>=0 else None

def tracks(rows,e,mode):
    return {h:[dict(start_s=r['observed_at'],end_s=rows[i+1]['observed_at'] if i+1<len(rows) else e['length']/e['fps'],
        anchor_frame=r['frame_index'],**r['heads'][h]) for i,r in enumerate(rows)] for h in MODES[mode]}
