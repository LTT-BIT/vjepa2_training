"""Frozen-cache outcome verifier scaffold; refuses unreviewed supervision."""
import numpy as np
from .tomato_operation_memory import EVENTS


def validate_label(row):
    if (row.get('event_type') not in EVENTS or row.get('verdict') not in ('positive','negative','unknown')
        or row.get('label_source')!='independent_operation_review' or row.get('stage_labels_used_as_truth') is not False):raise ValueError('Independent operation review required')
    if row.get('status')!='approved' or len(set(row.get('reviewers',[])))<2:raise ValueError('Two independent human approvals required')
    if any(type(row.get(k)) is not int for k in ('start_frame','end_frame','evidence_end_frame')):raise ValueError('Frame indices must be integers')
    if not 0<=row['start_frame']<row['end_frame'] or row['evidence_end_frame']<row['start_frame'] or row['evidence_end_frame']>=row['end_frame']:raise ValueError('Invalid observable interval')
    if not row.get('visible_cameras') or not row.get('evidence_note') or not row.get('annotation_sha256'):raise ValueError('Missing visual evidence/source identity')
    if row.get('review_schema')=='tomato-operation-review-v2':
        if row.get('outcome')!={'positive':'success','negative':'failure','unknown':'unknown'}[row['verdict']]:raise ValueError('Outcome/verdict mismatch')
        if any(c not in ('cam0','cam1','cam2') for c in row['visible_cameras']):raise ValueError('Invalid camera source')
        first=row.get('first_confirmable_frame')
        if row['verdict']=='unknown':
            if first is not None:raise ValueError('Unknown first-confirmable time must be null')
        elif type(first) is not int or first!=row['evidence_end_frame']:raise ValueError('First result evidence must determine causal release')
    return row


def causal_features(current,previous=None,*,mode='C',timestamp,previous_timestamp=None):
    x=np.asarray(current,np.float32)
    if x.shape!=(4000,) or not np.isfinite(x).all():raise ValueError('Expected existing 3072 visual +16x58 state features')
    if mode=='A':return x[3072:]
    if mode=='B':return x[:3072]
    if mode=='C':return x
    if mode!='D':raise ValueError('Unknown ablation')
    if previous is None:return np.r_[x,np.zeros_like(x)]
    p=np.asarray(previous,np.float32)
    if previous_timestamp is None or not previous_timestamp<timestamp or p.shape!=x.shape or not np.isfinite(p).all():raise ValueError('Previous same-episode causal feature required')
    return np.r_[x,x-p]


def masked_bce(logits,targets,known):
    import torch
    if logits.shape!=targets.shape or known.shape!=targets.shape or not bool(known.any()):raise ValueError('Unknown is masked, never a negative example')
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[known],targets[known])


def build_model(mode):
    from torch import nn
    dims={'A':928,'B':3072,'C':4000,'D':8000}
    return nn.Sequential(nn.Linear(dims[mode],64),nn.ReLU(),nn.Linear(64,len(EVENTS)))


class OutcomeVerifier:
    def __init__(self,checkpoint=None):
        # A real trained and calibrated artifact must be introduced separately.
        # No silent random classifier, stage-label fallback or heuristic grasp success.
        self.saved=self.model=None
        if checkpoint is not None:
            import torch
            self.saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if self.saved.get('format')!='tomato-outcome-verifier-v1' or self.saved.get('independent_labels') is not True:raise ValueError('Independent outcome checkpoint required')
            self.model=build_model(self.saved['mode']);self.model.load_state_dict(self.saved['state_dict']);self.model.eval()
    def predict(self,*,observation_id,timestamp,features=None):
        if self.model is not None:
            import torch
            if features is None:raise ValueError('Explicit causal input features required')
            x=np.asarray(features,np.float32)
            if x.shape!=tuple(self.saved['mean'].shape) or not np.isfinite(x).all():raise ValueError('Verifier feature schema')
            with torch.inference_mode():p=torch.sigmoid(self.model(torch.from_numpy((x-np.asarray(self.saved['mean']))/np.asarray(self.saved['std'])))).tolist()
            return [dict(event_type=e,event_probability=p[i],evidence_timestamp=timestamp,observation_id=observation_id,evidence_validity=False,status='research_candidate_not_qualified',reason='requires_independent_calibration_before_physical_confirmation') for i,e in enumerate(EVENTS)]
        return [dict(event_type=e,event_probability=None,evidence_timestamp=timestamp,observation_id=observation_id,evidence_validity=False,status='insufficient_evidence',reason='no_qualified_outcome_checkpoint') for e in EVENTS]


def training_readiness(labels):
    rows=[validate_label(r) for r in labels]
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('Duplicate review intervals do not add independent supervision')
    counts={e:{v:sum(r['event_type']==e and r['verdict']==v for r in rows) for v in ('positive','negative','unknown')} for e in EVENTS}
    timed=[r for r in rows if r.get('review_schema')=='tomato-operation-review-v2']
    qualified={e:{v:sum(r['event_type']==e and r['verdict']==v for r in timed) for v in ('positive','negative','unknown')} for e in EVENTS}
    eligible=[e for e,c in qualified.items() if c['positive']>=10 and c['negative']>=10 and len({r['source_recording_id'] for r in timed if r['event_type']==e and r['verdict']!='unknown'})>=6]
    return dict(counts=counts,timed_v2_counts=qualified,eligible_events=eligible,can_train=bool(eligible),rule='At least10 independently approved v2 positive/negative intervals across6 recordings; explicit first evidence required. Development smoke only, not final sufficiency')
