"""Request/result checks for an offline harness, never a robot/RTC executor."""
import numpy as np


def prediction_packet(bridge,current,future,session,oid,t,version):
    return dict(protocol=bridge.PROTOCOL,labels=list(bridge.LABELS),valid=True,error=None,observation_id=oid,session_id=session,model_version=version,
                prediction_mode='joint',future_horizon_s=1.5,input_timestamp=float(t),result_timestamp=float(t)+.04,
                current_probabilities=np.asarray(current,dtype=float).tolist(),future_probabilities=np.asarray(future,dtype=float).tolist(),
                current_logits=np.log(np.maximum(current,1e-30)).tolist(),future_logits=np.log(np.maximum(future,1e-30)).tolist())


def check_request_response(packet,*,observation_id,input_timestamp,model_version):
    if packet.get('observation_id')!=observation_id or packet.get('input_timestamp')!=input_timestamp or packet.get('model_version')!=model_version:
        raise ValueError('Locked request timestamp/model/observation mismatch')
    if packet.get('future_horizon_s')!=1.5:raise ValueError('Locked Future horizon mismatch')


class OfflineChunkGate:
    """Mock admission and continuity *measurement*. Never changes action values."""
    def __init__(self,bridge,max_age_s=2.):
        self.bridge=bridge;self.max_age_s=max_age_s;self.requests={};self.consumed=set();self.last_action=None

    def request(self,context,chunk_id):
        if chunk_id in self.requests:raise ValueError('Unique chunk request ID required')
        binding=self.bridge.bind_chunk(context,chunk_id);self.requests[chunk_id]=binding.copy();return binding

    def result(self,binding,actions,context,now,*,safe_boundary):
        if not isinstance(binding,dict):return dict(accepted=False,reason='invalid_request_binding')
        if binding.get('chunk_id') in self.consumed:return dict(accepted=False,reason='duplicate_result')
        if self.requests.get(binding.get('chunk_id'))!=binding:return dict(accepted=False,reason='request_binding_mismatch')
        if not self.bridge.chunk_matches_prompt(binding,context):return dict(accepted=False,reason='obsolete_prompt_or_session_model')
        try:age=float(now)-float(binding['input_timestamp'])
        except (TypeError,ValueError):return dict(accepted=False,reason='invalid_result_timestamp')
        if not 0<=age<=self.max_age_s:return dict(accepted=False,reason='expired_or_future_result')
        try:x=np.asarray(actions,float)
        except (TypeError,ValueError):return dict(accepted=False,reason='invalid_action_chunk')
        if x.shape!=(50,58) or not np.isfinite(x).all():return dict(accepted=False,reason='invalid_action_chunk')
        if not safe_boundary:return dict(accepted=False,reason='deferred_until_mock_safe_boundary')
        delta=float(np.max(abs(x[0]-self.last_action))) if self.last_action is not None else None
        self.last_action=x[14].copy();self.consumed.add(binding['chunk_id'])
        return dict(accepted=True,reason='mock_admitted_no_robot_execution',first_action_max_delta=delta,
                    mock_executed_prefix_steps=15,action_values_modified=False,continuity_is_policy_guarantee=False)
