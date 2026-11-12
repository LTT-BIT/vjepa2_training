"""Versioned compact SHADOW context, not an instruction or learned decision.

The byte budget is exact UTF-8, NOT an LLM tokenizer estimate. The audit journal
and robot vectors are never automatically placed into a VLA language Prompt.
"""
import math
from .tomato_durable_memory import canonical


def compact_context(memory,*,now,recent_limit=4,max_bytes=8192):
    if type(recent_limit) is not int or not 0<=recent_limit<=16 or type(max_bytes) is not int or not 256<=max_bytes<=65536:
        raise ValueError('Explicit bounded context request required')
    if not isinstance(now,(int,float)) or isinstance(now,bool) or not math.isfinite(now) or now<0 or memory.arrival_clock is not None and now<memory.arrival_clock:
        raise ValueError('Context query must use current nondecreasing observation clock')
    view=memory.view(now=now);facts=view['controller_facts'];rows=[r for r in view['perception']['short_term'] if r.get('valid')]
    recent=[]
    if recent_limit:
        for r in rows[-recent_limit:]:
            recent.append(dict(observation_id=r['observation_id'],observation_timestamp=r['timestamp'],current_probabilities=r['current'],future_probabilities=r['future'],future_horizon_s=1.5,scope='historical_prediction_not_physical_outcome'))
    result=dict(schema='tomato-task-memory-context-v1',session_id=memory.identity['session_id'],episode_id=memory.identity['episode_id'],model_version=memory.identity['model_version'],manifest_sha256=memory.identity['manifest_sha256'],controller_reported_phase=view['controller_reported_phase'],controller_prompt_version=view['controller_prompt_version'],ordered_decision_history=[dict(decision_id=f['decision_id'],from_phase=f['from_phase'],to_phase=f['to_phase'],confirmation_timestamp=f['decision_timestamp'],evidence_timestamp=f['observation_timestamp'],prompt_version=f['prompt_version']) for f in facts],recent_recognition_history=recent,fresh_current=None,suggests_prompt_update=False,physical_success='unknown',learned_memory=False,scope='shadow_context_only')
    if view['perception_is_fresh'] and rows:
        latest=rows[-1];result['fresh_current']=dict(observation_id=latest['observation_id'],timestamp=latest['timestamp'],age_s=now-latest['timestamp'],probabilities=latest['current'])
    while len(canonical(result).encode())>max_bytes and result['recent_recognition_history']:
        result['recent_recognition_history'].pop(0)
    if len(canonical(result).encode())>max_bytes:raise ValueError('Budget too small for identity and complete confirmed prefix; never silently truncate facts')
    return result
