"""Optional outcome/memory sidecar to an unchanged stage prediction service."""
from copy import deepcopy
from .tomato_task_memory import Observation
from .tomato_operation_memory import OperationMemory,public_memory
from .tomato_outcome_verifier import OutcomeVerifier
from .tomato_state_recovery import StateRecovery


class OperationSidecar:
    def __init__(self,session_id,episode_id,model_version):
        self.memory=OperationMemory(session_id,episode_id,model_version)
        self.b0=StateRecovery('baseline',session_id=session_id,model_id=model_version)
        self.verifier=OutcomeVerifier();self.sequence=0

    def process(self,response,*,now):
        if response.get('valid') is not True or response.get('session_id')!=self.memory.session_id or response.get('labels')!=['pickup','harvest','return','adjust'] or response.get('prediction_mode')!='joint':raise ValueError('Valid same-session joint prediction required')
        t=response['input_timestamp'];c=response['current_probabilities'];f=response['future_probabilities'];mid=response['model_version']
        b=self.b0.update(c,f,t,now=now,session_id=response['session_id'],model_id=mid)
        if not b['valid']:raise ValueError('Invalid/stale/repeated prediction')
        self.sequence+=1;canonical=f'{self.memory.session_id}:{self.sequence}'
        self.memory.stage.observe(Observation(self.memory.session_id,self.sequence,canonical,t,now,tuple(c),tuple(f),mid),b0_phase=b['phase'])
        self.memory.expire(now)
        payload=public_memory(self.memory,observation_id=response['observation_id'],timestamp=t,now=now)
        return dict(protocol='tomato-agent-memory-sidecar-v1',prediction=deepcopy(response),operation_evidence=self.verifier.predict(observation_id=response['observation_id'],timestamp=t),memory=payload,b0=deepcopy(b))
