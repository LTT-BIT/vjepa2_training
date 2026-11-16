from .contracts import DecisionProposal,LABELS

class RulePolicy:
 def __init__(self,name):self.name=name
 def propose(self,o,features,core,task_policy):
  stage=LABELS[max(range(4),key=lambda i:o.future[i])];i=LABELS.index(stage);active=LABELS.index(core.active);prob=o.future[i]
  legal=stage in task_policy.get(core.active,[]) and prob>=.65 and prob>=o.future[active]
  count=(core.pending or {}).get('count',0)+1 if (core.pending or {}).get('stage')==stage else 1
  request=prob>=.8
  if self.name=='R1':request=prob>=.95 or (prob>=.8 and count>=2)
  if self.name=='R2':request=prob>=.8 or (o.current[i]>=.8 and prob>=.65)
  return DecisionProposal(o.observation_id,o.observed_at,o.available_at,stage if legal else None,bool(legal and request),float(prob),self.name,o.model_version,o.policy_version)
