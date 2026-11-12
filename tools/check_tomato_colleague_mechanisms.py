"""Isolated synthetic mechanism comparison; never loads recordings or models."""
import importlib.util,json,hashlib,sys
from pathlib import Path
from tools.tomato_state_recovery import B0State
import argparse
a=argparse.ArgumentParser();a.add_argument('--colleague-source',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args();p=args.colleague_source
spec=importlib.util.spec_from_file_location('colleague_handshake',p);h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
def probs(i):return tuple(.97 if j==i else .01 for j in range(4))
def row(t,c,f):return dict(observed_at=t,heads={k:dict(probabilities=dict(zip(h.LABELS,probs(v)))) for k,v in [('current',c),('future',f)]})
r=h.replay([row(i*.5,0,1) for i in range(7)],{},3.5)
b=B0State();ours=[b.update(probs(0),probs(1),i*.5) for i in range(7)]
classes=[0,1,2,3,0,1,2,3];rows=[row(j+i*.5,c,c) for j,c in enumerate(classes) for i in range(2)]
cy=h.replay(rows,{'ack_enabled':False},8)
b=B0State()
for c in classes:
 for i in range(2):b.update(probs(c),probs(c),0)
result=dict(scope='synthetic mechanism probes only; not model accuracy or real recording scores',colleague_sha='07f93115d5427b3ebaaf1735e9bd18a1e1240aa5',source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),future_only_support=dict(colleague_events=[{k:v for k,v in e.items() if k in ['kind','time','active_stage']} for e in r['events'] if e['kind'] in ['commit','timeout']],ours_final_phase=ours[-1]['phase']),after_seventh_position=dict(ack_enabled=False,reason='isolate legal-edge topology from ACK gating',colleague_commits=[dict(time=e['time'],stage=e['to_stage']) for e in cy['events'] if e['kind']=='commit'],ours_final_phase=b.phase))
assert ours[-1]['phase']==0 and any(e['kind']=='commit' for e in r['events'])
assert cy['summary']['commits']==7 and b.phase==6
args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
