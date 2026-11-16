import subprocess
from pathlib import Path
from ..learned_handshake.identity import check as previous_check,load
from .contract import *


API_SHA='ba907d56d01320087d09b3f37ac48fe97ac7c363af181c6d78dcc5593b897339'


def check(c,out):
 import numpy as np
 root=Path(c['baseline_root']);mem=Path(c['memory_experiment']);ref=Path(c['reference_experiment']);core=REPO/'tools/tomato_review_ui/handshake.py'
 if sha(core)!=API_SHA:raise ValueError('Unexpected public API source, requires separate audit')
 original=subprocess.check_output(['git','show','db4cdad6e87af7c077f5be16a00bc277efeb54d9:tools/tomato_review_ui/handshake.py'],cwd=REPO,text=True);(out/'original_handshake.py').write_text(original)
 protected=read(mem/'protected_sha256.json');original_sha=protected.pop(str(core.resolve()),None)
 if original_sha!=__import__('hashlib').sha256(original.encode()).hexdigest():raise ValueError('Original production baseline identity mismatch')
 for p,h in protected.items():
  if not Path(p).is_file() or sha(p)!=h:raise ValueError('Original protected input changed '+p)
 for previous in [mem,ref]:
  for rel,h in read(previous/'artifacts_sha256.json').items():
   p=previous/rel
   if sha(p)!=h:raise ValueError('Prior artifact changed '+str(p))
   protected[str(p.resolve())]=h
  protected[str(previous/'handoff.zip')]=sha(previous/'handoff.zip')
 for name in ['TOMATO_JOINT_MEMORY_ABLATION_REPORT_20261112.md','TOMATO_LEARNED_HANDSHAKE_POLICY_ABLATION_REPORT_20261112.md']:
  p=REPO/'docs'/name;protected[str(p)]=sha(p)
 for p in (REPO/'tools/tomato_review_ui/learned_handshake').glob('*.py'):protected[str(p)]=sha(p)
 m=read(root/'joint/manifest.json');fc=read(mem/'feature_contract.lock.json');split=read(root/'joint/split.json')
 if len(m['episodes'])!=93 or {k:len(v) for k,v in split.items()}!={'train':75,'val':9,'test':9} or read(mem/'split.lock.json')!=split:raise ValueError('LOCAL93 split identity mismatch')
 if sha(root/'joint/manifest.json')!=fc['manifest_sha256'] or fc['contract']!=m['contract'] or m['prediction']['prediction_mode']!='joint' or m['prediction']['future_horizon_s']!=1 or len(m['contract']['joint_names'])!=58:raise ValueError('Joint/58D/+1s contract mismatch')
 folds=[];held=[]
 for f in sorted((mem/'teachers').glob('fold_*')):
  members=read(f/'oof_membership.json');meta=read(f/'manifest.json');stats=read(f/'train_stats.json');sets=[set(members[k]) for k in ['train','val','held']]
  if [len(x) for x in sets]!=[54,6,15] or any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)) or set.union(*sets)!=set(split['train']):raise ValueError('OOF source leakage')
  if meta['group']!='teacher' or meta['parameters']!=1026312 or sha(f/'best_head.pt')!=meta['checkpoint_sha256'] or json_sha(stats)!=meta['bindings']['stats_sha256']:raise ValueError('OOF model/stat identity')
  held+=members['held'];folds.append(dict(path=str(f),membership=members,weight_sha256=meta['checkpoint_sha256'],stats_sha256=sha(f/'train_stats.json'),probability_sha256={q.name:sha(q) for q in f.glob('*.npz')}))
 if len(folds)!=5 or len(set(held))!=75 or set(held)!=set(split['train']):raise ValueError('OOF coverage')
 write(out/'protected_sha256.json',protected);lock(out/'split.lock.json',split);write(out/'oof_teacher_manifest.json',dict(reused=True,encoder_calls=0,folds=folds,full_teacher_sha256=sha(root/'joint/best_head.pt'),source_manifest_sha256=sha(root/'joint/manifest.json')))
 identity=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),baseline_checkpoint_sha256=sha(root/'joint/best_head.pt'),baseline_original_predictions_sha256=sha(root/'joint/predictions.jsonl'),canonical_predictions='Prior B0-frozen same checkpoint; previous max float discrepancy5.64e-7 vs original split batching; exact historical H0 parity required',sources=93,split_counts={k:len(v) for k,v in split.items()},manifest_sha256=sha(root/'joint/manifest.json'),protected_files=len(protected),dirty_status=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True).strip())
 write(out/'input_identity.json',identity);write(out/'source_identity.json',identity);write(out/'public_interface_change.json',dict(path=str(core),before_sha256=original_sha,after_sha256=sha(core),change='Extract ACK/TTL receive_evidence and original propose_default_candidate; default path executes both unchanged; real Future is never replaced. Full parity required.'))
 from .verification import parity
 parts={s:[] for s in ['train','val','test']}
 for e in load(c):parts[e['metadata']['split']].append(e)
 parity(c,out,parts)


def verify(out):
 protected=read(out/'protected_sha256.json');bad=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h]
 api=read(out/'public_interface_change.json');core=REPO/'tools/tomato_review_ui/handshake.py'
 if sha(core)!=api['after_sha256']:bad.append(str(core))
 result=dict(originals_unchanged=not bad,protected_files=len(protected),differences=bad,explicit_API_refactor=api)
 write(out/'logs/integrity.json',result)
 if bad:raise ValueError('Protected files changed: '+str(bad[:10]))
 return result
