"""Strict reuse of existing source inventory and train-only OOF teachers."""
from pathlib import Path
import json,numpy as np
from .contracts import *


def verify_sources(protected):
 bad=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h]
 if bad:raise ValueError('Protected identity changed: '+str(bad[:10]))
 return len(protected)


def check(c,out):
 root=Path(c['baseline_root']);mem=Path(c['memory_experiment']);m=read(root/'joint/manifest.json');p=read(mem/'protocol.lock.json');fc=read(mem/'feature_contract.lock.json');split=read(root/'joint/split.json')
 assert len(m['episodes'])==93 and {s:len(v) for s,v in split.items()}=={'train':75,'val':9,'test':9}
 if sha(root/'joint/manifest.json')!=fc['manifest_sha256'] or fc['contract']!=m['contract'] or read(mem/'split.lock.json')!=split:raise ValueError('OOF feature/source/split contract mismatch')
 if m['prediction']['prediction_mode']!='joint' or m['prediction']['future_horizon_s']!=1 or len(m['contract']['joint_names'])!=58:raise ValueError('Not original Joint 58D/+1s')
 if p['config']['learning_rate']!=3e-4 or p['config']['weight_decay']!=.01:raise ValueError('OOF optimizer mismatch')
 protected=read(mem/'protected_sha256.json');count=verify_sources(protected)
 # Seal the complete preceding experiment, including every OOF weight/statistic/probability.
 for rel,h in read(mem/'artifacts_sha256.json').items():
  path=mem/rel
  if sha(path)!=h:raise ValueError('Prior evidence changed: '+str(path))
  protected[str(path)]=h
 for path in [mem/'handoff.zip',REPO/'docs/TOMATO_JOINT_MEMORY_ABLATION_REPORT_20261112.md']:
  protected[str(path)]=sha(path)
 train_sources=set(split['train']);held_all=[];folds=[]
 for f in sorted((mem/'teachers').glob('fold_*')):
  members=read(f/'oof_membership.json');meta=read(f/'manifest.json');stats=read(f/'train_stats.json')
  if len(members['train'])!=54 or len(members['val'])!=6 or len(members['held'])!=15 or set(members['train'])&set(members['held']) or set(members['val'])&set(members['held']) or set(members['train']+members['val']+members['held'])!=train_sources:raise ValueError('OOF membership leakage')
  if meta['group']!='teacher' or meta['parameters']!=1026312 or sha(f/'best_head.pt')!=meta['checkpoint_sha256'] or json_sha(stats)!=meta['bindings']['stats_sha256']:raise ValueError('OOF Joint/statistics identity')
  held_all+=members['held'];folds.append(dict(path=str(f),membership=members,weight_sha256=meta['checkpoint_sha256'],stats_sha256=sha(f/'train_stats.json'),probability_sha256={q.name:sha(q) for q in f.glob('*.npz')}))
 if len(folds)!=5 or len(set(held_all))!=75 or set(held_all)!=train_sources:raise ValueError('OOF coverage')
 write(out/'protected_sha256.json',protected);lock(out/'split.lock.json',split)
 write(out/'oof_teacher_manifest.json',dict(reused=True,encoder_calls=0,folds=folds,full_teacher_sha256=sha(root/'joint/best_head.pt'),source_manifest_sha256=sha(root/'joint/manifest.json')))
 write(out/'input_identity.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),baseline_checkpoint_sha256=sha(root/'joint/best_head.pt'),baseline_original_predictions_sha256=sha(root/'joint/predictions.jsonl'),canonical_predictions='Prior B0-frozen probabilities, same checkpoint; earlier reproduction maximum float error5.64e-7 vs original split batching. Enables exact historical H0 event parity.',sources=93,split_counts={k:len(v) for k,v in split.items()},previous_protected_count=count,full_protected_count=len(protected),prior_experiment=str(mem),manifest_sha256=sha(root/'joint/manifest.json')))


def load(c):
 root=Path(c['baseline_root']);mem=Path(c['memory_experiment']);m=read(root/'joint/manifest.json');by={e['episode_id']:dict(metadata=e,rows=[],oof=[]) for e in m['episodes']}
 for split in ['train','val','test']:
  for line in (mem/'evaluations/B0-frozen'/split/'predictions.jsonl').open():
   r=json.loads(line);eid=r['episode_id'];e=by[eid]['metadata']
   if r['source_recording_id']!=e['source_recording_id'] or r['split']!=e['split']:raise ValueError('B0 probability source mismatch')
   by[eid]['rows'].append(dict(observed_at=r['observed_at'],frame_index=r['frame_index'],heads={h:dict(probabilities=dict(zip(LABELS,r[h+'_probabilities']))) for h in ['current','future']}))
 for e in by.values():
  rows=e['rows'];meta=e['metadata']
  if [r['frame_index'] for r in rows]!=meta['anchors']:raise ValueError('B0 anchor alignment mismatch')
  if meta['split']=='train':
   matches=list((mem/'teachers').glob('fold_*/'+meta['episode_id']+'.npz'))
   if len(matches)!=1:raise ValueError('OOF missing/duplicated')
   if meta['source_recording_id'] not in read(matches[0].parent/'oof_membership.json')['held']:raise ValueError('OOF held identity')
   with np.load(matches[0]) as f:p=f['probabilities'].copy()
   if p.shape!=(len(rows),8) or not np.isfinite(p).all():raise ValueError('OOF shape')
   e['oof']=[dict(observed_at=r['observed_at'],frame_index=r['frame_index'],heads={h:dict(probabilities=dict(zip(LABELS,p[i,j:j+4].astype(float)))) for h,j in [('current',0),('future',4)]}) for i,r in enumerate(rows)]
 return list(by.values())
