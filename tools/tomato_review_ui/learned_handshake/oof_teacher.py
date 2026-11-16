"""Teacher-distribution diagnosis only; labels never enter the execution shell."""
import numpy as np
from tools import tomato_dataset as ds,tomato_metrics as metrics
from .contracts import LABELS,write


def quality(episodes,out):
 result={};disagreements={}
 for split in ['train','val','test']:
  if split=='test':continue  # historical diagnostic not opened before selection
  parts=[e for e in episodes if e['metadata']['split']==split]
  for source in (['oof','rows'] if split=='train' else ['rows']):
   head={}
   for h in ['current','future']:
    probabilities=[];truth=[];near=[]
    for e in parts:
     m=e['metadata'];anchors=np.array(m['anchors']);targets=ds.build_phase_targets(anchors,m['annotation']['segments'],m['length'],m['fps'],1.);mask=np.ones(len(anchors),bool) if h=='current' else targets['future_valid_mask'];p=np.array([[r['heads'][h]['probabilities'][s] for s in LABELS] for r in e[source]])
     frame=anchors if h=='current' else targets['future_target_frames'];bs=np.array([s['start_frame'] for s in m['annotation']['segments'][1:]]);close=np.min(abs(frame[:,None]-bs),axis=1)<=m['fps'];probabilities.append(p[mask]);truth.append(targets[h+'_labels'][mask]);near.append(close[mask])
    p=np.concatenate(probabilities);y=np.concatenate(truth);keep=np.concatenate(near);sc=metrics.scores(np.log(p.clip(1e-30)),y);confidence=p.max(1);correct=p.argmax(1)==y;ece=0.
    for lo in np.arange(0,1,.1):
     k=(confidence>=lo)&(confidence<lo+.1 if lo<.9-1e-9 else confidence<=1)
     if k.any():ece+=k.mean()*abs(confidence[k].mean()-correct[k].mean())
    head[h]=dict(metrics=sc,brier=float(((p-np.eye(4)[y])**2).sum(1).mean()),ece=float(ece),confidence_histogram=np.histogram(confidence,np.linspace(0,1,11))[0].tolist(),near_boundary_metrics=metrics.scores(np.log(p[keep].clip(1e-30)),y[keep]))
   result[split+'_'+source]=head
 for h in ['current','future']:
  differences=[]
  for e in episodes:
   if e['metadata']['split']!='train':continue
   pp=lambda rows:np.array([[r['heads'][h]['probabilities'][s] for s in LABELS] for r in rows])
   a,b=pp(e['oof']),pp(e['rows']);differences.extend((a.argmax(1)!=b.argmax(1)).tolist())
  disagreements[h]=float(np.mean(differences))
 write(out/'teacher_quality.json',dict(groups=result,oof_full_train_disagreement=disagreements,warning='OOF teachers train on54/inner-val6 sources; full teacher75. Quality/representation shift can limit policy transfer; negative learned results do not prove all learned policies ineffective. No filtering based on held quality.'))
