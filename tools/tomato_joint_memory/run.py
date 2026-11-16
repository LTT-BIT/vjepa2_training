"""python -m tools.tomato_joint_memory.run --config ... --output-dir ... --run"""
from __future__ import annotations
import argparse,csv,json,subprocess,time,traceback,sys
from pathlib import Path
import numpy as np
import torch
from tools import tomato_phase_model as baseline
from tools.tomato_metrics import probabilities
from tools.tomato_runtime import initialize_runtime
from .contracts import config,read,write,sha,digest_json,locked,validate_baseline,verify,load_entries,REPO,LABELS
from .train import fit,predict_entries
from .models import load_memory_checkpoint
from .cache_reader import ema
from .prediction_memory import folds,crossfit
from .evaluate import score_split
from .handshake_compare import compare,C0,C1
from .benchmark import benchmark


def source_hashes():
    paths=list((REPO/'tools/tomato_joint_memory').glob('*.py'))+[REPO/'configs/tomato_joint_memory_ablation_20261112.yaml',REPO/'tools/tomato_review_ui/handshake.py']
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(paths)}

def protocol(c,out,entries):
    train=[e for e in entries if e['episode']['split']=='train']
    p=dict(format='joint-memory-ablation-v1',head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),config=c,source_sha256=source_hashes(),folds=folds(train,c),handshake=dict(C0=C0,C1=C1),architecture='four-head latest query attention, h128, relative elapsed time, latest residual; identical T0/T1/T2 parameterization',smoothing='EMA alpha .6 per episode, no GT; B0-frozen postprocessing only',warning_definition='burst onset of predicted current/future disagreement; raw k1 threshold0, stable k2 Future>=.8; timestamp of kth observation, strict from->to one-to-one [-1,0), no boundary-time/post-boundary credit',class_weights='sqrt(train_count/class_count), mean normalized separately by head',normalization='train-only float64 moments -> FP32, std clip .01; K1 own train motion stats',selection='(F1current+.5F1future)/1.5 then (NLLcurrent+.5NLLfuture)/1.5, shared checkpoint',test='previously exposed historical holdout; evaluated once after all main weights and val results locked',oof='5 folds of train sources; inner source validation only, held sources excluded from teacher training and model selection; fixed existing full-train B0 for val/test beliefs',optional=dict(E1='NOT_RUN: not in initial batch, may consider only after stable P1 benefit',N16='NOT_RUN: not initial batch',physical_success='NOT_EVALUABLE: no independent labels'),latency='fixed simulated 90ms all backends; measured cached readout separate from NOT_MEASURED online Encoder+readout',endpoints=['Current MacroF1','Future MacroF1','strict raw Future event Recall/FPmin','C0 narrow matches / early>=.5'],promotion=dict(**c['promotion'],fp_definition='raw strict warning FP/min increase <=0.1',seed_comparison='paired against same seed B0R; history attribution additionally versus T0/T1'))
    locked(out/'experiment_protocol.lock.json',p);locked(out/'protocol.lock.json',p)
    return dict(protocol_sha256=sha(out/'protocol.lock.json'),feature_sha256=sha(out/'feature_contract.lock.json'),split_sha256=sha(out/'split.lock.json'))


def save_predictions(folder,entries,pred):
    with (folder/'predictions.jsonl').open('w') as f:
        for e in entries:
            eid=e['episode']['episode_id'];pc=probabilities(pred[eid]['current']);pf=probabilities(pred[eid]['future'])
            for i,t in enumerate(e['times']):
                row=dict(episode_id=eid,source_recording_id=e['episode']['source_recording_id'],split=e['episode']['split'],observed_at=float(t),frame_index=int(e['anchors'][i]),current_probabilities=pc[i].tolist(),future_probabilities=pf[i].tolist(),current_true=int(e['current_labels'][i]),future_true=int(e['future_labels'][i]) if e['future_valid_mask'][i] else None,future_valid=bool(e['future_valid_mask'][i]))
                f.write(json.dumps(row,separators=(',',':'))+'\n')


def evaluate_model(name,folder,entries,pred,c,n,split):
    folder.mkdir(parents=True,exist_ok=False);save_predictions(folder,entries,pred)
    ss=score_split(entries,pred,c,n);write(folder/'classification.json',ss)
    write(folder/'event_metrics.json',ss['future_events']);hh=compare(entries,pred,folder/'handshake');write(folder/'handshake_metrics.json',hh)
    with (folder/'episode_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['episode_id','source_recording_id','current_f1','future_f1','raw_recall','raw_fp_per_min','strict_hits','early_ge05','wrong_active_s']);w.writeheader()
        for ep,ev,h in zip(ss['episodes'],ss['future_events']['raw']['episodes'],hh['C0']['episodes']):w.writerow(dict(episode_id=ep['episode_id'],source_recording_id=ep['source_recording_id'],current_f1=ep['heads']['current']['overall']['macro_f1'],future_f1=ep['heads']['future']['overall']['macro_f1'],raw_recall=ev['recall'],raw_fp_per_min=ev['fp_per_min'],strict_hits=h['strict_hits'],early_ge05=h['early_ge05'],wrong_active_s=h['wrong_active_s']))
    print('evaluated',name,split,ss['heads']['current']['macro_f1'],ss['heads']['future']['macro_f1'],flush=True)
    return dict(classification=ss,handshake=hh)


def run(c,out,smoke=False):
    entries,m=load_entries(c);by={s:[e for e in entries if e['episode']['split']==s] for s in ['train','val','test']};bindings=protocol(c,out,entries)
    if smoke:
        for group in ['B0R','T0','T1','T2','K1']:
            fit(by['train'],by['val'],c,group,42,out/'smoke'/group,bindings,smoke=True)
        write(out/'smoke_receipt.json',dict(seed=42,epochs=2,groups=['B0R','T0','T1','T2','K1'],test_scored=False,purpose='correctness only, no architecture/parameter selection'))
        return
    if not (out/'smoke_receipt.json').exists():raise ValueError('Run --smoke first')
    initialize_runtime(dict(device=c['device'],seed=42));torch.set_num_threads(4)
    saved,base=baseline.load_checkpoint(Path(c['baseline_root'])/'joint/best_head.pt',device=c['device']);base_stats={k:saved[k].cpu().numpy() for k in ['mean','std']}
    base_saved=dict(stats=base_stats);bp={}
    for e in by['train']+by['val']:
        zz=baseline.predict_heads(base,e['features'],saved['mean'],saved['std']);bp[e['episode']['episode_id']]={h:zz[h+'_logits'] for h in ['current','future']}
    # Recompute only train/val now; independently compare with saved original probabilities.
    original={}
    with (Path(c['baseline_root'])/'joint/predictions.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            if row['split']!='test':original.setdefault(row['episode_id'],[]).append(row)
    max_error=0.
    for eid,z in bp.items():
        for h in ['current','future']:
            old=np.asarray([[r[h+'_probabilities'][label] for label in LABELS] for r in original[eid]])
            err=np.max(abs(probabilities(z[h])-old));max_error=max(max_error,float(err))
            if err>1e-6:raise ValueError(f'B0 saved predictions differ {eid} {h}: {err}')
    write(out/'B0_reproduction.json',dict(train_val_sources=len(bp),maximum_probability_error=max_error,tolerance=1e-6,test_recomputed=False))
    models={};summary={};teacher_beliefs={eid:np.concatenate([probabilities(z[h]) for h in ['current','future']],1).astype(np.float32) for eid,z in bp.items()}
    def register(name,group,seed,model,sav,meta,pred):
        summary[name]=dict(group=group,seed=seed,parameters=meta['parameters'],training=meta,latency=benchmark(model,sav,by['val'][0],c,group,teacher_beliefs if group=='P1' else None),splits={})
        for split in ['train','val']:
            summary[name]['splits'][split]=evaluate_model(name,out/'evaluations'/name/split,by[split],pred,c,c['memory_lengths'].get(group,1),split)
        models[name]=dict(group=group,seed=seed,folder=str(out/'models'/name),saved=sav)
        write(out/'summary.partial.json',summary)
    frozen_meta=dict(parameters=sum(p.numel() for p in base.parameters()),checkpoint_sha256=sha(Path(c['baseline_root'])/'joint/best_head.pt'),best_epoch=saved['best_epoch'],epochs=None,updates=None,training_s=0.,encoder_new_calls=0)
    register('B0-frozen','B0',42,base,base_saved,frozen_meta,bp)
    smooth={eid:{h:np.log(ema(probabilities(z[h]),next(e['times'] for e in entries if e['episode']['episode_id']==eid),c['ema_alpha']).clip(1e-30)) for h in z} for eid,z in bp.items()}
    register('S1','B0',42,base,base_saved,{**frozen_meta,'postprocess':'EMA alpha .6'},smooth)
    for group in c['groups']:
        if group=='P1':
            oof=crossfit(by['train'],c,out/'teachers',bindings);teacher_beliefs.update(oof)
            write(out/'teacher_bindings.json',dict(full_train_teacher_sha256=frozen_meta['checkpoint_sha256'],folds=read(out/'teachers/folds.lock.json'),normalization='each teacher inner train only',val_test_teacher='fixed original full-train B0',train_oof_episode_ids=sorted(oof),val_test_no_gt_history=True))
        for seed in c['seeds']:
            name=f'{group}_s{seed}';folder=out/'models'/name
            model,sav,meta,pred=fit(by['train'],by['val'],c,group,seed,folder,bindings,teacher_beliefs if group=='P1' else None)
            register(name,group,seed,model,sav,meta,pred);del model
    # The only test entry point; freeze all weights, parameters, val outcomes first.
    receipt=dict(protocol_sha256=bindings['protocol_sha256'],models={name:row['training']['checkpoint_sha256'] for name,row in summary.items()},val_results_sha256=digest_json({name:r['splits']['val'] for name,r in summary.items()}),test_source_ids=read(out/'split.lock.json')['test'],historically_exposed=True,no_test_tuning=True)
    locked(out/'test_lock_receipt.json',receipt)
    # Original teacher test beliefs generated only after receipt; no test labels involved.
    test_base={}
    for e in by['test']:
        zz=baseline.predict_heads(base,e['features'],saved['mean'],saved['std']);test_base[e['episode']['episode_id']]={h:zz[h+'_logits'] for h in ['current','future']}
    teacher_beliefs.update({eid:np.concatenate([probabilities(z[h]) for h in ['current','future']],1).astype(np.float32) for eid,z in test_base.items()})
    for name,rec in models.items():
        group=rec['group']
        if name=='B0-frozen':pred=test_base
        elif name=='S1':pred={e['episode']['episode_id']:{h:np.log(ema(probabilities(test_base[e['episode']['episode_id']][h]),e['times'],c['ema_alpha']).clip(1e-30)) for h in ['current','future']} for e in by['test']}
        else:
            sav,model=load_memory_checkpoint(Path(rec['folder'])/'best_head.pt',c,rec['saved']['bindings'],summary[name]['training']['checkpoint_sha256']);model.to(c['device']);pred=predict_entries(model,sav,by['test'],c,group,teacher_beliefs if group=='P1' else None);del model
        summary[name]['splits']['test']=evaluate_model(name,out/'evaluations'/name/'test',by['test'],pred,c,c['memory_lengths'].get(group,1),'test')
    write(out/'summary.json',summary);write(out/'run_complete.json',dict(models=len(summary),test_passes_per_model=1,encoder_new_calls=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/tomato_joint_memory_ablation_20261112.yaml');p.add_argument('--output-dir',required=True,type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    for flag in ['check-only','smoke','run','report-only','verify']:mode.add_argument('--'+flag,action='store_true')
    a=p.parse_args();c=config(a.config);out=a.output_dir.resolve()
    if out==Path(c['baseline_root']).resolve() or out.is_relative_to(Path(c['baseline_root']).resolve()):raise ValueError('Must not write inside protected baseline')
    out.mkdir(parents=True,exist_ok=True)
    try:
        if a.check_only:
            if (out/'baseline_inventory.json').exists():raise ValueError('Inventory exists; use --verify')
            validate_baseline(c,out)
        elif a.verify:print(verify(out))
        elif a.report_only:
            from .report import generate
            generate(c,out)
        else:
            if not (out/'baseline_inventory.json').exists():raise ValueError('Run --check-only first')
            run(c,out,a.smoke)
    except Exception:
        text=traceback.format_exc();write(out/f'failure_{time.time_ns()}.json',dict(command=sys.argv,traceback=text));raise

if __name__=='__main__':main()
