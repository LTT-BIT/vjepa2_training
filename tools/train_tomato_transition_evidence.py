"""Group-OOF E0/E1/E2 evidence probes; no B0/encoder retraining, no Prompt changes."""
import argparse
from pathlib import Path
from collections import Counter
import json
import time
import subprocess
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_frozen_protocol import verify
from .tomato_runtime import initialize_runtime
from .tomato_transition_evidence import ARMS,change_features,training_examples,EvidenceReadout
from .tomato_state_recovery import B0State

CONFIG=Path('configs/tomato_transition_evidence_20261112/protocol.json')
LOCK=Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json')
BASE=Path('runs/tomato/disagreement_20261112_v2')


def binary_scores(y,p,w=None):
    y=np.asarray(y,int);p=np.asarray(p,float);w=np.ones(len(y)) if w is None else np.asarray(w,float)
    if not len(y):return dict(samples=0,auroc=None,ap=None,nll=None,brier=None,thresholds={})
    order=np.argsort(-p,kind='stable');yp,pp,ww=y[order],p[order],w[order];ends=np.r_[np.flatnonzero(np.diff(pp)!=0),len(pp)-1]
    tp=np.cumsum(ww*yp)[ends];fp=np.cumsum(ww*(1-yp))[ends];pos=float(np.sum(w*y));neg=float(np.sum(w*(1-y)))
    precision=tp/np.maximum(tp+fp,1e-15);recall=tp/max(pos,1e-15)
    auc=float(np.sum(np.diff(np.r_[0,fp/max(neg,1e-15)])*(np.r_[0,recall][:-1]+recall)/2)) if pos and neg else None
    ap=float(np.sum(np.diff(np.r_[0,recall])*precision)) if pos else None
    return dict(samples=len(y),positives=int(y.sum()),negatives=int((1-y).sum()),auroc=auc,ap=ap,
        nll=float(np.average(-(y*np.log(np.clip(p,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-p,1e-7,1))),weights=w)),
        brier=float(np.average((p-y)**2,weights=w)),
        thresholds={str(th):dict(precision=float(np.sum(w*y*(p>=th))/max(np.sum(w*(p>=th)),1e-15)),recall=float(np.sum(w*y*(p>=th))/max(pos,1e-15)),false_positive_rate=float(np.sum(w*(1-y)*(p>=th))/max(neg,1e-15)),selected=int((p>=th).sum())) for th in (.5,.8)})


def fit(rows,arm,cfg,seed,out):
    initialize_runtime(dict(device='cuda',seed=seed));parts=[];ph=[];ys=[];ws=[]
    for r in rows:
        ex=r['examples'];parts.append(r['features'][arm][ex['index']]);ph.append(ex['phase']);ys.append(ex['label']);ws.append(ex['weight'])
    raw=np.concatenate(parts);phase=np.concatenate(ph);y=np.concatenate(ys);w=np.concatenate(ws);w=w/w.sum()
    mean=raw.mean(0,dtype=np.float64).astype('float32');std=raw.std(0,dtype=np.float64).clip(.01).astype('float32')
    x=torch.from_numpy((raw-mean)/std).cuda();pt=torch.from_numpy(phase).cuda();yt=torch.from_numpy(y).cuda();wt=torch.from_numpy(w).cuda()
    net=EvidenceReadout(raw.shape[1]).cuda();opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);history=[];torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(cfg['epochs']):
        opt.zero_grad(set_to_none=True);loss=(torch.nn.functional.binary_cross_entropy_with_logits(net(x,pt),yt,reduction='none')*wt).sum();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();history.append(float(loss))
    torch.cuda.synchronize();elapsed=time.perf_counter()-start;net.eval()
    state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    torch.save(dict(format_version=1,architecture='six_conditional_linear_transition_evidence',arm=arm,state_dict=state,mean=torch.from_numpy(mean),std=torch.from_numpy(std),seed=seed,epochs=cfg['epochs'],training_sources=[r['episode']['source_recording_id'] for r in rows],protocol=cfg,input_dimension=raw.shape[1],class_semantics='next ordered stage has begun, not physical success',source_sha256={p:ds.digest(p) for p in ['tools/tomato_transition_evidence.py',str(Path(__file__))]}),out)
    return net,mean,std,dict(training_s=elapsed,parameters=sum(p.numel() for p in net.parameters()),loss=history,checkpoint_sha256=ds.digest(out),samples=len(y))


@torch.inference_mode()
def predict(net,mean,std,rows,arm):
    result={}
    for r in rows:
        x=torch.from_numpy((r['features'][arm]-mean)/std).cuda();result[r['episode']['episode_id']]=net.all_phases(x).cpu().numpy()
    return result


def evidence_score(rows,pred):
    labels=[];probs=[];weights=[];near_y=[];near_p=[];near_w=[];commits=[];online_y=[];online_p=[];out_of_domain=0;by_type={};per={}
    for r in rows:
        e=r['episode'];eid=e['episode_id'];ex=r['examples'];prob=pred[eid];p=prob[ex['index'],ex['phase']];y=ex['label'];w=ex['weight'];labels.extend(y);probs.extend(p);weights.extend(w)
        starts=np.array([s['start_frame']/e['fps'] for s in e['annotation']['segments']]);offset=r['times'][ex['index']]-starts[ex['phase']+1];mask=np.abs(offset)<=.5
        near_y.extend(y[mask]);near_p.extend(p[mask]);near_w.extend(w[mask]);per[eid]=binary_scores(y,p,w)
        for k in range(6):
            name=e['annotation']['segments'][k]['label']+'->'+e['annotation']['segments'][k+1]['label'];q=ex['phase']==k
            by_type.setdefault(name,[]).append((y[q],p[q],w[q]))
        # Runtime phase is B0's maintained predecessor, never ground-truth phase.
        policy=B0State();gt=np.searchsorted(starts,r['times'],side='right')-1
        for i,t in enumerate(r['times']):
            previous=policy.phase;v=policy.update(r['current'][i],r['future'][i],float(t))
            if previous==6:continue
            if gt[i] not in (previous,previous+1):out_of_domain+=1;continue
            target_y=int(gt[i]==previous+1);score=float(prob[i,previous]);online_y.append(target_y);online_p.append(score)
            if v['changed']:
                commits.append(dict(episode_id=eid,source_recording_id=e['source_recording_id'],ordinal=previous+1,time=float(t),boundary_s=float(starts[previous+1]),offset_s=float(t-starts[previous+1]),already_begun=bool(target_y),evidence_probability=score,base_current_probability=float(r['current'][i].max()),transition=e['annotation']['segments'][previous]['label']+'->'+e['annotation']['segments'][previous+1]['label']))
    return dict(boundary_probe=binary_scores(labels,probs,weights),near_boundary_05=binary_scores(near_y,near_p,near_w),
        online_b0_phase_shadow=binary_scores(online_y,online_p),out_of_domain_online_windows=out_of_domain,
        by_transition={k:binary_scores(np.concatenate([a for a,b,c in parts]),np.concatenate([b for a,b,c in parts]),np.concatenate([c for a,b,c in parts])) for k,parts in by_type.items()},
        at_b0_commit=binary_scores([r['already_begun'] for r in commits],[r['evidence_probability'] for r in commits]),
        commits=commits,per_episode=per,scope='boundary_probe selects candidate by truth only for diagnostics; online shadow always uses B0 predecessor; no Prompt modified')


def run(out):
    if out.exists():raise FileExistsError('Preserve previous runs')
    cfg=ds.read_json(CONFIG);lock=verify(LOCK);m=ds.read_json(lock['manifest']);by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']}
    old=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');seen=set(old['splits']['train']);train=[e for e in m['episodes'] if e['split']=='train'];dev=[e for e in m['episodes'] if e['split']=='val']
    ids=sorted(e['source_recording_id'] for e in train);order=np.random.default_rng(cfg['fold_seed']).permutation(len(ids));foldids=[[ids[i] for i in ix] for ix in np.array_split(order,cfg['folds'])]
    assert len(ids)==len(set(ids))==84 and len(dev)==8
    out.mkdir(parents=True);ds.write_json(out/'protocol.json',cfg);ds.write_json(out/'folds.json',foldids)
    (out/'executed_runner.py').write_bytes(Path(__file__).read_bytes());(out/'executed_features.py').write_bytes(Path('tools/tomato_transition_evidence.py').read_bytes())
    inputs={str(CONFIG):ds.digest(CONFIG),str(LOCK):ds.digest(LOCK),lock['manifest']:ds.digest(lock['manifest'])};raw={};start=time.perf_counter()
    for e in train+dev:
        key=e['episode_id'];path=Path(by[key]['feature_path']);assert ds.digest(path)==by[key]['feature_sha256'];z=np.load(path);assert np.array_equal(z['anchors'],e['anchors']);inputs[str(path)]=ds.digest(path);raw[key]=np.asarray(z['features'],np.float32)
    results={}
    for seed in cfg['training_seeds']:
        initialize_runtime(dict(device='cuda',seed=seed));torch.cuda.reset_peak_memory_stats();folder=out/f'seed{seed}';folder.mkdir();rows=[]
        for e in train+dev:
            eid=e['episode_id'];p=BASE/f'seed{seed}'/f'{eid}_input.npz';z=np.load(p);inputs[str(p)]=ds.digest(p);assert np.array_equal(z['anchors'],e['anchors']) and np.allclose(z['times'],z['anchors']/e['fps']);features=change_features(raw[eid],z['current'],z['future'],z['times'],m['contract']['joint_names'])
            rows.append(dict(episode=e,times=z['times'],current=z['current'],future=z['future'],features=features,examples=training_examples(z['times'],e['annotation']['segments'],e['fps'])))
        pool=[r for r in rows if r['episode']['split']=='train'];development=[r for r in rows if r['episode']['split']=='val'];results[str(seed)]={}
        for arm in ARMS:
            oof={};foldlog=[]
            for k,outerids in enumerate(foldids):
                fitrows=[r for r in pool if r['episode']['source_recording_id'] not in outerids];outer=[r for r in pool if r['episode']['source_recording_id'] in outerids]
                assert not {r['episode']['source_recording_id'] for r in fitrows}&set(outerids)
                net,mean,std,info=fit(fitrows,arm,cfg,seed,folder/f'{arm}_fold{k}.pt');oof.update(predict(net,mean,std,outer,arm));foldlog.append(info);del net;torch.cuda.empty_cache()
            assert len(oof)==84
            evaluations={co:evidence_score([r for r in pool if condition(r)],oof) for co,condition in [('oof84',lambda r:True),('oof74_base_seen',lambda r:r['episode']['source_recording_id'] in seen),('oof10_base_unseen_exposed',lambda r:r['episode']['source_recording_id'] not in seen)]}
            # Dev scores are obtained only after fixed-budget training and OOF; not used to select checkpoints.
            net,mean,std,info=fit(pool,arm,cfg,seed,folder/f'{arm}_final84.pt');dp=predict(net,mean,std,development,arm);evaluations['development8_exposed']=evidence_score(development,dp)
            for r in rows:
                eid=r['episode']['episode_id'];np.savez_compressed(folder/f'{arm}_{eid}.npz',times=r['times'],six_phase_evidence=oof[eid] if eid in oof else dp[eid])
            # Resident single-observation classifier-only timing; no encoder/capture/transport.
            x=torch.from_numpy(((pool[0]['features'][arm][:1]-mean)/std).astype('float32')).cuda();lat=[]
            for i in range(60):
                torch.cuda.synchronize();t0=time.perf_counter_ns();net.all_phases(x);torch.cuda.synchronize()
                if i>=10:lat.append((time.perf_counter_ns()-t0)/1e6)
            results[str(seed)][arm]=dict(evaluation=evaluations,fold_training=foldlog,final_training=info,latency_ms=dict(p50=float(np.median(lat)),p95=float(np.percentile(lat,95)),maximum=float(max(lat))))
            ds.write_json(folder/f'{arm}_result.json',results[str(seed)][arm]);print('RESULT',seed,arm,{co:round(v['near_boundary_05']['auroc'],4) if v['near_boundary_05']['auroc'] is not None else None for co,v in evaluations.items()},flush=True)
            del net;torch.cuda.empty_cache()
        ds.write_json(folder/'resource.json',dict(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        del rows
    ds.write_json(out/'results.json',results);ds.write_json(out/'receipt.json',dict(input_sha256=inputs,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={str(p):ds.digest(p) for p in [Path(__file__),Path('tools/tomato_transition_evidence.py')]},wall_s=time.perf_counter()-start,encoder_calls=0,B0_retrained=False,prompt_changed=False,historical_test_scored=False,normalization='each fold training samples only',real_latency_measured=False,run_role='Exploratory group-OOF evidence probe, no deployment promotion'))
    verify(LOCK);print('DONE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
