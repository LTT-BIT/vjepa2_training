"""Matched old100 local/history readouts; frozen encoder, fixed final epoch."""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from . import tomato_dataset as ds,tomato_metrics as metrics
from .tomato_frozen_protocol import verify
from .tomato_runtime import initialize_runtime
from .tomato_rgb_memory_model import RGBMemoryReader
from .tomato_rgb_memory_training import history_indices,gather_inputs
from .train_tomato_vjepa_phase import fit_class_weights
from .evaluate_tomato_disagreement import score,summarize

CONFIG=Path('configs/tomato_rgb_memory_train_20261116/protocol.json')
BASE=Path('runs/tomato/disagreement_20261112_v2')


def load_rows(cache,cfg,manifest):
    if ds.read_json(cache/'protocol.json')!=cfg:raise ValueError('Cache protocol mismatch: do not mix right-anchor and recorded r11 ROI')
    index=ds.read_json(cache/'index.json');assert index['manifest_sha256']==cfg['manifest_sha256'];stats=np.load(index['statistics_path']);assert ds.digest(index['statistics_path'])==index['statistics_sha256'];by={e['episode_id']:e for e in manifest['episodes']};rows=[]
    for entry in index['episodes']:
        e=by[entry['episode_id']];assert e['split'] in ('train','val') and e['split']==entry['split'];assert ds.digest(entry['path'])==entry['sha256'];z=np.load(entry['path']);assert str(z['cache_signature'])==index['signature'];assert np.array_equal(z['anchors'],z['input_indices'][:,-1]) and np.all(z['input_indices']<=z['anchors'][:,None]);target=ds.build_phase_targets(z['anchors'],e['annotation']['segments'],e['length'],e['fps'],cfg['future_horizon_s']);rows.append(dict(episode=e,visual=z['visual'],positions=z['positions'].reshape(-1,928),metadata=z['metadata'],times=z['times'],anchors=z['anchors'],target=target))
    assert sum(r['episode']['split']=='train' for r in rows)==84 and sum(r['episode']['split']=='val' for r in rows)==8
    return rows,index,stats['mean'],stats['std']


def bank(rows,history):
    v=[];q=[];m=[];t=[];ix=[];y=[];f=[];offset=0
    for r in rows:
        local=history_indices(r['times'],history);ix.append(np.where(local<0,-1,local+offset));offset+=len(local);v.append(r['visual']);q.append(r['positions']);m.append(r['metadata']);t.append(r['times']);y.append(r['target']['current_labels']);f.append(r['target']['future_labels'])
    def tensor_with_sentinel(arrays):
        x=np.concatenate(arrays);return torch.from_numpy(np.concatenate([np.zeros((1,*x.shape[1:]),dtype=x.dtype),x])).cuda()
    return dict(visual=tensor_with_sentinel(v),positions=tensor_with_sentinel(q),metadata=tensor_with_sentinel(m),times=tensor_with_sentinel(t),indices=torch.from_numpy(np.concatenate(ix)).cuda(),current=torch.from_numpy(np.concatenate(y)).cuda(),future=torch.from_numpy(np.concatenate(f)).cuda())


def features(b,ids,method):return gather_inputs(b['visual'],b['positions'],b['metadata'],b['times'],b['indices'][ids],method)


def fit(rows,method,seed,cfg,mean,std,path,identity):
    initialize_runtime(dict(device='cuda',seed=seed));b=bank(rows,8 if method=='G2_local_history8' else 1);cw,_=fit_class_weights(b['current'].cpu().numpy());future_np=b['future'].cpu().numpy();fw,_=fit_class_weights(future_np[future_np>=0]);cw=torch.from_numpy(cw).cuda();fw=torch.from_numpy(fw).cuda();net=RGBMemoryReader(mean,std,cfg['hidden']).cuda();initial=ds.fingerprint({k:v.detach().cpu().numpy().tolist() for k,v in net.state_dict().items()});opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);rng=np.random.default_rng(seed);losses=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    for epoch in range(cfg['epochs']):
        net.train();epoch_loss=[]
        for batch in np.array_split(rng.permutation(len(b['current'])),int(np.ceil(len(b['current'])/cfg['batch_size']))):
            ids=torch.as_tensor(batch,device='cuda');opt.zero_grad(set_to_none=True);out=net(**features(b,ids,method));loss=torch.nn.functional.cross_entropy(out['current_logits'],b['current'][ids],weight=cw);valid=b['future'][ids]>=0
            future_loss=torch.nn.functional.cross_entropy(out['future_logits'][valid],b['future'][ids][valid],weight=fw) if valid.any() else out['future_logits'].sum()*0
            loss=cfg['current_loss_weight']*loss+cfg['future_loss_weight']*future_loss;loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();epoch_loss.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_loss)))
    torch.cuda.synchronize();info=dict(training_s=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in net.parameters()),initialization_sha256=initial,loss=losses)
    torch.save(dict(format_version=1,architecture='old100_rgb_memory_reader_v2',method=method,seed=seed,model_version=f'old100-{method}-s{seed}-{identity["cache_signature"][:12]}',head={k:v.detach().cpu() for k,v in net.state_dict().items()},state_mean=torch.from_numpy(mean),state_std=torch.from_numpy(std),current_class_weights=cw.cpu(),future_class_weights=fw.cpu(),training_sources=[r['episode']['source_recording_id'] for r in rows],protocol=cfg,identity=identity,training=info,input_schema=dict(visual=[4,1024],positions=[16,58],metadata=116,joint_names=identity['joint_names'],labels=ds.LABELS,history=8 if method=='G2_local_history8' else 1,language_prompt=False,stage_oracle=False)),path);info['checkpoint_sha256']=ds.digest(path);del b;return net,info


@torch.inference_mode()
def predict(net,rows,method,cfg):
    net.eval();pred={}
    for r in rows:
        b=bank([r],8 if method=='G2_local_history8' else 1);c=[];f=[]
        for ids in torch.arange(len(r['times']),device='cuda').split(cfg['batch_size']):
            z=net(**features(b,ids,method));c.append(z['current_logits'].softmax(-1).cpu().numpy());f.append(z['future_logits'].softmax(-1).cpu().numpy())
        pred[r['episode']['episode_id']]=dict(current=np.concatenate(c),future=np.concatenate(f));del b
    return pred


def evaluate(rows,pred):
    per={};events=[];ys=[];fs=[];cs=[];fps=[]
    for r in rows:
        e=r['episode'];eid=e['episode_id'];p=pred[eid];result,_=score(e,r['times'],r['anchors'],p['current'],p['future'],'B0');per[eid]=result;events.extend(dict(episode_id=eid,**x) for x in result['prompt']['events']);ys.append(r['target']['current_labels']);valid=r['target']['future_valid_mask'];fs.append(r['target']['future_labels'][valid]);cs.append(p['current']);fps.append(p['future'][valid])
    summary=summarize(per);summary['current']=metrics.scores(np.log(np.maximum(np.concatenate(cs),1e-30)),np.concatenate(ys));summary['future']=metrics.scores(np.log(np.maximum(np.concatenate(fps),1e-30)),np.concatenate(fs));summary['within_delay']={str(t):sum(e['offset_s'] is not None and 0<=e['offset_s']<=t+1e-9 for e in events) for t in [1.,1.5,2.]};summary['severe_early_gt1s']=sum(e['offset_s'] is not None and e['offset_s']< -1 for e in events);summary['late_gt1p5s']=sum(e['offset_s'] is not None and e['offset_s']>1.5 for e in events);summary['warnings']['false_per_minute']=summary['warnings']['false_events']/(sum(p['warnings']['duration_s'] for p in per.values())/60)
    return dict(summary=summary,per_episode=per,events=events)


def run(cache,out):
    if out.exists():raise FileExistsError('Preserve historical weights')
    cfg=ds.read_json(CONFIG);lock=verify(Path(cfg['frozen_lock']));manifest=ds.read_json(lock['manifest']);rows,index,mean,std=load_rows(cache,cfg,manifest);train=[r for r in rows if r['episode']['split']=='train'];dev=[r for r in rows if r['episode']['split']=='val'];assert not {r['episode']['source_recording_id'] for r in train}&{r['episode']['source_recording_id'] for r in dev};out.mkdir(parents=True);ds.write_json(out/'protocol.json',cfg);ds.write_json(out/'cache_binding.json',index);sources=[Path(__file__),Path('tools/tomato_rgb_memory_model.py'),Path('tools/tomato_rgb_memory_training.py')]
    for p in sources:(out/p.name).write_bytes(p.read_bytes())
    identity=dict(manifest_sha256=lock['manifest_sha256'],cache_signature=index['signature'],cache_index_sha256=ds.digest(cache/'index.json'),statistics_sha256=index['statistics_sha256'],joint_names=index['joint_names'],source_sha256={str(p):ds.digest(p) for p in sources});results={};start=time.perf_counter()
    for seed in cfg['seeds']:
        folder=out/f'seed{seed}';folder.mkdir();results[str(seed)]={};baseline={}
        for r in dev:
            e=r['episode'];path=BASE/f'seed{seed}'/f'{e["episode_id"]}_input.npz';z=np.load(path);ix=np.searchsorted(z['anchors'],r['anchors']);assert np.array_equal(z['anchors'][ix],r['anchors']);baseline[e['episode_id']]=dict(current=z['current'][ix],future=z['future'][ix])
        results[str(seed)]['B0_common_startup']=dict(evaluation=evaluate(dev,baseline))
        for method in cfg['methods']:
            net,info=fit(train,method,seed,cfg,mean,std,folder/f'{method}.pt',identity);pred=predict(net,dev,method,cfg);evaluation=evaluate(dev,pred);results[str(seed)][method]=dict(training=info,evaluation=evaluation)
            for r in dev:
                eid=r['episode']['episode_id'];np.savez_compressed(folder/f'{method}_{eid}.npz',**pred[eid],times=r['times'],anchors=r['anchors'])
            ds.write_json(folder/f'{method}_result.json',results[str(seed)][method]);del net;torch.cuda.empty_cache();print('TRAINED',seed,method,'sec',round(info['training_s'],2),'current',round(evaluation['summary']['current']['macro_f1'],5),'future',round(evaluation['summary']['future']['macro_f1'],5),'prompt',evaluation['summary']['prompt_counts'],flush=True)
        assert len({results[str(seed)][m]['training']['initialization_sha256'] for m in cfg['methods']})==1
    ds.write_json(out/'results.json',results);ds.write_json(out/'receipt.json',dict(identity=identity,config_sha256=ds.digest(CONFIG),cache_path=str(cache),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),wall_s=time.perf_counter()-start,trained_models=9,encoder_unfrozen=False,language_prompt_input=False,old100_only=True,historical_test_evaluated=False,default_changed=False));verify(Path(cfg['frozen_lock']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();run(a.cache,a.out)
