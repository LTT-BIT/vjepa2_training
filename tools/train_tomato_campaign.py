"""Locked validation-only comparison of frozen-encoder tomato heads, then held-out evaluation.

No test prediction is computed until all recipes and seeds finish and selection.json
is locked. Resume states belong only to their exact data/split/recipe/source version.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from . import tomato_dataset as ds, tomato_metrics as metrics, tomato_phase_model as model
from . import train_tomato_vjepa_phase as core
from .tomato_runtime import initialize_runtime, hardware_report

RECIPES = (
    dict(name='current', prediction_mode='current_only', transition_weight=1.),
    dict(name='future', prediction_mode='future_only', transition_weight=1.),
    dict(name='joint', prediction_mode='joint', transition_weight=1.),
    dict(name='joint_transition', prediction_mode='joint', transition_weight=2.),
)


def atomic_save(path, value):
    path = Path(path);tmp = path.with_suffix('.tmp.pt')
    torch.save(value, tmp);tmp.replace(path)


def transition_loss(outputs, yc, yf, valid, weights, prediction, multiplier=1.):
    """Train-only label-defined anticipation emphasis. multiplier=1 exactly matches baseline CE."""
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError('Transition multiplier must be finite and >=1')
    active = ds.active_heads(prediction['prediction_mode'])
    scale = torch.where(valid & (yc != yf), float(multiplier), 1.)
    parts = {}
    for name in active:
        keep = torch.ones_like(valid) if name == 'current' else valid
        labels = yc[keep] if name == 'current' else yf[keep]
        if not len(labels):
            raise ValueError('No valid active-head supervision')
        per = nn.functional.cross_entropy(outputs[name+'_logits'][keep], labels, weight=weights[name], reduction='none')
        parts[name] = (per * scale[keep]).sum() / (weights[name][labels] * scale[keep]).sum()
    total = sum(prediction['lambda_'+h] * parts[h] for h in active) / sum(prediction['lambda_'+h] for h in active)
    if not torch.isfinite(total):
        raise ValueError('Nonfinite training loss')
    return total


def concatenate(entries):
    return {s:{k:np.concatenate([r[k] for r in rows]) for k in
               ('features','current_labels','future_labels','future_valid_mask')}
            for s,rows in entries.items()}


def identity(config, parent, recipe, seed):
    return dict(parent_manifest_sha256=ds.digest(Path(config['output_dir'])/'manifest.json'),
                parent_validation_receipt_sha256=ds.digest(Path(config['output_dir'])/'validation_receipt.json'),
                split_sha256=parent['split_sha256'], recipe=recipe, training_seed=seed,
                epochs=config['epochs'], patience=config['patience'], batch_size=config['batch_size'],
                learning_rate=config['learning_rate'], weight_decay=config['weight_decay'],
                prediction=ds.prediction_contract({**config,**recipe}), device=config['device'],
                torch_version=str(torch.__version__), numpy_version=np.__version__, cuda_version=torch.version.cuda,
                runner_sha256=ds.digest(__file__), model_source_sha256=ds.digest(model.__file__),
                training_core_sha256=ds.digest(core.__file__))


def train_one(config, parent, arrays, recipe, seed, directory):
    directory = Path(directory);directory.mkdir(parents=True,exist_ok=True)
    spec = identity(config,parent,recipe,seed)
    spec_path = directory/'recipe.json'
    if spec_path.exists() and ds.read_json(spec_path) != spec:
        raise ValueError('Recipe/data/code changed; use new campaign directory')
    if not spec_path.exists(): ds.write_json(spec_path,spec)
    if (directory/'validation_result.json').exists():
        result=ds.read_json(directory/'validation_result.json')
        if result['checkpoint_sha256'] != ds.digest(directory/'best_head.pt'):
            raise ValueError('Completed checkpoint changed')
        return result
    initialize_runtime({**config,'seed':seed})
    prediction=ds.prediction_contract({**config,**recipe})
    manifest=copy.deepcopy(parent);manifest['prediction']=prediction
    manifest['validation_reused_from']=dict(parent_manifest_sha256=spec['parent_manifest_sha256'],
                                            parent_manifest=str(Path(config['output_dir'])/'manifest.json'),
                                            receipt_sha256=spec['parent_validation_receipt_sha256'],
                                            full_source_validation='Identical input bytes and parent receipt revalidated before campaign')
    manifest['training_recipe']=spec
    manifest_path=directory/'manifest.json'
    if manifest_path.exists() and ds.read_json(manifest_path)!=manifest:
        raise ValueError('Locked child manifest changed')
    ds.write_json(manifest_path,manifest);ds.write_json(directory/'split.json',parent['splits'])
    core._create_validation_receipt(directory,manifest)
    parent_index=ds.read_json(Path(config['output_dir'])/'feature_index.json')
    if (parent_index['manifest_sha256']!=spec['parent_manifest_sha256']
            or parent_index['pretrained']!=parent['pretrained']):
        raise ValueError('Parent feature index identity mismatch')
    ds.write_json(directory/'feature_index.json',{**parent_index,'manifest_sha256':ds.digest(manifest_path)})
    cfg={**config,**prediction,'output_dir':str(directory)}
    ds.write_json(directory/'config.json',cfg)
    x=arrays['train']['features'];mode=prediction['prediction_mode'];active=ds.active_heads(mode)
    mean=x.mean(axis=0,dtype=np.float64).astype(np.float32)
    std=x.std(axis=0,dtype=np.float64).clip(.01).astype(np.float32)
    weight_arrays={}
    for h in active:
        y=arrays['train'][h+'_labels']
        if h=='future': y=y[arrays['train']['future_valid_mask']]
        weight_arrays[h]=core.fit_class_weights(y)[0]
    weights={h:torch.from_numpy(v).to(config['device']) for h,v in weight_arrays.items()}
    head=model.mode_head_model(len(parent['contract']['joint_names']),mode).to(config['device'])
    optimizer=torch.optim.AdamW(head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    rng=np.random.default_rng(seed);history=[];best=(-1.,-float('inf'));stale=0;best_epoch=0;first=1
    last=directory/'last_state.pt'
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=True)
        if state['identity']!=spec: raise ValueError('Resume identity mismatch')
        head.load_state_dict(state['head']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
        if config['device'].startswith('cuda'): torch.cuda.set_rng_state_all(state['cuda_rng'])
        history=state['history'];best=tuple(state['best']);stale=state['stale'];best_epoch=state['best_epoch'];first=state['epoch']+1
    ds.write_json(directory/'training_config.json',{**cfg,'training_seed':seed,'recipe':recipe,'runtime':hardware_report(),
        'normalization_fit_split':'train','head_parameters':sum(p.numel() for p in head.parameters()),
        'class_weights':{h:w.tolist() for h,w in weight_arrays.items()},'test_used_for_selection':False})
    for epoch in range(first,config['epochs']+1):
        if stale>=config['patience']: break
        head.train();losses=[];began=time.monotonic()
        batches=core.training_batches(arrays['train']['future_labels'],arrays['train']['future_valid_mask'],len(x),config['batch_size'],rng,mode)
        for ids in batches:
            z=torch.from_numpy((x[ids]-mean)/std).to(config['device'])
            yc=torch.from_numpy(arrays['train']['current_labels'][ids]).to(config['device'])
            yf=torch.from_numpy(arrays['train']['future_labels'][ids]).to(config['device'])
            valid=torch.from_numpy(arrays['train']['future_valid_mask'][ids]).to(config['device'])
            optimizer.zero_grad(set_to_none=True)
            loss=transition_loss(model.forward_heads(head,z),yc,yf,valid,weights,prediction,recipe['transition_weight'])
            loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()))
        outputs=model.predict_heads(head,arrays['val']['features'],mean,std)
        val=dict(current=None,future=None)
        for h in active:
            keep=np.ones(len(arrays['val']['features']),dtype=bool) if h=='current' else arrays['val']['future_valid_mask']
            val[h]=metrics.scores(outputs[h+'_logits'][keep],arrays['val'][h+'_labels'][keep])
        f1,nll=core.validation_selection(val['current'],val['future'],prediction);key=(f1,-nll)
        row=dict(epoch=epoch,loss=float(np.mean(losses)),val=val,selection_score=f1,selection_nll=nll,elapsed_s=time.monotonic()-began)
        history.append(row)
        if key>best:
            best=key;best_epoch=epoch;stale=0
            model.save_checkpoint(directory/'best_head.pt',head,mean,std,manifest,ds.digest(manifest_path),epoch,val)
        else: stale+=1
        # An independent inference checkpoint at the last epoch; never overwrites best.
        model.save_checkpoint(directory/'last_head.pt',head,mean,std,manifest,ds.digest(manifest_path),epoch,val)
        state=dict(identity=spec,epoch=epoch,head={k:v.detach().cpu() for k,v in head.state_dict().items()},
                   optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if config['device'].startswith('cuda') else [],
                   best=list(best),best_epoch=best_epoch,stale=stale,history=history)
        atomic_save(last,state)
        tmp=directory/'metrics.tmp';tmp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in history));tmp.replace(directory/'metrics.jsonl')
        ds.status(directory,'training',epoch=epoch,best_epoch=best_epoch,validation_selection_score=f1)
    result=dict(name=recipe['name'],training_seed=seed,best_epoch=best_epoch,epochs_executed=len(history),
                validation_selection_score=best[0],validation_selection_nll=-best[1],
                checkpoint=str(directory/'best_head.pt'),checkpoint_sha256=ds.digest(directory/'best_head.pt'),
                test_evaluated=False)
    ds.write_json(directory/'validation_result.json',result);ds.status(directory,'validation_complete',**result)
    return result


def lock_selection(root, results, expected):
    root=Path(root)
    if len(results)!=expected: raise ValueError('All predefined runs must finish before selection')
    families={name:np.mean([r['validation_selection_score'] for r in results if r['name']==name])
              for name in ('joint','joint_transition')}
    winner=max(families,key=lambda n:(families[n],n=='joint'))
    chosen=next(r for r in results if r['name']==winner and r['training_seed']==42)
    selection=dict(joint_family_validation_scores=families,winner=winner,representative_seed=42,
                   checkpoint=chosen['checkpoint'],checkpoint_sha256=chosen['checkpoint_sha256'],
                   policy='Family selected by mean validation score across predefined seeds; fixed seed42 deployment representative',
                   results_sha256=ds.fingerprint(results),test_accessed_for_selection=False)
    path=root/'selection.json'
    if path.exists() and ds.read_json(path)!=selection: raise ValueError('Selection is already locked')
    ds.write_json(path,selection);return selection


def evaluate_selected_runs(root, results, entries, arrays, config):
    selection=ds.read_json(Path(root)/'selection.json')
    if selection['results_sha256']!=ds.fingerprint(results): raise ValueError('Selection/results mismatch')
    reports=[]
    for result in results:
        directory=Path(result['checkpoint']).parent
        if ds.digest(result['checkpoint'])!=result['checkpoint_sha256']: raise ValueError('Checkpoint changed')
        path=directory/'heldout_evaluation.json'
        if path.exists():
            existing=ds.read_json(path)
            if (existing['checkpoint_sha256']!=result['checkpoint_sha256']
                    or existing['selection_sha256']!=ds.digest(Path(root)/'selection.json')):
                raise ValueError('Saved evaluation identity changed')
            reports.append(existing);continue
        saved,head=model.load_checkpoint(result['checkpoint'],device=config['device'])
        # Test is accessed for model evaluation only after the immutable selection exists.
        logits=model.predict_heads(head,arrays['test']['features'],saved['mean'],saved['std'])
        report=dict(name=result['name'],training_seed=result['training_seed'],checkpoint_sha256=result['checkpoint_sha256'],
                    selection_sha256=ds.digest(Path(root)/'selection.json'),
                    metrics=metrics.evaluate_heads(entries['test'],logits,saved['prediction']))
        rows=[];offset=0
        for part in entries['test']:
            n=len(part['features']);z={h:None if logits[h+'_logits'] is None else logits[h+'_logits'][offset:offset+n] for h in ('current','future')}
            probs={h:None if v is None else metrics.probabilities(v) for h,v in z.items()}
            rows+=core.prediction_rows(part,probs,'test',saved['prediction'],z);offset+=n
        core.write_predictions(directory,rows)
        ds.write_json(path,report);reports.append(report)
    ds.write_json(Path(root)/'heldout_comparison.json',reports)
    summary={}
    for recipe in RECIPES:
        parts=[r for r in reports if r['name']==recipe['name']]
        summary[recipe['name']]={}
        for h in ('current','future'):
            values=[r['metrics'][h] for r in parts if r['metrics'][h] is not None]
            if not values:
                summary[recipe['name']][h]=None;continue
            summary[recipe['name']][h]={k:dict(mean=float(np.mean([v[k] for v in values])),
                                                   std=float(np.std([v[k] for v in values],ddof=1)))
                                       for k in ('accuracy','macro_f1','nll')}
    ds.write_json(Path(root)/'heldout_summary.json',dict(seeds=[42,43,44],summary=summary,
                selection=selection,test_source_count=len(entries['test']),test_used_for_tuning=False))
    return reports


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--campaign-dir',required=True)
    a=p.parse_args(argv);cfg=ds.load_config(a.config);initialize_runtime(cfg)
    parent=core.load_locked_manifest(cfg)
    if not (Path(cfg['output_dir'])/'feature_index.json').exists(): core.extract(cfg,parent)
    entries=core.load_training_arrays(cfg,parent);arrays=concatenate(entries)
    root=Path(a.campaign_dir).resolve();ds.ensure_output_outside_sources(root,[e['dataset_root'] for e in parent['episodes']]);root.mkdir(parents=True,exist_ok=True)
    plan=dict(recipes=list(RECIPES),seeds=[42,43,44],parent_manifest_sha256=ds.digest(Path(cfg['output_dir'])/'manifest.json'),
              selection='Validation-only; heldout untouched until all runs finish',runner_sha256=ds.digest(__file__))
    if (root/'plan.json').exists() and ds.read_json(root/'plan.json')!=plan:raise ValueError('Campaign plan changed')
    ds.write_json(root/'plan.json',plan);results=[]
    for recipe in RECIPES:
        for seed in plan['seeds']:
            ds.status(root,'training_campaign',recipe=recipe['name'],seed=seed,completed=len(results),total=12)
            result=train_one(cfg,parent,arrays,recipe,seed,root/f'{recipe["name"]}_s{seed}')
            results.append(result);ds.write_json(root/'validation_results.json',results)
    selection=lock_selection(root,results,12)
    reports=evaluate_selected_runs(root,results,entries,arrays,cfg)
    ds.status(root,'complete',runs=len(results),selected=selection['checkpoint'],reports=str(root/'heldout_comparison.json'))
    print(json.dumps(dict(selection=selection,runs=len(reports))),flush=True)


if __name__=='__main__':main()
