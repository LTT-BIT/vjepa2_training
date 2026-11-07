import copy
import numpy as np
import pytest
import torch
from tools import tomato_phase_model as model, tomato_independent_heads as dual
from tools import train_tomato_vjepa_phase as tr, tomato_dataset as ds
from tests.tomato_helpers import make_config


def test_bundle_exact_parity_and_reject_normalization_or_schema(tmp_path):
    cfg=make_config(tmp_path);manifest=tr.prepare(cfg);dimension=3072+16*len(cfg['joint_names'])
    paths=[];expected={};x=np.random.default_rng(1).normal(size=(3,dimension)).astype(np.float32)
    for name in ('current','future'):
        p=tmp_path/(name+'.pt');m=copy.deepcopy(manifest)
        m['prediction']=ds.prediction_contract({**cfg,'prediction_mode':name+'_only'})
        h=model.mode_head_model(len(cfg['joint_names']),name+'_only').eval()
        model.save_checkpoint(p,h,np.zeros(dimension),np.ones(dimension),m,'abc',1,{})
        expected[name+'_logits']=model.predict_heads(h,x,np.zeros(dimension),np.ones(dimension))[name+'_logits'];paths.append(p)
    output=tmp_path/'dual.pt';dual.pack(*paths,output)
    saved,h=dual.load_checkpoint(output)
    actual=model.predict_heads(h,x,saved['mean'],saved['std'])
    for key in expected: np.testing.assert_array_equal(actual[key],expected[key])
    assert saved['prediction_mode']=='joint'
    assert set(h.current.parameters()).isdisjoint(set(h.future.parameters()))
    original=torch.load(output,weights_only=True)
    changed=copy.deepcopy(original);changed['future']['std'][0]=2;torch.save(changed,output)
    with pytest.raises(ValueError,match='normalization'): dual.load_checkpoint(output)
    changed=copy.deepcopy(original);changed['future']['label_order'].reverse();torch.save(changed,output)
    with pytest.raises(ValueError,match='compatible'): dual.load_checkpoint(output)
    changed=copy.deepcopy(original);changed['future']['split_sha256']='x'*64;torch.save(changed,output)
    with pytest.raises(ValueError,match='split'): dual.load_checkpoint(output)


def test_horizon_derivative_preserves_features_and_updates_summary(tmp_path,monkeypatch):
    from tools import optimize_tomato_agent as opt
    cfg=make_config(tmp_path);manifest=tr.prepare(cfg)
    from pathlib import Path
    parent=Path(cfg['output_dir'])
    ds.write_json(parent/'feature_index.json',dict(episodes=[],pretrained=manifest['pretrained']))
    entries={s:[] for s in ('train','val','test')}
    for e in manifest['episodes']:
        a=np.array(e['anchors']);targets=ds.build_phase_targets(a,e['annotation']['segments'],e['length'],e['fps'],1.)
        entries[e['split']].append(dict(episode=e,features=np.zeros((len(a),3328),np.float32),times=a/e['fps'],anchors=a,**{k:v for k,v in targets.items() if k!='anchors'}))
    monkeypatch.setattr(opt,'ROOT',tmp_path/'derived')
    _,m,new,_=opt.horizon_data(cfg,manifest,entries,.5)
    for s in entries:
        mask=np.concatenate([r['future_valid_mask'] for r in new[s]])
        assert m['summary'][s]['future_valid']==int(mask.sum())
        assert m['summary'][s]['future_invalid']==int((~mask).sum())
        assert new[s][0]['features'] is entries[s][0]['features']
    assert m['split_sha256']==manifest['split_sha256']
