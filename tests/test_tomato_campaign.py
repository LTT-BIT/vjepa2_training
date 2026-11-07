"""Synthetic loss/resume tests; never accuracy evidence."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from tools import tomato_dataset as ds, tomato_phase_model as model
from tools import train_tomato_vjepa_phase as core
from tools import train_tomato_campaign as campaign
from tomato_helpers import make_config


@pytest.mark.parametrize('mode',['current_only','future_only','joint'])
def test_unit_transition_weight_matches_original_masked_class_weighted_loss(mode):
    cfg=ds.prediction_contract(dict(prediction_mode=mode))
    torch.manual_seed(4)
    output={h+'_logits':torch.randn(5,4,requires_grad=True) for h in ('current','future')}
    yc=torch.tensor([0,1,2,3,0]);yf=torch.tensor([1,1,3,-100,-100]);mask=torch.tensor([1,1,1,0,0],dtype=torch.bool)
    weights={h:torch.tensor([.5,1.,2.,1.5]) for h in ('current','future')}
    old,_=model.supervision_loss(output,yc,yf,mask,weights,cfg)
    new=campaign.transition_loss(output,yc,yf,mask,weights,cfg,1.)
    torch.testing.assert_close(old,new)
    new.backward()
    if mode!='current_only':
        assert torch.all(output['future_logits'].grad[~mask]==0)


def test_transition_weight_increases_hard_change_example_without_using_invalid_tail():
    out={'future_logits':torch.tensor([[6.,-6.,-6.,-6.],[6.,-6.,-6.,-6.],[0.,0.,0.,0.]])}
    yc=torch.tensor([0,0,0]);yf=torch.tensor([0,1,-100]);valid=torch.tensor([True,True,False])
    weights={'future':torch.ones(4)};cfg=ds.prediction_contract(dict(prediction_mode='future_only'))
    ordinary=campaign.transition_loss(out,yc,yf,valid,weights,cfg,1.)
    emphasis=campaign.transition_loss(out,yc,yf,valid,weights,cfg,2.)
    assert emphasis>ordinary


def test_resume_restores_optimizer_dropout_and_sampling_exactly(tmp_path,monkeypatch):
    cfg={**make_config(tmp_path,length=300), 'prediction_mode':'joint'}
    manifest=core.prepare(cfg)
    ds.write_json(Path(cfg['output_dir'])/'feature_index.json',dict(
        manifest_sha256=ds.digest(Path(cfg['output_dir'])/'manifest.json'),
        pretrained=manifest['pretrained'],episodes=[]))
    rng=np.random.default_rng(7);arrays={}
    for split in ('train','val','test'):
        e=next(e for e in manifest['episodes'] if e['split']==split)
        anchors=np.array(e['anchors']);targets=ds.build_phase_targets(anchors,e['annotation']['segments'],e['length'],e['fps'],1.)
        arrays[split]={k:targets[k] for k in ('current_labels','future_labels','future_valid_mask')}
        arrays[split]['features']=rng.normal(size=(len(anchors),3072+16*len(cfg['joint_names']))).astype(np.float32)
    recipe=dict(name='joint',prediction_mode='joint',transition_weight=1.)
    campaign.train_one(cfg,manifest,arrays,recipe,42,tmp_path/'continuous')
    original=campaign.atomic_save
    def crash_after_commit(path,value):
        original(path,value)
        raise RuntimeError('simulated process stop after epoch commit')
    monkeypatch.setattr(campaign,'atomic_save',crash_after_commit)
    with pytest.raises(RuntimeError,match='simulated'):
        campaign.train_one(cfg,manifest,arrays,recipe,42,tmp_path/'resumed')
    monkeypatch.setattr(campaign,'atomic_save',original)
    campaign.train_one(cfg,manifest,arrays,recipe,42,tmp_path/'resumed')
    a=torch.load(tmp_path/'continuous/last_state.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed/last_state.pt',weights_only=True)
    for name in a['head']: torch.testing.assert_close(a['head'][name],b['head'][name],rtol=0,atol=0)
    assert [r['loss'] for r in a['history']]==[r['loss'] for r in b['history']]
    assert not (tmp_path/'resumed/heldout_evaluation.json').exists()
    changed={**recipe,'transition_weight':2.}
    with pytest.raises(ValueError,match='changed'):
        campaign.train_one(cfg,manifest,arrays,changed,42,tmp_path/'resumed')


def test_selection_requires_complete_runs_and_uses_validation_only(tmp_path):
    with pytest.raises(ValueError,match='All predefined'):
        campaign.lock_selection(tmp_path,[],12)
    rows=[]
    for name,score in [('joint',.7),('joint_transition',.8)]:
        for seed in [42,43,44]:
            rows.append(dict(name=name,training_seed=seed,validation_selection_score=score,
                             checkpoint=f'{name}_{seed}.pt',checkpoint_sha256='a'*64))
    result=campaign.lock_selection(tmp_path,rows,6)
    assert result['winner']=='joint_transition' and result['representative_seed']==42
    assert result['test_accessed_for_selection'] is False
