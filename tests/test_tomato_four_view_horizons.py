import numpy as np
from tools.compare_tomato_four_view_horizons import relabel, common_scores


def row():
    labels = ['pickup','harvest','return','adjust','pickup','harvest','return']
    episode = dict(episode_id='synthetic', fps=30, length=420,
        annotation=dict(segments=[dict(start_frame=i*60,end_frame=(i+1)*60,
            label=label,reviewed=True) for i,label in enumerate(labels)]))
    anchors = np.arange(45,420,15)
    return dict(episode=episode,anchors=anchors,times=anchors/30,
        visual=np.zeros((len(anchors),4,1024),np.float32))


def test_horizon_changes_only_targets_and_masks_tail():
    source = row(); a = relabel([source],1.)[0]; b = relabel([source],1.5)[0]
    assert a['visual'] is source['visual'] and b['visual'] is source['visual']
    assert a['anchors'] is b['anchors']
    assert 'target' not in source
    np.testing.assert_array_equal(a['target']['current_labels'],b['target']['current_labels'])
    assert a['target']['future_labels'][-1] == -100
    assert a['target']['future_valid_mask'].sum() == b['target']['future_valid_mask'].sum()+1
    valid = b['target']['future_valid_mask']
    np.testing.assert_array_equal(b['target']['future_target_frames'][valid]-a['target']['future_target_frames'][valid],15)


def test_common_window_counts_and_scores_same_support():
    source = row(); rows = relabel([source],1.)
    pred = {source['episode']['episode_id']: dict(
        current=np.tile([.9,.04,.03,.03],(len(source['anchors']),1)),
        future=np.tile([.9,.04,.03,.03],(len(source['anchors']),1)))}
    result = common_scores(rows,pred,1.)
    for window in ['1.0','1.5']:
        assert result['windows'][window]['total']==6
        assert result['windows'][window]['hits']==0
        assert result['windows'][window]['false_events']==0
    assert len(result['per_episode_events'])==2


def test_horizon_model_version_is_distinct_without_weight_changes(tmp_path):
    import torch
    from tools.compare_tomato_four_view_horizons import version_checkpoint
    path=tmp_path/'head.pt';weights=torch.arange(8,dtype=torch.float32)
    torch.save(dict(model_version='shared-cache-model',head=dict(weight=weights)),path)
    first=version_checkpoint(path,dict(future_horizon_s=1.))
    saved=torch.load(path,weights_only=True)
    assert '-h10-' in saved['model_version']
    torch.testing.assert_close(saved['head']['weight'],weights)
    assert version_checkpoint(path,dict(future_horizon_s=1.))==first
