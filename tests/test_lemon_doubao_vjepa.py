import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('pyarrow')
pytest.importorskip('torch')

from tools import run_lemon_doubao_vjepa as pipeline
from tools import train_lemon_vjepa_phase as trainer


def make_inputs(tmp_path):
    previous, labels, run, root = [tmp_path / k for k in ('previous', 'labels', 'run', 'dataset')]
    for p in (previous / 'features', previous / 'pretrained', labels / 'predictions', run):
        p.mkdir(parents=True)
    original = dict(model_id=trainer.MODEL_ID, revision=trainer.REVISION, clip_frames=trainer.CLIP,
                    image_size=trainer.SIZE, stride_frames=trainer.STRIDE, cameras=list(trainer.CAMERAS),
                    joints=list(trainer.JOINTS), labels=list(trainer.LABELS), boundaries=list(trainer.BOUNDARIES),
                    causal=True, seed=trainer.SEED, dataset=str(root), annotations_sha256='original-human-file-hash',
                    demo_episode=2, episodes=[])
    inputs = {}
    for i, split in enumerate(('train','val','test')):
        videos, parquet = trainer.paths(root,i)
        for p in [*videos,parquet]:
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(f'original-{p.name}-{p.parent.name}'.encode())
        e = dict(episode=i, source=f'bag{i}', frames=60, split=split, boundaries=[.2,1.2,2.2,3.2,4.2],
                 video_sha256=[trainer.digest(v) for v in videos],parquet_sha256=trainer.digest(parquet))
        original['episodes'].append(e)
        times=np.arange(0,60,trainer.STRIDE)/10
        np.savez_compressed(previous/'features'/f'{i:06d}.npz',features=np.full((len(times),3264),i,dtype=np.float32),
                            indices=np.arange(0,60,trainer.STRIDE),times=times,
                            labels=np.full(len(times),99),fingerprint=np.array(trainer.feature_fingerprint(e)))
        inputs[str(i)] = dict(episode=i,frames=60,last_frame_s=5.9,video_sha256=e['video_sha256'],fingerprint=f'input{i}')
        b = {k:dict(time_s=t,confidence=.9,evidence='visible video evidence')
             for k,t in zip(trainer.BOUNDARIES,[.7,1.7,2.7,3.7,4.7])}
        trainer.write_json(labels/'predictions'/f'{i:06d}.json',dict(episode=i,input_fingerprint=f'input{i}',boundaries=b))
    trainer.write_json(previous/'manifest.json',original)
    trainer.write_json(previous/'training_config.json',dict(learning_rate=3e-4))
    trainer.write_json(labels/'config.json',dict(dataset=str(root),annotations_sha256=original['annotations_sha256']))
    trainer.write_json(labels/'inputs.json',inputs)
    trainer.write_json(labels/'doubao_annotations.json',dict(schema_version=1,episodes=[]))
    return previous,labels,run,original


def test_relabels_without_reusing_human_targets_and_preserves_heldout_split(tmp_path):
    previous,labels,run,original=make_inputs(tmp_path)
    manifest,_=pipeline.prepare_training(run,previous,labels)
    assert [e['split'] for e in manifest['episodes']]==['train','val','test']
    assert manifest['label_source']=='豆包'
    assert not (run/'best_head.pt').exists()
    assert (run/'pretrained').resolve()==(previous/'pretrained').resolve()
    for e in manifest['episodes']:
        with np.load(run/'features'/f'{e["episode"]:06d}.npz',allow_pickle=False) as d:
            np.testing.assert_array_equal(d['labels'],trainer.labels_at(d['times'],[.7,1.7,2.7,3.7,4.7]))
            np.testing.assert_array_equal(d['features'],np.full(d['features'].shape,e['episode']))
            assert str(d['fingerprint'])==trainer.feature_fingerprint(e)
    assert json.loads((previous/'manifest.json').read_text())==original
    pipeline.prepare_training(run,previous,labels)


def test_missing_prediction_blocks_training_before_manifest_creation(tmp_path):
    previous,labels,run,_=make_inputs(tmp_path)
    (labels/'predictions/000001.json').unlink()
    with pytest.raises(ValueError,match='Annotation not finished'):
        pipeline.prepare_training(run,previous,labels)
    assert not (run/'manifest.json').exists()


def test_unresolved_boundary_is_excluded_without_human_fill(tmp_path):
    previous,labels,run,_=make_inputs(tmp_path)
    p=labels/'predictions/000001.json'
    d=json.loads(p.read_text());d['boundaries'][trainer.BOUNDARIES[0]]['time_s']=None
    trainer.write_json(p,d)
    with pytest.raises(ValueError,match='required split'):
        pipeline.prepare_training(run,previous,labels)
    assert not (run/'manifest.json').exists()


@pytest.mark.parametrize('fault',['cache','video','input','split'])
def test_changed_cache_video_input_or_leaking_split_is_rejected(tmp_path,fault):
    previous,labels,run,original=make_inputs(tmp_path)
    if fault=='cache':
        p=previous/'features/000000.npz'
        with np.load(p,allow_pickle=False) as d: data={k:d[k] for k in d.files}
        data['fingerprint']=np.array('stale');np.savez_compressed(p,**data)
    elif fault=='video':
        trainer.paths(Path(original['dataset']),0)[0][0].write_bytes(b'changed')
    elif fault=='input':
        p=labels/'predictions/000000.json';d=json.loads(p.read_text());d['input_fingerprint']='wrong';trainer.write_json(p,d)
    else:
        original['episodes'][1]['source']=original['episodes'][0]['source']
        trainer.write_json(previous/'manifest.json',original)
    with pytest.raises(ValueError,match='mismatch|changed|leaked'):
        pipeline.prepare_training(run,previous,labels)

