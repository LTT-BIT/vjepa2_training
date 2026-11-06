from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip('pyarrow')
pytest.importorskip('torch')

from tools import train_lemon_vjepa_phase as model


def test_causal_history_never_reads_future_and_only_pads_the_past():
    for end in (0,1,10,15,100):
        ids=model.history_indices(end)
        assert len(ids)==16 and ids[-1]==end
        assert np.all(ids>=0) and np.all(ids<=end)
        assert np.all(np.diff(ids)>=0)
    np.testing.assert_array_equal(model.history_indices(2)[-3:],[0,1,2])
    with pytest.raises(ValueError):model.history_indices(-1)


def test_five_timestamps_make_six_causal_labels():
    times=np.array([0.,1.,1.9,2.,3.,4.,5.,6.])
    np.testing.assert_array_equal(model.labels_at(times,[1,2,3,4,5]),[0,1,1,2,3,4,5,5])
    for b in ([1,2,2,4,5],[1,2,3,4,float('nan')],[-1,2,3,4,5]):
        with pytest.raises(ValueError):model.labels_at(times,b)


def test_split_keeps_original_recordings_together_and_is_reproducible():
    episodes=[dict(episode=i,source=f'bag{i//2}') for i in range(40)]
    split=model.grouped_split(episodes)
    assert split==model.grouped_split(list(reversed(episodes)))[::-1]
    assignments={}
    for e in split:assignments.setdefault(e['source'],set()).add(e['split'])
    assert all(len(s)==1 for s in assignments.values())
    assert {e['split'] for e in split}=={'train','val','test'}


def test_quality_scores_report_class_failures_not_only_accuracy():
    y=np.arange(6)
    good=model.scores(np.eye(6)*10,y)
    bad=model.scores(np.tile([10,0,0,0,0,0],(6,1)),y)
    assert good['accuracy']==1 and good['macro_f1']==1
    assert bad['accuracy']==pytest.approx(1/6)
    assert bad['macro_f1']<.05
    with pytest.raises(ValueError):model.scores(np.full((6,6),np.nan),y)


def test_extraction_uses_fixed_schedule_and_aligned_joint_history_without_label_timing(monkeypatch,tmp_path):
    root=tmp_path/'dataset';run=tmp_path/'run';run.mkdir()
    times=np.arange(20)*.1
    q=np.arange(20*12,dtype=np.float32).reshape(20,12)
    views=[np.full((20,1,1,3),c,dtype=np.uint8) for c in range(3)]
    episode=dict(episode=7,source='bag',frames=20,split='train',boundaries=[.1,.3,.7,1.1,1.7])
    manifest=dict(dataset=str(root),episodes=[episode],joint_columns=list(range(12)))
    monkeypatch.setattr(model,'read_state',lambda *a:(times,q))
    monkeypatch.setattr(model,'decode_videos',lambda *a:views)
    seen=[]
    def encode(encoder,clips):
        seen.append(clips.copy())
        return clips[:,0,0,0,:1].astype(np.float32)
    monkeypatch.setattr(model,'encode_clips',encode)
    model.extract(run,manifest,None,2)
    with np.load(run/'features/000007.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(data['indices'],[0,5,10,15])
        np.testing.assert_array_equal(data['features'][:,:3],np.tile([0,1,2],(4,1)))
        np.testing.assert_array_equal(data['features'][0,3:].reshape(16,12),np.tile(q[0],(16,1)))
        np.testing.assert_array_equal(data['features'][-1,3:].reshape(16,12),q[:16])
        np.testing.assert_array_equal(data['labels'],model.labels_at(times[[0,5,10,15]],episode['boundaries']))
    model.extract(run,manifest,None,2)
    assert len(seen)==2
    episode['frames']=21
    with pytest.raises(ValueError,match='cache mismatch'):model.extract(run,manifest,None,2)


def test_decode_rejects_corrupt_or_short_video_and_closes(monkeypatch,tmp_path):
    closed=[]
    monkeypatch.setattr(cv2,'VideoCapture',lambda *a:SimpleNamespace(read=lambda:(False,None),release=lambda:closed.append(True)))
    with pytest.raises(ValueError,match='decode/state mismatch'):model.decode_videos(tmp_path,0,20)
    assert closed==[True]


def test_demo_rejects_same_recording_in_training_before_loading_features(tmp_path):
    manifest=dict(demo_episode=7,episodes=[dict(episode=7,source='same',split='test'),dict(episode=8,source='same',split='train')])
    with pytest.raises(ValueError,match='leaked'):model.render_demo(tmp_path,manifest,None,None)
