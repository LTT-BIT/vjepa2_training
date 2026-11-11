from copy import deepcopy
from tools.select_tomato_complete_episodes import ORDER, select_complete


def episode(eid, split, labels):
    return dict(episode_id=eid, source_recording_id='source-'+eid, split=split,
                dataset_root='/fixture', episode_index=0, anchors=[0, 15, 30], length=len(labels)*15, fps=30.,
                annotation=dict(sha256='fixture', segments=[dict(label=l, reviewed=True, start_frame=i*15, end_frame=(i+1)*15) for i,l in enumerate(labels)]))


def test_complete_selection_preserves_parent_and_split_assignments():
    parent=dict(episodes=[episode('a','train',ORDER), episode('b','val',ORDER[:3]), episode('c','val',ORDER), episode('d','test',ORDER)],
                splits=dict(train=['source-a'],val=['source-b','source-c'],test=['source-d']))
    before=deepcopy(parent);new,excluded=select_complete(parent)
    assert parent==before
    assert [e['episode_id'] for e in new['episodes']]==['a','c','d']
    assert new['splits']==dict(train=['source-a'],val=['source-c'],test=['source-d'])
    assert excluded[0]['episode_id']=='b' and excluded[0]['segment_count']==3
    assert new['summary']['val']['episodes']==1


def test_seven_segments_with_wrong_order_are_not_complete_task():
    e=episode('a','train',['pickup']*7)
    new,excluded=select_complete(dict(episodes=[e],splits=dict(train=['source-a'],val=[],test=[])))
    assert not new['episodes'] and len(excluded)==1
