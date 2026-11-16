import pytest
from tools.publish_tomato100_roi import validate_info,safe,publish,CAMERAS


def test_accepts_four_view_container_only():
    info=dict(features={k:dict(dtype='video') for k in CAMERAS},total_episodes=100,total_frames=425556,fps=30)
    validate_info(info)
    del info['features'][CAMERAS[-1]]
    with pytest.raises(ValueError):validate_info(info)


def test_paths_cannot_escape_dataset(tmp_path):
    (tmp_path/'ok').write_text('x')
    assert safe(tmp_path,'ok')==tmp_path/'ok'
    for value in ['/etc/passwd','../outside']:
        with pytest.raises(ValueError):safe(tmp_path,value)


def test_existing_publication_never_overwritten(tmp_path):
    source=tmp_path/'src';source.mkdir();target=tmp_path/'out';target.mkdir();(target/'keep').write_text('unchanged')
    with pytest.raises(FileExistsError):publish(source,target,tmp_path/'identity.json')
    assert (target/'keep').read_text()=='unchanged'
