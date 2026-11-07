from pathlib import Path
import pytest
from tools import tomato_dataset as ds
from tools.setup_tomato_local import create_run
from tomato_helpers import make_config


def test_setup_locks_strict_sources_and_shared_split_without_overwrite(tmp_path):
    cfg = make_config(tmp_path)
    root=tmp_path/'new-experiment'
    paths=create_run(cfg['annotation_index'],root,tmp_path/'weights',epochs=3)
    values=[ds.load_config(p) for p in paths.values()]
    assert {v['prediction_mode'] for v in values}=={'current_only','future_only','joint'}
    assert all(v['allow_subset_metadata_mismatch'] is False for v in values)
    assert len({v['split_file'] for v in values})==1
    assert len({v['feature_cache_dir'] for v in values})==1
    groups=ds.read_json(root/'split.json')
    assert sum(map(len,groups.values()))==len({g for v in groups.values() for g in v})==3
    with pytest.raises(ValueError,match='already exists'):
        create_run(cfg['annotation_index'],root,tmp_path/'weights')
