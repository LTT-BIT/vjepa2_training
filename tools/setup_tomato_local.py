"""Create a strict local run from an explicit annotation selection; no training/downloads."""
import argparse
from pathlib import Path
import yaml
from . import tomato_dataset as ds
from .build_tomato_annotation_index import lock_selection
from .train_tomato_vjepa_phase import grouped_split


def create_run(selection, run_root, pretrained_dir, *, template=None, seed=42, epochs=None):
    root = Path(run_root).resolve()
    if root.exists():
        raise ValueError('Run root already exists; select a new path')
    rows = ds.load_annotation_index(selection)
    ds.ensure_output_outside_sources(root, [r['dataset_root'] for r in rows],
                                     [r['coarse_annotation'] for r in rows])
    cfg = yaml.safe_load(Path(template or ds.REPO/'configs/tomato_four_stage.yaml').read_text())
    root.mkdir(parents=True)
    lock_selection(selection, root/'annotation_index.json', allow_subset_metadata_mismatch=False)
    episodes = [ds.load_episode_info(r['dataset_root'], r['episode_index']) for r in rows]
    splits = grouped_split(episodes, seed)
    ds.write_json(root/'split.json', splits)
    cfg.update(annotation_index=str(root/'annotation_index.json'), split_file=str(root/'split.json'),
               pretrained_dir=str(Path(pretrained_dir).resolve()), allow_subset_metadata_mismatch=False,
               feature_cache_dir=str(root/'shared_feature_cache'), seed=seed, batch_windows=4)
    if epochs is not None:
        cfg['epochs'] = epochs
    paths = {}
    for mode in ('current_only', 'future_only', 'joint'):
        value = {**cfg, 'prediction_mode': mode, 'output_dir': str(root/mode)}
        path = root/(mode+'.yaml');path.write_text(yaml.safe_dump(value, sort_keys=False))
        paths[mode] = str(path)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', required=True)
    p.add_argument('--run-root', required=True)
    p.add_argument('--pretrained-dir', required=True)
    p.add_argument('--epochs', type=int)
    a = p.parse_args()
    print(create_run(a.selection, a.run_root, a.pretrained_dir, epochs=a.epochs))


if __name__ == '__main__':
    main()
