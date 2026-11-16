"""Offline train/validation experiment; never opens test feature/annotation files.

This is a diagnostic runner, not a replacement for the production trainer.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds
from .tomato_runtime import initialize_runtime


def common_batches(valid, seed, epoch, batch_size):
    from .train_tomato_vjepa_phase import training_batches
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    return training_batches(None, valid, len(valid), batch_size, rng, 'joint')


def gradient_statistics(gc, gf, lc=1., lf=.5):
    import torch
    dot = sum((a * b).sum() for a, b in zip(gc, gf))
    nc = torch.sqrt(sum(a.square().sum() for a in gc))
    nf = torch.sqrt(sum(b.square().sum() for b in gf))
    return dict(current_norm=float(nc), future_norm=float(nf),
                cosine=float(dot / (nc * nf).clamp_min(1e-30)),
                weighted_future_current_ratio=float(lf * nf / (lc * nc).clamp_min(1e-30)))


def read_inputs(base):
    manifest = ds.read_json(base / 'joint/manifest.json')
    paths = {e['episode_id']: e['path'] for e in ds.read_json(base / 'joint/feature_index.json')['episodes']}
    entries = dict(train=[], val=[])
    hashes = {}
    for e in manifest['episodes']:
        if e['split'] not in entries:
            continue  # No test annotation, parquet or feature file is opened.
        path = Path(paths[e['episode_id']])
        for p, expected in [(Path(e['annotation']['path']), e['annotation']['sha256']),
                            (Path(e['parquet']), e['input_sha256'][str(Path(e['parquet']).relative_to(e['dataset_root']))]),
                            (path, ds.read_json(path.with_suffix('.json'))['content_sha256'])]:
            actual = ds.digest(p)
            if actual != expected:
                raise ValueError(f'Locked input changed: {p}')
            hashes[str(p)] = actual
        ann = ds.read_json(e['annotation']['path'])
        if ann['segments'] != e['annotation']['segments']:
            raise ValueError('Annotation snapshot changed')
        with np.load(path, allow_pickle=False) as f:
            features, anchors, times = f['features'].copy(), f['anchors'].copy(), f['times'].copy()
        if anchors.tolist() != e['anchors']:
            raise ValueError('Anchor grid changed')
        targets = ds.build_phase_targets(anchors, ann['segments'], e['length'], e['fps'], 1.)
        entries[e['split']].append(dict(episode=e, features=features, times=times,
                                        labels=targets['current_labels'], **targets))
    arrays = {s: {k: np.concatenate([e[k] for e in ee]) for k in
                 ['features', 'current_labels', 'future_labels', 'future_valid_mask']} for s, ee in entries.items()}
    return manifest, entries, arrays, hashes


def run_one(out, seed, mode, epochs, config, entries, arrays):
    import torch
    from . import tomato_phase_model as model, tomato_metrics as metrics
    from .train_tomato_vjepa_phase import fit_class_weights, validation_selection
    out.mkdir(parents=True, exist_ok=False)
    runtime = initialize_runtime(dict(device='cuda', seed=seed))
    cfg = {**config, 'seed': seed, 'prediction_mode': mode}
    # Construct the same two heads for all modes; each active head starts identically.
    head = model.joint_head_model(len(cfg['joint_names'])).cuda()
    active = ds.active_heads(mode)
    params = list(head.trunk.parameters()) + [p for h in active for p in getattr(head, h).parameters()]
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    x = arrays['train']['features']
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    std = x.std(0, dtype=np.float64).clip(.01).astype(np.float32)
    xt = torch.from_numpy((x - mean) / std).cuda()
    yc = torch.from_numpy(arrays['train']['current_labels']).cuda()
    yf = torch.from_numpy(arrays['train']['future_labels']).cuda()
    mask = torch.from_numpy(arrays['train']['future_valid_mask']).cuda()
    weights = {}
    for h in active:
        y = arrays['train'][h + '_labels']; w, _ = fit_class_weights(y[y >= 0])
        weights[h] = torch.from_numpy(w).cuda()
    # Reset dropout stream after construction; same shape/order in every mode.
    torch.manual_seed(seed + 100000)
    ds.write_json(out / 'config.json', {**cfg, 'runtime': runtime, 'epochs': epochs,
        'early_stopping': False, 'evaluate_test': False, 'diagnostic_only': True,
        'active_parameters': sum(p.numel() for p in params)})
    best = (-1., -float('inf')); best_state = None; best_epoch = 0
    order_hash = hashlib.sha256(); grad_rows = []; norm_rows = []
    began = time.monotonic()
    for epoch in range(1, epochs + 1):
        head.train(); losses = []
        batches = common_batches(arrays['train']['future_valid_mask'], seed, epoch, cfg['batch_size'])
        for step, ids in enumerate(batches):
            order_hash.update(ids.astype('<i8').tobytes())
            ix = torch.from_numpy(ids).cuda(); optimizer.zero_grad(set_to_none=True)
            outputs = head(xt[ix])
            total, parts = model.supervision_loss(outputs, yc[ix], yf[ix], mask[ix], weights, cfg)
            if mode == 'joint' and step % 10 == 0:
                trunk = list(head.trunk.parameters())
                gc = torch.autograd.grad(parts['current'], trunk, retain_graph=True)
                gf = torch.autograd.grad(parts['future'], trunk, retain_graph=True)
                grad_rows.append(dict(epoch=epoch, step=step, **gradient_statistics(gc, gf)))
            total.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
            norm_rows.append(dict(epoch=epoch, step=step, norm=norm, clipped=norm > 1.,
                                  scale=min(1., 1. / max(norm, 1e-30))))
            optimizer.step(); losses.append(float(total.detach()))
        logits = model.predict_heads(head, arrays['val']['features'], mean, std)
        val = dict(current=None, future=None)
        for h in active:
            y = arrays['val'][h + '_labels']; keep = y >= 0
            val[h] = metrics.scores(logits[h + '_logits'][keep], y[keep])
        f1, nll = validation_selection(val['current'], val['future'], cfg)
        row = dict(epoch=epoch, updates=epoch * len(batches), train_loss=float(np.mean(losses)),
                   val=val, selection_f1=f1, selection_nll=nll)
        with (out / 'epochs.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        if (f1, -nll) > best:
            best = (f1, -nll); best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if epoch % 10 == 0:
            print(seed, mode, epoch, 'val', round(f1, 5), 'elapsed', round(time.monotonic()-began), flush=True)
    reports = {}
    for name, state in [('final', None), ('best_validation', best_state)]:
        if state is not None: head.load_state_dict(state)
        logits = model.predict_heads(head, arrays['val']['features'], mean, std)
        selected = {h + '_logits': logits[h + '_logits'] if h in active else None for h in ['current', 'future']}
        reports[name] = metrics.evaluate_heads(entries['val'], selected, cfg)
    # Research-only checkpoint is deliberately distinct from production format.
    torch.save(dict(diagnostic_only=True, head=best_state, mean=mean, std=std,
                    best_epoch=best_epoch, config=cfg), out / 'research_best.pt')
    for name, rr in [('gradients', grad_rows), ('clipping', norm_rows)]:
        with (out / (name + '.jsonl')).open('w') as f:
            for row in rr: f.write(json.dumps(row) + '\n')
    ds.write_json(out / 'result.json', dict(seed=seed, mode=mode, best_epoch=best_epoch,
        epochs=epochs, updates=len(norm_rows), order_sha256=order_hash.hexdigest(),
        duration_s=time.monotonic()-began, reports=reports,
        gradient_samples=len(grad_rows), clip_fraction=float(np.mean([v['clipped'] for v in norm_rows]))))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--output', type=Path, default=ds.REPO/'runs/tomato/controlled_20261108')
    args = p.parse_args()
    initialize_runtime(dict(device='cuda', seed=42))
    import torch
    torch.set_num_threads(4)
    base = ds.REPO/'runs/tomato/local93_20261107'
    cfg = ds.read_json(base/'joint/training_config.json')
    manifest, entries, arrays, hashes = read_inputs(base)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(seeds=args.seeds, modes=['current_only', 'future_only', 'joint'], epochs=args.epochs,
        early_stopping=False, split_sha256=manifest['split_sha256'], input_sha256=hashes,
        common_batches='joint valid/invalid distributed, seed+epoch; identical all modes',
        initialization='identical trunk and corresponding output head; identical dropout stream',
        gradient_sample='joint steps 0,10,20,... every epoch; same forward graph before total backward',
        clipping='all active parameters every update; preclip norm and multiplier',
        selection='original lambda-normalized validation macro-F1 then NLL; final fixed budget also reported',
        test_features_labels_predictions_opened=False)
    pp = args.output/'protocol.json'
    if pp.exists() and ds.read_json(pp) != protocol: raise ValueError('Protocol changed; use new output')
    ds.write_json(pp, protocol)
    for seed in args.seeds:
        for mode in protocol['modes']:
            out = args.output/f'seed{seed}_{mode}'
            if (out/'result.json').exists(): continue
            if out.exists(): raise ValueError(f'Incomplete run exists: {out}; use new output, do not overwrite')
            run_one(out, seed, mode, args.epochs, cfg, entries, arrays)
    ds.write_json(args.output/'complete.json', dict(runs=len(args.seeds)*3, test_evaluated=False))


if __name__ == '__main__': main()
