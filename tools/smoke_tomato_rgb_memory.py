"""Real-video four-view wiring smoke with explicit OFFLINE manual ROI provenance.

Runs frozen encoding only, no training, no stage decisions, no robot calls.
Requires the existing OpenPI preview annotations as an explicit input file;
verifies the recording identity and every original camera SHA before use.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from tools import tomato_dataset as ds
from tools.tomato_interaction_rgb import crop_rgb, encode_four_views
from tools.tomato_phase_model import load_encoder
from tools.tomato_rgb_memory import RGBStateMemory, RGBStateWindow


def main():
    import av
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ds.REPO/'runs/tomato/complete100_20261111/dataset_manifest.json')
    parser.add_argument('--pretrained', type=Path, default=ds.REPO/'pretrained/vjepa2-vitl-fpc64-256')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new receipt directory')
    spec, manifest = ds.read_json(args.annotations), ds.read_json(args.manifest)
    matches = [e for e in manifest['episodes'] if
               f"{Path(e['dataset_root']).parent.name}/{Path(e['dataset_root']).name}/episode_{e['episode_index']:06d}"
               == spec['source_recording']]
    if len(matches) != 1:
        raise ValueError('Exact preview source recording not found uniquely')
    e = matches[0]
    if e['split'] != 'train' or e['fps'] != 30 or spec['fps'] != 30:
        raise ValueError('Only known training recording at30Hz for this smoke')
    for cam, path in zip(ds.CAMERAS, e['videos']):
        if ds.digest(path) != spec['source_sha256'][cam]:
            raise ValueError('Preview source camera hash mismatch')
    for relative, digest in e['input_sha256'].items():
        if ds.digest(Path(e['dataset_root'])/relative) != digest:
            raise ValueError('Manifest source identity changed')
    times, q = ds.read_state(e, manifest['contract']['joint_names'])
    # Two successive windows in each existing preview interval. No extrapolation.
    requests = []
    for clip in spec['clips']:
        first = ((clip['start']+45+14)//15)*15
        for anchor in (first, first+15):
            if anchor >= clip['end']:
                raise ValueError('Preview too short for two complete windows')
            requests.append((clip, np.arange(anchor-45, anchor+1, 3)))
    wanted = set(int(i) for _, indices in requests for i in indices)
    decoded = []
    for path in e['videos']:
        frames = {}
        with av.open(path) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i in wanted:
                    frames[i] = frame.to_ndarray(format='rgb24')
                if i >= max(wanted):
                    break
        if set(frames) != wanted:
            raise ValueError('Missing real source frames')
        decoded.append(frames)
    encoder, provenance = load_encoder(args.pretrained, device=args.device)
    signature = ds.fingerprint(dict(encoder=provenance, image_size=256,
        frames=16, sample_hz=10, views=[*ds.CAMERAS, 'interaction'], pooling='mean'))
    m = RGBStateMemory('offline-smoke', e['episode_id'], manifest['contract']['joint_names'],
        signature, ds.digest(args.annotations), mode='offline_diagnostic')
    features, receipts = [], []
    for clip, indices in requests:
        local, boxes, available = [], [], []
        keys = clip['keyframes']; keyframes = [k['frame'] for k in keys]
        if keyframes != sorted(set(keyframes)) or keyframes[0] > indices[0] or keyframes[-1] < indices[-1]:
            raise ValueError('Ordered annotation coverage required; never extrapolate')
        for i in indices:
            components = []
            for role in clip['roles']:
                components.append([np.interp(i, keyframes, [k['boxes'][role][axis] for k in keys])
                                   for axis in range(4)])
            components = np.asarray(components)
            tight = [*components[:, :2].min(0), *components[:, 2:].max(0)]
            native = decoded[0][int(i)]
            if [native.shape[1], native.shape[0]] != spec['source_size']:
                raise ValueError('Source dimensions mismatch')
            crop = crop_rgb(native, tight)
            local.append(crop['rgb']); boxes.append(crop['crop_box'])
            available.append(next(k for k in keyframes if k >= i)/30.)
        global_rgb = np.asarray([[cv2.resize(cam[int(i)], (256, 256), interpolation=cv2.INTER_AREA)
                                  for i in indices] for cam in decoded], dtype=np.uint8)
        start = time.perf_counter()
        visual = encode_four_views(encoder, global_rgb[None], np.asarray(local)[None])[0]
        elapsed = (time.perf_counter()-start)*1000
        # Use nominal frame timeline; verify measured timestamps match it, allowing
        # recorded float32 timestamp rounding but not future state measurements.
        timeline = indices/30.
        if not np.allclose(times[indices], timeline, atol=1e-4):
            raise ValueError('Measured state timeline differs')
        w = RGBStateWindow('offline-smoke', e['episode_id'], str(indices[-1]), signature,
            ds.digest(args.annotations), float(timeline[-1]+.01), timeline, timeline,
            visual[:3], visual[3], q[indices], boxes, np.ones(16, dtype=bool), available,
            'cam0', tuple(spec['source_size']), 'manual_offline')
        stored = m.observe(w); snapshot = m.read(now=timeline[-1]+.01)
        np.testing.assert_array_equal(snapshot['rows'][-1]['positions'], q[indices])
        features.append(visual)
        receipts.append(dict(clip=clip['name'], anchor=int(indices[-1]), indices=indices.tolist(),
            four_view_encoder_ms=elapsed, gap_reset=stored['gap_reset'],
            retained_windows=len(snapshot['rows']), roi_mode='manual_offline', state_shape=[16, 58]))
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out/'features.npz', visual=np.asarray(features))
    receipt = dict(scope='real-video frozen four-view encoding and full-state memory wiring only',
        manifest_sha256=ds.digest(args.manifest), annotation_sha256=ds.digest(args.annotations),
        episode_id=e['episode_id'], split=e['split'], source_recording=spec['source_recording'],
        source_sha256=spec['source_sha256'], encoder=provenance, device=args.device,
        feature_sha256=ds.digest(args.out/'features.npz'), windows=receipts,
        visual_shape=[len(features), 4, 1024], training=False, robot_calls=0,
        automatic_roi=False, physical_success_inferred=False, performance_improvement_claim=False,
        timing_scope='encoding helper including preprocessing/transfers; excludes decode, ROI, policy and capture',
        state_timestamp_handling='verified recorded float32 times against nominal30Hz; represented on nominal timeline')
    ds.write_json(args.out/'receipt.json', receipt)
    print(f'{len(features)} real four-view windows encoded; manual offline ROI only, no training')


if __name__ == '__main__':
    main()
