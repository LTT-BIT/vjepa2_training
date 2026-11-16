"""Read-only baseline validation and immutable experiment identity."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import yaml
from tools import tomato_dataset as ds,tomato_phase_model as baseline
from tools.train_tomato_vjepa_phase import feature_contract

REPO=Path(__file__).resolve().parents[2]
LABELS=list(ds.LABELS)

def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return ds.digest(p)
def digest_json(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def locked(p,v):
    p=Path(p)
    if p.exists() and read(p)!=v:raise ValueError(f'Immutable lock mismatch: {p}')
    if not p.exists():write(p,v)
def config(path):
    c=yaml.safe_load(Path(path).read_text())
    for k in ['baseline_root','baseline_config','pretrained_dir','output_root']:
        p=Path(c[k]);c[k]=str(p if p.is_absolute() else REPO/p)
    if c['seeds']!=[42,43,44] or c['groups']!=['B0R','T0','T1','T2','K1','P1']:raise ValueError('Unexpected main protocol')
    if c['prediction_hz']!=2 or c['future_horizon_s']!=1 or c['lambda_current']!=1 or c['lambda_future']!=.5:raise ValueError('Joint target contract')
    return c

def validate_baseline(c,out):
    """Rehash every registered source; do not prepare/extract/write baseline labels."""
    out=Path(out);root=Path(c['baseline_root']);run=root/'joint';m=read(run/'manifest.json')
    idx=read(run/'feature_index.json');contract=m['contract'];splits=read(run/'split.json')
    if len(m['episodes'])!=93 or {s:len(v) for s,v in splits.items()}!={'train':75,'val':9,'test':9}:raise ValueError('Not locked LOCAL93')
    if splits!=m['splits'] or sha(run/'split.json')!=m['split_sha256']:raise ValueError('Split mismatch')
    if contract['cameras']!=['cam0','cam1','cam2'] or contract['labels']!=LABELS or len(contract['joint_names'])!=58 or len(set(contract['joint_names']))!=58 or any(not n.endswith('.position') for n in contract['joint_names']):raise ValueError('Not three full views / 58 measured positions')
    for k,v in dict(image_size=256,history_frames=16,history_sample_hz=10,prediction_hz=2,variant='normal',robot='sharpa').items():
        if contract[k]!=v:raise ValueError(f'Contract mismatch {k}')
    if idx['manifest_sha256']!=sha(run/'manifest.json') or idx['pretrained']!=m['pretrained']:raise ValueError('Cache manifest binding')
    receipt=read(run/'validation_receipt.json');lock=read(run/'validation_lock.json')
    if lock!=dict(receipt_sha256=sha(run/'validation_receipt.json'),manifest_sha256=sha(run/'manifest.json'),split_file_sha256=sha(run/'split.json')) or receipt['full_media_validation'] is not True:raise ValueError('Validation receipt')
    saved,_=baseline.load_checkpoint(run/'best_head.pt',sha(run/'manifest.json'),split_sha256=m['split_sha256'])
    if saved['prediction_mode']!='joint':raise ValueError('Not Joint')
    baseline.verify_pretrained(c['pretrained_dir'])
    paths={r['episode_id']:Path(r['path']) for r in idx['episodes']}
    if set(paths)!={e['episode_id'] for e in m['episodes']} or len(paths)!=93:raise ValueError('Cache membership')
    sources=set();assets={};entries=[];protected={};identities=[];counts={s:0 for s in splits}
    for e in m['episodes']:
        sid=e['source_recording_id']
        if sid in sources or sid not in splits[e['split']]:raise ValueError('Source leakage')
        sources.add(sid)
        actual=ds.load_episode_info(e['dataset_root'],e['episode_index'])
        if any(actual[k]!=e[k] for k in actual):raise ValueError('Source metadata changed')
        ann=ds.load_reviewed_coarse(e['annotation']['path'],actual,m['allow_subset_metadata_mismatch'])
        if ann!=e['annotation']:raise ValueError('Annotation changed')
        baseline.assert_source_unchanged(e)
        source_hashes={}
        for rel,expected in e['input_sha256'].items():
            p=Path(e['dataset_root'])/rel;source_hashes[str(p)]=expected;protected[str(p)]=expected
            if expected in assets:raise ValueError(f'Duplicate video/Parquet content: {p}, {assets[expected]}')
            assets[expected]=str(p)
        for r in ann['source_hashes']:
            p=Path(e['dataset_root'])/r['path'];protected[str(p)]=r['actual']
        protected[ann['path']]=sha(ann['path'])
        data=baseline.load_cache(paths[e['episode_id']],e,feature_contract(m),m['pretrained'])
        times,q=ds.read_state(actual,contract['joint_names'],contract['joint_mode'])
        if not np.allclose(times,np.arange(len(times))/e['fps'],atol=1e-4,rtol=0) or e['fps']!=30:raise ValueError('Irregular timestamps/fps')
        if not np.array_equal(data['features'][:,3072:],q[data['input_indices']].reshape(-1,928).astype(np.float32)):raise ValueError('Cached joints not measured causal history')
        targets=ds.build_phase_targets(data['anchors'],ann['segments'],e['length'],e['fps'],1.)
        entries.append(dict(episode=e,**{**data,**targets}))
        counts[e['split']]+=len(data['anchors'])
        identities.append(dict(episode_id=e['episode_id'],source_recording_id=sid,split=e['split'],fps=e['fps'],frames=e['length'],anchors=len(data['anchors']),future_invalid=int((~targets['future_valid_mask']).sum()),video_parquet_sha256=source_hashes,annotation_sha256=ann['sha256'],cache_path=str(paths[e['episode_id']]),cache_sha256=sha(paths[e['episode_id']]),max_timestamp_grid_error_s=float(np.max(abs(times-np.arange(len(times))/e['fps']))),pts_validation=e['media_validation']))
        print('identity',len(entries),e['episode_id'],flush=True)
    if counts!={'train':21706,'val':3007,'test':2580}:raise ValueError(f'Anchor deviation: {counts}')
    # Original reports/checkpoints/caches and selected raw sources are all protected.
    for p in root.rglob('*'):
        if p.is_file():protected[str(p.resolve())]=sha(p)
    for p in [REPO/'tools/train_tomato_vjepa_phase.py',REPO/'tools/tomato_phase_model.py',REPO/'tools/tomato_dataset.py',REPO/'tools/tomato_review_ui/handshake.py',Path(c['baseline_config']),Path(c['pretrained_dir'])/'model.safetensors']:
        protected[str(p.resolve())]=sha(p)
    write(out/'baseline_inventory.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),baseline_checkpoint_sha256=sha(run/'best_head.pt'),summary=m['summary'],feature_dim=4000,encoder_new_calls=0,verified_sources=93,source_bytes=sum(Path(p).stat().st_size for p in protected),existing_full_decode_receipt_sha256=sha(run/'validation_receipt.json'),timestamp_check='actual Parquet all frames; cached indices and measured q verified; unchanged video SHA binds original full PTS decode receipt'))
    locked(out/'split.lock.json',splits);locked(out/'feature_contract.lock.json',dict(contract=contract,pretrained=m['pretrained'],manifest_sha256=sha(run/'manifest.json'),feature_index_sha256=sha(run/'feature_index.json')))
    write(out/'source_identity.json',identities);write(out/'protected_sha256.json',protected)
    return entries,m

def load_entries(c):
    root=Path(c['baseline_root'])/'joint';m=read(root/'manifest.json');paths={r['episode_id']:r['path'] for r in read(root/'feature_index.json')['episodes']};out=[]
    for e in m['episodes']:
        with np.load(paths[e['episode_id']],allow_pickle=False) as f:data={k:f[k].copy() for k in ['features','anchors','times','input_indices']}
        targets=ds.build_phase_targets(data['anchors'],e['annotation']['segments'],e['length'],e['fps'],1.)
        out.append(dict(episode=e,**{**data,**targets}))
    return out,m

def verify(out):
    expected=read(Path(out)/'protected_sha256.json');bad=[p for p,h in expected.items() if not Path(p).is_file() or sha(p)!=h]
    result=dict(protected_files=len(expected),unchanged=not bad,differences=bad);write(Path(out)/'integrity_final.json',result)
    if bad:raise ValueError(f'Protected files changed: {bad}')
    return result
