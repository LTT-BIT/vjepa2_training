"""Provenance, train-only statistics and fresh chronological inference acceptance."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from . import tomato_dataset as ds
from .tomato_runtime import initialize_runtime
from .tomato_frozen_protocol import verify
from .tomato_rgb_memory_inference import RGBMemorySession


def run(cache,root):
    initialize_runtime(dict(device='cuda',seed=42));cfg=ds.read_json(root/'protocol.json');lock=verify(Path(cfg['frozen_lock']));manifest=ds.read_json(lock['manifest']);index=ds.read_json(cache/'index.json');receipt=ds.read_json(root/'receipt.json');assert ds.digest(cache/'index.json')==receipt['identity']['cache_index_sha256'];assert ds.digest('configs/tomato_rgb_memory_train_20261116/protocol.json')==receipt['config_sha256'];cached=ds.read_json(cache/'receipt.json')
    for entry in index['episodes']:
        assert ds.digest(entry['path'])==entry['sha256']
    for p,sha in cached['inputs'].items():assert ds.digest(p)==sha
    for p,sha in cached['source_sha256'].items():assert ds.digest(p)==sha
    for p,sha in receipt['identity']['source_sha256'].items():assert ds.digest(p)==sha
    # Independently recompute raw measured train-only normalization.
    total=np.zeros(58);square=np.zeros(58);n=0;train=[e for e in manifest['episodes'] if e['split']=='train']
    for e in train:
        _,q=ds.read_state(e,index['joint_names'],manifest['contract']['joint_mode']);total+=q.sum(0,dtype=np.float64);square+=np.square(q.astype(np.float64)).sum(0);n+=len(q)
    mean=(total/n).astype('float32');std=np.sqrt(np.maximum(square/n-(total/n)**2,0)).clip(.01).astype('float32');stats=np.load(index['statistics_path']);np.testing.assert_array_equal(mean,stats['mean']);np.testing.assert_array_equal(std,stats['std']);assert set(index['statistics_training_sources'])=={e['source_recording_id'] for e in train}
    rows=[];results=ds.read_json(root/'results.json')
    for seed in cfg['seeds']:
        initial=[]
        for method in cfg['methods']:
            cp=root/f'seed{seed}'/f'{method}.pt';s=torch.load(cp,map_location='cpu',weights_only=True);assert ds.digest(cp)==results[str(seed)][method]['training']['checkpoint_sha256'];assert set(s['training_sources'])=={e['source_recording_id'] for e in train};np.testing.assert_array_equal(s['state_mean'].numpy(),mean);np.testing.assert_array_equal(s['state_std'].numpy(),std);initial.append(s['training']['initialization_sha256']);errors=[];latency=[];observations=0
            for entry in index['episodes']:
                if entry['split']!='val':continue
                eid=entry['episode_id'];z=np.load(entry['path']);pred=np.load(root/f'seed{seed}'/f'{method}_{eid}.npz');session=RGBMemorySession(cp,session_id=eid,episode_id=eid,device='cuda',offline_replay=bool(cfg.get('roi_uses_future_coordinates')));pc=[];pf=[]
                for i,t in enumerate(z['times']):
                    torch.cuda.synchronize();start=time.perf_counter();r=session.observe(z['visual'][i],z['positions'][i],z['metadata'][i],timestamp=float(t),arrival_timestamp=float(t)+.04,session_id=eid,episode_id=eid,cache_signature=index['signature']);torch.cuda.synchronize();latency.append((time.perf_counter()-start)*1000);assert r['valid'] and not r['suggests_prompt_update'];pc.append(r['current_probabilities']);pf.append(r['future_probabilities']);observations+=1
                for actual,expected in [(np.stack(pc),pred['current']),(np.stack(pf),pred['future'])]:np.testing.assert_allclose(actual,expected,atol=5e-6,rtol=1e-4);errors.append(float(abs(actual-expected).max()))
                # Invalid/stale/cross-session data cannot add a history window.
                count=len(session.rows);last=float(z['times'][-1]);bad=session.observe(z['visual'][-1],z['positions'][-1],z['metadata'][-1],timestamp=last+.5,arrival_timestamp=last+2,session_id=eid,episode_id=eid,cache_signature=index['signature']);assert not bad['valid'] and len(session.rows)==count
                bad=session.observe(z['visual'][-1],z['positions'][-1],z['metadata'][-1],timestamp=last+.5,arrival_timestamp=last+.54,session_id='other',episode_id=eid,cache_signature=index['signature']);assert not bad['valid'] and len(session.rows)==count
            rows.append(dict(seed=seed,method=method,checkpoint_sha256=ds.digest(cp),streamed_observations=observations,stream_batch_max_abs_error=max(errors),head_only_latency_ms=dict(p50=float(np.median(latency)),p95=float(np.percentile(latency,95)),max=float(max(latency))),train_only_normalization_exact=True,train_dev_source_overlap=0))
        assert len(set(initial))==1
    ds.write_json(root/'verification.json',dict(models=rows,normalization_train_raw_frames=n,paired_initializations_identical=True,chronological_stream_batch_parity=True,strict_online_causality_validated=not bool(cfg.get('roi_uses_future_coordinates')),recorded_roi_uses_future_coordinates=bool(cfg.get('roi_uses_future_coordinates')),latency_scope='local host frozen-feature bank to current/future probabilities; no video encoding or network',historical_test_evaluated=False));print('VERIFIED',len(rows),'models')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True,type=Path);p.add_argument('--run',required=True,type=Path);a=p.parse_args();run(a.cache,a.run)
