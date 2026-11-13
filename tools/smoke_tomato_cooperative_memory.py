"""Read-only recorded-cache wiring/parity smoke, not outcome-model evaluation."""
from pathlib import Path
import json
import argparse
import time
import numpy as np
from tools.tomato_cooperative_memory import CooperativeMemory, Frame
from tools import tomato_dataset as ds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=ds.REPO/'runs/tomato/complete100_20261111/dataset_manifest.json')
    parser.add_argument('--history-root',type=Path,default=ds.REPO/'runs/tomato/complete100_history_20261111_final')
    parser.add_argument('--predictions-root',type=Path,default=ds.REPO/'runs/tomato/transition_time_20261109/seed42')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Fresh local receipt required')
    repo=ds.REPO
    manifest=args.manifest
    m=ds.read_json(manifest)
    identity=ds.read_json(manifest.parent/'data_identity.json')
    registered={r['episode_id']:r for r in identity['episodes']}
    old=ds.read_json(args.history_root/'seed42_per_episode.json')['none']
    model=ds.read_json(args.history_root/'results.json')['42']['model']
    assert ds.digest(repo/model['path'])==model['sha256']
    signature=ds.fingerprint(dict(encoder=ds.MODEL_ID,revision=ds.REVISION,
        pooling='last_hidden_state.mean(dim=1)',cameras=['cam0','cam1','cam2'],frames=16,sample_hz=10))
    total=0;timings=[];rows=[]
    for e in m['episodes']:
        if e['split']!='val':continue
        eid=e['episode_id'];r=registered[eid]
        assert ds.digest(r['feature_path'])==r['feature_sha256']
        cache=np.load(r['feature_path'])
        z=np.load(args.predictions_root/f'{eid}.npz')
        assert np.array_equal(cache['anchors'],z['anchors'])
        assert np.array_equal(cache['input_indices'][:,-1],z['anchors'])
        mem=CooperativeMemory(eid,eid,model['sha256'],m['contract']['joint_names'],signature)
        for i,t in enumerate(z['times']):
            x=cache['features'][i];q=x[3072:].reshape(16,58)[-1]
            frame=Frame(eid,eid,str(i),float(t),float(t)+.01,float(t),float(t),
                        x[:3072].reshape(3,1024),q,z['current_probabilities'][i],z['future_probabilities'][i])
            start=time.perf_counter_ns();got=mem.observe(frame,encoder_signature=signature);read=mem.read(now=float(t)+.01)
            timings.append((time.perf_counter_ns()-start)/1e6)
            assert got['decision']['phase']==old[eid]['trace'][i]['phase']
            assert read['result_status']=='unknown' and read['robot_action'] is None
            assert read['recent'][-1]['time']==float(t)
            total+=1
        rows.append(dict(episode_id=eid,observations=len(z['times']),phase_parity=True,
                         feature_sha256=r['feature_sha256'],maximum_recent_frames=mem.recent_size))
    result=dict(test='recorded_cache_wiring_only',validation_recordings=len(rows),observations=total,
        b0_phase_parity=True,independent_outcome_evidence=0,outcome_status='unknown',
        encoder_calls=0,training=False,robot_calls=0,performance_improvement_claim=False,
        manifest_sha256=ds.digest(manifest),baseline_checkpoint=model,
        cpu_increment_ms=dict(p50=float(np.median(timings)),p95=float(np.percentile(timings,95)),maximum=max(timings)),
        timing_scope='Python prototype update and deep-copy read; excludes encoder and action model',episodes=rows)
    out=args.out
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Fresh local receipt required')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Real cached observations:',total,'B0 phases unchanged; no physical result inferred')


if __name__=='__main__':main()
