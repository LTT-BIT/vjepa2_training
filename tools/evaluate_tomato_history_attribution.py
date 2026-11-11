"""H0/H1/H2/H3: fixed prior, fixed emissions, only history information differs."""
from pathlib import Path
import argparse
import subprocess
import numpy as np
from . import tomato_dataset as ds
from .evaluate_tomato_complete100_history import freeze_dataset, score_arm, distribution, SOURCE
from .evaluate_tomato_state_recovery import summarize

ARMS={'H0':'none','H1':'anchor','H2':'oracle','H3':'predicted'}


def run(out):
    if out.exists():raise FileExistsError('New output required')
    m,lock=freeze_dataset();out.mkdir(parents=True);ds.write_json(out/'dataset_lock.json',lock)
    old=Path('runs/tomato/complete100_history_20261111_final');frozen=ds.read_json(old/'results.json');results={};equalities=[];lat={a:[] for a in ARMS};events=[];inputs={}
    for seed in (42,43,44):
        model=frozen[str(seed)]['model'];assert ds.digest(model['path'])==model['sha256'];inputs[model['path']]=model['sha256']
        per={a:{} for a in ARMS}
        for e in m['episodes']:
            if e['split']!='val':continue
            path=SOURCE/f'seed{seed}'/(e['episode_id']+'.npz');assert ds.digest(path)==ds.read_json(old/'receipt.json')['inputs'][str(path)];inputs[str(path)]=ds.digest(path);z=np.load(path)
            for a,mode in ARMS.items():
                per[a][e['episode_id']],cpu=score_arm(e,z,mode);lat[a]+=cpu
            eid=e['episode_id'];h0,h1,h2,h3=[per[a][eid] for a in ARMS]
            equalities.append(dict(seed=seed,episode_id=eid,observations=len(z['times']),H1_H2_trace_equal=h1['trace']==h2['trace'],H0_H3_trace_equal=h0['trace']==h3['trace']))
            for i in range(6):events.append(dict(seed=seed,episode_id=eid,ordinal=i+1,**{a:per[a][eid]['prompt']['events'][i] for a in ARMS}))
        summary={a:summarize(rows) for a,rows in per.items()}
        for a,rows in per.items():
            mat=np.sum([r['position_confusion'] for r in rows.values()],axis=0);s=summary[a]
            s.update(position_accuracy=float(mat.trace()/mat.sum()),position_confusion=mat.tolist(),
                     error_run_duration_s=distribution([v['duration_s'] for r in rows.values() for v in r['state']['runs']]),
                     maximum_error_run_s=max(v['duration_s'] for r in rows.values() for v in r['state']['runs']),
                     prompt_offset_s=distribution([v['offset_s'] for r in rows.values() for v in r['prompt']['events'] if v['offset_s'] is not None]))
        assert all(v==frozen[str(seed)]['summary']['none']['warnings'][k] for k,v in summary['H0']['warnings'].items())
        assert summary['H0']['mismatch_s']==frozen[str(seed)]['summary']['none']['mismatch_s']
        results[str(seed)]=summary;ds.write_json(out/f'seed{seed}_replay.json',per)
    aggregate={a:{key:dict(values=v,mean=float(np.mean(v)),std=float(np.std(v))) for key,v in {
        'position_accuracy':[r[a]['position_accuracy'] for r in results.values()],
        'mismatch_s':[r[a]['mismatch_s'] for r in results.values()],
        'persistent_mismatch_s':[r[a]['persistent_mismatch_s'] for r in results.values()],
        'on_time_prompts':[r[a]['prompt_counts'].get('on_time',0) for r in results.values()]}.items()} for a in ARMS}
    ds.write_json(out/'results.json',results);ds.write_json(out/'aggregate.json',aggregate);ds.write_json(out/'trace_equalities.json',equalities);ds.write_json(out/'event_comparison.json',events)
    ds.write_json(out/'receipt.json',dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),manifest_sha256=lock['manifest_sha256'],arms=ARMS,
        input_sha256=inputs,old8_results_sha256=ds.digest(old/'results.json'),training=False,test_values_read=False,encoder_calls=0,robot_calls=0,
        oracle_scope='Offline only. Previous observation label, no current truth/future boundaries/end times; same one-step forward-only state correction in H1/H2.',
        cpu_ms={a:distribution(v) for a,v in lat.items()},all_H1_H2_equal=all(r['H1_H2_trace_equal'] for r in equalities),all_H0_H3_equal=all(r['H0_H3_trace_equal'] for r in equalities)))
    print('ATTRIBUTION',aggregate,'H1/H2 equal',all(r['H1_H2_trace_equal'] for r in equalities))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
