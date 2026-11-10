"""Readable playback of saved observations and memory; no model or robot calls."""
import argparse
from pathlib import Path
from . import tomato_dataset as ds
from .tomato_state_recovery import PHASES


def rows(source,episode,scenario,start,end):
    replay=ds.read_json(source/'seed42'/scenario/'replay.json')[episode]
    for i,row in enumerate(replay['input_trace']):
        t=row['schedule_time']
        if not start<=t<=end:continue
        v=replay['views']['M2'][i]
        yield dict(time=t,received=len(row['messages']),B0=PHASES[replay['phases']['M0'][i]],memory=PHASES[v['confirmed_phase']],candidate=PHASES[v['candidate_phase']] if v['candidate_phase'] is not None else None,flags=v['uncertainty'],physical_completion=v['progress']['physical_completion'],suggests_prompt_update=v['suggests_prompt_update'],memory_version=v['memory_version'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--episode',required=True);p.add_argument('--scenario',default='normal');p.add_argument('--start',type=float,default=0.);p.add_argument('--end',type=float,default=float('inf'));a=p.parse_args()
    for r in rows(a.run,a.episode,a.scenario,a.start,a.end):
        print(f"t={r['time']:7.2f}s  B0={r['B0']:12s} memory={r['memory']:12s} pending={str(r['candidate']):12s} flags={r['flags']} physical={r['physical_completion']} prompt_update={r['suggests_prompt_update']}")
