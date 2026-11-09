"""Paired event changes and compute-availability sensitivity, no selection search."""
from collections import Counter
import numpy as np
from . import tomato_dataset as ds
from .audit_tomato_transitions import ROOT,MODELS,match_events


def run():
    comparison=ds.read_json(ROOT/'comparison.json');runtime=ds.read_json(ROOT/'extraction_runtime.json');pairs={};availability={}
    for candidate in ['independent_h15_4hz_k2','independent_h15_4hz_k3']:
        baseline=comparison['independent_h15_2hz_k2'];new=comparison[candidate];result={}
        for head in ['current','future']:
            counts=Counter();deltas=[];by_episode={}
            for eid,b in baseline['per_episode'].items():
                c=new['per_episode'][eid];local=Counter()
                for old,now in zip(b[head]['boundaries'],c[head]['boundaries']):
                    label=('hit' if old['detected'] else 'miss')+'->'+('hit' if now['detected'] else 'miss');counts[label]+=1;local[label]+=1
                    if old['detected'] and now['detected']:deltas.append(now['event_time']-old['event_time'])
                by_episode[eid]=dict(local)
            result[head]=dict(paired_counts=dict(counts),common_hits=len(deltas),median_new_minus_old_detection_time_s=float(np.median(deltas)) if deltas else None,
                              mean_new_minus_old_detection_time_s=float(np.mean(deltas)) if deltas else None,by_episode=by_episode)
        pairs[candidate]=result
    for name,r in comparison.items():
        hz='4' if '_4hz_' in name else '2';latency=runtime['runtime'][hz]['latency']['p95_ms']/1000;horizon=MODELS[name.rsplit('_',2)[0]][1];result={}
        for head in ['current','future']:
            hits=0;false=0;total=0
            for eid,e in r['per_episode'].items():
                original=e[head];truth=[{k:v[k] for k in ['time','end','previous','label']} for v in original['boundaries']]
                shifted=[dict(v,time=v['time']+latency) for v in original['events']]
                matched,unmatched=match_events(shifted,truth,head,horizon);hits+=sum(v['detected'] for v in matched);false+=len(unmatched);total+=len(truth)
            result[head]=dict(hits=hits,total=total,recall=hits/total,false_events=false,assumed_compute_delay_s=latency)
        availability[name]=result
    ds.write_json(ROOT/'paired_rates.json',dict(pairs=pairs,availability_p95_sensitivity=availability,
        note='Same56 truth boundaries, fixed operating points, no parameter tuning. Availability adds measured selected-head GPU-path P95 to ALL model events as a sensitivity scenario, not per-request measured latency. Capture/decode/transport and physical confirmation not included.'))
    print(pairs)


if __name__=='__main__':run()
