"""Explicit EXPLORATORY val-only V2 experiment. No test scores or tuning."""
import argparse,json,sys,csv
from functools import lru_cache
from pathlib import Path
from .handshake import replay,DEFAULT
from .catalog import Catalog,read,write,sha
from .timeline import validate_rows,tracks
from .prediction_store import sha_bytes

def score(e,r):
    gt=e['annotation']['segments'];fps=e['fps'];commits=[v for v in r['events'] if v['kind']=='commit'];bs=[dict(time=b['start_frame']/fps,pair=(a['label'],b['label'])) for a,b in zip(gt,gt[1:])]
    @lru_cache(None)
    def dp(i,j):
        if i==len(bs) or j==len(commits):return ()
        opts=[dp(i+1,j),dp(i,j+1)];delta=commits[j]['time']-bs[i]['time']
        if bs[i]['pair']==(commits[j]['from_stage'],commits[j]['to_stage']) and -1-1e-9<=delta<=2+1e-9:opts.append(((i,j),)+dp(i+1,j+1))
        return min(opts,key=lambda z:(-len(z),sum(abs(commits[y]['time']-bs[x]['time']) for x,y in z),z))
    pairs=dp(0,0);d=[commits[j]['time']-bs[i]['time'] for i,j in pairs];wrong=0
    for a in r['active_intervals']:
        for g in gt:
            if a['stage']!=g['label']:wrong+=max(0,min(a['end_s'],g['end_frame']/fps)-max(a['start_s'],g['start_frame']/fps))
    transitions=[]
    for i,b in enumerate(bs):
        j=dict(pairs).get(i);ce=commits[j] if j is not None else None
        ack=next((v for v in r['events'] if ce and v['kind']=='ack' and v['commit_at']==ce['time']),None)
        transitions.append(dict(boundary_index=i,gt_boundary_time=b['time'],from_stage=b['pair'][0],to_stage=b['pair'][1],matched=ce is not None,
            commit_time=ce['time'] if ce else None,commit_delta_s=ce['time']-b['time'] if ce else None,ack_time=ack['time'] if ack else None,
            first_future_candidate_time=ce['evidence']['first_candidate_at'] if ce else None,pending_ready_time=ce['evidence']['first_qualified_at'] if ce else None,
            request_time=ce['evidence']['request_at'] if ce else None,return_occurrence=sum(g['label']=='return' for g in gt[:i+2]) if b['pair'][1]=='return' else None,
            final_return=b['pair'][1]=='return' and i==len(bs)-1,short_return=b['pair'][1]=='return' and (gt[i+1]['end_frame']-gt[i+1]['start_frame'])/fps<=1.5))
    actual=['pickup']+[c['to_stage'] for c in commits]
    def percentile(v,p):
        if not v:return None
        a=sorted(v);x=(len(a)-1)*p;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
    return dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],matched=len(pairs),eligible=len(bs),missed=len(bs)-len(pairs),unmatched_commits=len(commits)-len(pairs),
        early_gt02=sum(x<-.2 for x in d),early_ge05=sum(x<=-.5 for x in d),strict_hits=sum(-.2<=x<=.5 for x in d),deltas=d,
        wrong_active_s=wrong,exact_order=actual==[g['label'] for g in gt],short_return_missed=sum(i not in dict(pairs) and g['label']=='return' and (g['end_frame']-g['start_frame'])/fps<=1.5 for i,g in enumerate(gt[1:])),
        transitions=transitions,late_gt05=sum(x>.5 for x in d),duplicate_commits=max(0,len(commits)-len(bs)),illegal_candidates=sum(v['kind']=='illegal_candidate' for v in r['events']),canceled=sum(v['kind']=='cancel' for v in r['events']),stale_canceled=sum(v['kind']=='cancel' and v['reason']=='stale_evidence' for v in r['events']),**r['summary'],ack_waits=[dict(commit_at=v['commit_at'],ack_at=v['time'],committed_to_ack=v['time']-v['commit_at']) for v in r['events'] if v['kind']=='ack'],waits=[dict(prepare_to_request=c['evidence']['request_at']-c['evidence']['first_available_at'],request_to_commit=c['time']-c['evidence']['request_at'],first_candidate_at=c['evidence']['first_candidate_at'],first_qualified_at=c['evidence']['first_qualified_at'],first_observed_at=c['evidence']['first_observed_at'],first_available_at=c['evidence']['first_available_at'],last_confirm_available_at=c['evidence']['last_confirm_available_at'],request_at=c['evidence']['request_at'],commit_at=c['time']) for c in commits])

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    root=a.run_root;c=Catalog(root);protected={str(p):sha(p) for m in ['current_only','future_only','joint'] for p in (root/m).rglob('*') if p.is_file()}
    candidates=[dict(name=f'H{(2 if k==2 else 1)+(2 if ack else 0)}_t{t}_ack{timeout}',config={**DEFAULT,'K':k,'ack_enabled':ack,'theta_commit':t,'ack_timeout_s':timeout}) for t in [.8,.9,.95] for k in [1,2] for ack in [False,True] for timeout in [1.5]]
    candidates += [dict(name=f'H{(2 if k==2 else 1)+2}_t0.8_ack{timeout}',config={**DEFAULT,'K':k,'ack_enabled':True,'ack_timeout_s':timeout}) for k in [1,2] for timeout in [1.,2.]]
    candidates += [dict(name='BASE_H0_raw',config={**DEFAULT,'theta_prepare':0.,'theta_commit':0.,'ack_enabled':False,'legal_edges':False}),dict(name='LEGAL_OFF_K2_ACK',config={**DEFAULT,'K':2,'legal_edges':False})]
    write(out/'protocol.lock.json',dict(status='EXPLORATORY_VAL_ONLY',candidates=candidates,predictions_sha256=sha(root/'joint/predictions.jsonl'),checkpoint_sha256=sha(root/'joint/best_head.pt'),five_hz='NOT_RUN',camera_degradation='NOT_RUN',test_scored=False,source_sha256=sha(Path(__file__).with_name('handshake.py'))))
    groups={e:[] for e in c.episodes}
    for line in (root/'joint/predictions.jsonl').open():
        v=json.loads(line);groups[v['episode_id']].append(v)
    clean={e:validate_rows(v,c.get(e),'joint') for e,v in groups.items()};allmetrics=[]
    episode_csv=[];transition_csv=[];failures=[]
    for candidate in candidates:
        rr=[];directory=out/candidate['name'];directory.mkdir()
        for e in c.episodes.values():
            if e['split']!='val':continue
            r=replay(clean[e['episode_id']],candidate['config'],e['length']/e['fps']);m=score(e,r);rr.append(m)
            episode_csv.append(dict(candidate=candidate['name'],**{k:v for k,v in m.items() if not isinstance(v,list)}))
            transition_csv.extend(dict(candidate=candidate['name'],episode_id=e['episode_id'],**t) for t in m['transitions'])
            failures.extend(dict(candidate=candidate['name'],episode_id=e['episode_id'],kind='missed_or_early',**t) for t in m['transitions'] if not t['matched'] or t['commit_delta_s']<-.2)
            failures.extend({**v,'policy_name':candidate['name'],'episode_id':e['episode_id']} for v in r['events'] if v['kind'] in ['timeout','illegal_candidate','cancel'])
            write(directory/(e['episode_id']+'.json'),dict(replay=r,metrics=m))
        deltas=sorted(x for v in rr for x in v['deltas'])
        def quant(p):
            x=(len(deltas)-1)*p;i=int(x);return deltas[i]+(deltas[min(i+1,len(deltas)-1)]-deltas[i])*(x-i) if deltas else None
        totals={k:sum(v[k] for v in rr) for k in ['matched','eligible','missed','unmatched_commits','early_gt02','early_ge05','strict_hits','wrong_active_s','exact_order','commits','acks','timeouts','short_return_missed','external_recoveries']}
        allmetrics.append(dict(name=candidate['name'],**totals,delta_p50=quant(.5),delta_p95=quant(.95),config=candidate['config']))
    write(out/'val_summary.json',allmetrics)
    for name,values in [('per_episode',episode_csv),('per_transition',transition_csv)]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in values for k in v)));writer.writeheader();writer.writerows(values)
    (out/'failure_cases.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in failures))
    from .video_export import export_video
    for length in [7,3]:
        e=min((e for e in c.episodes.values() if len(e['annotation']['segments'])==length),key=lambda e:e['length']);rows=clean[e['episode_id']]
        sim=replay(rows,DEFAULT,e['length']/e['fps']);sid=sha_bytes(dict(config=DEFAULT,rows=rows));sim.update(simulation_id=sid,config_sha256=sha_bytes(DEFAULT))
        result=dict(episode_id=e['episode_id'],prediction_mode='joint',result_id=sid,provenance=dict(checkpoint_sha256=sha(root/'joint/best_head.pt')),rows=rows,tracks=tracks(rows,e,'joint'),handshake=sim)
        write(out/f'sample_{length}_timeline.json',sim);video=export_video(c,result,out)
        write(out/f'sample_{length}_video.json',video)
    assert all(sha(p)==h for p,h in protected.items());write(out/'integrity.json',dict(original_files=len(protected),unchanged=True,torch_imported='torch' in sys.modules))
    print(out)
if __name__=='__main__':main()
