"""Use actual production state machine, fixed clocks and policies for every model."""
import numpy as np
from tools.tomato_review_ui.handshake import replay,DEFAULT
from tools.tomato_metrics import probabilities
from .contracts import LABELS,write
from .evaluate import boundaries,match_events

C0={**DEFAULT,'late_ack_enabled':True,'late_ack_timeout_s':5.}
C1={**C0,'ack_enabled':False,'late_ack_enabled':False}

def score(entry,result):
    e=entry['episode'];gt=boundaries(e);duration=e['length']/e['fps'];commits=[dict(ev,time=ev['time']) for ev in result['events'] if ev['kind']=='commit']
    strict=match_events(gt,commits,-.2,.5);wide=match_events(gt,commits,-1,2);used=dict(wide);deltas=[commits[j]['time']-gt[i]['time'] for i,j in wide];wrong=0.
    for a in result['active_intervals']:
        for g in e['annotation']['segments']:
            if a['stage']!=g['label']:wrong+=max(0,min(a['end_s'],g['end_frame']/e['fps'])-max(a['start_s'],g['start_frame']/e['fps']))
    events=[]
    for i,b in enumerate(gt):
        j=used.get(i);events.append(dict(**b,wide_matched=j is not None,strict_matched=i in dict(strict),commit_at=commits[j]['time'] if j is not None else None,delta_s=commits[j]['time']-b['time'] if j is not None else None))
    evs=result['events'];recov=0.;stuck=0.
    for i,ev in enumerate(evs):
        end=evs[i+1]['time'] if i+1<len(evs) else duration
        if ev['policy_state']=='RECOVERY_REQUIRED':recov+=max(0,end-ev['time'])
        if ev['policy_state'] in ['WAIT_CURRENT_ACK','WAIT_LATE_ACK','RECOVERY_REQUIRED']:stuck+=max(0,end-ev['time'])
    return dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],strict_hits=len(strict),wide_hits=len(wide),eligible=len(gt),unmatched_commits=len(commits)-len(wide),early_ge025=sum(d<=-.25 for d in deltas),early_ge05=sum(d<=-.5 for d in deltas),late_gt05=sum(d>.5 for d in deltas),late_gt1=sum(d>1 for d in deltas),wrong_active_s=wrong,duration_s=duration,active_agreement=1-wrong/duration,exact_order=['pickup']+[v['to_stage'] for v in commits]==[s['label'] for s in e['annotation']['segments']],pending_canceled=sum(v['kind']=='cancel' for v in evs),recovery_required_s=recov,ack_locked_s=stuck,short_return_missed=sum(v['to_stage']=='return' and v['short_to'] and not v['wide_matched'] for v in events),events=events,**result['summary'])

def compare(entries,pred,out):
    result={}
    for name,cfg in [('C0',C0),('C1',C1)]:
        reports=[]
        for entry in entries:
            eid=entry['episode']['episode_id'];pp={h:probabilities(pred[eid][h]) for h in ['current','future']}
            # Whitelist only observation timestamps and the paired probability vectors.
            rows=[dict(observed_at=float(t),heads={h:dict(probabilities=dict(zip(LABELS,pp[h][i].tolist()))) for h in pp}) for i,t in enumerate(entry['times'])]
            rr=replay(rows,cfg,entry['episode']['length']/entry['episode']['fps']);ss=score(entry,rr);reports.append(ss)
            write(out/name/(eid+'.json'),dict(replay=rr,metrics=ss))
        sums={k:sum(r.get(k,0) for r in reports) for k in ['strict_hits','wide_hits','eligible','unmatched_commits','early_ge025','early_ge05','late_gt05','late_gt1','wrong_active_s','duration_s','exact_order','pending_canceled','recovery_required_s','ack_locked_s','short_return_missed','commits','acks','late_acks','timeouts','hard_timeouts']}
        sums.update(active_agreement=1-sums['wrong_active_s']/sums['duration_s'],episodes=reports,config=cfg);result[name]=sums
    return result
