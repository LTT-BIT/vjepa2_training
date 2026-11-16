"""Post-hoc GT-only continuous-time scoring, monotone one-to-one matching and CI."""
from functools import lru_cache
import math
import random
from .tomato_prompt_policy import LABELS, EDGES


def percentile(values,p):
    if not values:return None
    a=sorted(values);x=(len(a)-1)*p/100;i=int(x);j=min(i+1,len(a)-1)
    return a[i]*(j-x)+a[j]*(x-i) if i!=j else a[i]


def phases(episode):
    result=[]
    for s in episode['annotation']['segments']:
        a=dict(stage=s['label'],start_s=s['start_frame']/episode['fps'],end_s=s['end_frame']/episode['fps'])
        if result and result[-1]['stage']==a['stage']:result[-1]['end_s']=a['end_s']
        else:result.append(a)
    if not result or result[0]['start_s']!=0 or abs(result[-1]['end_s']-episode['length']/episode['fps'])>1e-7:
        raise ValueError('Incomplete reviewed timeline')
    for a,b in zip(result,result[1:]):
        if abs(a['end_s']-b['start_s'])>1e-7:raise ValueError('Annotation gap')
    return result


def match(boundaries,commits,window=(-1.,2.)):
    """Maximize matches, then minimize absolute delta, then lexicographic pair indices."""
    @lru_cache(None)
    def solve(i,j):
        if i==len(boundaries) or j==len(commits):return ()
        options=[solve(i+1,j),solve(i,j+1)]
        b,c=boundaries[i],commits[j];delta=c['time']-b['time']
        if (b['from_stage'],b['to_stage'])==(c['from_stage'],c['to_stage']) and window[0]-1e-9<=delta<=window[1]+1e-9:
            options.append(((i,j),)+solve(i+1,j+1))
        return min(options,key=lambda pairs:(-len(pairs),sum(abs(commits[y]['time']-boundaries[x]['time']) for x,y in pairs),pairs))
    return solve(0,0)


def score_episode(episode, replay):
    gt=phases(episode);duration=episode['length']/episode['fps'];commits=replay['commits'];intervals=replay['intervals']
    bounds=[dict(time=b['start_s'],from_stage=a['stage'],to_stage=b['stage'],index=i,
        ambiguous_for_fixed_horizon=any(abs(b['start_s']-q['start_s'])<1.-1e-9 for q in gt[1:] if q is not b),
        phase_duration_s=b['end_s']-b['start_s']) for i,(a,b) in enumerate(zip(gt,gt[1:]))]
    wrong=0.;early=0.;per_class={s:dict(duration_s=0.,wrong_s=0.) for s in LABELS}
    for i,g in enumerate(gt):
        per_class[g['stage']]['duration_s']+=g['end_s']-g['start_s']
        for a in intervals:
            overlap=max(0.,min(g['end_s'],a['end_s'])-max(g['start_s'],a['start_s']))
            if a['stage']!=g['stage']:
                wrong+=overlap;per_class[g['stage']]['wrong_s']+=overlap
                if i+1<len(gt) and a['stage']==gt[i+1]['stage']:early+=overlap
    def truth(t):return next(g['stage'] for g in gt if g['start_s']<=t<g['end_s'])
    pairs=match(bounds,commits);matched=dict(pairs);used={j for _,j in pairs}
    transition_rows=[]
    for i,b in enumerate(bounds):
        j=matched.get(i);c=commits[j] if j is not None else None
        transition_rows.append(dict(**b,matched=c is not None,commit_time=c['time'] if c else None,
            commit_delta_s=c['time']-b['time'] if c else None,
            return_occurrence=sum(x['stage']=='return' for x in gt[:i+2]) if b['to_stage']=='return' else None,
            final_return=b['to_stage']=='return' and i==len(bounds)-1))
    deltas=[r['commit_delta_s'] for r in transition_rows if r['matched']]
    wrong_commits=[c for c in commits if c['to_stage']!=truth(c['time'])]
    recovery=None;recovery_reason='no_wrong_commit'
    if wrong_commits:
        t=wrong_commits[0]['time'];recovery_reason='never_recovered_before_input_end'
        for a in intervals:
            for g in gt:
                lo=max(t,a['start_s'],g['start_s']);hi=min(a['end_s'],g['end_s'])
                if lo<hi and a['stage']==g['stage']:
                    recovery=lo-t;break
            if recovery is not None:recovery_reason=None;break
    event_counts={k:sum(e['kind']==k for e in replay['events']) for k in ['pending','cancel','illegal_edge','expire','gate_reject']}
    timing=[]
    for c in commits:
        p=c['request'];timing.append(dict(observation_to_available_s=p['first_available_at']-p['first_observed_at'],
            confirmation_wait_s=p['created_at']-p['confirmation_start_at'],
            deferred_release_wait_s=p['release_at']-p['created_at'],
            planning_wait_s=c['time']-p['release_at'],request_to_commit_s=c['time']-p['created_at'],
            observation_to_commit_s=c['time']-p['first_observed_at'],
            prediction_age_s=c['time']-p['source_observed_at']))
    actual=[replay['initial_prompt']]+[c['to_stage'] for c in commits]
    metrics=dict(source_recording_id=episode['source_recording_id'],episode_id=episode['episode_id'],duration_s=duration,
        consumed_anchors=replay['consumed_anchors'],input_anchors=replay['input_anchors'],gt_sequence=[g['stage'] for g in gt],
        active_sequence=actual,protocol_anomaly=gt[0]['stage']!='pickup',total_commits=len(commits),extra_commits=max(0,len(commits)-len(bounds)),
        wrong_to_stage_commits=len(wrong_commits),unmatched_commits=len(commits)-len(used),
        reverse_or_skip_commits=sum(EDGES[c['from_stage']]!=c['to_stage'] for c in commits),
        matched_transitions=len(pairs),eligible_transitions=len(bounds),missed_transitions=len(bounds)-len(pairs),
        strict_window_matched=len(match(bounds,commits,(-.2,.5))),
        unambiguous_eligible=sum(not b['ambiguous_for_fixed_horizon'] for b in bounds),
        unambiguous_matched=sum(not bounds[i]['ambiguous_for_fixed_horizon'] for i,j in pairs),
        active_prompt_time_accuracy=1-wrong/duration,wrong_prompt_duration_s=wrong,wrong_prompt_fraction=wrong/duration,
        early_wrong_prompt_duration_s=early,other_wrong_prompt_duration_s=wrong-early,per_class=per_class,
        full_episode_order_exact=actual==[g['stage'] for g in gt],
        post_first_error_recovery_s=recovery,post_first_error_recovery_reason=recovery_reason,
        canceled_pending=event_counts['cancel'],illegal_edge_attempts=event_counts['illegal_edge'],
        stale_prediction_count=event_counts['expire'],gated_predictions=event_counts['gate_reject'],pending_requests=event_counts['pending'],
        expired_pending_count=sum(e['kind']=='cancel' and e['reason']=='prediction_expired' for e in replay['events']),
        pending_commit_ratio=len(commits)/event_counts['pending'] if event_counts['pending'] else None,
        negative_delta_count=sum(d< -1e-9 for d in deltas),positive_delta_count=sum(d>1e-9 for d in deltas),
        zero_delta_count=sum(abs(d)<1e-9 for d in deltas),
        valid_early_commits=sum(-.25+1e-9<d< -1e-9 for d in deltas),
        premature_ge_025=sum(d<=-.25+1e-9 for d in deltas),premature_ge_05=sum(d<=-.5+1e-9 for d in deltas),
        premature_ge_1=sum(d<=-1.+1e-9 for d in deltas),on_time_count=sum(-.2-1e-9<=d<=.5+1e-9 for d in deltas),
        deltas=deltas,timing=timing,event_counts=event_counts,transitions=transition_rows,
        finish_status='not_supported',physical_safety='not_evaluated')
    return metrics


def aggregate(rows):
    add=['duration_s','consumed_anchors','input_anchors','total_commits','extra_commits','wrong_to_stage_commits','unmatched_commits',
         'reverse_or_skip_commits','matched_transitions','eligible_transitions','missed_transitions','strict_window_matched',
         'wrong_prompt_duration_s','early_wrong_prompt_duration_s','other_wrong_prompt_duration_s','canceled_pending',
         'illegal_edge_attempts','stale_prediction_count','gated_predictions','pending_requests','negative_delta_count','positive_delta_count',
         'zero_delta_count','valid_early_commits','premature_ge_025','premature_ge_05','premature_ge_1','on_time_count','unambiguous_eligible','unambiguous_matched','expired_pending_count']
    result={k:sum(r[k] for r in rows) for k in add};result['episodes']=len(rows)
    result.update(active_prompt_time_accuracy=1-result['wrong_prompt_duration_s']/result['duration_s'],
        transition_recall=result['matched_transitions']/result['eligible_transitions'],
        on_time_ratio=result['on_time_count']/result['eligible_transitions'],
        full_episode_order_exact_count=sum(r['full_episode_order_exact'] for r in rows),
        full_episode_order_exact_ratio=sum(r['full_episode_order_exact'] for r in rows)/len(rows),
        pending_commit_ratio=result['total_commits']/result['pending_requests'] if result['pending_requests'] else None)
    result.update(pending_matched_transition_ratio=result['matched_transitions']/result['pending_requests'] if result['pending_requests'] else None,
        canceled_pending_ratio=result['canceled_pending']/result['pending_requests'] if result['pending_requests'] else None,
        queue_congestion_count=0,queue_congestion_reason='Fixed latency below 0.5s serial service interval; overloaded queue NOT_RUN')
    dd=[d for r in rows for d in r['deltas']]
    result['commit_delta_s']=dict(p50=percentile(dd,50),p95=percentile(dd,95),min=min(dd) if dd else None,max=max(dd) if dd else None,
                                reason=None if dd else 'no_matched_commit')
    result['mean_absolute_matched_delta_s']=sum(abs(d) for d in dd)/len(dd) if dd else None
    keys=['observation_to_available_s','confirmation_wait_s','deferred_release_wait_s','planning_wait_s','request_to_commit_s','observation_to_commit_s','prediction_age_s']
    result['timing']={k:dict(p50=percentile([t[k] for r in rows for t in r['timing']],50),
        p95=percentile([t[k] for r in rows for t in r['timing']],95),reason=None if any(r['timing'] for r in rows) else 'no_commits') for k in keys}
    result['max_prediction_age_s']=max([t['prediction_age_s'] for r in rows for t in r['timing']],default=None)
    result['per_transition_pair']={}
    for a,b in EDGES.items():
        tt=[t for r in rows for t in r['transitions'] if (t['from_stage'],t['to_stage'])==(a,b)]
        ds=[t['commit_delta_s'] for t in tt if t['matched']]
        result['per_transition_pair'][a+'->'+b]=dict(eligible=len(tt),matched=len(ds),delta_p50_s=percentile(ds,50),
            premature_ge_05=sum(d<=-.5+1e-9 for d in ds),final_return_matched=sum(t['matched'] and t['final_return'] for t in tt))
    return result


def calibration(messages,episode,head='future'):
    gt=phases(episode);nll=[];brier=[];bins=[[] for _ in range(10)];per_class={s:[] for s in LABELS}
    for m in messages:
        target=m.observed_at+m.horizon_s
        g=next((g for g in gt if g['start_s']<=target<g['end_s']),None)
        if g is None:continue  # Only the independent scorer uses future recording support.
        y=LABELS.index(g['stage']);p=m.future_probabilities;k=max(range(4),key=lambda i:p[i])
        nll.append(-math.log(max(p[y],1e-300)));brier.append(sum((v-(i==y))**2 for i,v in enumerate(p)))
        bins[min(int(p[k]*10),9)].append((p[k],int(k==y)))
        per_class[LABELS[y]].append((nll[-1],brier[-1],p[k],int(k==y)))
    n=len(nll)
    return dict(samples=n,nll=sum(nll)/n if n else None,brier=sum(brier)/n if n else None,
        ece_10_equal_width=sum(abs(sum(v-c for v,c in b)) for b in bins)/n if n else None,
        bins=[dict(count=len(b),mean_confidence=sum(v for v,c in b)/len(b) if b else None,
                   accuracy=sum(c for v,c in b)/len(b) if b else None) for b in bins],
        per_true_class={s:dict(samples=len(v),nll=sum(x[0] for x in v)/len(v) if v else None,
            brier=sum(x[1] for x in v)/len(v) if v else None,
            confidence_bins=[dict(count=len(z),confidence_sum=sum(x[2] for x in z),correct_sum=sum(x[3] for x in z))
                for i in range(10) for z in [[x for x in v if min(int(x[2]*10),9)==i]]],
            reason=None if v else 'no_class_targets') for s,v in per_class.items()},
        definition='Forecast target classification calibration, not probability of a safe prompt switch')


def bootstrap_pair(b,c,seed=20261108,n=5000):
    if [r['source_recording_id'] for r in b]!=[r['source_recording_id'] for r in c]:raise ValueError('Unpaired sources')
    rng=random.Random(seed);differences={k:[] for k in ['time_accuracy','transition_recall','wrong_duration_s','premature_ge_05']}
    for _ in range(n):
        ids=[rng.randrange(len(b)) for _ in b]
        def totals(rr):
            duration=sum(rr[i]['duration_s'] for i in ids);eligible=sum(rr[i]['eligible_transitions'] for i in ids)
            return dict(time_accuracy=1-sum(rr[i]['wrong_prompt_duration_s'] for i in ids)/duration,
                transition_recall=sum(rr[i]['matched_transitions'] for i in ids)/eligible,
                wrong_duration_s=sum(rr[i]['wrong_prompt_duration_s'] for i in ids),premature_ge_05=sum(rr[i]['premature_ge_05'] for i in ids))
        aa,bb=totals(b),totals(c)
        for k in differences:differences[k].append(bb[k]-aa[k])
    return dict(seed=seed,replicates=n,source_count=len(b),direction='Joint Future minus Future-only',
        exploratory_only=True,intervals={k:[percentile(v,2.5),percentile(v,97.5)] for k,v in differences.items()})
