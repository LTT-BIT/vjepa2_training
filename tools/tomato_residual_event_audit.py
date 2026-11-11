"""Truth is used only after streaming decisions, for diagnostics and pairing."""
import numpy as np
from .tomato_state_recovery import StableLabel


def accepted_events(prob,times):
    rule=StableLabel();result=[]
    for p,t in zip(prob,times):
        event=rule.update(p,float(t))
        if event:result.append(event)
    return result


def boundary_evidence(times,prob,boundary,previous_start,events):
    t=np.asarray(times);p=np.asarray(prob);b=boundary['time'];target=boundary['label']
    ok=(p.argmax(1)==target)&(p[:,target]>=.5)
    scope=(t>=previous_start)&(t<boundary['end'])
    window=(t>=b-1.5)&(t<b)
    first=np.flatnonzero(scope&ok);inside=np.flatnonzero(window&ok)
    pairs=np.flatnonzero(scope[1:]&scope[:-1]&ok[1:]&ok[:-1])+1
    related=[e for e in events if (e['previous'],e['label'])==(boundary['previous'],target) and max(0,previous_start-1.5)<=e['time']<boundary['end']]
    crosses=False
    for event in related:
        i=int(np.searchsorted(t,event['time']))
        if i>0 and i<len(t) and t[i]==event['time'] and t[i-1]<b<=t[i] and ok[i-1] and ok[i]:crosses=True
    return dict(first_qualifying_s=float(t[first[0]]) if len(first) else None,
                first_target_K2_candidate_s=float(t[pairs[0]]) if len(pairs) else None,
                first_accepted_exact_edge_s=related[0]['time'] if related else None,
                first_in_strict_window_s=float(t[inside[0]]) if len(inside) else None,
                cross_boundary_K2=crosses,
                potential_target_pair_spans_boundary=bool(any(t[i-1]<b<=t[i] for i in pairs)),
                early_accepted_events=[e for e in related if e['time']<b-1.5],
                related_accepted_events=related,
                window=[dict(time=float(t[i]),offset_s=float(t[i]-b),argmax=int(p[i].argmax()),target_probability=float(p[i,target]),qualifies=bool(ok[i]),probabilities=p[i].tolist()) for i in np.flatnonzero(window)])


def false_changes(before,after):
    """Exact false-event differences, with one-to-one shift hints, never re-scoring."""
    key=lambda e:(e['time'],e['previous'],e['label'])
    old={key(e):e for e in before};new={key(e):e for e in after}
    removed=[old[k] for k in old.keys()-new.keys()];added=[new[k] for k in new.keys()-old.keys()]
    candidates=sorted((abs(a['time']-b['time']),i,j) for i,a in enumerate(removed) for j,b in enumerate(added)
                      if (a['previous'],a['label'])==(b['previous'],b['label']) and abs(a['time']-b['time'])<=1.5)
    useda=set();usedb=set();shifts=[]
    for delta,i,j in candidates:
        if i in useda or j in usedb:continue
        useda.add(i);usedb.add(j);shifts.append(dict(before=removed[i],after=added[j],delta_s=added[j]['time']-removed[i]['time']))
    return dict(preserved_exact=len(old.keys()&new.keys()),removed_exact=sorted(removed,key=lambda e:e['time']),added_exact=sorted(added,key=lambda e:e['time']),
                shifted_false_to_false_diagnostic=shifts,scope='Exact set differences include shifted events. Shift hints pair once within1.5s only for attribution; never alter strict truth matching.')


def nested_folds(source_ids,seed=20261112):
    ids=sorted(source_ids)
    if len(ids)!=10 or len(set(ids))!=10:raise ValueError('Exactly10 unique source recordings required')
    order=np.random.default_rng(seed).permutation(10);blocks=[[str(v) for v in np.asarray(ids)[ix]] for ix in np.array_split(order,5)]
    folds=[]
    for k in range(5):
        outer=blocks[k];selection=blocks[(k+1)%5];fit=[v for v in ids if v not in outer+selection]
        assert not(set(outer)&set(selection) or set(outer)&set(fit) or set(selection)&set(fit))
        folds.append(dict(outer=outer,selection=selection,fit=fit,refit=[v for v in ids if v not in outer]))
    return folds
