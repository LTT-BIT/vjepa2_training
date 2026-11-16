"""Automatic plots from predictions/events; no video access or human review."""
import math
from pathlib import Path
from .tomato_prompt_policy import LABELS
from .evaluate_tomato_prompt_switch import phases
from .tomato_prompt_replay import write

COLORS=['#3b82f6','#16a34a','#f59e0b','#a855f7']


def visualize(out,manifests,streams,policy,val,test,config):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    folder=Path(out)/'plots';folder.mkdir(exist_ok=False);index=[];count=0
    def raw_intervals(mm,end):
        return [dict(start_s=m.observed_at,end_s=mm[i+1].observed_at if i+1<len(mm) else end,
            stage=LABELS[max(range(4),key=lambda j:m.future_probabilities[j])]) for i,m in enumerate(mm)]
    def pending_intervals(replay):
        pp=[]
        for event in replay['events']:
            if event['kind'] in ['commit','cancel']:
                p=event.get('request') or event.get('canceled')
                if p:pp.append(dict(start_s=p['created_at'],end_s=event['time'],stage=p['stage']))
        return pp
    def bands(ax,rr,lo,hi):
        for y,items in enumerate(rr):
            for a in items:
                start=max(lo,a['start_s']);stop=min(hi,a['end_s'])
                if stop>start:ax.broken_barh([(start,stop-start)],(y-.35,.7),facecolors=COLORS[LABELS.index(a['stage'])])
        ax.set_xlim(lo,hi);ax.set_yticks(range(7),['GT','B raw @obs','C raw @obs','B Active','C Active','B Pending','C Pending'])
        ax.invert_yaxis()
    for split,models in [('val',val),('test',test)]:
        for e in manifests['joint']['episodes']:
            if e['split']!=split:continue
            eid=e['episode_id'];source=e['source_recording_id'];duration=e['length']/e['fps'];gt=phases(e)
            b=models['future_only']['replays'][eid];c=models['joint']['replays'][eid]
            base=f'{split}_{source}_{eid}'
            rr=[gt,raw_intervals(streams['future_only'][eid],duration),raw_intervals(streams['joint'][eid],duration),
                b['intervals'],c['intervals'],pending_intervals(b),pending_intervals(c)]
            fig,ax=plt.subplots(figsize=(16,5));bands(ax,rr,0,duration)
            for y,r in [(3,b),(4,c)]:
                for event in r['events']:
                    if event['kind']=='commit':ax.plot(event['time'],y,'kv',ms=5)
                    elif event['kind']=='cancel':ax.plot(event['time'],y+.25,'rx',ms=4)
                    elif event['kind'] in ['illegal_edge','gate_reject']:ax.plot(event['time'],y+.25,'r|',ms=5)
                times=[ev['time'] for ev in r['events'] if ev['kind']=='planning']
                ax.vlines(times,y-.35,y+.35,colors='black',alpha=.07,lw=.3)
            for g in gt[1:]:ax.axvline(g['start_s'],color='black',ls=':',lw=.8)
            ax.set_title(f'{split} | {source} | {policy["name"]}\nRaw at observation t; forecast target=t+1s. Black triangles=commit; red x=cancel. AUTO; NO MANUAL REVIEW')
            ax.set_xlabel('Episode seconds: observation and actual commit on same clock')
            ax.legend(handles=[Patch(color=co,label=label) for co,label in zip(COLORS,LABELS)],loc='upper right',ncol=4)
            fig.tight_layout();fig.savefig(folder/(base+'_timeline.png'),dpi=120);plt.close(fig);count+=1
            fig,axes=plt.subplots(2,1,figsize=(16,7),sharex=True)
            for ax,model,repl in zip(axes,['future_only','joint'],[b,c]):
                mm=streams[model][eid];tt=[x.observed_at for x in mm]
                for k,label in enumerate(LABELS):ax.plot(tt,[x.future_probabilities[k] for x in mm],color=COLORS[k],label=label,lw=.8)
                for ev in repl['events']:
                    if ev['kind']=='commit':ax.axvline(ev['time'],color='black',lw=.6,alpha=.6)
                for g in gt[1:]:ax.axvline(g['start_s'],color='red',ls=':',lw=.6)
                ax.set(title=model+' | probabilities issued at observation anchor (target=anchor+1s)',ylim=(-.05,1.05),ylabel='Softmax');ax.legend(ncol=4)
            axes[-1].set_xlabel('Observation seconds; black=actual commit, red dotted=annotation boundary')
            fig.tight_layout();fig.savefig(folder/(base+'_probabilities.png'),dpi=120);plt.close(fig);count+=1
            boundaries=[g['start_s'] for g in gt[1:]]
            fig,axes=plt.subplots(math.ceil(len(boundaries)/2),2,figsize=(15,3*math.ceil(len(boundaries)/2)),squeeze=False)
            for ax,t in zip(axes.flat,boundaries):
                lo=max(0,t-3);hi=min(duration,t+3);bands(ax,rr,lo,hi);ax.axvline(t,color='red',ls=':')
                for y,rep in [(3,b),(4,c)]:
                    for ev in rep['events']:
                        if ev['kind']=='commit' and lo<=ev['time']<=hi:ax.plot(ev['time'],y,'kv',ms=5)
                ax.set_title(f'Annotation boundary {t:.3f}s; commit triangles')
            for ax in list(axes.flat)[len(boundaries):]:ax.set_visible(False)
            fig.tight_layout();fig.savefig(folder/(base+'_boundaries.png'),dpi=120);plt.close(fig);count+=1
            index.append(dict(split=split,source_recording_id=source,episode_id=eid,files=[base+suffix+'.png' for suffix in ['_timeline','_probabilities','_boundaries']],
                B_matched=next(v['matched_transitions'] for v in models['future_only']['per_episode'] if v['episode_id']==eid),
                C_matched=next(v['matched_transitions'] for v in models['joint']['per_episode'] if v['episode_id']==eid),
                failure_indexes=[str(Path('..')/(('val_grid/'+policy['name']) if split=='val' else 'test_main')/model/'failure_cases.jsonl') for model in ['future_only','joint']]))
    # Verify PNG structure mechanically; no human semantic or physical-safety assessment.
    from PIL import Image
    for p in folder.glob('*.png'):
        with Image.open(p) as im:im.verify()
    write(folder/'visualization_integrity.json',dict(episodes=len(index),png_count=count,png_decode_passed=True,
        P5_manual_review='NOT_PERFORMED / DEFERRED',index=index))
    (folder/'timeline_index.md').write_text('# Automatic timeline index\n\nP5: AUTO_GENERATED; MANUAL_REVIEW_DEFERRED. No MP4 was opened.\n\n'+
        '\n\n'.join(f"- {v['split']} source {v['source_recording_id']} episode {v['episode_id']}; B/C matched={v['B_matched']}/{v['C_matched']}\n  "+' | '.join(f'[{f}]({f})' for f in v['files'])+'\n  Failures: '+' | '.join(f'[{p}]({p})' for p in v['failure_indexes']) for v in index)+'\n')
