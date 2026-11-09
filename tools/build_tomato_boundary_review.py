"""Read-only41-case human review package, with30Hz videos and named joint traces."""
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import tomato_dataset as ds,tomato_metrics as metrics,tomato_boundary_metrics as bm
from .audit_tomato_transitions import manifest,PRIOR
from .tomato_decision_attribution import ROOT,SOURCE


def cases():
    truth=ds.read_json(SOURCE/'events.json')['events'];false=ds.read_json(SOURCE/'false_events.json')['events'];selected=[]
    for e in truth:
        tags=[]
        if not e['strict']['detected']:tags.append('strict_miss')
        if e['from_stage']=='adjust' and e['to_stage']=='pickup':tags.append('all_adjust_pickup')
        if e['event_id']=='3e645bf4e137d0f836c9d7ff_b4':tags.append('early_prompt_1p3s')
        if tags:selected.append(dict(case_id=e['event_id'],kind='truth',target_event_id=e['event_id'],episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],tags=tags,original_boundary_frame=e['boundary_frame'],boundary_s=e['boundary_s'],prediction_time=e['strict']['event_time'],from_stage=e['from_stage'],to_stage=e['to_stage']))
    for e in false:
        near=min([v for v in truth if v['episode_id']==e['episode_id']],key=lambda v:abs(v['boundary_s']-e['prediction']['time']))
        selected.append(dict(case_id=e['false_event_id'],kind='false',target_event_id=near['event_id'],episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],tags=['unmatched_warning'],original_boundary_frame=near['boundary_frame'],boundary_s=near['boundary_s'],prediction_time=e['prediction']['time'],from_stage=e['from_stage'],to_stage=e['to_stage']))
    assert len(selected)==41 and sum(c['kind']=='truth' for c in selected)==18
    return selected


def make_video(e,lo,hi,out):
    if out.exists():return
    cmd=['ffmpeg','-v','error','-y']
    for p in e['videos']:cmd+=['-ss',str(lo/e['fps']),'-t',str((hi-lo)/e['fps']),'-i',p]
    filters=';'.join(f'[{i}:v]scale=384:256:force_original_aspect_ratio=decrease,pad=384:256:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]' for i in range(3))+';[v0][v1][v2]hstack=inputs=3[v]'
    cmd+=['-filter_complex_threads','1','-filter_complex',filters,'-map','[v]','-an','-r',str(e['fps']),'-c:v','libx264','-preset','veryfast','-crf','24',str(out)]
    subprocess.run(cmd,check=True)


def run():
    _,episodes=manifest();byid={e['episode_id']:e for e in episodes};selected=cases();out=ROOT/'review_package';out.mkdir(exist_ok=True)
    manifest_data=ds.read_json(PRIOR/'prepared_h10/manifest.json');names=manifest_data['contract']['joint_names']
    logs=np.load(PRIOR/'evaluations/independent_h15_s42_val_logits.npz');original=ds.read_json(SOURCE/'comparison.json')['independent_h15_2hz_k2']['per_episode'];trace=ds.read_json(ROOT/'interface_traces.json')['episodes'];offset=0
    for e in episodes:
        a=np.array(e['anchors']);n=len(a);z={h:logs[h+'_logits'][offset:offset+n] for h in ['current','future']};offset+=n
        local=[v for v in selected if v['episode_id']==e['episode_id']]
        if not local:continue
        assert ds.digest(e['annotation']['path'])==e['annotation']['sha256']
        t,q=ds.read_state(e,names,'arms-and-hands');velocity=np.zeros_like(q);velocity[1:]=np.diff(q,axis=0)/np.diff(t)[:,None];velocity[0]=np.nan
        for case in local:
            b=case['boundary_s'];pred=case['prediction_time'];central=[b]+([pred] if pred is not None else [])
            lo=max(0,int(np.floor((min(central)-4)*e['fps'])));hi=min(e['length'],int(np.ceil((max(central)+4)*e['fps'])))
            # Include original and predicted markers for premature/late same-path misses.
            if case['kind']=='truth':
                candidates=[ev for ev in original[e['episode_id']]['future']['events'] if ev['previous']==ds.LABELS.index(case['from_stage']) and ev['label']==ds.LABELS.index(case['to_stage'])]
                near=min(candidates,key=lambda ev:abs(ev['time']-b)) if candidates else None
                if near and abs(near['time']-b)<=5:
                    case['nearest_model_event_s']=near['time'];lo=max(0,min(lo,int((near['time']-2)*e['fps'])));hi=min(e['length'],max(hi,int((near['time']+2)*e['fps'])))
            lo_s=lo/e['fps'];hi_s=hi/e['fps'];folder=out/case['case_id'];folder.mkdir(exist_ok=True)
            make_video(e,lo,hi,folder/'views30hz.mp4')
            ix=np.arange(lo,hi,dtype=np.int64);anchor_t=a/e['fps'];pk=(a>=lo)&(a<hi)
            probs={h:metrics.probabilities(z[h])[pk] for h in z}
            nearby={h:[v for v in original[e['episode_id']][h]['events'] if lo_s<=v['time']<hi_s] for h in ['current','future']}
            gt=[v for v in bm.boundaries(dict(episode=e)) if lo_s<=v['time']<hi_s]
            fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
            for ax,h in zip(axes,['current','future']):
                for j,label in enumerate(ds.LABELS):ax.plot(anchor_t[pk],probs[h][:,j],label=label)
                ax.axhline(.5,ls=':',c='gray');ax.axhline(.8,ls='-.',c='gray');ax.set_ylabel(h+' p')
                for v in gt:ax.axvline(v['time'],ls='--',c='black')
                for v in nearby[h]:ax.axvline(v['time'],ls=':',c='magenta')
                ax.grid(alpha=.2)
            axes[0].legend(ncol=4);axes[-1].set_xlabel('Episode seconds: black dashed=human boundary; magenta dotted=model accepted change .5/K2');fig.suptitle(case['case_id']);fig.tight_layout();fig.savefig(folder/'probabilities.png',dpi=120);plt.close(fig)
            fig,axs=plt.subplots(4,2,figsize=(14,12),sharex=True)
            for r,(title,start,end) in enumerate([('Left arm',0,7),('Right arm',7,14),('Left hand',14,36),('Right hand',36,58)]):
                for j in range(start,end):
                    label=names[j].replace('.position','').replace('left_','L:').replace('right_','R:')
                    axs[r,0].plot(t[ix],q[ix,j],lw=.7,label=label);axs[r,1].plot(t[ix],velocity[ix,j],lw=.7)
                axs[r,0].set_ylabel(title+' position rad');axs[r,1].set_ylabel('Velocity rad/s');axs[r,0].legend(ncol=2,fontsize=5,loc='best')
                for ax in axs[r]:
                    for v in gt:ax.axvline(v['time'],ls='--',c='black',lw=.8)
                    for h,color in [('current','blue'),('future','magenta')]:
                        for v in nearby[h]:ax.axvline(v['time'],ls=':',c=color,lw=.6)
                    ax.grid(alpha=.15)
            axs[-1,0].set_xlabel('Episode seconds');axs[-1,1].set_xlabel('Episode seconds');fig.suptitle(case['case_id']+' named measured positions; finite difference at real timestamps');fig.tight_layout();fig.savefig(folder/'joints.png',dpi=110);plt.close(fig)
            # Keep raw30Hz values locally, never commit measured state arrays to Git.
            with (folder/'state.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['source_frame','timestamp_s']+names+[v.replace('.position','.derived_velocity') for v in names])
                for i in ix:w.writerow([int(i),float(t[i])]+q[i].astype(float).tolist()+[None if np.isnan(v) else float(v) for v in velocity[i]])
            csv_rows=[r for r in trace[e['episode_id']] if lo_s<=r['time']<hi_s]
            ds.write_json(folder/'prompt_trace.json',dict(rows=csv_rows,source='Historical33ce417 interface; full Episode replay then sliced'))
            case.update(annotation_sha256=e['annotation']['sha256'],annotation_path=e['annotation']['path'],fps=e['fps'],episode_length=e['length'],clip_start_frame=lo,clip_end_frame=hi,clip_start_s=lo_s,clip_end_s=hi_s,
                video=str((folder/'views30hz.mp4').relative_to(out)),position_velocity_plot=str((folder/'joints.png').relative_to(out)),probability_plot=str((folder/'probabilities.png').relative_to(out)),state_csv=str((folder/'state.csv').relative_to(out)),trace_json=str((folder/'prompt_trace.json').relative_to(out)),
                truth_boundaries=gt,model_boundaries=nearby,current_future_times_s=anchor_t[pk].tolist(),current_probabilities=probs['current'].tolist(),future_probabilities=probs['future'].tolist())
            ds.write_json(folder/'case.json',case)
            print('REVIEW',case['case_id'],lo,hi,flush=True)
    data=dict(format_version=1,scope='Validation review selected using model errors; hide-model mode is NOT a genuinely blinded independent test',joint_names=names,convention='tomato-boundary-review-v1-proposal',cases=selected)
    ds.write_json(out/'cases.json',data)
    templates=[dict(case_id=c['case_id'],target_event_id=c['target_event_id'],source_annotation_sha256=c['annotation_sha256'],original_boundary_frame=c['original_boundary_frame'],reviewer=None,model_seen=False,proposed_boundary_frame=None,earliest_plausible_frame=None,latest_plausible_frame=None,reason=None,status='pending_human_review',revision_history=[]) for c in selected]
    ds.write_json(out/'review_opinions_template.json',dict(convention=data['convention'],reviews=templates))
    write_html(out,data)
    print('COMPLETE41_REVIEW_CASES',flush=True)


def write_html(out,data):
    # Preserve the frozen catalogue and case IDs; exclusions affect only the UI/export.
    policy=Path('configs/tomato_curated91_20261109/quality_exclusions.json')
    data=dict(data)
    if policy.exists():
        excluded=ds.read_json(policy)
        data['excluded_source_recording_ids']=[e['source_recording_id'] for e in excluded['excluded_episodes']]
    text=(Path(__file__).with_name('tomato_boundary_review_viewer.html')).read_text()
    embedded=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    (out/'index.html').write_text(text.replace('__CASE_DATA__',embedded))


if __name__=='__main__':run()
