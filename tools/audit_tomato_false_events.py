"""Export every unmatched candidate2Hz Future change, including stable-phase false alarms."""
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import tomato_dataset as ds,tomato_metrics as metrics
from .audit_tomato_transitions import ROOT,PRIOR,manifest


def run():
    m,episodes=manifest();comparison=ds.read_json(ROOT/'comparison.json')['independent_h15_2hz_k2']['per_episode']
    z=np.load(PRIOR/'evaluations/independent_h15_s42_val_logits.npz');offset=0;rows=[];out=ROOT/'false_events';out.mkdir(exist_ok=True)
    for e in episodes:
        eid=e['episode_id'];a=np.array(e['anchors']);t=a/e['fps'];n=len(t);probs={h:metrics.probabilities(z[h+'_logits'][offset:offset+n]) for h in ['current','future']};offset+=n
        ts,q=ds.read_state(e,m['contract']['joint_names'],'arms-and-hands');vel=np.diff(q,axis=0)/np.diff(ts)[:,None];truth=comparison[eid]['future']['boundaries']
        for j,pred in enumerate(comparison[eid]['future']['false_events']):
            key=eid+f'_fp{j+1}';at=pred['time'];lo=max(0,at-4);hi=min(e['length']/e['fps'],at+4);keep=(t>=lo)&(t<=hi)
            fig,axs=plt.subplots(3,1,sharex=True,figsize=(10,7))
            for k,h in enumerate(['current','future']):
                for c,label in enumerate(ds.LABELS):axs[k].plot(t[keep]-at,probs[h][keep,c],label=label)
                axs[k].axhline(.5,color='gray',ls=':');axs[k].set_ylabel(h+' probability')
            axs[0].legend(ncol=4);axs[0].set_title(key+' unmatched '+ds.LABELS[pred['previous']]+' -> '+ds.LABELS[pred['label']])
            use=(ts[1:]>=lo)&(ts[1:]<=hi)
            for title,sl in [('arms14',slice(0,14)),('hands44',slice(14,58))]:axs[2].plot(ts[1:][use]-at,np.sqrt(np.mean(vel[use,sl]**2,axis=1)),label=title+' RMS rad/s')
            axs[2].legend();axs[2].set_xlabel('Seconds relative to false accepted-change event (black dashed)')
            for ax in axs:
                ax.axvline(0,color='k',ls='--');ax.grid(alpha=.2)
                for b in truth:
                    if lo<=b['time']<=hi:ax.axvline(b['time']-at,color='red',ls='-.')
            fig.tight_layout();curve=out/(key+'.png');fig.savefig(curve,dpi=110);plt.close(fig)
            video=out/(key+'.mp4')
            if not video.exists():
                cmd=['ffmpeg','-v','error','-y']
                for path in e['videos']:cmd+=['-ss',str(lo),'-t',str(hi-lo),'-i',path]
                filters=';'.join(f'[{i}:v]scale=384:256:force_original_aspect_ratio=decrease,pad=384:256:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]' for i in range(3))+';[v0][v1][v2]hstack=inputs=3[v]'
                cmd+=['-filter_complex_threads','1','-filter_complex',filters,'-map','[v]','-an','-r','15','-c:v','libx264','-preset','veryfast','-crf','26',str(video)]
                subprocess.run(cmd,check=True)
            closest=min(truth,key=lambda b:abs(b['time']-at));same=[b for b in truth if b['previous']==pred['previous'] and b['label']==pred['label']];exact=min(same,key=lambda b:abs(b['time']-at)) if same else None
            category='wrong_from_to_path' if exact is None else 'too_early' if at<exact['time']-1.5 else 'late_or_at_boundary'
            rows.append(dict(false_event_id=key,episode_id=eid,source_recording_id=e['source_recording_id'],prediction=pred,from_stage=ds.LABELS[pred['previous']],to_stage=ds.LABELS[pred['label']],nearest_boundary_s=closest['time'],offset_to_nearest_s=at-closest['time'],category=category,
                video=str(video.resolve()),curve=str(curve.resolve()),false_event_clip_offset_s=at-lo,annotation_revised=False,review=dict(status='unreviewed',source_annotation_sha256=e['annotation']['sha256'],proposed_revision=None,revision_history=[]),
                probability_curve=dict(times_s=t[keep].tolist(),**{h:probs[h][keep].tolist() for h in probs})))
    assert len(rows)==23
    ds.write_json(ROOT/'false_events.json',dict(model='independent_h15_s42',threshold=.5,confirmations=2,horizon_s=1.5,events=rows))
    ds.write_json('docs/validation/tomato_transition_20261109/false_events.json',dict(model='independent_h15_s42',events=rows))
    lines=['# 全部23个未匹配Future转换事件','', '黑色虚线为错误预测事件，红色点划线为图内真实边界。包括远离真实边界的稳定阶段误报，避免只展示边界漏检。视频中预测事件通常在+4秒，精确偏移见JSON。','', '|事件|源录制|时间秒|预测转移|距最近真实边界秒|归因|产物|','|---|---|---:|---|---:|---|---|']
    for r in rows:lines.append(f'|{r["false_event_id"]}|{r["source_recording_id"]}|{r["prediction"]["time"]}|{r["from_stage"]}→{r["to_stage"]}|{r["offset_to_nearest_s"]:.3f}|{r["category"]}|[视频]({r["video"]}) [概率/状态]({r["curve"]})|')
    text='\n'.join(lines)+'\n';(ROOT/'FALSE_EVENTS.md').write_text(text);Path('docs/validation/tomato_transition_20261109/FALSE_EVENTS.md').write_text(text)
    print('ALL23_FALSE_EVENT_VIDEOS_AND_CURVES_SAVED')


if __name__=='__main__':run()
