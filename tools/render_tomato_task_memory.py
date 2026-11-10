"""Scientific evidence plots from frozen episode memory replay; no model calls."""
import argparse
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds, tomato_boundary_metrics as bm
from .evaluate_tomato_reviewed_boundaries import revise_episode
from .tomato_state_recovery import PHASES


def run(source,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if out.exists():raise FileExistsError('Keep old evidence; choose a new directory')
    out.mkdir(parents=True);manifest=ds.read_json('runs/tomato/campaign94_20261107/prepared/manifest.json');by={e['episode_id']:e for e in manifest['episodes']};proposals=ds.read_json('docs/validation/tomato_reviewed_boundaries_20261109/proposals.json')['truth_proposals']
    results=ds.read_json(source/'results.json')['42'];index=[];colors=['#2864b4','#d17812','#287c47','#984898']
    for scenario in ('normal','consecutive_class_errors'):
        replay=ds.read_json(source/'seed42'/scenario/'replay.json')
        for eid,row in replay.items():
            e=revise_episode(by[eid],proposals.get(eid,[]));truth=bm.boundaries(dict(episode=e));ts=np.array([r['schedule_time'] for r in row['input_trace']]);gt=np.searchsorted([b['time'] for b in truth],ts,side='right')
            c=np.array([r['messages'][0]['current'] for r in row['input_trace']]);f=np.array([r['messages'][0]['future'] for r in row['input_trace']]);views=row['views']['M2']
            fig,ax=plt.subplots(4,1,figsize=(12,10),sharex=True)
            for k,label in enumerate(ds.LABELS):
                ax[0].plot(ts,c[:,k],label=label,color=colors[k]);ax[1].plot(ts,f[:,k],label=label,color=colors[k])
            ax[0].set_title('Current: actual received probabilities (including corruption in fault replay)',loc='left');ax[1].set_title('Future 1.5s: fixed original probabilities',loc='left')
            for a in ax[:2]:a.set_ylim(-.02,1.02);a.legend(ncol=4,fontsize=8)
            ax[2].step(ts,gt,where='post',label='Human proposal (scoring only)',color='black',linewidth=2)
            ax[2].step(ts,row['phases']['M0'],where='post',label='B0',color='#2864b4')
            ax[2].step(ts,row['phases']['M2'],where='post',label='M2 ledger (offline estimate)',color='#d17812',linestyle='--')
            ax[2].set_yticks(range(7),PHASES,fontsize=8);ax[2].legend(fontsize=8,ncol=3)
            ax[3].step(ts,[v['progress']['decision_fraction'] for v in views],where='post',label='Ordered decision progress; not physical success')
            ax[3].fill_between(ts,0,[float('history_current_conflict' in v['uncertainty']) for v in views],alpha=.2,color='red',label='History/current conflict')
            ax[3].scatter(ts,[v['candidate_phase']/6 if v['candidate_phase'] is not None else np.nan for v in views],s=8,color='purple',label='Pending stage; not confirmed')
            ax[3].legend(fontsize=8,ncol=2);ax[3].set_ylim(-.05,1.05);ax[3].set_xlabel('Scheduled observation time (seconds)')
            for a in ax:
                a.grid(alpha=.2)
                if scenario!='normal':a.axvspan(ts[row['fault_start_index']],ts[row['fault_end_index']]+.5,color='red',alpha=.12)
            fig.suptitle(f'{eid} / {scenario}\nShadow never changes active B0; classifier confidence is not a physical success observation',fontsize=10)
            fig.tight_layout();path=out/f'{eid}_{scenario}.png';fig.savefig(path,dpi=125);plt.close(fig)
            index.append(dict(episode_id=eid,scenario=scenario,plot=str(path)))
            if scenario=='normal':
                b=next((b for b in truth if b['previous']==3 and b['label']==0),None)
                if b is not None:
                    # A separate focused figure for every one of the eight difficult events.
                    keep=(ts>=b['time']-4)&(ts<=b['time']+4);fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
                    for k,label in enumerate(ds.LABELS):axes[0].plot(ts[keep],c[keep,k],label=label,color=colors[k]);axes[1].plot(ts[keep],f[keep,k],label=label,color=colors[k])
                    for aa in axes:aa.axvline(b['time'],color='black',linestyle=':');aa.grid(alpha=.2);aa.legend(ncol=4,fontsize=8)
                    changes=results['normal']['per_episode']['M2'][eid]['prompt']['events'];arrival=next(r['arrival_s'] for r in changes if r['transition']=='adjust->pickup')
                    if arrival is not None:
                        for aa in axes:aa.axvline(arrival,color='red',linestyle='--')
                    axes[0].set_title('Current / black proposed boundary; red ledger confirmation');axes[1].set_title('Future 1.5s / memory does not move this warning');axes[1].set_xlabel('Observation time (seconds)');fig.suptitle(eid+' / adjust to second pickup',fontsize=10);fig.tight_layout();path=out/f'{eid}_adjust_pickup.png';fig.savefig(path,dpi=125);plt.close(fig);index.append(dict(episode_id=eid,scenario='adjust_pickup',plot=str(path)))
    ds.write_json(out/'index.json',index);print(f'{len(index)} saved evidence figures')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.run,a.out)
