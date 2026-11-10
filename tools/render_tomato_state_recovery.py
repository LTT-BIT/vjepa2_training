"""Render fixed replay evidence; no video encoding, training or online truth inputs."""
import argparse
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds
from .tomato_state_recovery import PHASES


def run(source, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if out.exists():raise FileExistsError('Use a new evidence directory')
    out.mkdir(parents=True)
    results=ds.read_json(source/'results.json')['42'];cases=[r for r in ds.read_json(source/'attribution.json') if r['seed']==42]
    index=[];seen=set();colors=['#2864b4','#d17812','#287c47','#984898']
    for case in cases:
        eid=case['episode_id'];root=results['B2_roots'][eid];z=np.load('runs/tomato/transition_time_20261109/seed42/'+eid+'.npz');t=z['times'];b=case['boundary_s']
        fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
        for ax,key,title in zip(axes[:2],['current_probabilities','future_probabilities'],['Current four-class probabilities','Future 1.5s four-class probabilities']):
            for k,label in enumerate(ds.LABELS):ax.plot(t,z[key][:,k],label=label,color=colors[k])
            ax.set_ylim(-.02,1.02);ax.set_ylabel('Probability');ax.set_title(title,loc='left');ax.legend(ncol=4,fontsize=8)
        rt=np.asarray(root['times']);axes[2].step(rt,root['truth_phase_offline'],where='post',color='black',label='Human proposal (scoring only)',linewidth=2)
        for mode,style in [('baseline','-'),('hypotheses','--')]:
            trace=results['per_episode'][mode][eid]['trace'];axes[2].step(rt,[r['phase'] for r in trace],where='post',label=mode,linestyle=style)
        axes[2].step(rt,root['B2_phase'],where='post',label='Historical B2',color='red')
        axes[2].set_yticks(range(7),PHASES,fontsize=8);axes[2].legend(fontsize=8,ncol=2);axes[2].set_xlabel('Observation timestamp (seconds)')
        for ax in axes:
            ax.axvline(b,color='black',linestyle=':',label='Boundary')
            ax.axvline(case['B1_event_time'],color='#269999',linestyle='--')
            ax.grid(alpha=.2);ax.set_xlim(max(0,b-5),b+3)
        fig.suptitle(f"{eid} / transition {case['ordinal']} / {case['transition']}\nBlack dotted: proposed boundary; teal: correct B1 warning; no online oracle",fontsize=10)
        fig.tight_layout();name=f"{eid}_event{case['ordinal']}.png";fig.savefig(out/name,dpi=135);plt.close(fig)
        index.append(dict(episode_id=eid,ordinal=case['ordinal'],plot=str(out/name),boundary_s=b))
        if eid in seen:continue
        seen.add(eid)
        fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=True)
        for k,label in enumerate(ds.LABELS):
            axes[0].plot(t,z['current_probabilities'][:,k],label=label,color=colors[k])
            axes[1].plot(t,z['future_probabilities'][:,k],label=label,color=colors[k])
        axes[0].set_title('Current probabilities');axes[1].set_title('Future probabilities')
        for ax in axes[:2]:ax.set_ylim(-.02,1.02);ax.legend(ncol=4,fontsize=8)
        axes[2].step(rt,root['truth_phase_offline'],where='post',label='Human proposal (offline only)',color='black',linewidth=2)
        axes[2].step(rt,root['B2_phase'],where='post',label='Historical B2',color='red')
        for mode in ['baseline','hypotheses']:
            axes[2].step(rt,[r['phase'] for r in results['per_episode'][mode][eid]['trace']],where='post',label=mode)
        axes[2].set_yticks(range(7),PHASES,fontsize=8);axes[2].legend(ncol=2,fontsize=8)
        rr=root['root'];start=root['first_persistent_disagreement']['start_s']
        for ax in axes:
            ax.axvline(start,color='red',linestyle=':');ax.axvline(rr['current_opportunities'][0]['time_s'],color='#269999',linestyle='--');ax.grid(alpha=.2)
        axes[2].set_xlabel('Observation timestamp (seconds)');fig.suptitle(eid+' / first persistent drift: red; first blocked Current evidence: teal',fontsize=10)
        fig.tight_layout();fig.savefig(out/(eid+'_root.png'),dpi=135);plt.close(fig)
    ds.write_json(out/'index.json',index)
    print(f'{len(index)} event plots + {len(seen)} root timelines')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();run(args.run,args.out)
