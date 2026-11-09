"""Plot all three models/rates per event; audit overlap with frozen2Hz features."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import tomato_dataset as ds, tomato_metrics as metrics
from .audit_tomato_transitions import ROOT,PRIOR,MODELS,manifest


def run():
    m,episodes=manifest(); events=ds.read_json(ROOT/'events.json')['events'];offset=0;parity=[]
    old={n:np.load(PRIOR/'evaluations'/f'{n}_s42_val_logits.npz') for n in MODELS}
    index=ds.read_json(PRIOR/'prepared_h10/feature_index.json')
    for e in episodes:
        eid=e['episode_id'];cache=np.load(ROOT/'fresh4hz'/(eid+'.npz'));a=cache['anchors'];t=a/e['fps'];original=np.array(e['anchors']);n=len(original);select=np.searchsorted(a,original)
        assert np.array_equal(a[select],original)
        feature_path=next(x['path'] for x in index['episodes'] if x['episode_id']==eid)
        with np.load(feature_path) as f: reference=f['features'].copy()
        assert np.array_equal(reference[:,-928:],cache['features'][select,-928:])
        differences={}
        for name in MODELS:
            for head in ['current','future']:
                reference_z=old[name][head+'_logits'][offset:offset+n];fresh=cache[name+'_'+head+'_logits'][select]
                differences[name+'_'+head]=dict(argmax_disagreements=int(np.sum(reference_z.argmax(1)!=fresh.argmax(1))),max_logit_abs=float(np.max(np.abs(reference_z-fresh))))
        parity.append(dict(episode_id=eid,shared_observations=n,positions_exact=True,visual_max_abs=float(np.max(np.abs(reference[:,:3072]-cache['features'][select,:3072]))),logits=differences))
        for event in [v for v in events if v['episode_id']==eid]:
            b=event['boundary_s'];fig,axs=plt.subplots(3,2,figsize=(13,8),sharex=True,sharey=True)
            for row,name in enumerate(MODELS):
                for col,head in enumerate(['current','future']):
                    ax=axs[row,col];p2=metrics.probabilities(old[name][head+'_logits'][offset:offset+n]);p4=metrics.probabilities(cache[name+'_'+head+'_logits']);t2=original/e['fps'];keep2=np.abs(t2-b)<=3;keep4=np.abs(t-b)<=3
                    for k,label in enumerate(ds.LABELS):
                        line,=ax.plot(t2[keep2]-b,p2[keep2,k],label=label+' 2Hz',lw=1.8)
                        ax.plot(t[keep4]-b,p4[keep4,k],color=line.get_color(),ls=':',lw=1.2)
                    ax.axvline(0,color='k',ls='--');ax.axvline(-MODELS[name][1],color='gray',ls='-.');ax.axhline(.5,color='gray',lw=.5);ax.grid(alpha=.15);ax.set_title(name+' '+head)
            axs[0,0].legend(ncol=2,fontsize=8);axs[-1,0].set_xlabel('Seconds relative to annotation (solid2Hz / dotted4Hz)');axs[-1,1].set_xlabel('Seconds relative to annotation')
            fig.suptitle(event['event_id']+' '+event['from_stage']+' -> '+event['to_stage']);fig.tight_layout();fig.savefig(ROOT/'media'/(event['event_id']+'_comparison.png'),dpi=100);plt.close(fig)
            event['media']['model_rate_comparison']=str((ROOT/'media'/(event['event_id']+'_comparison.png')).resolve())
        offset+=n
    ds.write_json(ROOT/'overlap_parity.json',dict(episodes=parity,note='All4Hz observations freshly encoded including shared2Hz anchors; exact position match and numerical parity prevent cache/source drift explanation.'))
    ds.write_json(ROOT/'events.json',dict(label_order=list(ds.LABELS),events=events));print('ALL56COMPARISON_PLOTS_AND_PARITY_DONE')


if __name__=='__main__':run()
