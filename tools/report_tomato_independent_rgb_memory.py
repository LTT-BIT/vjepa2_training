"""Independent-readout ablations, all seeds, plus explicit historical references."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .report_tomato_local_only_memory import read_json,write_json,digest

METHODS={'I0_three_global':'G0_global_current','I1_four_current':'G1_local_current',
         'I2_local_history8':'G3_local_only_history8'}


def aggregate(results,method):
    rows=[results[str(seed)][method]['evaluation'] for seed in [42,43,44]]
    ss=[r['summary'] for r in rows];counts=Counter()
    for s in ss:counts.update(s['prompt_counts'])
    events=[e for r in rows for e in r['events']]
    offsets=[e['offset_s'] for e in events if e['offset_s'] is not None]
    warnings=[b for r in rows for p in r['per_episode'].values() for b in p['warnings']['boundaries']]
    lead=[-b['offset_s'] for b in warnings if b['detected']]
    hits=sum(s['warnings']['hits'] for s in ss);false=sum(s['warnings']['false_events'] for s in ss)
    return dict(current_f1_mean=float(np.mean([s['current']['macro_f1'] for s in ss])),
        current_f1_sd=float(np.std([s['current']['macro_f1'] for s in ss],ddof=1)),
        future_f1_mean=float(np.mean([s['future']['macro_f1'] for s in ss])),
        future_f1_sd=float(np.std([s['future']['macro_f1'] for s in ss],ddof=1)),
        prompt_counts=dict(counts),warnings=dict(hits=hits,total=len(warnings),
            recall=hits/len(warnings),false_events=false,precision=hits/max(hits+false,1),
            false_per_minute_mean=float(np.mean([s['warnings']['false_per_minute'] for s in ss])),
            lead_p50_s=float(np.median(lead)) if lead else None),
        prompt_offset_p50_s=float(np.median(offsets)),
        prompt_offset_p95_s=float(np.percentile(offsets,95)),
        severe_early_gt1s=sum(s['severe_early_gt1s'] for s in ss),
        non_early_within_1p5s=sum(s['within_delay']['1.5'] for s in ss),
        mismatch_s_per_seed_mean=float(np.mean([s['mismatch_s'] for s in ss])),
        prompt_by_transition={t:dict(Counter(e['outcome'] for e in events if e['transition']==t)) for t in sorted({e['transition'] for e in events})},
        seed_summaries={str(seed):s for seed,s in zip([42,43,44],ss)})


def run(root):
    r=read_json(root/'results.json');protocol=read_json(root/'protocol.json')
    prior=read_json('runs/tomato/rgb_memory_train_20261116_r11_v1/results.json')
    local=read_json('runs/tomato/local_only_memory_20261116_v1/results.json')
    methods={name:aggregate(r,method) for name,method in METHODS.items()}
    methods['original_B0_reference']=aggregate(prior,'B0_common_startup')
    for name,method in METHODS.items():
        data=local if method=='G3_local_only_history8' else prior
        methods[name.replace('I','J',1)+'_joint_reference']=aggregate(data,method)
    checkpoints=[];events=[]
    for seed in [42,43,44]:
        for name,method in METHODS.items():
            item=r[str(seed)][method];info=item['training']
            checkpoints.append(dict(arm=name,seed=seed,path=item['checkpoint_path'],sha256=info['checkpoint_sha256'],
                parameters=info['parameters'],training_s=info['training_s'],branches=info['branches']))
            events.extend(dict(arm=name,seed=seed,**e) for e in item['evaluation']['events'])
    contrasts={}
    for left,right in [('I1_four_current','I0_three_global'),('I2_local_history8','I1_four_current')]:
        m1,m0=METHODS[left],METHODS[right];ids=list(r['42'][m1]['evaluation']['per_episode']);per=[]
        for eid in ids:
            delta=[]
            for seed in [42,43,44]:
                counts=[]
                for method in [m0,m1]:
                    es=r[str(seed)][method]['evaluation']['per_episode'][eid]['prompt']['events']
                    counts.append(sum(e['outcome']=='on_time' for e in es))
                delta.append(counts[1]-counts[0])
            per.append(dict(episode_id=eid,seed_count_deltas=delta,mean_count_delta=float(np.mean(delta))))
        d=np.array([p['mean_count_delta'] for p in per])/6*100
        rng=np.random.default_rng(20261116);boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
        contrasts[left+'_minus_'+right]=dict(recordings=per,mean_on_time_percentage_points=float(d.mean()),
            recording_bootstrap_95_interval=np.percentile(boot,[2.5,97.5]).tolist())
    report=dict(methods=methods,paired_contrasts=contrasts,checkpoints=checkpoints,
        verification=read_json(root/'verification.json'),receipt=read_json(root/'receipt.json'),
        trained_branches=18,bundles=9,tests='370 passed,1 skipped',
        newly_trained_gpu_seconds=sum(x['training_s'] for x in checkpoints),
        protocol_sha256=digest(root/'protocol.json'),results_sha256=digest(root/'results.json'),
        caveats=['48 events repeated across3 seeds, not144 independent events; exposed development8.',
            'I0/I1/I2 have equal independent capacity and matched per-target initialization, batches, epochs.',
            'Independent readouts have roughly twice trainable parameters of joint references: training regime vs capacity is not isolated.',
            'Original B0 is a frozen historical architecture/data-version reference, not I0 retrained.',
            'Recorded r11 ROI coordinates are future-assisted; no online or robot performance claim.',
            'History8 affects only older local RGB/ROI/time; global RGB and state are latest-window only.'],
        protocol=protocol)
    write_json(root/'summary.json',report);write_json(root/'all_prompt_events.json',events)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(METHODS);x=np.arange(3);fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for offset,key,label in [(-.18,'current_f1_mean','Current'),(.18,'future_f1_mean','Future +1.5s')]:
        bars=axes[0].bar(x+offset,[methods[n][key]*100 for n in names],.36,label=label)
        axes[0].bar_label(bars,fmt='%.2f',fontsize=9)
    axes[0].set_ylim(85,100);axes[0].set_ylabel('Macro-F1 (%)');axes[0].legend()
    bottom=np.zeros(3)
    for key,color in [('on_time','#3a9869'),('early','#d88f44'),('late','#b45d5d'),('missing','#777777')]:
        vals=np.array([methods[n]['prompt_counts'].get(key,0) for n in names])
        bars=axes[1].bar(x,vals,bottom=bottom,label=key,color=color)
        axes[1].bar_label(bars,labels=[str(v) if v else '' for v in vals],label_type='center');bottom+=vals
    axes[1].set_ylabel('48 boundaries x3 seeds');axes[1].legend(fontsize=8)
    bars=axes[2].bar(x,[methods[n]['warnings']['false_per_minute_mean'] for n in names],color='#bb8261')
    axes[2].bar_label(bars,fmt='%.3f');axes[2].set_ylim(0,2.5);axes[2].set_ylabel('Unmatched warnings / min')
    for ax in axes:ax.set_xticks(x,['I0: 3 views','I1: 4 views','I2: local memory']);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Independent Current/Future readouts | old100 | train84/dev8 | frozen encoder')
    fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    for name,s in methods.items():print(name,s['current_f1_mean'],s['future_f1_mean'],s['prompt_counts'],s['warnings'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);run(p.parse_args().run)
