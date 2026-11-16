"""Summarize the three preregistered input arms without selecting a best seed."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import json
import hashlib


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


METHODS={'A_three_global':'G0_global_current','B_four_current':'G1_local_current',
         'C_local_history8':'G3_local_only_history8'}


def run(root):
    results=read_json(root/'results.json');cfg=read_json(root/'protocol.json')
    verification=read_json(root/'verification.json');summary={};events=[];checkpoints=[]
    for name,method in METHODS.items():
        ss=[results[str(s)][method]['evaluation']['summary'] for s in cfg['seeds']]
        counts=Counter()
        for s in ss:counts.update(s['prompt_counts'])
        summary[name]=dict(current_f1_mean=float(np.mean([s['current']['macro_f1'] for s in ss])),
            current_f1_seed_sd=float(np.std([s['current']['macro_f1'] for s in ss],ddof=1)),
            future_f1_mean=float(np.mean([s['future']['macro_f1'] for s in ss])),
            future_f1_seed_sd=float(np.std([s['future']['macro_f1'] for s in ss],ddof=1)),
            prompt_counts=dict(counts),future_hits=sum(s['warnings']['hits'] for s in ss),
            repeated_events=sum(s['warnings']['total'] for s in ss),
            unmatched_warnings=sum(s['warnings']['false_events'] for s in ss),
            false_per_minute_mean=float(np.mean([s['warnings']['false_per_minute'] for s in ss])),
            state_mismatch_s_mean=float(np.mean([s['mismatch_s'] for s in ss])),
            non_early_within_1p5s=sum(s['within_delay']['1.5'] for s in ss),
            by_transition={t:dict(Counter({k:sum(s['prompt_by_transition'].get(t,{}).get(k,0) for s in ss) for k in ['on_time','early','late','missing']})) for t in ss[0]['prompt_by_transition']},
            seed_summaries={str(seed):s for seed,s in zip(cfg['seeds'],ss)})
        for seed in cfg['seeds']:
            r=results[str(seed)][method]
            checkpoints.append(dict(arm=name,seed=seed,path=r['checkpoint_path'],sha256=r['training']['checkpoint_sha256'],reused=r['reused'],parameters=r['training']['parameters'],training_s=r['training']['training_s']))
            events.extend(dict(arm=name,seed=seed,**e) for e in r['evaluation']['events'])
    deltas=[]
    ids=list(results['42'][METHODS['B_four_current']]['evaluation']['per_episode'])
    for eid in ids:
        per_seed=[]
        for seed in cfg['seeds']:
            values=[]
            for method in [METHODS['B_four_current'],METHODS['C_local_history8']]:
                es=results[str(seed)][method]['evaluation']['per_episode'][eid]['prompt']['events']
                values.append(sum(e['outcome']=='on_time' for e in es))
            per_seed.append(values[1]-values[0])
        deltas.append(dict(episode_id=eid,seed_on_time_count_deltas=per_seed,mean_delta=float(np.mean(per_seed))))
    d=np.array([r['mean_delta'] for r in deltas])/6*100
    rng=np.random.default_rng(20261116);boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
    report=dict(methods=summary,paired_C_minus_B=dict(recordings=deltas,
        mean_on_time_percentage_points=float(d.mean()),
        recording_bootstrap_95_interval=np.percentile(boot,[2.5,97.5]).tolist(),
        scope='Repeatedly exposed development8; diagnostic interval, not independent generalization'),
        checkpoints=checkpoints,verification=verification,
        protocol_sha256=digest(root/'protocol.json'),results_sha256=digest(root/'results.json'),
        scope='48 boundaries repeated3 seeds, not144 independent events; train84/dev8 only',
        conclusions=['Only historical local RGB, ROI geometry and relative time enter C; historical global/state cannot influence output.',
            'Equal725640 trainable parameters and identical per-seed initialization/batches/30epochs; A/B reused.',
            'C has no consistent overall benefit over B in this trial; no default or deployment weights changed.',
            'Recorded r11 ROI coordinates are future-assisted: offline same-source experiment, not online ROI acceptance.'])
    write_json(root/'summary.json',report);write_json(root/'all_prompt_events.json',events)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(METHODS);x=np.arange(3);fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    ax=axes[0,0]
    for offset,key,label in [(-.18,'current_f1_mean','Current'),(.18,'future_f1_mean','Future +1.5s')]:
        bars=ax.bar(x+offset,[summary[n][key]*100 for n in names],.36,label=label)
        ax.bar_label(bars,fmt='%.2f',fontsize=9)
    ax.set_ylim(85,100);ax.set_ylabel('Macro-F1 (%)');ax.legend()
    ax=axes[0,1];bottom=np.zeros(3)
    for key,color in [('on_time','#3a9869'),('early','#d88f44'),('late','#b45d5d'),('missing','#777777')]:
        vals=np.array([summary[n]['prompt_counts'].get(key,0) for n in names]);bars=ax.bar(x,vals,bottom=bottom,label=key,color=color)
        ax.bar_label(bars,labels=[str(v) if v else '' for v in vals],label_type='center',fontsize=9);bottom+=vals
    ax.set_ylabel('Prompt events (48 boundaries x3 seeds)');ax.legend(fontsize=8)
    ax=axes[1,0];bars=ax.bar(x,[summary[n]['future_hits']/144*100 for n in names],color='#6276b8');ax.bar_label(bars,fmt='%.1f%%');ax.set_ylim(0,100);ax.set_ylabel('Strict future recall (%)')
    ax=axes[1,1];bars=ax.bar(x,[summary[n]['false_per_minute_mean'] for n in names],color='#bb8261');ax.bar_label(bars,fmt='%.3f');ax.set_ylim(0,2.2);ax.set_ylabel('Unmatched warnings / minute')
    for ax in axes.flat:ax.set_xticks(x,['A: 3 views','B: 4 views','C: local memory']);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Old100 matched ablation: only fourth-view history in C\nFrozen V-JEPA2 | 84 train / 8 development | seeds42/43/44')
    fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    print({k:{m:v for m,v in s.items() if m in ['current_f1_mean','future_f1_mean','prompt_counts','future_hits','unmatched_warnings']} for k,s in summary.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);run(p.parse_args().run)
