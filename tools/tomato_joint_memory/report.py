"""Real result tables, source-cluster paired uncertainty, integrity and handoff."""
from __future__ import annotations
import csv,json,zipfile
from pathlib import Path
import numpy as np
from .contracts import read,write,sha,REPO,verify,LABELS


def save_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bootstrap(a,b,c):
    aa=a['classification']['episodes'];bb=b['classification']['episodes']
    if [x['source_recording_id'] for x in aa]!=[x['source_recording_id'] for x in bb]:raise ValueError('Paired source order')
    rng=np.random.default_rng(c['bootstrap_seed']);n=len(aa);weights=rng.multinomial(n,np.ones(n)/n,size=c['bootstrap_replicates']);out={}
    for h in ['current','future']:
        def boot(ep):
            cm=np.array([x['heads'][h]['overall']['confusion_true_rows'] for x in ep]);ss=np.einsum('ri,ijk->rjk',weights,cm);den=ss.sum(1)+ss.sum(2);tp=np.diagonal(ss,axis1=1,axis2=2)
            return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(1)
        delta=boot(aa)-boot(bb);per=np.array([x['heads'][h]['overall']['macro_f1']-y['heads'][h]['overall']['macro_f1'] for x,y in zip(aa,bb)])
        out[h]=dict(delta=a['classification']['heads'][h]['macro_f1']-b['classification']['heads'][h]['macro_f1'],ci95=np.percentile(delta,[2.5,97.5]).tolist(),source_wins=int((per>1e-9).sum()),ties=int((abs(per)<=1e-9).sum()),losses=int((per< -1e-9).sum()))
    for key in ['strict_hits','early_ge05','wrong_active_s']:
        delta=np.array([x[key]-y[key] for x,y in zip(a['handshake']['C0']['episodes'],b['handshake']['C0']['episodes'])]);out[key]=dict(delta=float(delta.sum()),ci95=np.percentile(weights@delta,[2.5,97.5]).tolist(),source_positive=int((delta>1e-9).sum()),source_negative=int((delta< -1e-9).sum()),ties=int((abs(delta)<=1e-9).sum()))
    for metric in ['recall','fp_per_min']:
        ae=a['classification']['future_events']['raw']['episodes'];be=b['classification']['future_events']['raw']['episodes'];num='tp' if metric=='recall' else 'fp';den=np.array([r['tp']+r['fn'] if metric=='recall' else r['duration_s']/60 for r in ae]);delta=(weights@np.array([r[num] for r in ae])-weights@np.array([r[num] for r in be]))/(weights@den)
        out['warning_'+metric]=dict(delta=a['classification']['future_events']['raw'][metric]-b['classification']['future_events']['raw'][metric],ci95=np.percentile(delta,[2.5,97.5]).tolist())
    return out


def table_rows(summary,split):
    rows=[]
    for name,r in summary.items():
        s=r['splits'][split];cl=s['classification'];hh=s['handshake']['C0'];baseline=summary.get('B0R_s'+str(r['seed']),r)['splits'][split]['classification'];future=cl['future_events']['raw'];lat=r['latency']
        rows.append(dict(model=name,seed=r['seed'],parameters=r['parameters'],current_f1=cl['heads']['current']['macro_f1'],future_f1=cl['heads']['future']['macro_f1'],delta_current_vs_B0R=cl['heads']['current']['macro_f1']-baseline['heads']['current']['macro_f1'],delta_future_vs_B0R=cl['heads']['future']['macro_f1']-baseline['heads']['future']['macro_f1'],future_event_recall=future['recall'],fp_per_min=future['fp_per_min'],strict_hits=hh['strict_hits'],eligible=hh['eligible'],early_ge05=hh['early_ge05'],wrong_active_s=hh['wrong_active_s'],readout_p95_ms=lat['cached_readout_ms']['p95'],best_epoch=r['training']['best_epoch'],epochs=r['training']['epochs'],updates=r['training']['updates'],training_s=r['training']['training_s']))
    return rows


def generate(c,out):
    summary=read(out/'summary.json');protocol=read(out/'protocol.lock.json');paired={};promotion={};tables={}
    for split in ['val','test','train']:
        tables[split]=table_rows(summary,split);save_csv(out/(('test_locked_comparison' if split=='test' else split+'_comparison')+'.csv'),tables[split])
    for group in c['groups'][1:]:
        paired[group]={}
        for seed in c['seeds']:
            name=f'{group}_s{seed}';paired[group][str(seed)]={}
            for split in ['val','test']:
                paired[group][str(seed)][split]=bootstrap(summary[name]['splits'][split],summary[f'B0R_s{seed}']['splits'][split],c)
                for comparator in (['T0'] if group in ['T1','T2'] else ['T1'] if group in ['K1','P1'] else []):
                    paired[group][str(seed)][split+'_vs_'+comparator]=bootstrap(summary[name]['splits'][split],summary[f'{comparator}_s{seed}']['splits'][split],c)
        deltas=np.array([[paired[group][str(seed)]['val'][h]['delta'] for h in ['current','future']] for seed in c['seeds']]);mean=deltas.mean(0);wins=(deltas>0).sum(0);fp=np.mean([paired[group][str(seed)]['val']['warning_fp_per_min']['delta'] for seed in c['seeds']]);threshold=c['promotion']
        passes=bool(((wins[0]>=threshold['min_seed_wins'] and mean[0]>=threshold['min_f1_gain'] and mean[1]>=-threshold['max_other_head_loss']) or (wins[1]>=threshold['min_seed_wins'] and mean[1]>=threshold['min_f1_gain'] and mean[0]>=-threshold['max_other_head_loss'])) and fp<=threshold['max_fp_per_min_increase'])
        promotion[group]=dict(val_only_pass=passes,mean_current_delta=float(mean[0]),mean_future_delta=float(mean[1]),current_seed_wins=int(wins[0]),future_seed_wins=int(wins[1]),mean_raw_warning_fp_per_min_delta=float(fp))
    write(out/'source_paired_bootstrap.json',paired);write(out/'val_promotion.json',promotion)
    compact={}
    for name,row in summary.items():
        compact[name]={k:v for k,v in row.items() if k!='splits'};compact[name]['splits']={}
        for split,v in row['splits'].items():
            cl=v['classification'];compact[name]['splits'][split]=dict(heads=cl['heads'],bucket_pooled=cl['bucket_pooled'],future_persistence=cl['future_persistence'],future_events={k:{a:b for a,b in ev.items() if a!='episodes'} for k,ev in cl['future_events'].items()},handshake={k:{a:b for a,b in hh.items() if a!='episodes'} for k,hh in v['handshake'].items()})
    write(out/'summary.compact.json',compact)
    # All displayed values derive from saved score/episode evidence; source SHA validates inputs.
    integrity=verify(out)
    lines=['# LOCAL93 Joint + Memory 受控消融报告（2026-11-12）','',f'实验目录：`{out}`。基线 `{protocol["head"]}`；实验分支 `{protocol["branch"]}`。参考分支锁定 `{c["reference_sha"]}`。代码未自动提交/推送，生产模型和界面默认不变。','', '本轮真实使用原 93 条 normal 数据、75/9/9 来源划分、三路完整 RGB 冻结特征与每锚点 16×58 个实测位置。2Hz、Future +1.0s、联合监督权重 1/.5 固定。没有 ROI、第四路图像、GT 阶段记忆、重划分或原产物覆盖。','', '## 方法与执行边界','', 'B0-frozen 只读复核原 Joint；B0R 为匹配三种子原 Joint 重训。T0/T1/T2 为完全同构的 128 隐层、四头最新查询 Attention，加最新锚点 residual，分别 N=1/4/8。相对时间编码只读过去观测。K1 为 T1 加实测位置差分速度摘要（非 action/目标速度）；P1 为 T1 加严格过去的双头概率，train 使用 5-fold 来源 OOF（每折54训练/6内部验证/15留出），val/test 使用固定原 B0 全训练 teacher。S1 为 B0-frozen 因果 EMA(alpha=.6)，单独报告过滤后的结果。','', '先运行 seed42 两 epoch 烟雾检验，随后全部六组、三个 seed、最多100 epoch/patience15，val 加权双头 F1、NLL 破同分，同一 epoch 两头。模型、参数和 val 结果全部锁定后才进行一次 test 评估。原 test 已在历史实验中被观察，本报告是历史保留集诊断，不是新盲测。','', 'E1/N16 为初批协议外可选项，本轮 NOT_RUN；物理成功 OperationMemory 为 NOT_EVALUABLE（缺独立结果标签）。没有接机器人、π0.5 或完成守卫。','', '## 验证集主表','', '|方案|Seed|参数量|Current F1|Future F1|ΔCurrent vs B0R|ΔFuture|严格预警Recall|FP/min|握手窄窗命中|提前≥.5s|Wrong Active(s)|读出P95(ms)|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in tables['val']:lines.append(f"|{r['model']}|{r['seed']}|{r['parameters']}|{r['current_f1']:.5f}|{r['future_f1']:.5f}|{r['delta_current_vs_B0R']:+.5f}|{r['delta_future_vs_B0R']:+.5f}|{r['future_event_recall']:.4f}|{r['fp_per_min']:.3f}|{r['strict_hits']}/{r['eligible']}|{r['early_ge05']}|{r['wrong_active_s']:.3f}|{r['readout_p95_ms']:.3f}|")
    lines+=['', '## 预定验证门槛及归因','', '门槛仅用于下一阶段研究：同一头至少2/3 seed改善、均值提升≥.005、另一头平均退化不超过.005、raw严格预警 FP/min 增量≤.1。没有把多个终点合成总分。','']
    for group,p in promotion.items():lines.append(f"- {group}：{'通过' if p['val_only_pass'] else '未通过'}；平均 ΔCurrent={p['mean_current_delta']:+.5f}，ΔFuture={p['mean_future_delta']:+.5f}，seed 改善数 {p['current_seed_wins']}/3、{p['future_seed_wins']}/3，ΔFP/min={p['mean_raw_warning_fp_per_min_delta']:+.3f}。")
    lines+=['','T1/T2 另与同参数 T0 比较，K1/P1 另与 T1 比较，详见 source_paired_bootstrap.json；B0R 与 T0 的容量不同，不能将二者差异归因于历史。每个来源为抽样单元，5000次配对 bootstrap，固定种子；只有9条val/9条已暴露test，多候选结果不能宣称全新统计显著泛化。','', '## 历史 test 固定方案对照','', '|方案|Current F1|Future F1|严格预警Recall|FP/min|窄窗命中|提前≥.5s|Wrong Active(s)|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in tables['test']:lines.append(f"|{r['model']}|{r['current_f1']:.5f}|{r['future_f1']:.5f}|{r['future_event_recall']:.4f}|{r['fp_per_min']:.3f}|{r['strict_hits']}/{r['eligible']}|{r['early_ge05']}|{r['wrong_active_s']:.3f}|")
    lines+=['', '## 证据、局限及复现','', '每个 evaluations/<model>/<split> 保存完整概率、逐 Episode 分类和边界分桶、两类预警原始/稳定事件、C0/C1真实状态机回放及配对事件。classification.json 的 bucket_pooled 区分当前/未来目标、±.5/1/2秒、稳定区、短/长阶段、冷启动和重复阶段出现次数；每头另有四类 P/R/F1/support、混淆矩阵、NLL/Brier/ECE。Oracle persistence 仅是独立评分诊断，不能进入模型或状态机。','', '严格预警按预测 Current→Future 分歧事件开始计时：必须匹配真实相邻 from→to 且发生于边界前[-1,0)；稳定预警用第二个锚点时刻，不回填第一个时刻。过早持续告警不会在之后滑入1秒窗口时被重记成成功。这比“窗口内曾出现正确未来标签”更严格。歧义短间隔边界显式标注。','', '所有模型 C0 为固定 WAIT_LATE_ACK；C1 仅关闭 ACK 的诊断，不是上线推荐。90ms 是统一模拟结果延迟，不是本轮测得的模型时延。','', '本轮所有训练/评估复用缓存，Encoder 新提取0次。P50/P95/P99是batch1、GPU FP32、特征已驻留的读出耗时，CPU缓存更新时间另记；P1还必须加固定B0读出成本。真正在线 Encoder+Memory 总耗时和 π0.5 并发为 NOT_MEASURED，不能把缓存读出速度当端到端速度。各模型训练时长、更新步数、最佳epoch、峰值分配显存及历史容量见 models/*/manifest.json 与 summary.json。','', f'原文件完整性：{integrity["protected_files"]} 个文件 SHA 复核，unchanged={integrity["unchanged"]}。来源身份、58个关节字段、原视频/Parquet/标注/缓存SHA、时间戳最大网格误差记录在 source_identity.json；沿用既有1e-4秒时间戳容差以兼容长序列float32量化，没有补帧或改标注。', '', '完整命令（新实验先使用新的空目录，已有训练目录拒绝覆盖）：','', '```bash', 'python -m tools.tomato_joint_memory.run --config configs/tomato_joint_memory_ablation_20261112.yaml --output-dir "$EXPERIMENT_DIR" --check-only','python -m tools.tomato_joint_memory.run --config configs/tomato_joint_memory_ablation_20261112.yaml --output-dir "$EXPERIMENT_DIR" --smoke','python -m tools.tomato_joint_memory.run --config configs/tomato_joint_memory_ablation_20261112.yaml --output-dir "$EXPERIMENT_DIR" --run','python -m tools.tomato_joint_memory.run --config configs/tomato_joint_memory_ablation_20261112.yaml --output-dir "$EXPERIMENT_DIR" --verify','python -m tools.tomato_joint_memory.run --config configs/tomato_joint_memory_ablation_20261112.yaml --output-dir "$EXPERIMENT_DIR" --report-only','python -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py','```','', '使用本仓库 .venv-tomato/bin/python。大权重留在本机 models/*/best_head.pt，准确SHA见各 checkpoint.sha256；teacher 权重和OOF成员在 teachers/fold_*。handoff.zip只含代码、报告、CSV/JSON摘要和逐事件证据，不含视频/Encoder/NPZ/权重。']
    report='\n'.join(lines)+'\n'
    if (out/'ANALYSIS_SUPPLEMENT.md').exists():
        supplement=(out/'ANALYSIS_SUPPLEMENT.md').read_text();lead,tail=supplement.split('## 其他正常来源的回归',1)
        report=report.replace('## 方法与执行边界',lead+'\n## 方法与执行边界',1)+'\n## 其他正常来源的回归'+tail
    write(out/'status.json',dict(complete=True,models=len(summary),encoder_new_calls=0))
    path=REPO/'docs/TOMATO_JOINT_MEMORY_ABLATION_REPORT_20261112.md';path.write_text(report);(out/'REPORT.md').write_text(report)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        groups=c['groups'];fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,h in zip(axes,['current_f1','future_f1']):
            values=[[r[h] for r in tables['val'] if r['model'].startswith(g+'_')] for g in groups];ax.bar(groups,[np.mean(v) for v in values],yerr=[np.std(v) for v in values]);ax.set_title('Val '+h+' (3 seeds)');ax.set_ylim(.5,1)
        fig.tight_layout();fig.savefig(out/'val_f1.png',dpi=160);plt.close(fig)
    except ImportError:write(out/'plot_NOT_RUN.json',dict(reason='matplotlib missing'))
    hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['artifacts_sha256.json','handoff.zip']}
    write(out/'artifacts_sha256.json',hashes)
    with zipfile.ZipFile(out/'handoff.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if not p.is_file() or p.name in ['handoff.zip','summary.json'] or p.suffix in ['.pt','.npz'] or p.name=='predictions.jsonl' or p.name=='summary.partial.json':continue
            if 'smoke' in p.parts:continue
            if 'evaluations' in p.parts and 'train' in p.parts and p.name in ['classification.json','event_metrics.json','handshake_metrics.json']:continue
            z.write(p,str(Path('evidence')/p.relative_to(out)))
        for p in list((REPO/'tools/tomato_joint_memory').glob('*.py'))+list((REPO/'tests').glob('test_tomato_joint_memory*.py'))+[REPO/'configs/tomato_joint_memory_ablation_20261112.yaml']:
            z.write(p,str(Path('source')/p.relative_to(REPO)))
    print(path,flush=True)
