"""Summarize predeclared controlled runs, without opening held-out results."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from . import tomato_dataset as ds


def rows(p): return [json.loads(l) for l in p.read_text().splitlines()]
def table(headers, data):
    esc = lambda v: str(v).replace('|', '\\|')
    return '\n'.join(['| '+' | '.join(map(esc,headers))+' |', '|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(esc,r))+' |' for r in data])


def main():
    base = ds.REPO/'runs/tomato/controlled_20261108'
    protocol = ds.read_json(base/'protocol.json')
    assert ds.read_json(base/'complete.json')['runs'] == 9
    modes = protocol['modes']; seeds = protocol['seeds']
    result = {(s,m):ds.read_json(base/f'seed{s}_{m}/result.json') for s in seeds for m in modes}
    epochs = {(s,m):rows(base/f'seed{s}_{m}/epochs.jsonl') for s in seeds for m in modes}
    for s in seeds:
        assert len({result[s,m]['order_sha256'] for m in modes}) == 1
        assert all(result[s,m]['updates'] == 17000 for m in modes)
    reproduce = {}
    for m in modes:
        a=base/f'seed42_{m}'; b=base/'repro_seed42'/f'seed42_{m}'
        reproduce[m] = {n:ds.digest(a/n)==ds.digest(b/n) for n in ['epochs.jsonl','gradients.jsonl','clipping.jsonl']}
        assert all(reproduce[m].values())
    old=ds.read_json(ds.REPO/'runs/tomato/local93_20261107/analysis/original_result_sha256.json')
    protection={p:ds.digest(ds.REPO/p)==h for p,h in old.items()}
    assert all(protection.values())
    grads={}; clips={}
    for s in seeds:
        gg=rows(base/f'seed{s}_joint/gradients.jsonl')
        groups={}
        for label,rr in [('all',gg),('epoch1_10',[g for g in gg if g['epoch']<=10]),('epoch11_100',[g for g in gg if g['epoch']>10])]:
            valid=[g for g in rr if min(g['current_norm'],g['future_norm'])>1e-8]
            groups[label]=dict(samples=len(rr),nonzero_samples=len(valid),
                cosine_median=float(np.median([g['cosine'] for g in valid])),
                negative_fraction=float(np.mean([g['cosine']<0 for g in valid])),
                current_norm_median=float(np.median([g['current_norm'] for g in rr])),
                future_norm_median=float(np.median([g['future_norm'] for g in rr])),
                weighted_ratio_median=float(np.median([g['weighted_future_current_ratio'] for g in valid])))
        grads[s]=groups
        for m in modes:
            rr=rows(base/f'seed{s}_{m}/clipping.jsonl')
            clips[f'{s}_{m}']=dict(fraction=float(np.mean([g['clipped'] for g in rr])),
                first10_fraction=float(np.mean([g['clipped'] for g in rr if g['epoch']<=10])),
                norm_p50_p95=np.percentile([g['norm'] for g in rr],[50,95]).tolist())
    comparison={}
    for selection in ['best_validation','final']:
        comparison[selection]={}
        for s in seeds:
            a=result[s,'current_only']['reports'][selection]['current']
            b=result[s,'future_only']['reports'][selection]['future']
            c=result[s,'joint']['reports'][selection]
            comparison[selection][s]=dict(current_delta=c['current']['macro_f1']-a['macro_f1'],
                future_delta=c['future']['macro_f1']-b['macro_f1'],
                future_vs_oracle=b['macro_f1']-b['persistence_baselines']['oracle_current_persistence']['macro_f1'],
                joint_future_vs_oracle=c['future']['macro_f1']-c['future']['persistence_baselines']['oracle_current_persistence']['macro_f1'])
    # Only reload our own research checkpoints, never an external/untrusted pickle.
    from .test_tomato_controlled_experiment import read_inputs
    from .tomato_runtime import initialize_runtime
    initialize_runtime(dict(device='cpu',seed=42))
    import torch
    from . import tomato_phase_model as model
    torch.set_num_threads(4)
    _,_,arrays,_=read_inputs(ds.REPO/'runs/tomato/local93_20261107')
    va=arrays['val']; valid=va['future_valid_mask']
    changed=va['current_labels']!=va['future_labels']
    decomposition={}
    for s in seeds:
        for m in ['future_only','joint']:
            saved=torch.load(base/f'seed{s}_{m}/research_best.pt',map_location='cpu',weights_only=False)
            head=model.joint_head_model(58).eval();head.load_state_dict(saved['head'])
            z=model.predict_heads(head,va['features'],saved['mean'],saved['std'])['future_logits']
            from .tomato_metrics import scores
            assert abs(scores(z[valid],va['future_labels'][valid])['macro_f1']-
                       result[s,m]['reports']['best_validation']['future']['macro_f1']) < 1e-12
            correct=z.argmax(1)==va['future_labels']
            decomposition[f'{s}_{m}']=dict(stable_errors=int(np.sum(valid & ~changed & ~correct)),
                stable_total=int(np.sum(valid & ~changed)),changed_correct=int(np.sum(valid & changed & correct)),
                changed_total=int(np.sum(valid & changed)))
    summary=dict(protocol=protocol,comparisons=comparison,gradients=grads,clipping=clips,
        reproduction=reproduce,decomposition=decomposition,original_files_unchanged=len(old),test_evaluated=False)
    ds.write_json(base/'summary.json',summary)
    text=['# 番茄93条数据：多seed受控实验验收报告（2026-11-08）',
        '本轮依据《TOMATO_LOCAL93_RESULT_ANALYSIS.md》第8节执行训练/验证受控对照。未使用9条测试来源进行重新推理、选模或调参；结果均为真实冻结特征上的训练/验证结果。原模型结构、损失与生产训练流程保持原样，新增独立诊断入口。没有执行git commit或push。',
        '## 执行协议与范围',
        '- 预先固定seed=42/43/44，三模式共9次训练；原75训练来源、9验证来源；原冻结三视角V-JEPA 2与58位置关节特征。未来1秒，λ_current=1、λ_future=0.5，AdamW lr=0.0003、weight_decay=0.01，batch=128，梯度裁剪1.0。',
        '- 三模式每轮共同使用同一个有效/无效尾部分摊批次表，170步/轮、100轮=17,000步。每个当前锚点每轮一次，Future只对有效目标求损失，不填补尾部。每个seed的完整批次顺序SHA相同。',
        '- 同seed从同一JointHead初始化：共享层相同，对应Current/Future输出层也分别相同。单头只优化共享层和对应输出层。所有模式在构造后重置相同Dropout随机流；输入顺序、张量形状相同。Future单头的初始输出层因此与旧单头初始化不同，这是控制变量设计，不是旧训练的逐值复现。',
        '- 各模式完整运行100轮，不启用早停，既报告既定验证目标的最佳checkpoint，也报告相同17,000步终点；无学习率/λ/阈值扫描。Joint最佳目标仍为(Val_Current_F1+0.5*Val_Future_F1)/1.5，NLL仅用于同F1时择优。',
        '- 每10个训练batch记录一次Joint共享层两头梯度，共每seed 1,700次；在同一次含Dropout的forward图上测量，然后执行原组合损失backward。每步记录全部有效参数的裁剪前范数与裁剪倍率。梯度统计没有额外forward。',
        '- 训练和验证共84来源的标注、Parquet及NPZ哈希重新核验；复用冻结缓存，不解码视频、不重新提取Encoder。载入的原manifest含测试来源元数据/锁定标注快照，但程序不访问其数据文件，也不将其标签用于训练/评估；未读取旧测试预测或重新评估测试集。',
        '## 验证结果',
        'F1为固定四类Macro-F1。下表“最佳”分别使用各模式原有验证目标，两个Joint头来自同一轮，不能独立拼接。验证集结果用于诊断，不构成无偏的最终泛化精度。']
    for selection in ['best_validation','final']:
        text += [f'### {selection}',table(['seed','模式','最佳epoch','Current F1','Future F1','Current NLL','Future NLL'],[
            [s,m,result[s,m]['best_epoch'] if selection=='best_validation' else 100,
             *[f"{result[s,m]['reports'][selection][h]['macro_f1']:.6f}" if result[s,m]['reports'][selection][h] else '—' for h in ['current','future']],
             *[f"{result[s,m]['reports'][selection][h]['nll']:.6f}" if result[s,m]['reports'][selection][h] else '—' for h in ['current','future']]] for s in seeds for m in modes])]
        text += [table(['头','跨seed F1均值 ± 样本标准差','最小–最大'],[
            [m+'/'+h,f"{np.mean(v):.6f} ± {np.std(v,ddof=1):.6f}",f'{min(v):.6f}–{max(v):.6f}']
            for m,h in [('current_only','current'),('future_only','future'),('joint','current'),('joint','future')]
            for v in [[result[s,m]['reports'][selection][h]['macro_f1'] for s in seeds]]])]
    text += ['### Joint各头独立验证上限（仅诊断，不拼接或部署权重）',
        table(['seed','头','该头最佳epoch','该头最高验证F1'],[
            [s,h,b['epoch'],f"{b['val'][h]['macro_f1']:.6f}"] for s in seeds for h in ['current','future']
            for b in [max(epochs[s,'joint'],key=lambda e:(e['val'][h]['macro_f1'],-e['val'][h]['nll']))]]),
        'seed44的Joint当前头独立最高F1=0.954459，高于该seed独立Current-only最佳0.952101；但Joint联合目标选中epoch26的当前F1约0.94788。因此，当前头的部分下降可以由联合选模取舍解释，不能全部归因于共享层表征退化。seed42两头最优同在epoch5，仍存在当前头差距，说明选模取舍也不是全部解释。这里仅重读已有验证日志，没有改变选模或用测试挑轮数。']
    text += ['### 配对差值（百分点）',table(['选择','seed','Joint Current − A','Joint Future − B','B − Oracle持久性','Joint Future − Oracle'],[
        [sel,s,*[f'{v[k]*100:+.3f}' for k in ['current_delta','future_delta','future_vs_oracle','joint_future_vs_oracle']]] for sel,dd in comparison.items() for s,v in dd.items()]),
        'Oracle持久性使用真实当前阶段，不能部署。最终结论应结合可部署当前头持久性、事件表现及误报；完整指标保存在每次result.json，未用测试集决定哪种模型胜出。',
        '### 验证集最佳checkpoint的提前识别与误报',table(['seed','Future头','切换事件识别','提前量P50秒','窗外误报次数','误报/分钟','稳定区错误/总数','切换锚点正确/总数'],[
            [s,m,f"{a['detected_events']}/{a['evaluable_events']}",f"{a['lead_time_p50_s']:.3f}",a['out_of_window_false_alarm_count'],f"{a['out_of_window_false_alarms_per_minute']:.3f}",
             f"{decomposition[f'{s}_{m}']['stable_errors']}/{decomposition[f'{s}_{m}']['stable_total']}",
             f"{decomposition[f'{s}_{m}']['changed_correct']}/{decomposition[f'{s}_{m}']['changed_total']}"]
            for s in seeds for m in ['future_only','joint'] for a in [result[s,m]['reports']['best_validation']['future']['anticipation']]]),
        '事件召回指真实边界前1秒内至少一个锚点正确预测新阶段；提前量只统计成功事件。窗外误报是依赖真实当前阶段定义的离线诊断，不等于部署告警率。',
        '## 梯度与裁剪',
        '下表两头范数为未乘λ、未除1.5且未裁剪的共享层CE梯度；加权范数比为0.5×Future范数/Current范数。只有两范数均>1e-8的采样点进入cosine/比值统计，以避免近零梯度方向失真。负cosine表示该步两个任务梯度方向冲突，不能单独证明负迁移导致验证下降。',
        table(['seed','阶段','样本/有效','cosine中位数','负cosine比例','Current范数中位','Future范数中位','加权范数比中位'],[
            [s,g,f"{v['samples']}/{v['nonzero_samples']}",f"{v['cosine_median']:.3f}",f"{v['negative_fraction']:.1%}",f"{v['current_norm_median']:.5f}",f"{v['future_norm_median']:.5f}",f"{v['weighted_ratio_median']:.3f}"] for s,dd in grads.items() for g,v in dd.items()]),
        table(['seed/模式','全部17,000步裁剪比例','前10轮裁剪比例','裁剪前norm P50/P95'],[
            [k,f"{v['fraction']:.2%}",f"{v['first10_fraction']:.2%}",'/'.join(f'{n:.4f}' for n in v['norm_p50_p95'])] for k,v in clips.items()]),
        '裁剪统计覆盖共享层及有效输出层。Joint组合损失包含/1.5，梯度还存在抵消或增强；其总裁剪率与单头不同，并不说明实现错误。不会据此自动修改λ、裁剪阈值或优化器。',
        '## 可重复性与回归',
        '- seed42的三个模式另外完整重跑100轮。每个模式的epochs.jsonl、gradients.jsonl、clipping.jsonl均逐字节相同。即训练/验证数值轨迹和梯度日志精确复现；耗时与进程信息不要求相同。',
        f'- 原93条实验的{len(old)}个输出文件SHA全部保持不变；源码中原训练/模型/数据/评估文件未修改。已有用户文档修改保留。新增3项针对公共批次覆盖、梯度观测不干扰backward/RNG、真实Dropout MLP与AdamW更新逐值不变的测试。',
        '回归测试实际输出：\n\n```text\n'+(base/'regression_tests.log').read_text().strip()+'\n```',
        '## 结论与开发决策边界',
        '1. 在这组受控实验的验证最佳checkpoint上，Joint Current三个seed均略低于独立Current；Joint Future在seed42、44高于Future-only，在seed43较低。方向并不完全一致，不能将旧测试集上Joint Future的优势视为稳定结论。相比只比较单一seed，本轮显示未来头对初始化/训练随机性的依赖。',
        '2. 按共同第100轮终点比较，Current三个seed均是Joint较低；Future仅seed42的Joint较高，其余两个较低。全部模式的固定终点F1均低于对应验证最佳，支持持续训练后的过拟合现象；继续增加训练轮数不是已被验证的改善方案。',
        '3. λ=2:1并不保证共享层梯度强度2:1。梯度统计提供任务方向关系的直接证据，但必须结合前10轮与后期统计及有效梯度计数阅读；小范数的方向冲突可能对实际更新影响很小。也不能将整段训练的cosine中位数概括为所有阶段始终冲突或始终协同。',
        '具体而言：前10轮cosine中位数约0.116–0.142，负方向采样约14.1%–21.8%；第11–100轮中位数降为0.012–0.016，负方向约37.4%–38.6%。前10轮加权Future/Current范数比中位数约0.664–0.733，后期升到1.615–1.822。后期Future更新作用更大且方向冲突更常见，与两任务收敛速度不同一致；这是机制线索，不构成“改变λ即可改善”的实验证明。',
        '本轮结果只涉及既有75/9划分、3个seed；100轮多次验证比较会产生验证选择偏差，不能把最佳验证分数作为测试精度，也不能据此认定跨日期/操作者泛化已解决。固定终点可能已经过拟合，因此必须与验证最佳结果一起阅读。',
        '原实验证据中的标签对齐、损失mask及选模逻辑没有因此被推翻。受控设计排除了批次顺序、更新上限、对应初始化和Dropout流的主要混杂，但仍不能仅以梯度cosine给出共享结构的单一因果解释。联合验证选模和每头独立目标存在取舍；本轮不更改选模逻辑、不增加独立部署checkpoint或新模型。',
        '下一步应由用户结合下表数值决定：是否继续使用Joint作候选、是否开展预先约定的跨采集会话验证、或是否需要更细的边界与误报检查。在新独立保留数据准备前，不对原测试集反复调参。新增数据采集、人工逐帧复审、部署验证未执行，本轮也没有声称完成这些工作。',
        '## 文件与运行命令',
        '- 新增 tools/test_tomato_controlled_experiment.py：受控训练、采样梯度、裁剪、锁定数据复用及验证报告。\n- 新增 tests/test_tomato_controlled_experiment.py：3项回归测试。\n- 新增 tools/report_tomato_controlled_experiment.py：汇总、轨迹图与证据包。\n- 新增 docs/TOMATO_CONTROLLED_EXPERIMENT_REPORT_20261108.md：本报告。\n- runs/tomato/controlled_20261108/：protocol.json、9次训练结果、重跑结果、逐轮指标、梯度/裁剪日志、研究用权重及汇总。研究checkpoint不是生产checkpoint格式。',
        '```bash\ncd /home/user/pi05/vjepa2_training\n.venv-tomato/bin/python -m tools.test_tomato_controlled_experiment\n.venv-tomato/bin/python -m tools.test_tomato_controlled_experiment --seeds 42 --output runs/tomato/controlled_20261108/repro_seed42\n.venv-tomato/bin/python -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py\n.venv-tomato/bin/python -m tools.report_tomato_controlled_experiment\n```',
        '已完成的目录只在协议相同且有result.json时跳过；发现未完成输出则拒绝覆盖。若要重跑请指定新输出目录。没有自动提交。']
    report=ds.REPO/'docs/TOMATO_CONTROLLED_EXPERIMENT_REPORT_20261108.md'
    report.write_text('\n\n'.join(text)+'\n')
    # Standalone scientific plot; no changes to the product or training code.
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,h,singles in [(axes[0,0],'current','current_only'),(axes[0,1],'future','future_only')]:
        for m in [singles,'joint']:
            yy=np.array([[e['val'][h]['macro_f1'] for e in epochs[s,m]] for s in seeds])
            ax.plot(np.arange(1,101),yy.mean(0),label=m)
            ax.fill_between(np.arange(1,101),yy.min(0),yy.max(0),alpha=.15)
        ax.set(title=h+' validation Macro-F1',xlabel='Epoch');ax.legend()
    for s in seeds:
        gg=rows(base/f'seed{s}_joint/gradients.jsonl')
        axes[1,0].plot(range(1,101),[np.median([g['cosine'] for g in gg if g['epoch']==i]) for i in range(1,101)],label=str(s))
    axes[1,0].axhline(0,color='grey',lw=.7);axes[1,0].set(title='Joint shared gradient cosine (median)',xlabel='Epoch');axes[1,0].legend(title='seed')
    for m in modes:
        vv=[rows(base/f'seed{s}_{m}/clipping.jsonl') for s in seeds]
        axes[1,1].plot(range(1,101),[np.mean([g['clipped'] for rr in vv for g in rr if g['epoch']==i]) for i in range(1,101)],label=m)
    axes[1,1].set(title='Fraction of updates clipped (seed mean)',xlabel='Epoch');axes[1,1].legend()
    fig.tight_layout();fig.savefig(base/'validation_and_gradients.png',dpi=160);plt.close(fig)
    with zipfile.ZipFile(base/'Tomato_Controlled_Experiment_Evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        files=[report,*[p for p in base.rglob('*') if p.is_file() and p.suffix not in ['.pt','.zip']]]
        files += [ds.REPO/p for p in ['tools/test_tomato_controlled_experiment.py','tools/report_tomato_controlled_experiment.py','tests/test_tomato_controlled_experiment.py']]
        for p in files:z.write(p,str(p.relative_to(ds.REPO)))
    print(report)


if __name__=='__main__': main()
