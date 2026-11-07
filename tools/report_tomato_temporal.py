"""Generate the second-stage audit report from actual completed artifacts."""
from pathlib import Path
import shutil
import numpy as np
from . import tomato_dataset as ds,analyze_tomato_temporal as analysis
ROOT=Path('runs/tomato/temporal_final_20261108');BASE=Path('runs/tomato/temporal_20261108_v2');PRIOR=Path('runs/tomato/optimization_20261108');LONG=Path('runs/tomato/long_history_20261108_v2')


def run():
 rows=analysis.read_results();global_choice=ds.read_json(ROOT/'global_selection.json');g=global_choice['winner'];families=ds.read_json(ROOT/'family_summary.json');bench=ds.read_json(ROOT/'real_runtime_check.json');probe=ds.read_json(BASE/'sampling_probe.json');norm=ds.read_json(ROOT/'normalization_audit.json')
 normal=[r for r in rows if r.get('training_s',0)>0];cv=[r for r in normal if '_fold' in r['name']];fixed=[r for r in normal if '_fold' not in r['name']];mixed=[r for r in rows if r['name'] in ['hybrid_new_future','ensemble_half']]
 current_refs=[ds.read_json(PRIOR/'evaluations'/f'current_s{s}.json') for s in [42,43,44]]
 current_best=dict(name='current_only_16x10',mean=float(np.mean([r['heads']['current']['overall']['macro_f1'] for r in current_refs])),checkpoint='/home/user/vjepa2_training/runs/tomato/campaign94_20261107/experiments/current_s42/best_head.pt')
 for name,v in families.items():
  if len(v['seeds'])==3 and v['heads']['current']['mean']>current_best['mean']:
   ref=next(r for r in rows if r['name']==name and r['seed']==42);current_best=dict(name=name,mean=v['heads']['current']['mean'],checkpoint=str(Path(ref['checkpoint']).resolve()))
 future_best={}
 for h,tag in [(.5,'h05'),(1.,'h10'),(1.5,'h15')]:
  name='future' if h==1. else 'future_'+tag
  vals=[ds.read_json(PRIOR/'evaluations'/f'{name}_s{s}.json')['heads']['future']['overall']['macro_f1'] for s in [42,43,44]]
  future_best[str(h)]=dict(name='future_only_'+tag,mean=float(np.mean(vals)),sample_std=float(np.std(vals,ddof=1)))
 for name,v in families.items():
  if len(v['seeds'])==3 and v['heads']['future']['mean']>future_best['1.5']['mean']:future_best['1.5']=dict(name=name,**v['heads']['future'])
 out=Path('docs/validation/tomato_temporal_20261108');out.mkdir(parents=True,exist_ok=True)
 summary=dict(training_runs=len(normal),fixed_split_training_runs=len(fixed),cv_training_runs=len(cv),mixture_evaluations=len(mixed),
 global_selection=global_choice,families=families,best_current=current_best,best_future_by_horizon=future_best,
 runtime=bench,normalization_audit=norm,results=ds.read_json(ROOT/'results_summary.json'),
 multitask_gradient_probe=ds.read_json(ROOT/'multitask_gradient_probe.json'),sampling_probe=probe,
 threshold_feasible_count=ds.read_json(ROOT/'threshold_diagnostics.json')['feasible_count'],
 actual_openpi_warning_audit=ds.read_json(ROOT/'actual_openpi_warning_audit.json'),availability=ds.read_json(ROOT/'availability_diagnostics.json'),paired_cluster_bootstrap=ds.read_json(ROOT/'paired_cluster_bootstrap.json'),
 extraction_cost=ds.read_json(ROOT/'extraction_cost.json'),integrity=ds.read_json(ROOT/'integrity_final.json'),baseline_file='docs/validation/tomato_temporal_baseline_20261108.json')
 ds.write_json(out/'summary.json',summary)
 for name in ['global_selection_protocol.json','global_selection.json','normalization_audit.json','validation_persistence.json','causal_stream_parity.json','mixture_runtime_check.json','paired_cluster_bootstrap.json','extraction_cost.json','integrity_final.json','stable_default_inference_config.json','actual_openpi_warning_audit.json']:
  shutil.copyfile(ROOT/name,out/name)
 for source,target in [('input_schema.json','selected_input_schema.json'),('inference_config.json','selected_inference_config.json'),('file_manifest.json','selected_bundle_file_manifest.json')]:shutil.copyfile(ROOT/'selected_research_bundle'/source,out/target)
 lines=[];a=lines.append
 a('# V-JEPA 2 番茄 Agent 第二阶段实测报告 · 2026-11-08\n')
 a(f"本轮完成 **{len(normal)} 次真实冻结编码器分类头 GPU 训练**：固定76/10划分 {len(fixed)} 次、86条来源三折诊断 {len(cv)} 次；另完成 {len(mixed)} 个无需训练的推理混合评估。复用了上一轮全部分类头和特征缓存，并为16帧×5Hz历史完整提取86条训练/验证录制的独立新缓存。测试8条未用于新增推理或调参。\n")
 a('**结论：没有发现足以替换现有综合研究候选的改善。** 仍保留独立双头1.5秒 seed42 为综合研究候选，原 Joint seed42 作为已接通接口的默认。新方法在部分分类指标上有小变化，但没有达到预先定义的阶段转换可靠性门槛。该结果支持继续采用较简单的冻结编码器＋独立分类头，而不支持直接升级到更复杂的时序头。\n')
 a('## 1. 基线与数据核查\n')
 a('工作分支为 `research-tomato-agent`。原12次正式训练、76/10/8分组、旧checkpoint和测试报告均保留；历史汇报0.955/0.935/0.914等已从旧结果文件核实，属于旧测试集均值，不能与本报告验证集数字直接混用。训练前基线见 `docs/TOMATO_TEMPORAL_BASELINE_20261108.md`，含边界±0.25/0.5/1秒质量与阶段持续时间JSON。\n')
 a('94条为94个不同录制UUID；原始录制分组、跨划分采集时间区间、完全相同视频/状态SHA检查均未发现交叉。已知Smoke/pilot有4条验证、2条训练、0条测试。验证集经历多轮早停与选模，不能视作新的独立测试。未知重编码副本和未记录历史使用仍需采集日志排除。\n')
 a(f"本轮数值核对 **{len(norm['checks'])} 个 checkpoint** 的mean/std，全部与各自训练来源重算结果逐值相等；CV每折单独拟合，旧测试来源不进入任何训练。新3秒历史的86份缓存逐一核对SHA、因果索引和真实58D位置顺序，均通过。V-JEPA 2官方权重固定revision与SHA，保持冻结；无环境升级、无OpenPI/RTC改动。\n")
 a('## 2. 方法、动机与实际执行范围\n')
 a('| 方法 | 动机 | 实际训练/评估 |\n|---|---|---|')
 for row in [
 ('原Current/Future/Joint/Boundary Joint','保留真实基线','复用原12次正式训练及此前0.5/1.5秒Future头'),
 ('Joint Loss 1:0.5、1:1、1:2','检查当前/未来目标权衡','主比较时距1.5秒；0.5、1.0权重三seed，2.0仅pilot'),
 ('Focal γ=2','聚焦难窗口、类别不均衡','seed42筛选，未晋级'),
 ('边界辅助A','预测未来1.5秒内是否有真实操作类别切换','共享干线＋二分类头，权重0.2，seed42筛选'),
 ('GRU / Causal TCN','保留缓存特征的短期运动信息','4个2Hz历史anchor，GRU三seed，TCN仅pilot'),
 ('三相机门控','按观测重新分配视角权重','seed42筛选，未晋级'),
 ('位置＋速度','显式提供真实时间差计算的运动量','额外58D速度，三seed'),
 ('14D双臂/44D双手分开编码','保留不同执行器的结构','Split融合三seed'),
 ('仅视觉/仅状态','检查模态贡献','各seed42单因素消融'),
 ('推理解耦/概率均值','保留Current-only，替换或平均Future输出','两种固定组合×三seed，无重复训练'),
 ('16×5Hz历史','先在编码前扩大原始观测时间范围','3秒历史，独立缓存，三seed')]:a('| '+' | '.join(row)+' |')
 a('\n所有新训练均使用AdamW3e-4、weight decay0.01、batch128、dropout0.3，最大100epoch/patience15；早停统一按(Current F1+0.5Future F1)/1.5，不随训练Loss权重改变。GRU/TCN和Split参数量不同，因此结果是实际候选的成本/性能比较，不能据此排除容量等混杂后断言时序结构本身无效。\n')
 a('边界标签A来自人工标注类别区间，定义为(t,t+1.5]内发生任一真实操作类别变化；相邻同类别标注段合并，非相邻重复操作仍计切换。缺失阶段不伪造标签，Episode尾部无法获得完整监督时屏蔽。类别不均衡使用训练集平方根逆频率CE，A使用训练集正负比例平方根的BCE正样本权重。未来标注只进损失，不进网络输入。没有多标注者边界不确定性资料，保留硬标签并通过多时间范围评估暴露敏感性，不声称边界精确到采样间隔。\n')
 a('## 3. 验证集汇总：同为未来1.5秒\n')
 a('以下为seed42/43/44均值±样本标准差，方差、每seed原值、Precision/Recall、混淆矩阵和逐Episode结果见 summary.json。所有事件用固定threshold0.5/K2。\n')
 a('| 方法 | Current Macro-F1 | Future Macro-F1 | Future边界召回 | 误报/分钟 | 参数量（头部） |\n|---|---:|---:|---:|---:|---:|')
 for c in global_choice['all_candidates']:
  n=c['name'];rs=[r for r in rows if r['name']==n]
  if n=='prior_independent_h15':
   cstd=.0021672540390449994;fstd=.004030217660335875;params=2050568
  else:cstd=families[n]['heads']['current']['sample_std'];fstd=families[n]['heads']['future']['sample_std'];params=rs[0]['parameters']
  a(f"| {n} | {c['current_f1']:.5f} ± {cstd:.5f} | {c['future_f1']:.5f} ± {fstd:.5f} | {c['future_recall']:.1%} | {c['future_false_per_min']:.3f} | {params:,} |")
 a('\n**逐次真实训练列表**（CV另表；单seed只是筛选结果）：\n')
 a('| 方法 | seed | 选中epoch / 已跑epoch | Current F1 | Future F1 | 训练秒 | 头部P95毫秒 |\n|---|---:|---:|---:|---:|---:|---:|')
 for r in sorted(fixed,key=lambda x:(x['name'],x['seed'])):
  a(f"| {r['name']} | {r['seed']} | {r['best_epoch']} / {r['epochs']} | {r['evaluation']['current']['overall']['macro_f1']:.6f} | {r['evaluation']['future']['overall']['macro_f1']:.6f} | {r['training_s']:.1f} | {r['head_latency_ms']['p95']:.3f} |")
 a('### 与原Joint seed42的真实差值\n')
 a('原Joint1秒 seed42的验证Current/Future F1为0.972677/0.933805；同为1秒的独立头seed42为0.974811/0.944451，但三seed Future均值差仅0.001255，不能由单seed优势断言稳定改善。未来1.5秒候选与Joint1秒的分类F1目标不同，不直接相减。\n')
 a('相同threshold0.5/K2下，旧Joint1秒seed42提前命中21/56，独立1.5秒seed42为40/56；误报约1.626→0.962/分钟，命中事件中位提前量0.400→0.567秒。这是此前验证集上的实用预警改善，同时改变架构与时距。本第二阶段的新方法没有继续超越这个既有候选。\n')
 a('与同为1.5秒的原独立Future基线比较，新Joint MLP均值Future F1约高0.261个百分点，却降低预警召回并提高误报；50:50集成F1约高0.301个百分点，也没有在线综合优势。\n')
 a('## 4. 哪些改进有效，哪些未得到支持\n')
 a('1:1 Loss在seed42的Future较好，但三seed复验平均0.92309，低于1:0.5的0.92875。1:2 pilot Future为0.9032，也未改善。最终checkpoint的局部干线梯度诊断显示两种Loss设置都有负余弦批次，均值仍为正；这说明存在局部目标竞争，不足以因果解释全部泛化差距。\n')
 a('原1秒任务的Future-only验证F1均值0.941436，Joint为0.940181，差距仅0.126个百分点；seed44反而Joint更高。旧测试均值不能用于这次选参。Joint早停倾向Current、共享干线竞争、参数容量、类别与边界分布、随机种子都会影响结果；没有证据支持“Joint结构必然有问题”。\n')
 a('A辅助头AP为0.71960，原MLP的Current/Future概率边界proxy AP为0.72362；Future分类/预警也未改善，未达到预定义晋级条件。因此没有继续叠加B剩余时间、C阶段进度。两者未实现或训练，不包装成已有功能。\n')
 a('GRU与TCN在本轮上下文和容量下未优于MLP。TCN pilot训练约188秒，MLP/GRU大多仅数秒到十余秒，推理仍很轻；本轮没有依据继续增加Causal Transformer。三相机门控未晋级；Split减少参数并降低一些误报，却损失Current/Future分类且未通过可靠性门槛。\n')
 a('位置＋速度的Current均值0.972902，较匹配Joint MLP的0.970315略高、波动也较小，但仍低于已有Current-only的0.973793，预警没有改善。速度由同名实测位置和真实历史时间差计算，主要是已有运动信息的显式表示；未使用未来状态或偷偷读取原始state中的非position字段。\n')
 a('50:50概率集成的Future F1均值0.929160，高于独立Future的0.926145，但边界召回从72.0%下降到70.2%，误报从0.948增加到1.046/分钟。该小幅整体分类提升不构成综合Agent升级理由。\n')
 a('## 5. 真实阶段转换与边界质量\n')
 a(f"最终仍选 `{g['name']}`：Current F1 {g['current_f1']:.5f}，Future F1 {g['future_f1']:.5f}；Current边界召回{g['current_recall']:.1%}，seed内P95检测延迟的均值{g['current_delay_p95_s']:.3f}秒。Future边界召回{g['future_recall']:.1%}、漏检{g['future_miss_rate']:.1%}、误报{g['future_false_per_min']:.3f}/分钟、命中事件中位提前量的seed均值{g['future_lead_p50_seed_mean_s']:.3f}秒。\n")
 a('事件定义严格：预测事件生成不读取真值；首次稳定类别初始化，之后按真实from/to一对一匹配。Future需在[边界−时距,边界)内，Current需在边界后2秒内且未进入下一阶段；未匹配事件为误报，未匹配边界为漏检。2Hz/K2有约0.5秒确认开销，提前量只针对命中事件，不能忽略漏检。\n')
 a('全局规则在新长历史结果前锁定：先检查Current下降≤0.005、召回≥90%、P95≤1.5秒、Future召回≥80%、误报≤1/分钟，再按预定义在线分数排序，代表seed固定42。另检查9个三seed候选×64个Current/Future工作点，共576组合，全部无可行解；没有事后放宽门槛。\n')
 a('新模型边界前后0.25、0.5、1.0秒的完整分类支持数、各类F1见 `boundary_neighborhoods.json`（本地）及版本化JSON。0.25秒只是2Hz真实anchor子集，未插值伪造4Hz预测；未来标签时间目标不同的F1不可直接横比。\n')
 a('考虑50ms结果可用时间后，所选候选的Future召回仍为72.0%；100ms敏感性条件下降为66.1%，误报约1.088/分钟。这是额外可用时延敏感性，不冒充实测网络/机器人延迟。\n')
 a('录制级表现继承完整10条验证分析：一个3阶段Episode缺adjust，固定四类Macro-F1最高0.75；不能按该数值直接判最差。先前难例`6d46...`贡献seed42候选23次误报中的9次。各Episode的类别支持、混淆和事件时间保留在独立结果文件。\n')
 boot=ds.read_json(ROOT/'paired_cluster_bootstrap.json')['results']['ensemble_half']['future']
 a(f"按10条原始录制聚类、5000次配对bootstrap，集成相对已有独立双头的等Episode Future F1差值为{boot['mean_equal_episode_f1_delta']:.5f}，95%区间[{boot['ci95'][0]:.5f}, {boot['ci95'][1]:.5f}]，跨0。该分析仍是复用验证集上的探索结果，不构成独立确认。\n")
 a('## 6. 分组交叉验证\n')
 a('使用原训练＋验证86条录制、三折原始来源分组、seed42；每折重新初始化头部并仅用本折训练来源拟合统计。原8条测试不参与。它是诊断性CV，不是独立确认实验。\n')
 a('| 方法/折 | 训练/验证来源 | Current F1 | Future F1 | Future边界召回 |\n|---|---:|---:|---:|---:|')
 for r in sorted(cv,key=lambda r:r['name']):
  identity=ds.read_json(Path(r['checkpoint']).parent/'identity.json');a(f"| {r['name']} | {len(identity['split']['train'])}/{len(identity['split']['val'])} | {r['evaluation']['current']['overall']['macro_f1']:.5f} | {r['evaluation']['future']['overall']['macro_f1']:.5f} | {r['evaluation']['future']['events']['0.5/2']['recall']:.1%} |")
 a('\nSplit相对MLP的差值在不同折改变方向，预警召回平均几乎相同，未形成稳定优势。CV分数低于固定验证集还受训练来源从76减少到57/58和折间难度影响，不能把差距直接归咎于泄漏。长历史未满足预定义的CV晋级条件，故没有盲目追加CV。\n')
 a('## 7. 时间窗口与预测频率\n')
 a('| 历史 | 实际编码兼容 | P95毫秒/窗口 | 完整精度试验 |\n|---|---|---:|---|')
 for n,d in probe['designs'].items():a(f"| {n} | 是，特征{d['episodes'][0]['dimension']}维 | {d['latency']['p95_ms']:.2f} | "+('三seed，独立86条缓存' if n=='16x5' else ('原基线缓存复用' if n=='16x10' else '仅20个真实窗口兼容性/成本探针'))+' |')
 a('\n8帧的3536D特征与旧4000D分类头不兼容，未偷偷填充后复用旧头。非均匀采样跨度3.2秒，保留最新帧；只做编码/成本探针，没有完整训练，不宣称准确率改善。所有新cache身份包含历史设计并由带完整joint_names的manifest绑定。\n')
 a('16×5Hz完整试验Current0.972162±0.005054，Future0.909047±0.008572，Future预警召回54.2%、误报1.478/分钟，明显不支持替换16×10Hz。它同时改变视觉和状态历史跨度，是历史采样单因素对照，不能解读成纯视觉效果。\n')
 a('0.5/1.0/1.5秒Future已用相同冻结特征与MLP比较；验证F1分别约0.96134/0.94144/0.92615，严格预警不随整体F1排序。0.25/0.75秒本轮未继续训练；若采用“请求时间之后第一个录制帧”，30Hz有效时距分别8帧=0.266667秒、23帧=0.766667秒。映射函数和测试已实现，不暗中四舍五入成原请求值。\n')
 a('| 更新频率 | 不同真实观测数 | P95毫秒 | 超过周期次数 |\n|---|---:|---:|---:|')
 for hz,d in probe['frequency_feasibility'].items():a(f"| {hz}Hz | {d['unique_observations']} | {d['latency']['p95_ms']:.2f} | {d['deadline_misses']} |")
 a('\n这些频率测试每次重新编码新的录制RGB/状态，未把过期特征冒充新观测；不含采集、网络、ROS和动作模型。计算预算支持4/5Hz，但精度研究仍在2Hz。GRU/TCN的上下文按2Hz训练，运行时拒绝静默改成高频上下文；尚未训练高频新缓存模型。\n')
 cost=ds.read_json(ROOT/'extraction_cost.json')
 a(f"新增完整缓存包含{cost['recordings']}条录制、{cost['windows']:,}个窗口，压缩后约{cost['compressed_cache_bytes']/1024**2:.1f}MiB。两次保留记录中已完成工作的编码/拼接约{cost['total_recorded_encode_s']/60:.1f}分钟、视频解码约{cost['total_recorded_decode_s']/60:.1f}分钟，累计记录wall约{(cost['new_run_wall_s']+cost['prior_completed_wall_s'])/60:.1f}分钟；7份完整缓存只读复用，不重复计入工作。提取batch16峰值PyTorch已分配显存约{cost['extraction_peak_allocated_bytes']/1024**3:.2f}GiB。未估算中断时少量未完成窗口成本。\n")
 a('## 8. 实际运行、缓存一致性与接口\n')
 a(f"新16×5Hz checkpoint实际回放60个不同观测、2Hz定时调用：P95 {bench['latency']['p95_ms']:.2f}ms，最大{bench['latency']['max_ms']:.2f}ms，超500ms为0。5次warm-up＋60次调用恰好65次encoder前向。冻结encoder参数{bench['encoder_parameters']:,}，三相机共享同一权重；PyTorch峰值已分配显存约{bench['peak_allocated_bytes']/1024**3:.3f}GiB，不含CUDA context。\n")
 a(f"batch16缓存与batch1真实编码在60窗口上状态部分逐值一致，视觉最大差{bench['cache_live_feature_max_abs_error']:.9f}、平均绝对差{bench['cache_live_visual_mean_abs_error']:.3g}，Current/Future argmax均0个分歧。该小差异与BF16批量计算精度一致，未伪称全部视觉逐位相同。真实100窗口GRU FIFO与批量因果历史logits最大差小于3e-6，通过。\n")
 a('新checkpoint输出实际进入未修改的OpenPI概率校验、七位置映射和chunk元数据绑定；独立混合头也经过真实encoder调用检查。时钟明确是离线回放重基准；不控制机器人、不调用动作模型、不改RTC。输入按完整58个.position名称对齐，5Hz历史需传真实历史时间戳，错误采样间隔、过期/乱序或错误会话会被拒绝。\n')
 a('另核验了一个容易混淆的接口细节：现有Provider的两次确认仅作用于Current阶段推进；Future是在threshold0.8且等于下一有序类别时立即给出level warning，没有Future K2。上面的K2事件指标属于独立评估协议，未写入现有接口，也不等于接口warning覆盖率。\n')
 actual=ds.read_json(ROOT/'actual_openpi_warning_audit.json')['results']
 for name in ['joint','independent_h15']:
  rs=[v for v in actual if v['name']==name]
  a(f"现有接口 {name} 的三seed离线回放：真实切换前任一窗口正确预警覆盖{np.mean([v['any_correct_warning_recall'] for v in rs]):.1%}，命中边界的最早预警中位时间均值{np.mean([v['conditional_lead_p50_s'] for v in rs]):.3f}秒；{np.mean([v['false_warning_window_fraction'] for v in rs]):.1%}的告警窗口不在正确提前区间，告警上升事件误报约{np.mean([v['false_warning_rising_per_minute'] for v in rs]):.3f}/分钟。\n")
 a('这些非提前窗口包括阶段切换后Current尚未推进时的延迟告警，以及过早/进度不符告警，不应称作同样比例的Future分类错误；详细分类计数见actual_openpi_warning_audit.json。宽松的“任一窗口覆盖”与严格from/to事件召回定义不同，不能拿前者替代锁定的模型选择规则或宣称部署可靠。\n')
 a('## 9. 最佳模型、完整配置与保存位置\n')
 a(f"- 最佳Current分类均值：**{current_best['name']}，{current_best['mean']:.6f}**；代表checkpoint `{current_best['checkpoint']}`。\n- 最佳1秒Future分类：原Future-only，验证均值0.941436；代表 `runs/tomato/campaign94_20261107/experiments/future_s42/best_head.pt`。\n- 最佳1.5秒Future分类均值：**{future_best['1.5']['name']}，{future_best['1.5']['mean']:.6f}**；整体F1小提升但预警更差，未作为综合升级。\n- 最佳综合研究候选：**独立Current/Future双头1.5秒 seed42**；SHA `{global_choice['checkpoint_sha256']}`。\n- 当前保留接口默认：`{global_choice['stable_default']}`。它已验证接口兼容，不等于已验证真实机器人转换可靠性；本轮没有自动替换默认。\n")
 a(f"现有默认的完整输入/时序/阶段映射配置另存 `docs/validation/tomato_temporal_20261108/stable_default_inference_config.json`。完整研究包在 `{(ROOT/'selected_research_bundle').resolve()}`：agent.pt嵌入两份头部权重和统计，另存normalization.npz、input_schema.json、inference_config.json与逐文件SHA。官方1.3GB编码器仍引用本地固定revision/权重SHA，不重复拷贝或上传。原实验的best.pt、last_state.pt、训练历史、优化器/RNG状态、数据与源码身份都保存在各自独立目录。\n")
 a('核心代码：`tomato_temporal_heads.py`、`train_tomato_temporal.py`、`tomato_time_sampling.py`、`train_tomato_long_history.py`、`tomato_causal_agent.py`、`tomato_agent_mixture.py`、`eval_tomato_mixture.py`、`probe_tomato_sampling.py`；汇总、审计和回放入口为`analyze_tomato_temporal.py`、`select_tomato_temporal.py`、`check_tomato_temporal_runtime.py`。独立配置在 `configs/tomato_temporal_20261108/`。\n')
 a('补充代码入口：`train_tomato_current_velocity.py`、`probe_tomato_multitask.py`、`cross_validate_tomato_history.py`、`check_tomato_openpi_warnings.py`、`report_tomato_temporal.py`，分别负责当前识别专门复验、梯度诊断、有条件CV、实际接口告警语义审计与结果报告。\n')
 a('中途发现原始state名称同时含非position字段，速度试验已按.position白名单修复；已完成8个pilot只读复用，未重复训练或覆盖。长历史清单在正式头部训练前补全实际监督统计，安全重启到新目录并复用7份已完成缓存，原日志保留。旧时序训练以f9e53e8为基准Git并保存实际源码SHA/snapshot，后续代码提交10c15c5；基准Git SHA不被冒充成完整训练源码版本。\n')
 a('106项番茄相关测试通过，真实新视频接口、缓存/在线数值比对、GRU流式因果回放和所有训练来源归一化审计通过。新runner保存完整恢复状态，但本轮没有额外启动重复训练来声称其所有架构中断恢复均已逐位验收。旧checkpoint、数据与测试结果的最终SHA复核见完整性清单。\n')
 a('## 10. 下一轮最值得做的事\n')
 a('先准备全新的独立录制，覆盖当前误报集中Episode同类场景、长稳定阶段、小动作干扰、不同操作速度与边界标注一致性。固定本轮选出的checkpoint、时距、阈值和事件定义后一次性评估，不再用旧8条测试或本10条验证继续证明可靠性。\n')
 a('低成本模型研究优先做等参数量GRU/TCN对照和边界标注盲审；目前时序头容量与MLP不同，不能把负结果概括为时序信息无用。4Hz新鲜输入可能减少确认延迟，但需相应准确率评估和时序上下文重采样；现有时延证据只证明算得过来。没有证据时不继续堆叠Transformer、完整Token/Attentive Readout或所有参数的笛卡尔积。\n')
 Path('docs/TOMATO_TEMPORAL_RESULTS_20261108.md').write_text('\n'.join(lines)+'\n')
 # Neighborhood details remain lightweight enough to version alongside seed summaries.
 shutil.copyfile(ROOT/'boundary_neighborhoods.json',out/'boundary_neighborhoods.json')
 print('FINAL REPORT WRITTEN',len(rows),'evaluation rows',len(normal),'training runs',flush=True)

if __name__=='__main__':run()
