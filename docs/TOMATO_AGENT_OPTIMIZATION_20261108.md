# 番茄 Agent 验证集优化实测报告 · 2026-11-08

本轮首选后续研究架构为**独立双头**：共享一次冻结 V-JEPA 2 编码，两套独立 MLP 分别选模。它隔离了两个训练目标，额外头部成本很小；但目前**没有任何方案通过预先规定的可靠转换预警门槛**。因此保存 1.5 秒独立双头 seed42 为研究候选，原 Joint seed42 继续作为稳定接口默认，不自动替换。

所有新增准确率结论均来自原有 **10 条验证录制**。旧测试集已在上一轮被查看，本轮只核验其完整性，没有重新做测试预测，也没有用旧测试均值调参。验证集也经历过 Smoke、早停和候选搜索，因此不能代替全新独立测试。

## 1. 数据划分、来源与冻结

| 划分 | 原始录制 / Episode | 2 Hz 窗口 | pickup | harvest | return | adjust |
|---|---:|---:|---:|---:|---:|---:|
| train | 76 | 22168 | 2908 | 14729 | 1480 | 3051 |
| val | 10 | 2898 | 327 | 1999 | 187 | 385 |
| test | 8 | 2406 | 280 | 1664 | 145 | 317 |

94 条对应 94 个不同 `source_recording_id`。按原始录制分组，跨划分 UUID、原始录制时间区间、完全相同视频/状态文件 SHA 的交叉检查均未发现重叠。历史窗口严格来自本 Episode 且所有输入帧 ≤ 当前 anchor；16 帧、10 Hz 历史，原始 30 Hz，标签时距 0.5/1/1.5 秒分别为 +15/+30/+45 原始帧，尾部越界样本屏蔽 Future 监督。窗口彼此重叠仅发生在同一划分内，不作为独立样本估计置信区间。

已知 6 条 Smoke / pilot 来源分配：

- `34a4b430af2c4fd6a1912ff03628ef3f` → train
- `471c02da39ac4fa7ba4fcd140f9cb1ed` → val
- `5befbcd6c7f143028144b68d754c6883` → val
- `76354615cf9746758c5c7e6e5ab579d4` → val
- `7d879f9d5d5a4be09beb55cf88591b0d` → val
- `e826c474e80547d5a3b24e5eb0f30977` → train

其中 4 条进入验证、2 条进入训练，测试中无已知 Smoke 来源。无法仅凭 UUID 和 SHA 排除未知重编码副本或未记录的历史使用，需采集日志进一步确认。

12 份旧 checkpoint 的 mean/std 均与 **22,168 个训练窗口**重新计算的统计逐值完全一致；没有使用验证/测试统计。冻结并复核旧 campaign **423 个文件**，包括旧 checkpoint、测试预测、报告及元数据；数据清单哈希不变。划分 SHA：`00783ed7fbcc45ba8d160049f5a6764e8cec832977055b72ba4d4f8a01d3e0e0`。

冻结文件清单见 `docs/validation/tomato_optimization_20261108/freeze.json`；数据字节版本以旧 prepared manifest 内逐文件 SHA 和 annotation SHA 为准。

## 2. 训练与公平比较

复用全部 12 次既有正式训练；没有重新提取 94 条视频特征。额外仅训练 6 个 Future-only MLP：0.5 秒与 1.5 秒各 seeds 42/43/44，共 192 个实际 epoch；100 epoch 上限、patience 15、AdamW 3e-4、weight decay 0.01、batch128、dropout0.3 与原方案相同。9 份独立双头包由同 seed 的已有 Current 与相应 Future 权重直接组合，未重复训练。

输入始终为同一 4000D 冻结特征（3×1024 视觉 +16×58 状态），同一 split、同一训练统计与相同种子集合。Current/Future 各自 MLP 为 4000→256→4；Joint 共享 4000→256 干线，两个 256→4 输出。Boundary Joint 为边界样本倍率2。独立头总参数 2,050,568，Joint 1,026,312；因此不把本次差异解释为排除了参数容量影响的纯因果实验。

| 架构 / 时距 | Current F1 均值±样本SD | Future F1 均值±样本SD |
|---|---:|---:|
| current / 1.0s | 0.97379 ± 0.00217 | — |
| future / 1.0s | — | 0.94144 ± 0.00268 |
| joint / 1.0s | 0.97003 ± 0.00321 | 0.94018 ± 0.00905 |
| joint_transition / 1.0s | 0.97094 ± 0.00202 | 0.93434 ± 0.00215 |
| independent_h05 / 0.5s | 0.97379 ± 0.00217 | 0.96134 ± 0.00133 |
| independent_h10 / 1.0s | 0.97379 ± 0.00217 | 0.94144 ± 0.00268 |
| independent_h15 / 1.5s | 0.97379 ± 0.00217 | 0.92615 ± 0.00403 |

12 次旧正式训练逐种子验证结果：

| 方案 | Seed | 最优 epoch | Current F1 | Future F1 |
|---|---:|---:|---:|---:|
| current | 42 | 6 | 0.974811 | — |
| current | 43 | 6 | 0.971304 | — |
| current | 44 | 30 | 0.975264 | — |
| future | 42 | 4 | — | 0.944451 |
| future | 43 | 21 | — | 0.939323 |
| future | 44 | 14 | — | 0.940534 |
| joint | 42 | 7 | 0.972677 | 0.933805 |
| joint | 43 | 5 | 0.970953 | 0.936192 |
| joint | 44 | 18 | 0.966466 | 0.950545 |
| joint_transition | 42 | 7 | 0.971123 | 0.936476 |
| joint_transition | 43 | 10 | 0.968842 | 0.934389 |
| joint_transition | 44 | 19 | 0.972869 | 0.932169 |

**Future-only 并非稳定优于 Joint。** 同为1秒时，三 seed 平均仅高 0.001255 F1（0.126 个百分点）；seed44 的 Joint Future 反而为 0.950545，高于独立 Future 的 0.940534。按 Episode 等权的配对 bootstrap，独立1秒 Future 差值均值 -0.00102，95%区间 [-0.01047, 0.00777]，不能确认有稳定改进。完整曲线和各头指标保存在 summary.json。

可核实的机制证据：Joint 按 (Current F1 +0.5×Future F1)/1.5 选 epoch；seed42 选 epoch7（Future0.933805），单看 Future 最优在 epoch15（0.941287），但 Current 从0.972677降到0.963415。seed43 也存在相同取舍；seed44 的加权最优恰好也是 Future 最优。独立头可分别保留各自最优 epoch。

训练集上对所选 Joint 干线做了24个固定 mini-batch 的梯度夹角诊断：seed42/43/44 负余弦比例分别16.7%/25.0%/45.8%，平均余弦仍为正。它证明最终 checkpoint 存在局部目标竞争，**不证明梯度冲突导致整体差距**。容量不同、选 epoch 规则、标签边界不确定性和视觉先兆不足仍是待控制验证的因素；本轮没有把推测当作结论。

## 3. 时序评估与选模规则

新评估生成预测事件时不读取真值：概率阈值达标并连续 K 帧后更新已接受类别，首次稳定类别只作初始化。随后才按真实 from/to 类别一对一匹配：Future 事件必须严格早于边界且位于 [边界−h, 边界)，Current 事件位于 [边界, min(边界+2秒, 下个边界))。未匹配事件计误报，未匹配边界计漏检；迟到 Future、过早 Current 均不获得正确预警信用。

验证集共56个真实类别边界。2 Hz 给出0.5秒时间分辨率，K=2 本身增加约0.5秒确认时间。提前量和检测延迟只在成功匹配事件上统计，同时报告漏检率与误报/分钟；不能把条件中位提前量解释为所有边界都成功预警。

固定 threshold=0.8、K=2 时，以下均为三个 seed 的平均：

| 方案 | Current 边界召回 | Future 边界召回 | Future 误报/分钟 |
|---|---:|---:|---:|
| joint | 91.7% | 26.2% | 1.779 |
| joint_transition | 90.5% | 36.9% | 1.557 |
| independent_h05 | 92.3% | 5.4% | 2.244 |
| independent_h10 | 92.3% | 29.8% | 1.710 |
| independent_h15 | 92.3% | 63.7% | 1.004 |

0.5秒方案的整体 Future F1 最高，但严格提前边界召回只有5.4%，说明整体 Macro-F1 无法替代时序可用性。1.5秒扩展了确认后的预警余量，但其 Future 标签目标已改变，不能将其优于 Joint 1秒的预警增益完全归因于双头架构；还缺 Joint 1.5秒等受控消融。

在计算本轮候选验证指标前已锁定 `selection_rule.json`，网格为时距0.5/1/1.5秒，threshold=0.5/0.65/0.8/0.9，K=1/2。三个 seed 的均值用于选家族和工作点，代表 seed 固定42。门槛包括：Current F1 较 Joint 平均下降≤0.005、Current边界召回≥90%、P95延迟≤1.5秒、Current误报≤1/分钟；Future边界召回≥80%、Future误报≤1/分钟。可行候选按 Future召回、提前量、Current召回、Future F1 顺序排序；没有可行候选则保持旧默认，仅导出研究候选。

实际无候选全部达标。排序得到研究候选 independent_h15，Current threshold0.65/K2，Future threshold0.5/K2；三 seed 平均 Current召回94.0%、P95延迟1.288秒、误报0.166/分钟，Future召回72.0%、漏检28.0%、误报0.948/分钟、成功事件中位提前量的 seed 均值0.600秒。未满足 Future召回80%门槛，不放宽规则。

与当前 Joint seed42 的直接对照：

| checkpoint | Current工作点 | Current召回 / P95延迟 | Future工作点 | Future命中 | 漏检率 | 误报/分钟 | 命中事件中位提前量 |
|---|---|---|---|---:|---:|---:|---:|
| joint seed42 | 0.8/2 | 92.9% / 1.530s | 0.8/2 | 18/56 | 67.9% | 1.668 | 0.317s |
| joint seed42 | 0.65/2 | 91.1% / 1.333s | 0.5/2 | 21/56 | 62.5% | 1.626 | 0.400s |
| independent_h10 seed42 | 0.65/2 | 92.9% / 1.215s | 0.5/2 | 25/56 | 55.4% | 1.376 | 0.233s |
| independent_h15 seed42 | 0.65/2 | 92.9% / 1.215s | 0.5/2 | 40/56 | 28.6% | 0.962 | 0.567s |

独立1秒 seed42 的整体 Current/Future F1 比 Joint seed42 高约0.213/1.065个百分点，但固定0.8/K2时 Future预警命中反而从18/56降为12/56；阈值改变后才为25/56。独立1.5秒在相同0.5/K2 Future工作点达到40/56，相较 Joint1秒21/56有明确的**本验证集预警改善**，同时改变了架构和时距，仍不能声称独立测试提升或可靠部署。

## 4. 边界距离、类别混淆与 Episode 差异

同为1秒 Future标签，seed42 按距离真实当前边界的有符号时间统计：

| 距边界区间/秒 | 窗口数 | Joint Current F1 | 独立 Current F1 | Joint Future F1 | 独立1秒 Future F1 |
|---|---:|---:|---:|---:|---:|
| (-2.0, -1.0) | 109 | 0.9789 | 0.9789 | 0.7666 | 0.8559 |
| (-1.0, -0.5) | 56 | 0.9657 | 0.9789 | 0.4116 | 0.4851 |
| (-0.5, 0.0) | 56 | 0.8615 | 0.8860 | 0.7447 | 0.7142 |
| (0.0, 0.5) | 56 | 0.7090 | 0.6544 | 0.9097 | 0.9097 |
| (0.5, 1.0) | 56 | 0.9437 | 0.9143 | 0.9721 | 0.9721 |
| (1.0, 2.0) | 110 | 0.9292 | 0.9437 | 0.9721 | 0.9721 |
| outside_2s | 2455 | 0.9915 | 0.9957 | 0.9570 | 0.9662 |

Future尾部无标签窗口被屏蔽，因此各区间 Future支持数与Current可能略有差别；精确各类支持和混淆矩阵见 JSON。独立 Current 在刚切换后的[0,0.5)秒 F1 为0.6544，低于 Joint的0.7090，即使整体分数较高也不能说所有边界都改善。验证窗口68.98%为harvest；边界±0.5秒仅112/2898窗口，长稳定阶段掩盖了切换困难。

四类指标（研究候选 seed42，固定四类顺序）：

| 类别 | Current支持/F1 | Future1.5秒支持/F1 |
|---|---:|---:|
| pickup | 327 / 0.9632 | 297 / 0.9022 |
| harvest | 1999 / 0.9967 | 1999 / 0.9885 |
| return | 187 / 0.9651 | 187 / 0.8658 |
| adjust | 385 / 0.9742 | 385 / 0.9378 |

研究候选 Future 混淆主要包括 pickup→adjust 20次、adjust→pickup 15次、return→harvest 17次、harvest→return 12次；return Future F1仅0.8658。不能只关注return/adjust：pickup/adjust以及harvest/return同样重要。

| 验证 Episode | 类别阶段数 | Current F1 | Future1.5 F1 | 提前命中 | 误报数 | 有序阶段位置准确率 |
|---|---:|---:|---:|---:|---:|---:|
| `3e645bf4e137d0f836c9d7ff` | 7 | 0.9528 | 0.9044 | 3/6 | 3 | 96.6% |
| `27d8bc0413a42341c7e8c3e5` | 7 | 0.9830 | 0.9479 | 5/6 | 1 | 97.1% |
| `0b12ac0fda9d7cd77dee6e60` | 7 | 0.9817 | 0.9076 | 3/6 | 5 | 96.4% |
| `61c420843f11ca54bfe804ee` | 7 | 0.9856 | 0.9757 | 6/6 | 0 | 97.6% |
| `5b82f4bb76e70d8bf12c0798` | 7 | 0.9703 | 0.9649 | 6/6 | 0 | 96.8% |
| `f53f0b6fb7f3888f87a0f8ff` | 7 | 0.9642 | 0.8906 | 3/6 | 3 | 96.7% |
| `ca784332e44d979f99018a52` | 3 | 0.7283 | 0.7164 | 2/2 | 0 | 95.8% |
| `a8b23394b642f0bb769a0391` | 7 | 0.9917 | 0.9283 | 5/6 | 1 | 96.1% |
| `6d46f3c8c5c61bd2c6d22973` | 7 | 0.9723 | 0.8782 | 2/6 | 9 | 97.7% |
| `d0f8727d35eced697dcf79fe` | 7 | 0.9634 | 0.9150 | 5/6 | 1 | 95.1% |

Episode `ca784...` 只有pickup/harvest/return三类，四类Macro-F1最高0.75；应结合支持数判读。`6d46...` 有9次误报，占研究候选seed42全部23次误报的39%，是下一轮分析重点。按原始录制聚类的5000次配对bootstrap保存在报告JSON；仅10个复用验证来源，区间也不构成独立确认。

## 5. 实际推理与 OpenPI 离线回放

RTX5090上，20个来自全部10条验证视频的真实三视角历史窗口，5次warm-up、120次同步CUDA推理：整体P50/P95/P99/max为 45.65/48.23/49.19/49.91ms；特征提取P95 47.33ms，双头合计P95 0.995ms。Current/Future单头平均分别0.545/0.534ms。

60次按2Hz定时调用的P95/max为 47.27/51.59ms，无500ms超时。hook实测185次请求恰好185次encoder前向，每次将3路相机放在一次batch内。PyTorch峰值allocated/reserved显存约0.763/0.881GiB，不含CUDA context等进程额外显存。

同一缓存特征的200次头部对比也均在约1ms量级，运行波动大于两种头部结构的差异，不宣称独立头更快。模型加载约1.99秒另计。计时包含H2D、归一化、编码、融合与logits回传；不包含相机采集、磁盘、网络、ROS、OpenPI动作模型。逐视频全量解码/选帧耗时单独保存在 benchmark.video_decode，不能拿这些短时离线测试声称完整机器人系统长期稳定。

通过现有未修改的 OpenPI `PredictionPromptProvider` 回放全部2898个验证窗口。使用真实checkpoint预测，回放时钟由录制anchor/fps构造；合成的+40ms产出/+50ms消费时间明确标记，不冒充实测延迟。每个Episode显式初始化pickup_1，四类概率通过现有有限状态进度映射至七个有序位置，9条完整录制到return_stem，1条三阶段录制停在return_truss；概率不会被当成七维分布。

有序位置准确率约95.1%–97.7%，不等于转换时序完全正确。过期、错误observation/session、会话中模型版本改变均被拒绝。另以真实验证视频NPZ启动JSONL服务，实际独立双头checkpoint输出经OpenPI接口接收通过。保留其默认threshold0.8/K2；研究分析选出的不同Current/Future工作点没有偷偷写入OpenPI。没有生成机器人动作、没有RTC修改、没有控制机器人。

## 6. 保存、可复现性和下一步

研究候选目录：`/home/user/vjepa2_training/runs/tomato/optimization_20261108/selected_bundle`。含agent.pt、独立current_head.pt/future_head.pt、normalization.npz、input_schema.json、inference_config.json与逐文件SHA。Agent模型版本：`7b9275832dc559ad5f9b17faa6380bb5b7571897bcee8588cf91ad5e3c260b00`。官方编码器沿用已验证的本地固定revision和权重SHA，不重复下载/复制1.3GB权重；bundle中明确记录依赖与本地路径。

实现入口：`tools/tomato_independent_heads.py`负责组合和严格输入/归一化一致性检查；`tomato_phase_model.load_checkpoint_payload`复用旧格式全部校验，旧单头/Joint保持兼容；`TomatoAgent.predict`仍只有一次fuse_features调用。`tools/tomato_boundary_metrics.py`提供因果事件指标，`optimize_tomato_agent.py`负责验证比较和锁定规则，`analyze_tomato_optimization.py`负责录制级与梯度诊断，`benchmark_tomato_agent.py`负责真实视频计时/服务/离线桥接。

复现应在仓库根目录使用 `.venv-tomato/bin/python -m tools.optimize_tomato_agent --output-root runs/tomato/optimization_fresh`，然后对同一output-root依次运行 `tools.analyze_tomato_optimization`、`tools.benchmark_tomato_agent`。复用已有官方权重及冻结缓存；新目录避免改变既有运行版本。旧12次训练目录只读。运行目录完整保存27组验证报告、逐窗口logits、阈值网格、6份新训练的best/last/优化器与RNG恢复状态，未提交Git的大文件留在本地。

元数据修正记录：最初0.5/1.5秒派生manifest的汇总字段继承了1秒值，实际逐Episode监督、mask、训练数组和checkpoint时距一直正确。已生成准确聚合侧车 `metadata_corrections.json`，不改写原训练来历；生成器已修复并用合成数据测试校验。它不改变模型参数或任何指标。

验证：95项番茄相关测试通过；真实encoder单次调用计数、JSONL服务、2898窗口OpenPI回放及异常拒绝通过；423旧文件与全部锁定输入复核通过。OpenPI工作区未改动。首次服务测试因子进程解释器误解析而缺numpy，已使用虚拟环境解释器修正并重测通过；这是测试启动问题，不是模型问题。

下一步优先级：

1. 收集15–20条全新独立录制、至少约100个切换边界，包含难例与长稳定阶段负样本；按原始录制封存，不用本轮旧测试或新数据来回调阈值。先固定checkpoint、时距、阈值、事件定义再一次性评估；样本量仍应根据预期改进和录制相关性计算，100边界不是可靠性保证。
2. 对现有误报集中Episode和return/harvest、pickup/adjust混淆复核边界标注与视觉先兆，保存盲审一致性记录。
3. 下一轮训练若继续，优先用冻结缓存做Joint1.5秒、等容量对照，以及只读过去2–4个anchor的因果时序头。当前所有分类头仅使用单个历史窗口特征；时序记忆是否改善需独立消融，不能提前宣称已实现。

