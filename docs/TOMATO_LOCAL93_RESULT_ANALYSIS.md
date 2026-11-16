# 番茄四阶段分类器：本地 93 条数据实验结果与实现核验

生成时间：2026-11-07T18:23:57.860321+08:00。代码 HEAD：`63d89e8700cfb9444e8d319ada34b5ba8f071a2e`。分支：`feat/tomato-four-stage`。

本文件面向无法访问本地仓库的网页端 GPT，包含任务背景、实验配置、实测结果、审计证据、局限和后续讨论约束。所有精度均来自真实数据及已保存的训练结果；合成测试仅用于实现回归。

## 1. 结论

1. 三样本实验中的“Future-only 低于持久性基线、Joint 两个头均明显退化”没有在本次测试集上复现。Future-only 测试 Macro-F1=0.922962，高于 Oracle 当前标签持久性基线 0.918360，增益 **0.46 个百分点**；Joint Future=0.933795，高于 Future-only **1.08 个百分点**。Joint Current=0.957637，较独立 Current-only=0.960806 低 **0.32 个百分点**。

2. 这不是稳定优越性的证明：验证集 Future-only 与 Oracle 持久性几乎持平，Joint Future 略低于 Future-only；9 条测试来源的配对 bootstrap 差值区间跨 0。当前证据支持“整体泛化明显改善，存在实际提前识别能力，但稳定收益仍待更多独立来源验证”。

3. 本次全量标签/因果特征核验、损失日志核算、验证选模重算及权重重载均通过，未发现解释当前差异的标签错位、未来泄露或选模实现错误。不能用这项审计替代对训练全过程梯度行为的观察。

4. 过去突出的训练/验证关节分布偏移大幅减轻：归一化关节 RMS 从此前验证约 17.83，降到本次约 0.969（训练约 1.000）。这与数据覆盖改善一致，但三样本与 93 条实验的划分、归一化、训练更新量均变化，不能仅凭对比确定单一因果。

5. 本轮未修改训练代码、数据、标注、权重或超参数，未重训、未基于测试集选择模型。只读取已保存结果、重载固定权重验证验证集、生成诊断及交接文件。

## 2. 背景与固定配置

任务仅限 `pickup / harvest / return / adjust` 四阶段分类。冻结 V-JEPA 2 ViT-L，三相机各 1024 维，与过去 16 帧 ×58 个关节位置融合为 4000 维；分类头共享/独立 256 维 MLP。历史窗口为 1.5 秒、10 Hz，预测锚点 2 Hz；不涉及 Agent、π0.5、HFSM、RTC/T3。

- 数据目录：`/home/user/pi05/tomato_sharpa_remote93/A` 与 `B`。实验目录：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/`。

- 预训练模型：`facebook/vjepa2-vitl-fpc64-256`，revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`。固定权重 SHA256 `25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b`。

- seed=42；AdamW；learning_rate=0.0003；weight_decay=0.01；batch_size=128；epochs 上限100；patience=15；dropout=0.3；梯度裁剪范数1.0。CUDA BF16 Encoder，RTX 4070 Ti SUPER；确定性设置见 training_config.json。

- 未来目标固定 1.0 秒（30fps 下 +30 帧）；lambda_current=1.0、lambda_future=0.5 独立配置。Current/Future 类权重分别由训练标签 `sqrt(N/n_class)` 再除权重均值得到；归一化仅拟合训练特征，std 下限0.01。

- `allow_subset_metadata_mismatch=true` 沿用现有行为，无修改；标签标注 SHA 与来源锁定。

- 三模式相同划分 SHA256：`5ccda5a1499c01082c5223fbdb04d780f8cfb02da7146d5c50cc748ab32281de`；相同 93 个 NPZ 特征路径；来源 ID 无跨集合交叉。跨集合相同 Parquet/视频文件哈希数量：0/0。这只排除已定义来源/相同文件泄露，不能证明同一天同场景采集之间完全独立。

### 数据规模（类顺序 pickup / harvest / return / adjust）

| 集合 | 来源/episode | 当前锚点 | 当前类计数 | 未来有效/屏蔽 | 未来类计数 |
|---|---|---|---|---|---|
| train | 75/75 | 21706 | [2829, 14413, 1450, 3014] | 21556/150 | [2679, 14413, 1450, 3014] |
| val | 9/9 | 3007 | [346, 2138, 180, 343] | 2989/18 | [328, 2138, 180, 343] |
| test | 9/9 | 2580 | [318, 1744, 171, 347] | 2562/18 | [300, 1744, 171, 347] |

全部 27,293 个锚点；未来有效 27,107 个，尾部屏蔽186个。采集日期跨 2026-11-01 至 11-06，训练/验证/测试多天交织；本次属于来源隔离划分，非跨天隔离泛化评估。训练 harvest 占约66.4%，因此必须同时观察 Macro-F1、少数类和切换指标。

### 与三样本实验配置是否相同

逐字段比较三模式的已保存 training_config.json：模型、输入、采样、seed、设备、优化器参数、batch_size、epoch/patience、未来时距与 λ 相同。不同项为输入/输出/索引/划分路径、运行进程信息，以及由新训练数据自动计算的类别计数/权重；完整差异见 result_audit.json。

但实际训练量不同：三样本每轮约3次更新，本次 Current/Joint 每轮170次、Future每轮169次。最佳模型的累计更新数分别约2890、3887、1190（旧实验约102、6、3）。因此“超参数相同”不意味着有效训练预算相同；不能把改善全部归因于样本数。各模式批次组织仍不同，Joint 将有效/尾部屏蔽样本分摊到每个批次。

## 3. 三模式真实离线结果

下表均为各模式按验证集既定规则选择的单个 checkpoint。NLL 为评估用未加权概率负对数似然，与训练加权 CE 不同。Current 与 Future 目标/有效样本不同，不应直接将两者 F1 差异解释为模型退化。

| 模式/头 | 最佳/执行epoch | Train F1 | Val F1 | Test F1 | Test Accuracy | Test NLL |
|---|---|---|---|---|---|---|
| current_only/current | 17/32 | 0.9986 | 0.9562 | 0.9608 | 98.18% | 0.0935 |
| future_only/future | 23/38 | 0.9973 | 0.9232 | 0.9230 | 96.17% | 0.1487 |
| joint/current | 7/22 | 0.9908 | 0.9527 | 0.9576 | 98.02% | 0.0676 |
| joint/future | 7/22 | 0.9766 | 0.9187 | 0.9338 | 96.64% | 0.0880 |

Current-only 与 Joint Current 测试错分分别47、51个；未来有效锚点中 Future-only 与 Joint Future 分别98、86个错分。Joint 的两个头并非同时退化，且本次 Joint 两头测试 NLL 均低于相应单头模型，说明 F1 与概率损失结论并不完全一致。

### 测试集各类别 F1

| 模式/头 | pickup | harvest | return | adjust |
|---|---|---|---|---|
| current_only/current | 0.9584 | 0.9966 | 0.9377 | 0.9506 |
| future_only/future | 0.8986 | 0.9905 | 0.8983 | 0.9045 |
| joint/current | 0.9579 | 0.9960 | 0.9302 | 0.9465 |
| joint/future | 0.9251 | 0.9885 | 0.8889 | 0.9327 |

Joint Future 相对 Future-only 的改进主要体现在 pickup 和 adjust；harvest/return 的 F1 略低。不能概括为所有类别均受益。

## 4. 持久性基线与稳定/切换区间拆解

Oracle baseline 用真实当前阶段直接当作未来阶段，是不可部署的诊断参照；A_current_hold 使用独立 Current-only 的预测，C_current_hold 使用 Joint 当前头预测，两者是可实现的持久性策略。以下所有策略在相同未来有效锚点上评分。稳定区间指 y(t)=y(t+1s)，切换区间指两者不同，不是预测是否发生切换。

### val 集合

| 策略 | 未来 F1 | Accuracy | 稳定区错分 | 切换区正确 | 总错分 |
|---|---|---|---|---|---|
| oracle_current_hold | 0.9232 | 96.39% | 0/2881 | 0/108 | 108 |
| A_current_hold | 0.9072 | 95.58% | 41/2881 | 17/108 | 132 |
| B_future | 0.9232 | 96.62% | 69/2881 | 76/108 | 101 |
| C_current_hold | 0.8983 | 95.22% | 49/2881 | 14/108 | 143 |
| C_future | 0.9187 | 96.25% | 72/2881 | 68/108 | 112 |

### test 集合

| 策略 | 未来 F1 | Accuracy | 稳定区错分 | 切换区正确 | 总错分 |
|---|---|---|---|---|---|
| oracle_current_hold | 0.9184 | 95.78% | 0/2454 | 0/108 | 108 |
| A_current_hold | 0.9186 | 95.82% | 23/2454 | 24/108 | 107 |
| B_future | 0.9230 | 96.17% | 75/2454 | 85/108 | 98 |
| C_current_hold | 0.9170 | 95.75% | 26/2454 | 25/108 | 109 |
| C_future | 0.9338 | 96.64% | 58/2454 | 80/108 | 86 |

测试未来锚点中，稳定区2454/2562=95.78%，切换区108/2562=4.22%。Future-only 相对 Oracle 正确预判85个切换锚点，但引入75个稳定区错误，净减少10个错分；Joint Future 正确预判80个切换锚点，引入58个稳定区错误，净减少22个错分。

Joint Future 比 Future-only 少正确5个切换锚点，但少17个稳定区错误，净减少12个错分。因此，本次 Joint Future 的整体 F1 优势主要来自更少稳定区错误；它的切换捕获率并不更高。

验证集 Future-only F1=0.923169 与 Oracle=0.923168 几乎相同（差0.00000184），尽管 Accuracy 有所改善；Joint Future=0.918726 略低于 Oracle。这说明测试集点估计的胜出没有在验证集上同等复现。

### 按来源估计差异的不确定性

配对 cluster bootstrap：每次有放回重采样9个来源，保留来源内全部相关锚点，重新合并混淆矩阵计算四类 Macro-F1；5000次，随机种子20261107。以下差异/区间单位为百分点。区间仅为此次固定模型、固定9来源上的探索性不确定性，不能包含训练种子、采集场景或标注误差的全部不确定性；也不用于选择超参数。

| 集合 | 对比 | 差值（百分点） | 95%百分位区间（百分点） |
|---|---|---|---|
| val | B_future minus oracle_current_hold | 0.0002 | [-2.37, 2.94] |
| val | B_future minus A_current_hold | 1.5929 | [-0.26, 3.99] |
| val | C_future minus B_future | -0.4443 | [-1.98, 1.86] |
| val | C_future minus C_current_hold | 2.0421 | [0.05, 4.48] |
| test | B_future minus oracle_current_hold | 0.4602 | [-2.60, 3.32] |
| test | B_future minus A_current_hold | 0.4390 | [-2.38, 3.04] |
| test | C_future minus B_future | 1.0833 | [-0.57, 2.82] |
| test | C_future minus C_current_hold | 1.6838 | [-0.27, 4.02] |

| 集合 | Joint Current − Current-only（百分点） | 95%来源bootstrap区间 |
|---|---|---|
| val | -0.3575 | [-1.58, 0.67] |
| test | -0.3169 | [-1.08, 0.49] |

测试中 Future-only 在5/9个来源上超过 Oracle；Joint Future 在6/9个来源上超过 Future-only。不能把数千个重叠锚点当作数千个独立证据。

## 5. 时间行为与误报

| 头/集合 | 切换事件识别 | 提前量P50/P95（秒） | 窗外误报次数 | 窗外误报/分钟 | 窗外误报持续（秒） |
|---|---|---|---|---|---|
| future_only/val | 48/54 | 0.600/1.000 | 30 | 1.2044 | 34.5 |
| future_only/test | 50/54 | 0.633/0.985 | 44 | 2.0609 | 38.0 |
| joint/val | 42/54 | 0.600/1.000 | 36 | 1.4453 | 36.0 |
| joint/test | 48/54 | 0.633/1.000 | 29 | 1.3583 | 29.0 |

Future 事件识别定义为：在真实边界前1秒窗口内，至少一个2Hz锚点正确预测下一阶段。它不保证整个1秒窗口持续正确，故事件召回率高于切换锚点准确率。提前量仅统计成功识别事件，漏报不记为0秒；窗外误报按现有指标定义统计，以真实当前阶段作为离线比较参照，不等于部署后基于预测当前阶段的实际告警误报率。测试 Future-only 漏4/54事件，Joint漏6/54；Joint误报较少，但并未更早或更完整地捕获事件。

| 当前头（test） | 首次正确P50/P95秒 | 连续2次确认P50/P95秒 | 首次/稳定漏检 | 误切换次数 | 错误阶段持续秒 |
|---|---|---|---|---|---|
| current_only | 0.283/0.557 | 0.783/1.057 | 0%/0% | 32 | 23.5 |
| joint | 0.283/0.748 | 0.783/1.248 | 0%/0% | 33 | 25.5 |

时间分辨率为0.5秒。连续2次的确认延迟包含第二个锚点等待时间。表中持续时间按2Hz指标规则汇总，属于离线估计。

## 6. 实现审计：标签、损失、选模

### 6.1 全量未来对齐和历史特征

独立逐区间查找人工审核标注的半开区间 `[start_frame,end_frame)`；对全部93个episode、27,293个锚点验证 y_current=y(frame)、y_future=y(frame+30)。全部三模式预测文件的真实标签、future_valid、目标帧与独立结果一致。每条尾部2个锚点无未来目标：future_valid=false、目标帧=-1、标签/null时间缺失，未填补。

从原始Parquet时间戳核验1秒偏移，最大绝对偏差 `0` 秒。NPZ 输入索引全部不晚于锚点；全部关节特征逐值等于原始 `observation.state` 中指定58个位置字段在历史索引上的取值，未混入速度或未来状态。93份标注SHA、Parquet SHA及NPZ内容SHA复核通过。

限制：本轮没有重新解码279路视频或重新运行Encoder；完整媒体验证依据原有 prepare_validation.log/local_preflight_summary.json。缓存哈希/历史索引/原始关节一致性通过不等于重新独立验证视觉编码的每一步。未逐帧重审人工边界语义。

### 6.2 双头损失

源码 `tools/tomato_phase_model.py::supervision_loss`：Current 用全部当前样本；Future先应用future_valid_mask；各自按训练类权重计算PyTorch weighted CE。Joint：`L=(1.0*L_current+0.5*L_future)/1.5`。独立重新计算所有epoch的训练损失组合：Current/Future残差0，Joint最大绝对残差约4.25e-9，符合浮点误差。训练类计数/权重从训练标签复算一致；三个checkpoint的mean/std均与训练特征重算逐值相等。

55项相关单元/回归测试通过，其中包含Future mask、模式头、损失、对齐和选模逻辑。训练CE是批次均值的epoch平均，不能与全量评估NLL直接对比；日志没有保存每步的梯度夹角/范数/裁剪比例，无法据此证明或排除训练过程的负迁移。λ=2:1也不等价于共享层梯度强度恒为2:1。

### 6.3 验证集选模

源码 `tools/train_tomato_vjepa_phase.py::validation_selection`：单头取对应Val Macro-F1；Joint取 `(Val_Current_F1+0.5*Val_Future_F1)/1.5`，仅在F1相同后比较更小的加权NLL。并非按最低CE/NLL选模型。逐epoch重算选择分数和排序，最佳epoch与checkpoint及evaluation一致；CPU重载固定权重仅在验证集重跑，F1精确复现，NLL误差<1e-5。未使用测试集选模。

Current-only选epoch17，Future-only选epoch23，Joint选epoch7；各自再持续15个未改善epoch后早停。Joint验证当前头单独最优epoch3（F1=0.953105），未来头单独最优epoch5（F1=0.921585），但这不是同一个checkpoint；Joint epoch7的组合分数约0.941347，高于epoch3约0.940842，按现有规则选择正确。单头独立最佳与共享双头加权最佳不同属于目标取舍，不能直接认定选模bug。

后三模式验证指标未继续改善：最终Val F1约Current-only0.9462、Future-only0.9028、Joint Current0.9288/Future0.9066。训练集近乎拟合而验证并未同步提升，仍存在过拟合；也提示100 epoch上限并非实际执行量。

## 7. 与上次结论如何衔接

旧三样本划分为1训练/1验证/1测试；仅357个训练窗口、6个训练阶段事件，验证关节标准化RMS约17.83，验证Current F1约0.50。新划分训练75来源、21,706窗口、446个阶段事件（其中部分边界不产生完整2个未来切换锚点），验证Current F1约0.95。

| 集合 | 视觉标准化RMS | 关节标准化RMS | 关节\|z\|>3比例 | 关节最大\|z\| |
|---|---|---|---|---|
| train | 1.0000 | 1.0000 | 1.59% | 18.2391 |
| val | 1.0115 | 0.9686 | 1.59% | 10.6678 |

本次未再出现旧验证集大面积几十倍标准差的关节偏移。这使“小数据训练未覆盖验证姿态/分布”得到更强支持。原有三条样本的来源均被本次划入训练集合，旧测试结果与本次测试结果不属于相同保留集，不能将分数变化作为严格配对的数据规模消融。新归一化也不同，仍无法断言旧手部偏移是传感器校准错误，或断言该问题已在数据源端修复。

Joint训练最佳epoch更早、批次组织不同、Future与Current梯度共享；本次未来头反而更好，说明旧“Joint共享结构必然导致两个头退化”的推断不成立。当前仅单种seed和单个划分，仍不足以量化共享训练的稳定收益。

## 8. 给网页端GPT的讨论边界与后续建议

请基于以上真实结果讨论，不沿用“Future-only必然低于持久性、Joint两个头均明显退化”的旧前提。请区分：已验证实现行为、当前数据上的统计事实、尚待新实验验证的机制假设。

1. 暂不基于这9条测试数据调学习率、λ、时距、平滑、告警阈值或checkpoint；也不挑测试最好的epoch。任何配置选择只使用训练/验证数据，后续最终结论应保留新的独立测试来源。

2. 首先在训练/验证上做预先约定的多seed、相同来源划分和相同批次/更新预算对照；若要分析负迁移，记录共享层各头梯度范数、夹角与梯度裁剪比例，避免仅凭最终F1归因。

3. 继续收集多日期、多操作者、多场景来源，设计按采集会话/日期隔离的验证；在不修改原标注的前提下抽查阶段边界，特别是pickup/adjust提前误报和return短阶段。

4. 将未来能力同时报告为总体F1、真实当前Oracle持久性、可部署模型持久性、切换锚点正确率、事件召回/提前量与窗外误报。若讨论“Joint更好”，须明确是整体误差/概率损失更好，还是切换提前识别更好；本次两者结论不同。NLL降低本身不能代替专门的校准评估。

5. 若讨论两头独立选checkpoint或更改联合验证目标，这是新设计，不是已证实bug修复；需预先定义目标、仅用验证集比较，并保留原行为兼容。

## 9. 审计范围、文件和复现

审计前后 `321` 个原始实验输出文件SHA完全一致（包含checkpoint、日志、标签缓存、评估和预测文件）。原训练代码未编辑，用户已有的 `docs/TOMATO_LOCAL93_TRAINING.md` 本地修改保留。

`local_preflight_summary.json` 中 training_started=false / encoder_extraction_started=false 是训练前的历史快照；当前三个status.json均为training_complete，应以实际日志和评估文件判断。此次产物只有离线锚点评估，未发现该目录内独立视频回放/部署性能报告，本分析不宣称已完成这些验收。

交接压缩包包含：本报告、result_audit.json、supplementary_audit.json、原始结果SHA清单、审计脚本/日志、55项测试日志、三模式完整evaluation.json/metrics.jsonl/training_config.json/split.json、当前相关实现和测试代码、数据概要。未打包原始视频、模型权重、NPZ或逐锚点预测文件。网页端GPT阅读单个Markdown即可理解结论，需要核算细节时再解压证据。

复现诊断（不训练）：

```bash
cd /home/user/pi05/vjepa2_training
.venv-tomato/bin/python -u runs/tomato/local93_20261107/analysis/analyze_local93.py > runs/tomato/local93_20261107/analysis/audit.log 2>&1
.venv-tomato/bin/python -m pytest -q tests/test_tomato_dataset.py tests/test_tomato_joint_model.py tests/test_tomato_joint_training.py tests/test_tomato_joint_metrics.py tests/test_tomato_runtime.py > runs/tomato/local93_20261107/analysis/targeted_tests.log 2>&1
.venv-tomato/bin/python runs/tomato/local93_20261107/analysis/build_report.py
```

现有训练的入口（仅记录，不建议覆盖已完成目录）：`bash tools/run_tomato_local93.sh all`。原程序会拒绝覆盖训练结果。本轮没有执行该命令。

### 测试来源明细（未来目标Macro-F1）

| source_recording_id | Oracle hold | A hold | Future-only | Joint Current hold | Joint Future |
|---|---|---|---|---|---|
| 0340d869ea074774b09441f2339cec30 | 0.9156 | 0.9210 | 0.9489 | 0.9340 | 0.9441 |
| 1b2e1a577afa442cb057c595cf27c48f | 0.9208 | 0.9211 | 0.9854 | 0.9007 | 0.9895 |
| 26a781329cd64ac2b99e19c41fb02f10 | 0.9195 | 0.9340 | 0.9055 | 0.9253 | 0.9364 |
| 3b7d153b61764f4b9920e382f7f1083b | 0.8968 | 0.8889 | 0.8955 | 0.8972 | 0.8887 |
| 5d8ad555472e4581a1768e9c27eb03e2 | 0.9383 | 0.9369 | 0.8626 | 0.9447 | 0.9200 |
| 666d070f4a194cefae4bf7e9437710c7 | 0.9053 | 0.9052 | 0.9489 | 0.9052 | 0.9634 |
| 9c6fd4d27e9e4713819c96a88264c853 | 0.9093 | 0.8684 | 0.8715 | 0.8552 | 0.8873 |
| 9e19bb249db04f8781577bc3ad22d753 | 0.9206 | 0.9400 | 0.9653 | 0.9400 | 0.9392 |
| b693693c0fd34115989b89177cbe78c2 | 0.9038 | 0.9191 | 0.9164 | 0.9110 | 0.9178 |
