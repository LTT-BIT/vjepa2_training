# LOCAL93：冻结预测驱动的离线Prompt切换仿真报告

## 1. 摘要与最终建议

实际HEAD `63d89e8700cfb9444e8d319ada34b5ba8f071a2e`；共享策略仅用9条val来源锁定为 **P3_d50**，参数 `{"name": "P3_d50", "id": "P3", "theta": 0.8, "K": 2, "margin": 0.0, "d": 0.5, "initial_prompt": "pickup", "prediction_ttl_s": 0.75, "seed": 20261108}`。主场景假设预测到达延迟90ms，规划周期0.5秒、相位0。风险约束是否满足：**False**。如果为false，该策略只是预声明fallback排序的最优项，不代表风险已达标。

| 模型/集合 | 实际commit匹配 | 时长一致率 | 错误Prompt秒 | commit时与GT不符 | 未匹配commit | 提前≥0.5秒 | 提前/同时/滞后 | Δt P50/P95秒 | 完整顺序正确 |
|---|---|---|---|---|---|---|---|---|---|
| Future-only / test | 45/54 | 94.98% | 64.767 | 8 | 9 | 2 | 4/0/41 | 0.867/1.667 | 9/9 |
| Joint Future / test | 48/54 | 94.75% | 67.667 | 7 | 8 | 1 | 5/0/43 | 1.033/1.598 | 8/9 |

建议仅继续离线/Shadow Mode研究，尚不足以采用“未来Softmax高置信度即执行切换”的真机路线。未来标签预测不是动作完成检测；GT时间一致率、事件匹配与完整顺序分别描述不同现象。以下结果不证明机器人安全，也不能从测试结果回头选择新策略。

## 2. 实验输入与可信性

预检93/93来源通过：75 train、9 val、9 test，三模式每个27,293个2Hz锚点，顺序 `pickup/harvest/return/adjust`。序列分布 `{'7': 92, '3': 1}`。没有排除来源或补预测。主对照实际执行9 val+9 test来源；额外三阶段资料在其原集合单独检查，不加入主测试。

配对键为(source_recording_id,episode_id,frame_index,timestamp_s)，允许1e-6秒验证容差；拒绝重复、缺失、倒序、未来输入索引、概率非法和split混淆。manifest、split、任务契约及来源/标注快照逐模式比较；reviewed标注文件SHA验证。checkpoint仅做SHA锁定，不加载或更改。完整输入SHA见 replay_integrity.json；原始输出保护见 protected_original_sha256.json。

三模式共保护 `321` 个历史输出文件，结束时全部SHA不变。冻结特征、原视频/Parquet/标注未写入。已有用户未提交文件保留；本轮新增模块独立于旧训练/评估代码。

| 状态项 | 实际状态 |
|---|---|
| P0 | PASS |
| P1 | PASS |
| P2 | PASS |
| P3 | PASS |
| P4_val | PASS |
| P4_test | PASS |
| P5_auto | PASS |
| P5_manual_review | DEFERRED / NOT_PERFORMED |
| P6 | PASS |
| real_robot | NOT_RUN |
| FINISH | NOT_SUPPORTED |
| FINAL_regression | PASS |

## 3. P1 因果回放审计

策略只接收frozen/slots PolicyInput白名单，不含任何GT、future_valid、future_target_time_s或录制长度。Forecast target由original observed_at+1s计算，Current-only反应式输入为observed_at+0s。状态由固定启动先验pickup开始，全程不按真实边界或错误重置。

事件时间四舍五入到1ns；同一时刻先接收结果，再处理expiry/release，最后规划边界提交，结束事件最后。每个确认必须来自不同新锚点；planning HOLD不能增加计数。TTL=0.75秒，从最后一条新结果可用时间计，严格超时才失效；Active继续保持，Pending取消，不借TTL自动切换。新证据撤回或候选变化会取消Pending并重置计数。

回放驱动器用Episode长度安排输入停止，但该长度从未传给策略。只处理t<数据终止的可用消息；无法在终止前完成的尾部结果不消费，其余尾部预测即使future_valid=false也照常处理。数据停止只产生stream_end，不产生FINISH。每个case全部Episode回放完成后才调用独立GT评分。

真实selected Joint val复跑：`{'replay_events.jsonl': True, 'active_prompt_intervals.jsonl': True, 'evaluation_control.json': True}`；完整事件、Active区间、评估JSON逐字节相同。全流程主进程torch_imported=False，transformers_imported=False。未启动Encoder/CUDA。

## 4. P2 策略与参数表

| name | 策略 | θ | K | margin | defer秒 |
|---|---|---|---|---|---|
| P0_raw | P0 | 0.0 | 1 | 0.0 | 0.0 |
| P1_t70 | P1 | 0.7 | 1 | 0.0 | 0.0 |
| P1_t80 | P1 | 0.8 | 1 | 0.0 | 0.0 |
| P1_t90 | P1 | 0.9 | 1 | 0.0 | 0.0 |
| P2_k1 | P2 | 0.8 | 1 | 0.0 | 0.0 |
| P2_k2 | P2 | 0.8 | 2 | 0.0 | 0.0 |
| P2_k2_m10 | P2 | 0.8 | 2 | 0.1 | 0.0 |
| P3_d0 | P3 | 0.8 | 2 | 0.0 | 0.0 |
| P3_d25 | P3 | 0.8 | 2 | 0.0 | 0.25 |
| P3_d50 | P3 | 0.8 | 2 | 0.0 | 0.5 |
| P4_legal | P4 | 0.8 | 2 | 0.0 | 0.25 |
| P4_legal_m10 | P4 | 0.8 | 2 | 0.1 | 0.25 |

P0直接argmax；P1增加置信度；P2增加新锚点连续确认/概率margin；P3增加Pending释放延迟；P4增加合法边pickup→harvest→return→adjust→pickup。P2_k1与P1_t80、P3_d0与P2_k2故意保留为实现等价性检查，不是更多独立搜索。未尝试全笛卡尔积、模型特定调参、TTL/时距扫描或温度校准。

## 5. P3 独立控制级指标

Active区间与GT半开区间精确求交得错误时长；时长一致率=1−错误秒/总秒。early_wrong是错误Active等于紧邻下一GT阶段的重叠时长，其余单列。错误commit定义为commit时新Active与GT不一致，因此包括相对GT提前的正确下一阶段，不等同于真实物理错误。extra_commits=max(0,commit数−GT边界数)；unmatched_commits按匹配结果计算。

边界匹配按每个Episode、精确阶段对、单调一对一动态规划：主窗口[-1,+2]秒内先最大化匹配数，再最小化|Δt|总和，最后按配对索引字典序破同分。首个pickup不算边界。独立严格窗口[-0.2,+0.5]重新匹配；on_time_ratio是主匹配中落入该窗口的数量/全部边界。提前≥0.25/0.5/1秒仅为人工边界相对风险诊断，不是任务提前完成。

| test模型 | strict匹配 | on_time | 提前≥0.25/0.5/1秒 | Δt最小/最大秒 | early wrong/other wrong秒 | 非法候选/撤销/过期 |
|---|---|---|---|---|---|---|
| future_only | 9/54 | 9/54 | 3/2/0 | -0.967/2.000 | 10.967/53.800 | 0/7/0 |
| joint | 5/54 | 5/54 | 3/1/0 | -0.633/2.000 | 3.167/64.500 | 0/1/0 |

首次错误commit后恢复时间按实际Active与GT再次一致的最早时间事后计算，未人工复位；没有错误或未恢复分别用null+reason。完整per_episode_metrics.csv包含每类时长、错误时长、序列、恢复状态。短阶段/1秒内多边界不从时长评分删除；事件有无歧义敏感性和return首次/最终标记保留在JSON/CSV。

## 6. P4 公平对照与验证锁定

selection_protocol.json在任何新test控制评分前保存。两个模型分别满足错误时长≤5%、提前≥0.5秒的匹配数≤10%边界、事件召回≥80%，才视为可行；可行项按B/C等权平均召回最大、平均错误时长比例最小、平均匹配|Δt|最小、名称排序。若均不可行，按两模型归一化约束超出总量、错误时长、提前比例、负召回及名称排序。阈值为预声明研究约束，不是安全标准。

| val候选 | 模型 | 匹配 | 时长一致率 | 提前≥0.5秒 | 完整顺序 |
|---|---|---|---|---|---|
| P0_raw | future_only | 46/54 | 96.66% | 8 | 4/9 |
| P0_raw | joint | 46/54 | 96.37% | 8 | 2/9 |
| P1_t70 | future_only | 46/54 | 96.51% | 7 | 4/9 |
| P1_t70 | joint | 47/54 | 96.99% | 7 | 4/9 |
| P1_t80 | future_only | 45/54 | 96.52% | 6 | 4/9 |
| P1_t80 | joint | 47/54 | 96.79% | 5 | 4/9 |
| P1_t90 | future_only | 46/54 | 96.74% | 6 | 6/9 |
| P1_t90 | joint | 46/54 | 96.92% | 3 | 7/9 |
| P2_k1 | future_only | 45/54 | 96.52% | 6 | 4/9 |
| P2_k1 | joint | 47/54 | 96.79% | 5 | 4/9 |
| P2_k2 | future_only | 49/54 | 96.01% | 8 | 7/9 |
| P2_k2 | joint | 49/54 | 96.23% | 3 | 8/9 |
| P2_k2_m10 | future_only | 49/54 | 96.01% | 8 | 7/9 |
| P2_k2_m10 | joint | 49/54 | 96.23% | 3 | 8/9 |
| P3_d0 | future_only | 49/54 | 96.01% | 8 | 7/9 |
| P3_d0 | joint | 49/54 | 96.23% | 3 | 8/9 |
| P3_d25 | future_only | 49/54 | 96.01% | 8 | 7/9 |
| P3_d25 | joint | 49/54 | 96.23% | 3 | 8/9 |
| P3_d50 | future_only | 46/54 | 94.84% | 1 | 9/9 |
| P3_d50 | joint | 46/54 | 94.87% | 2 | 9/9 |
| P4_legal | future_only | 48/54 | 93.99% | 8 | 9/9 |
| P4_legal | joint | 49/54 | 96.26% | 3 | 9/9 |
| P4_legal_m10 | future_only | 48/54 | 93.99% | 8 | 9/9 |
| P4_legal_m10 | joint | 49/54 | 96.26% | 3 | 9/9 |

共享锁文件SHA `84520cfe3b5ff7542b397cdb133a9253fcbcfef6ab12caffc4c042873720abb1`。主test及P6输出均绑定此SHA；辅助试验的固定规则也在锁中预声明。未依据test更改阈值、算法、checkpoint、时距或规划周期。

| 模型/集合 | 实际commit匹配 | 时长一致率 | 错误Prompt秒 | commit时与GT不符 | 未匹配commit | 提前≥0.5秒 | 提前/同时/滞后 | Δt P50/P95秒 | 完整顺序正确 |
|---|---|---|---|---|---|---|---|---|---|
| B / val | 46/54 | 94.84% | 77.567 | 6 | 8 | 1 | 5/2/39 | 1.033/1.750 | 9/9 |
| C / val | 46/54 | 94.87% | 77.000 | 4 | 8 | 2 | 4/0/42 | 1.083/1.833 | 9/9 |

| test source | B/C匹配 | B/C时长一致率 | C−B百分点 | B/C完整顺序 |
|---|---|---|---|---|
| 9e19bb249db04f8781577bc3ad22d753 | 6/5 of 6 | 95.04%/93.10% | -1.938 | True/True |
| 1b2e1a577afa442cb057c595cf27c48f | 6/6 of 6 | 96.64%/94.89% | -1.750 | True/True |
| 5d8ad555472e4581a1768e9c27eb03e2 | 4/6 of 6 | 95.67%/96.78% | +1.103 | True/True |
| 9c6fd4d27e9e4713819c96a88264c853 | 4/4 of 6 | 95.22%/92.19% | -3.028 | True/False |
| b693693c0fd34115989b89177cbe78c2 | 5/5 of 6 | 95.33%/95.57% | +0.243 | True/True |
| 26a781329cd64ac2b99e19c41fb02f10 | 5/6 of 6 | 92.44%/95.11% | +2.672 | True/True |
| 666d070f4a194cefae4bf7e9437710c7 | 5/5 of 6 | 94.28%/92.63% | -1.643 | True/True |
| 3b7d153b61764f4b9920e382f7f1083b | 5/6 of 6 | 94.78%/96.62% | +1.844 | True/True |
| 0340d869ea074774b09441f2339cec30 | 5/5 of 6 | 95.27%/95.82% | +0.548 | True/True |

95%配对来源cluster bootstrap（5000次，seed20261108），方向均为C−B：

| 指标 | 95%百分位CI |
|---|---|
| time_accuracy | [-0.014, 0.010] |
| transition_recall | [-0.037, 0.148] |
| wrong_duration_s | [-13.067, 18.867] |
| premature_ge_05 | [-3.000, 0.000] |

来源时长一致率胜/负/平 `{'wins': 5, 'losses': 4, 'ties': 0}`。只有9独立test来源，CI仅探索性；错误秒/提前计数CI为同样9个重采样来源总量，不能解读为每Episode值。

### 独立附加对照

| 模型/集合 | 实际commit匹配 | 时长一致率 | 错误Prompt秒 | commit时与GT不符 | 未匹配commit | 提前≥0.5秒 | 提前/同时/滞后 | Δt P50/P95秒 | 完整顺序正确 |
|---|---|---|---|---|---|---|---|---|---|
| val_joint_current_gate | 0/54 | 11.34% | 1331.700 | 0 | 0 | 0 | 0/0/0 | null/null | 0/9 |
| val_current_only_reactive | 35/54 | 92.39% | 114.367 | 2 | 21 | 1 | 1/0/34 | 1.733/2.000 | 8/9 |
| test_joint_current_gate | 0/54 | 12.05% | 1133.567 | 0 | 0 | 0 | 0/0/0 | null/null | 0/9 |
| test_current_only_reactive | 47/54 | 93.11% | 88.833 | 4 | 9 | 0 | 2/0/45 | 1.633/1.990 | 8/9 |

Joint Current辅助规则是：Current argmax与Active不同时冻结该条Future请求并撤回Pending，除此与共享策略相同。它可能抑制误切换，也可能在Active落后时阻止恢复；只作附加比较，不混入B/C公平主对照。Current-only使用相同提交骨架，输入目标是当前阶段，不把它冒充未来预测。

### Softmax概率诊断（未拟合校准）

| 模型/test | 有效未来目标 | NLL | Brier(四类平方和) | ECE(10等宽bin) |
|---|---|---|---|---|
| future_only | 2562 | 0.149 | 0.062 | 0.024 |
| joint | 2562 | 0.088 | 0.051 | 0.011 |

每真实类别的NLL/Brier/ECE见evaluation_control.json。这些指标仅评价forecast target分类概率，不是可以安全切换概率。只有独立评分器在未来目标超出资料时跳过概率诊断，策略仍消费可用的尾部预测。

## 7. P5 自动可视化与异常索引

自动生成18个val/test Episode ×3类图，共54张PNG：完整时轴、四类概率、每个边界±3秒细节。自动PNG解码检查通过。P5_manual_review=**AUTO_GENERATED; MANUAL_REVIEW_DEFERRED / NOT_PERFORMED**。没有打开、解码、重编码MP4或逐帧核验，没有改标注或据图调参。

图与异常索引：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/prompt_switch_sim_20261108_p1_p6_v2/plots/timeline_index.md`。每个case含failure_cases.jsonl，按风险、时间、来源排序；原始observed_at/available_at、固定target及actual commit时间分别保存。图上raw future位于原始observation时间，目标始终是该anchor+1秒；HOLD不伪造新预测。

## 8. P6 假设延迟与规划时钟敏感性

真实执行288个case：val/test各9来源 ×延迟[0,.09,.2,.3] ×周期[.25,.5,1] ×相位[0,周期/2] ×共享锁定策略及P0/P2参考 ×两模型。每case的输入/消费锚点数、实际Episode数、匹配、错误/提前、Pending、时延分解均见sensitivity_matrix.json/.csv。若参考与锁定策略同名会去重，真实case数以文件为准。

延迟是注入的结果可用时间假设，数据到达延迟假设为0；没有测量真实模型计算/传输时间。所有延迟<0.5秒锚点间隔，采用单服务串行无拥塞队列，overload/drop_late/jitter为NOT_RUN，不能外推超载吞吐。π0.5的15步规划仅由参数化时钟代替，未测每步时间，也未实现RTC/T3动作块。

| test:latency/period/offset秒 | B匹配 | C匹配 | B/C一致率 | B/C提前≥0.5秒 | B/C Δt P50秒 |
|---|---|---|---|---|---|
| 0.0/0.25/0.0 | 44/54 | 49/54 | 96.24%/96.30% | 2/4 | 0.417/0.533 |
| 0.0/0.25/0.125 | 44/54 | 49/54 | 95.98%/95.92% | 1/3 | 0.542/0.658 |
| 0.0/0.5/0.0 | 44/54 | 49/54 | 96.24%/96.30% | 2/4 | 0.417/0.533 |
| 0.0/0.5/0.25 | 45/54 | 50/54 | 95.69%/95.54% | 2/3 | 0.650/0.783 |
| 0.0/1.0/0.0 | 46/54 | 48/54 | 96.06%/95.41% | 2/3 | 0.533/0.833 |
| 0.0/1.0/0.5 | 44/54 | 49/54 | 95.36%/95.64% | 2/2 | 0.717/0.767 |
| 0.09/0.25/0.0 | 45/54 | 50/54 | 95.69%/95.54% | 2/3 | 0.650/0.783 |
| 0.09/0.25/0.125 | 44/54 | 49/54 | 95.98%/95.92% | 1/3 | 0.542/0.658 |
| 0.09/0.5/0.0 | 45/54 | 48/54 | 94.98%/94.75% | 2/1 | 0.867/1.033 |
| 0.09/0.5/0.25 | 45/54 | 50/54 | 95.69%/95.54% | 2/3 | 0.650/0.783 |
| 0.09/1.0/0.0 | 46/54 | 47/54 | 93.96%/93.88% | 4/1 | 1.000/1.233 |
| 0.09/1.0/0.5 | 43/54 | 46/54 | 94.56%/93.84% | 1/1 | 1.033/1.283 |
| 0.2/0.25/0.0 | 45/54 | 50/54 | 95.69%/95.54% | 2/3 | 0.650/0.783 |
| 0.2/0.25/0.125 | 45/54 | 48/54 | 95.34%/95.15% | 2/3 | 0.775/0.908 |
| 0.2/0.5/0.0 | 45/54 | 48/54 | 94.98%/94.75% | 2/1 | 0.867/1.033 |
| 0.2/0.5/0.25 | 45/54 | 50/54 | 95.69%/95.54% | 2/3 | 0.650/0.783 |
| 0.2/1.0/0.0 | 46/54 | 47/54 | 93.96%/93.88% | 4/1 | 1.000/1.233 |
| 0.2/1.0/0.5 | 43/54 | 46/54 | 94.56%/93.84% | 1/1 | 1.033/1.283 |
| 0.3/0.25/0.0 | 45/54 | 48/54 | 94.98%/94.75% | 2/1 | 0.867/1.033 |
| 0.3/0.25/0.125 | 45/54 | 48/54 | 95.34%/95.15% | 2/3 | 0.775/0.908 |
| 0.3/0.5/0.0 | 45/54 | 48/54 | 94.98%/94.75% | 2/1 | 0.867/1.033 |
| 0.3/0.5/0.25 | 44/54 | 46/54 | 94.20%/93.88% | 2/0 | 1.100/1.283 |
| 0.3/1.0/0.0 | 46/54 | 47/54 | 93.96%/93.88% | 4/1 | 1.000/1.233 |
| 0.3/1.0/0.5 | 43/54 | 46/54 | 94.56%/93.84% | 1/1 | 1.033/1.283 |

每次commit均自动检查不早于其证据available_at与release_at，且每规划边界最多一次。时延分解：首候选观测→可用、K确认等待、defer等待、规划等待，四者相加等于首候选观测→commit；最新支持锚点的预测年龄另算。置信度、取消和闭环Active状态会随相位改变，因此不同场景的匹配集合/整体指标不保证单调；不将不同证据的两个commit当成同一事件作时间倒流判断。孤立相同消息的0/90/200/300ms因果时钟另有单元测试。

## 9. 困难案例、特殊Episode与风险

| 特殊三阶段模型 | 来源ID | 原始集合 | 期望序列 | 实际Active序列 | 匹配 | FINISH |
|---|---|---|---|---|---|---|
| future_only | ef866ed4c8f949d7b1a7d7fb3ef8ffbb | train | ['pickup', 'harvest', 'return'] | ['pickup', 'harvest', 'return'] | 2/2 | not_supported |
| joint | ef866ed4c8f949d7b1a7d7fb3ef8ffbb | train | ['pickup', 'harvest', 'return'] | ['pickup', 'harvest', 'return'] | 2/2 | not_supported |

四类合法边只约束局部阶段对，不能区分第一/第二轮return，也无法确定最终return之后应停止；特殊三阶段按自然序列评分，没有伪装七阶段。first/final return和短阶段标记已保留。数据结束≠FINISH已检测。

风险重点是相对标注过早harvest→return/ pickup→harvest、非法跳转尝试、K确认造成漏检或迟滞、Pending证据撤销、合法边拒绝导致错误状态长期保持。完整failures和每类时长供后续独立复核，本轮不对物理抓取完成或放回安全作判断。

## 10. 与既有93条训练报告的衔接

旧报告Future-only原始未来事件50/54、Joint Future 48/54，只要求边界前1秒至少一次预测下一阶段；本报告统计已经经过门槛、K、defer、规划时钟而真正提交的Active。不能把旧50/54或48/54当控制器检出，也不能将test Future Macro-F1较高直接解释为切换更好。原分类训练报告、checkpoint、预测日志没有修改。

## 11. 下一步与证据边界

不推荐据本轮test再挑策略或规划相位。当前四类概率可继续用于离线或Shadow提示研究，但是否需要Transition Head/Completion Head应由新的独立来源与任务完成定义支持；本轮既不实现新头，也不能证明完整HFSM必需或可省略。优先补充独立保留来源、真实5090+π0.5并发可用时延/队列测量、动作块更新协议和人工物理完成核验，再作上线决策。

未执行：人工视频复核、重新标注、真实GPU并发推理测速、π0.5/ROS/RTC/T3/HFSM集成、终止检测、新数据采集、过载队列及jitter。所有0/90/200/300ms结果只能称为假设实时参数下的离线开放环轨迹回放；Prompt变更没有生成新的机器人动作轨迹。

## 12. 复现、测试与附件清单

仿真/自动图表阶段用时58.9秒（不含最终全回归与打包）。Python `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]`。核心回放/策略/评分仅标准库；绘图依赖matplotlib/Pillow，未导入PyTorch/transformers。

```bash
cd /home/user/pi05/vjepa2_training
.venv-tomato/bin/python -u -m tools.compare_tomato_prompt_policies --output-dir /home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/prompt_switch_sim_20261108_p1_p6_v2
```

该命令是本次实际入口；输出目录已存在会拒绝覆盖。若要重现请换新目录，协议不变。单case CLI：

```bash
.venv-tomato/bin/python -m tools.simulate_tomato_prompt_switch --run-root /home/user/pi05/vjepa2_training/runs/tomato/local93_20261107 --split test --model future_only --policy-config configs/tomato_prompt_switch_offline.yaml --policy-id P3_d50 --selected-policy-lock /home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/prompt_switch_sim_20261108_p1_p6_v2/selected_policy.lock.json --output-dir /tmp/tomato_prompt_reproduce_new --latency-s 0.09 --plan-period-s 0.5 --seed 20261108
```

完整测试实际输出：

```text
........................................................................ [ 55%]
.........................................................                [100%]
129 passed in 11.08s
```

新增文件：6个实现模块（policy/replay/scorer/CLI/comparison/visualize）及本报告生成器、独立offline配置、3个测试模块；仅新增，不修改旧模块。与建议目录相比，将latency/policy边界测试合并在test_tomato_prompt_replay.py，评估与比较各独立模块，避免重复fixture。

实施记录：首轮开发运行在val网格结束后、生成selected_policy.lock及任何test控制评分之前，因CSV导出重复model关键字失败。修复序列化并增加端到端编排回归；首轮产物保留，最终在新目录重跑。阈值、候选网格和选择排序未改动；没有依据test修补策略。取消事件另明确保存canceled旧请求与pending=null的变化后状态，决策行为不变。

完整产物根目录：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/prompt_switch_sim_20261108_p1_p6_v2`。val_grid含所有12候选B/C；test_main为锁定主对照；auxiliary为额外信息对照；special_three_stage为特殊资料；sensitivity为各时钟场景。每case提供evaluation_control.json、逐Episode/逐转换CSV、failure_cases.jsonl、replay_config.lock.json及完整events/Active区间；P6日志使用确定性gzip压缩(.jsonl.gz)节省空间，内容仍可独立重算。

selection_protocol.json、selected_policy.lock.json、val_selection_rankings.json、test_paired_sources.csv、test_cluster_bootstrap.json、sensitivity_matrix.csv/json、plots/timeline_index.md及artifact_sha256.json提供完整追踪。精简证据ZIP包含主对照、全部val网格汇总/指标、场景矩阵、图表和源码测试；完整P6事件留在本地并由SHA清单引用。没有自动commit或push。


## 13. 完成后的独立复算与结果解读（不改变策略）

本节在全部预注册实验结束后仅核验保存的事件和补充解读，不重放新策略、不修改参数或模型。

### 实际回答研究问题

1. **没有全面胜出的模型。** 主测试Joint Future匹配48/54，高于Future-only的45/54，提前≥0.5秒也更少（1 vs 2）；但时长一致率略低（94.75% vs 94.98%），完整顺序8/9低于9/9，匹配提交中位数更迟（1.033 vs 0.867秒）。配对来源CI跨0，不能声称稳定控制优势。

2. **置信度/合法边能改变行为，但当前风险与覆盖尚未同时达标。** val的P1_t90相较P0改善完整顺序（B:4→6，C:2→7），却仍未令两模型共同满足预声明提前风险限制。P4将两者顺序都约束到9/9，但B时长一致率降至93.99%，并仍有8个提前≥0.5秒的匹配。顺序更整齐不等于切换时序更安全。

3. **本次Current硬门控不可用。** 在已锁定defer=0.5秒的规则下，该门控在val/test都没有commit，转换匹配0/54，Active长期停留pickup；test错误Prompt达1133.567秒。等待期间Current已改变，门控撤回Pending；Active一旦滞后，又因Current不一致而无法恢复。这是该具体“与Active不一致就冻结”规则的实测失败，不能推广为所有Current辅助无价值。未根据此test结果实现替代门控。

4. **反应式基线与未来策略也有取舍。** 同后端下Current-only test匹配47/54，略高于Future-only45/54，但时长一致率93.11%较低、Δt中位数1.633秒较迟、完整顺序8/9较低；其提前≥0.5秒为0。不能仅用单一匹配数选择部署模型。

5. **主要损失已经从提前风险转为确认/规划迟滞。** 共享选中的P3策略需两个2Hz锚点，至少增加0.5秒确认，再defer 0.5秒，并等规划边界；1秒预测先行量可能被这些等待耗尽。严格[-0.2,+0.5]窗口仅B:9/54、C:5/54。没有证据支持“未来分类好，就可以现在安全切换”。

### 与旧分类指标严格分栏

| 原始冻结forecast（非控制） | Test Macro-F1 | 原始边界前事件 | NLL |
|---|---|---|---|
| future_only | 0.923 | 50/54 | 0.149 |
| joint | 0.934 | 48/54 | 0.088 |

上述分类/原始预警值来自原报告，仅作为参照；主报告45/54和48/54是后端actual commit的有序匹配，定义不同。

### P6范围解读（所有场景预先约定，并未选择测试最优配置）

| 锁定策略/test | 转换匹配范围 | 时长一致率范围 | 匹配Δt中位数范围秒 |
|---|---|---|---|
| future_only | 43–46 /54 | 93.96%–96.24% | 0.417–1.100 |
| joint | 46–50 /54 | 93.84%–96.30% | 0.533–1.283 |

主要失效模式是规划网格错过释放时刻而增加整段等待、确认/释放期间候选撤回、Active偏离后难以回到合法顺序，以及门控锁死。模型置信度变化和时钟相位共同影响状态路径；不能把矩阵最优相位当作已验证部署参数。

### 独立保存产物审计

复核321个真实case、2873条Episode回放、895801个可用输入事件、20306次commit。对保存的Active/GT所有断点做独立中点积分，错误时长与CSV一致；所有区间连续覆盖，commit均不早于支持证据/释放时刻，且位于规划边界、每边界最多一次。错误数均为0。

实际消费5334条未来监督目标已超出录像支持的尾部消息，证明回放没有用future_valid提前裁去尾部。策略消息严格白名单、原始target不随HOLD移动。原321个历史文件和锁定源码哈希保持不变。

复算JSON：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/prompt_switch_sim_20261108_p1_p6_v2/independent_artifact_audit.json`；原始分类参照：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/prompt_switch_sim_20261108_p1_p6_v2/raw_forecast_reference.json`。复现补充步骤：`.venv-tomato/bin/python -m tools.enrich_tomato_prompt_report /home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/prompt_switch_sim_20261108_p1_p6_v2`。新增该报告后处理工具，不改变锁中的任何策略、评分或仿真源码。
