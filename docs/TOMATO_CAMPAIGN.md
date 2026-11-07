# 固定划分的多方案 Agent 训练

用户已授权正式训练多个方案。本轮冻结官方视觉编码器，仅训练四类current/future头，不训练OpenPI、不运行机器人。

## 本轮划分与试验计划

原76/9/9划分中的一条test曾在smoke中查看结果。新实验将该来源移至验证，不改训练来源：76train、10val、8test。旧manifest与split保留。本轮入口 `configs/tomato_campaign94.yaml`，锁定索引/划分和来源审计已版本化在 `configs/tomato_campaign94/`；严格媒体验收产物在 `runs/tomato/campaign94_20261107/`。历史三条pilot与本轮smoke的已知来源均不在8条test中，审计见split_audit.json。

样本是固定2Hz因果历史锚点：train22168、val2898、test2406；未来尾部mask，不用未来观测。三集合按原始录制ID严格分组。正常/加速副本不混用。

预先固定四组，各用seed42、43、44：

| 组 | 监督 | 变化样本权重 |
|---|---|---:|
| current | 当前分类 | 1 |
| future | 未来1秒分类 | 1 |
| joint | 当前＋未来，λ=1/.5 | 1 |
| joint_transition | 同joint | 2 |

变化样本严格定义为训练锚点有效的future label与current label不同。仅对这些样本提高加权CE的贡献，类别权重仍只用训练集统计。该策略是待检验的工程假设，不声称算法新颖性或预先保证收益。权重1与原训练损失在单元测试中数值一致；未来尾部不贡献未来loss。

全部方案共享同一冻结特征缓存、历史窗口、模型、数据划分、学习率3e-4、weight decay .01、batch128、最多100epoch、patience15。train-only特征mean/std和sqrt逆频率类别权重，各seed初始化独立。

## 选择与测试隔离

每次训练只计算验证指标，以活跃λ归一化的val Macro-F1、再NLL选epoch。joint与joint_transition按三个seed的平均验证得分选家族，部署代表固定为seed42，避免事后挑幸运种子。current/future单头作为对照，不把单头分数与双头混合分数直接排名。

12次训练全部完成后先写不可变selection.json，再计算8条heldout上的所有预定义对照。输出均值/样本标准差、每类指标、混淆矩阵、时间边界指标和persistence对照。forecast_baseline_comparison.json额外比较同seed的Current-only模型持久预测与每个未来头，并单列未来1秒标签变化的锚点；不将oracle-current当作可部署基线。不得根据test改权重、阈值或重新选择家族。

## 启动与恢复

先完成官方冻结特征提取；不得并行写同一episode缓存。增大离线batch只改变计算批量，每个episode记录实际性能/批量；各训练方案读相同最终缓存。

```bash
cd /home/user/vjepa2_training
# 全量缓存若未完成，先按已有入口继续提取；已核验缓存按episode复用。
bash tools/run_tomato_local94.sh joint extract
# 新split已经严格prepare；新机器需先执行：
.venv-tomato/bin/python -m tools.train_tomato_vjepa_phase prepare --config configs/tomato_campaign94.yaml
# 运行/恢复同一锁定计划
bash tools/run_tomato_campaign.sh
```

run目录 `runs/tomato/campaign94_20261107/experiments`。每个recipe/seed保存best_head.pt、last_head.pt、last_state.pt、训练配置、recipe身份、manifest/验证来源证据和metrics。last_state含optimizer、CPU/CUDA与采样RNG、早停状态；同命令自动恢复。代码、数据、配置或环境身份不同会拒绝resume。既有baseline训练器仍不支持resume；新campaign训练器支持，不要混淆。

派生run沿用已逐帧核验的同源媒体证据，main启动时重新核对父manifest/receipt及所有输入SHA。子manifest记录父manifest和receiptSHA；不重新伪造或放松来源检查。

12次完成后的selection.json记录代表checkpoint。heldout_comparison.json含完整分组结果，heldout_summary.json汇总三seed均值/标准差；各run的heldout_evaluation.json和predictions.jsonl保存独立证据。

最终checkpoint仍是兼容的四类v2 head，可用现有serve_tomato_agent加载，或用每个run的config.json调用eval_tomato_vjepa_phase回放。Future-only不能驱动当前阶段Prompt。任何离线准确率和提前预测指标都不等于实机安全切换或任务完成。
