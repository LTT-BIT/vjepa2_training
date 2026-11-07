# 首轮 Agent 正式训练结果

2026-11-07完成。只训练冻结官方V-JEPA 2特征上的四类阶段识别/预测头；没有训练OpenPI或控制机器人。

## 数据与执行

- 94条normal，76train/10val/8test；原测试一条已用于smoke，转入验证。六个已知pilot/smoke来源均不在最终test中。
- 22168/2898/2406个2Hz锚点；未来标签为+30原始帧，尾部mask。
- 四组×seed42/43/44，共12次正式训练，全部完成；最多100epoch、patience15。
- 官方模型、相机、58D名称、因果采样及train-only归一化相同；各方案共用冻结缓存。
- CPU与RTX5090的中断恢复都通过参数/损失精确一致检查，专项测试91项通过。

## 最终独立测试结果

表中为三个随机种子的均值±样本标准差，不是对测试录制总体的置信区间。

| 方案 | 当前Macro-F1 | 未来1秒Macro-F1 |
|---|---:|---:|
| Current | 0.9546 ± 0.0030 | — |
| Future | — | 0.9348 ± 0.0081 |
| Joint | 0.9522 ± 0.0015 | 0.9137 ± 0.0104 |
| Joint Transition | 0.9551 ± 0.0021 | 0.9117 ± 0.0158 |

Current-only模型的持久预测（预测1秒后仍是模型当前输出）平均未来Macro-F1为0.9066。Future-only为0.9348，Joint为0.9137；这里比较的是可计算的模型基线，不是把真实当前标签当作可部署输入。

未来1秒类别变化的锚点上，三seed平均预测准确率：Current持久基线19.10%，Future-only70.49%，Joint68.40%，Joint Transition77.08%。边界强化具有局部收益，但整体未来F1及验证主指标未超过普通Joint，不能据此声称整体更优。

## 选择与可加载模型

先锁定验证选择，再执行test。Joint家族平均验证主分数0.960082，边界强化Joint为0.958745，因此选普通Joint；代表seed42事先固定，没有按test挑seed。

- checkpoint：`/home/user/vjepa2_training/runs/tomato/campaign94_20261107/experiments/joint_s42/best_head.pt`
- SHA256：`92e4b2dbd3dbdcb287fa0c4e29a6020679a90c58bf1777a324826f7ff53c5bcc`
- 该单个checkpoint的测试当前Accuracy/Macro-F1：0.9792/0.9537；未来Accuracy/Macro-F1：0.9628/0.9253。
- 其未来预警评估48个边界，39个检出，中位提前量0.667秒；另有32次窗口外预警，不能把未来头的正确分类等同于可安全提前切换。
- 已完成官方Encoder＋新head＋真实保留测试观测→JSONL→OpenPI Provider的离线验证，过期消息拒绝。没有调用π0.5动作模型。
- 独立eval/测试回放成功，errors.json为空。离线三窗口Encoder+head平均约44.8ms，不代表ROS/机器人端到端延迟或并发负载性能。

```bash
cd /home/user/vjepa2_training
.venv-tomato/bin/python -m tools.serve_tomato_agent \
  --checkpoint runs/tomato/campaign94_20261107/experiments/joint_s42/best_head.pt \
  --pretrained-dir pretrained/vjepa2-vitl-fpc64-256
```

## 原始产物

- run根目录：`runs/tomato/campaign94_20261107/experiments/`。
- 每个模型：best_head.pt、last_head.pt、last_state.pt、recipe/config、metrics.jsonl、validation_result.json、heldout_evaluation.json及逐锚点预测。
- 总表：heldout_summary.json、heldout_comparison.json、forecast_baseline_comparison.json、selection.json、source_snapshot.json。
- 可加载路径与校验值：`runs/tomato/campaign94_20261107/agent_bundle.json`。
- 回放：`joint_s42/offline_eval/14ee7fa242865e1d2d940612.mp4`。
- 本文数据摘要：[可核对JSON](validation/tomato_campaign94_20261107_summary.json)。

本轮测试已用于最终报告，后续新方法应继续只在训练/验证开发，新增独立数据用于新的最终验收；不得把本轮test反馈用于调参后仍称其未见测试。
