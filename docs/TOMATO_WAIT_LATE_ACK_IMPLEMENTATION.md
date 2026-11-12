# WAIT_LATE_ACK 实施与验收（2026-11-12）

本次仅实现用户批准的迟到 Current ACK 机制，提交到新分支 `feat/tomato-wait-late-ack`，基线为 `07f93115d5427b3ebaaf1735e9bd18a1e1240aa5`。原分支 `feat/tomato-offline-review-ui` 不新增提交。工作区已有的其他修改保留，不纳入本次提交；没有 push。

## 行为与默认配置

- GUI 默认开启 `late_ack_enabled=true`；普通 ACK 窗口仍为 1.5 秒。旧 API/JSON 未指定新开关时默认 false，维持旧逻辑；GUI 可取消勾选作对照。
- 普通窗口内，一个 commit 后的新 Current 观测满足 top1=Active、置信度≥`theta_ack`（默认 .7），仍可完成普通 ACK。
- 普通窗口超时后进入 `WAIT_LATE_ACK`，保留 Active 和 prompt_version，暂停下一阶段请求。连续两个不同的新 Current 锚点支持当前 Active 才记录 `late_ack`，恢复后续候选资格。仅解除感知等待，不直接把 Active 改成 Current/Future。
- 不支持 Active、低置信或无效观测清空迟到计数；间隔超过 `prediction_ttl_s` 重新计数；HELD、重复或倒序锚点拒绝。证据观察时刻必须晚于实际 commit。
- `late_ack_timeout_s=5.0` 是从实际 commit 起的总硬截止，**不是软超时后再加 5 秒**。超过硬截止进入 `RECOVERY_REQUIRED`，之后再多的一致预测也不能自动解锁，只能显式外部恢复。
- 硬截止同刻先处理已完成预测；没有新预测时计时器也会触发软/硬超时。每次新 commit 重置窗口与计数。
- GUI 配置下载、重放缓存、事件按钮、时间轴、详情及视频叠加识别 late_ack/hard_timeout。修改参数后需重新应用模拟。

配置示例（其余字段沿用既有默认值）：

```json
{"late_ack_enabled": true, "ack_timeout_s": 1.5, "late_ack_timeout_s": 5.0}
```

## 代码范围

| 文件 | 修改 |
| --- | --- |
| `tools/tomato_review_ui/handshake.py` | 状态、窗口计时、连续证据确认、硬异常保护、事件及汇总 |
| `tools/tomato_review_ui/static/index.html` | 开关、总硬截止输入及普通窗口文案 |
| `tools/tomato_review_ui/static/app.js` | 配置校验、默认值、事件/状态展示 |
| `tools/tomato_review_ui/video_export.py` | 视频叠加识别迟到 ACK 和硬超时事件 |
| `tools/tomato_review_ui/HANDSHAKE_README.md` | 行为和兼容性说明 |
| `tests/test_tomato_wait_late_ack.py` | 12 项新测试（含参数化） |
| 本报告及 `docs/validation/wait_late_ack/` | 实际验证记录 |

未纳入其他未提交功能：5Hz 握手、即时提交、模型/数据包管理、研究包装器。本次未实现任务轮数契约或物理完成守卫；不改模型、训练、冻结特征、LeRobot 数据、人工标注和 metadata mismatch 策略。

## 实际验证

1. **独立提交树：161 passed in 11.45s。** 从 Git 暂存树导出干净源码，仅包含本次修改和基线，运行全部番茄及切柠檬回归测试。日志见 `validation/wait_late_ack/clean_commit_tests.log`。
2. **现有完整工作区：189 passed in 11.44s。** 包含此前已有未提交功能的额外测试；不是本提交单独拥有的测试数。日志见 `validation/wait_late_ack/worktree_tests.log`。
3. **真实冻结概率回放：85 来源。** 9 条原 val、75 条 train 回归、1 条当前外部案例；未评分原 test。独立提交代码关闭开关的 Active 区间与旧基线逐值相同；开启后与已验证 `S5_lateACK2_hard5` 的 Active 区间逐值相同，均为 85/85。见 `validation/wait_late_ack/clean_staged_replay.json`。工作区另核对了关闭开关时 85 来源事件完全相同、321 个原训练产物 SHA 不变，见 `real_replay.json`。
4. **真实浏览器验证：通过。** 对独立提交树启动服务，验证默认开启、103 秒处显示 WAIT_LATE_ACK、随后有 late_ack；关闭开关并重放无 late_ack；恢复默认重新启用且总硬截止为 5 秒。见 `validation/wait_late_ack/browser.json`。截图保存在本机 `runs/tomato/local93_20261107/wait_late_ack_20261112_verification/ui_wait_late_ack.png`。

全套测试命令：

```bash
cd /home/user/pi05/vjepa2_training
.venv-tomato/bin/python -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py
# 只运行本次新增测试：
.venv-tomato/bin/python -m pytest -q tests/test_tomato_wait_late_ack.py
# 启动当前分支的本地界面，识别 Joint 2Hz 后点击“应用并重新模拟”：
.venv-tomato/bin/python -m tools.tomato_review_ui.server --run-root runs/tomato/local93_20261107 --port 8784
```

独立提交树测试时，归档起初缺少 Git 历史，导致检查原 metadata 策略的既有测试失败；补齐只读历史对象引用后全套 161 项通过。此处是测试夹具环境修复，没有修改该既有测试或 metadata 策略。

## 跨样本效果与限制

本次正式实现的 Active 区间与前轮研究 S5 结果逐条相同，因此可复用其配对结果；没有重新训练或根据 test 调参。

| 集合 | 旧 → 新 |
| --- | --- |
| 当前案例 | 错误 Active 8.367 → 3.367 秒；匹配切换 4/6 → 5/6 |
| 9 条 val | 错误 Active 273.733 → 98.233 秒；匹配 37/54 → 43/54；完整顺序 6/9 → 9/9 |
| 75 条 train（仅回归） | 1 条改善、74 条不变；错误 Active 252.733 → 176.600 秒 |
| 76 条基线正常来源 | Active 区间全部不变，未观察到新增退化 |

**代价与未解决项：** val 未匹配提交 9 → 11，train 未匹配提交 3 → 4；恢复后新增提交仍可能时序错误。当前案例约 75.59 秒完成迟到 ACK，113 秒能够提交 return，但 118 秒仍误提交 adjust，完整顺序仍不正确。本次没有实现其余两项建议，不能宣称解决尾部误切或物理完成问题。

5 秒来自先前 val/回归探索性敏感性分析，不是已证明最优或物理安全的参数；train 回放不是独立泛化精度。全部结论基于固定 Joint 2Hz 离线概率，不涉及实机控制、重新训练、模型精度提升或 test 调参。其余任务继续/完成机制留待用户决定。
