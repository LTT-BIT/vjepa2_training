# 服务器 A/B 的 93 条样本：三种模式训练

2026-11-07：已通过 tairan28 只读检查，A 有 64 条、B 有 29 条；先前 8 条的逐段 reviewed 状态已全部为 true，93 份 inherited 均为 false。配置创建时对全部 93 条执行了阶段区间检查、来源 episode/长度/记录 ID 对照，并锁定标注 revision/SHA256。尚未启动本轮全量视频核验、特征提取或训练。

## 配置及数据

- 配置：`configs/tomato_four_stage_remote93.yaml`。
- 显式的 93 条索引：`configs/tomato_remote93/annotation_index.json`。标注修改后会拒绝继续，不静默换版本。
- 固定划分：`configs/tomato_remote93/split.json`，沿用现有 seed=42 按 source_recording_id 分组的划分算法：75 train / 9 val / 9 test。三种模式必须共用该文件。A/B 共同参与划分，并非 A=train、B=test。
- 远程原始目录：`tairan28:/mnt/workspace/train_platform/datasets/tomato_sharpa/normal/no-block-more`。
- 本地只读挂载：`runs/tomato/remote93_mount`。不持久复制原始数据；远程读取会占用网络和系统 RAM 缓存。
- 共享冻结特征缓存：`runs/tomato/remote93_20261107/shared_feature_cache`。
- 原有本地三条样本实验和切柠檬代码不变。

只改变数据选择、划分文件和产物位置。已逐项对照此前三个真实实验的 training_config.json：仍为 58 个关节、16 帧/10 Hz 历史、2 Hz 预测、未来 1.0 秒、lambda_current=1.0、lambda_future=0.5、seed=42、batch_windows=1、batch_size=128、epochs=100、patience=15、learning_rate=0.0003、weight_decay=0.01。allow_subset_metadata_mismatch=true 维持原行为。

## 启动

在 Bash 终端中执行：

```bash
cd /home/user/pi05/vjepa2_training
bash tools/run_tomato_remote93.sh all
```

脚本会自动只读挂载，依次完成 Current-only、Future-only、Joint 的完整 run 流程，并在退出时卸载自己建立的挂载。使用系统 SSHFS；若未安装则使用上次测试在本地解包的客户端。使用现有 tairan28 SSH 登录配置（BatchMode，无交互密码）。建议在稳定终端或已有 tmux 会话中运行。

每个模式均通过原有 prepare 进行严格来源/标注/哈希/三相机视频核验，不跳过检查。第一种模式提取特征，后续模式复用缓存但仍执行来源完整性检查。因三个模式分别 prepare，全量用时会高于此前“两条性能测试”的简单线性外推；不会针对本轮测试集修改超参数。

也可分别启动：

```bash
bash tools/run_tomato_remote93.sh current_only
bash tools/run_tomato_remote93.sh future_only
bash tools/run_tomato_remote93.sh joint
```

若希望先验收全量数据再决定训练：

```bash
bash tools/run_tomato_remote93.sh all prepare
```

此命令执行完整来源和媒体核验，不加载 Encoder、不训练。之后仍可执行 `all`，已锁定的 manifest 会按原程序规则校验复用。

网络中断后，已完成的单条特征缓存可复用。若在训练前中断，可按已完成阶段使用 `模式 extract`、`模式 train`；若已有 metrics.jsonl/best_head.pt，原程序会拒绝覆盖或继续训练，不要删除现有结果强行重跑。已完成某个模式后，应单独启动尚未完成的模式，重复 `all` 会在已存在训练结果处拒绝执行。

## 结果位置

以下目录均位于 `/home/user/pi05/vjepa2_training/runs/tomato/remote93_20261107/`：

- `current_only_seed42/`
- `future_only_h1s_seed42/`
- `joint_h1s_lc1_lf0p5_seed42/`

各模式产物包含 `launcher_run_*.log`（终端完整日志）、`training_config.json`、`manifest.json`、`split.json`、`feature_index.json`、`metrics.jsonl`（逐 epoch 训练/验证）、`best_head.pt`、`evaluation.json`（选定模型的 train/val/test 离线评估）、`predictions.jsonl` 和 `predictions.csv`。遇到输入异常时查看 `prepare_rejections.json` 和 `status.json`。

训练完成后告知这三个输出目录即可继续分析，无须手工汇总指标。训练内部已执行离线分类/阶段事件评估，本启动器不额外生成逐帧视频或 GPU 延迟报告。

## 本次配置验证

- 93 条均能读取，先前 8 条逐段 reviewed 问题已修复。
- 全 93 条阶段区间与来源身份检查通过；固定划分为 75/9/9 个独立 source。
- YAML 原加载器解析、索引与划分一致性、三种 prediction contract 检查通过。
- Bash 语法检查和训练 CLI 帮助检查通过。
- 现有 test_tomato_joint_training.py 回归：9 passed in 2.56s。
- 尚未执行本轮全量文件哈希、媒体解码或训练，不将上述检查视为最终训练验收。
