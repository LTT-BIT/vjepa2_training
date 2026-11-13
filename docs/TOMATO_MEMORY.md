# Memory 协作开发入口

Memory 统一在本仓库 `research-tomato-agent` 分支开发和评审：
<https://github.com/LTT-BIT/vjepa2_training/tree/research-tomato-agent>。
本地旧 `memory_cooperation_v1` 目录仅保留历史备份，后续修改提交到本仓库。

## 当前研究：实测 state＋局部 RGB 历史

| 内容 | 入口 |
|---|---|
| 有界历史存储、时间和身份校验 | `tools/tomato_rgb_memory.py`：`RGBStateMemory`、`RGBStateWindow` |
| 可训练历史读出结构 | `tools/tomato_rgb_memory_model.py`：`RGBMemoryReader`、`reader_batch` |
| 四路窗口裁剪与冻结编码 | `tools/tomato_interaction_rgb.py` |
| 配置与历史长度候选 | `configs/tomato_rgb_memory_v2/experiment.json` |
| 合成契约、读出反传测试 | `tests/test_tomato_rgb_memory.py` |
| 有真实视频和权重时的接线检查 | `python -m tools.smoke_tomato_rgb_memory --help` |

每个观测保存原三路 `[3,1024]` 和一路局部 `[1024]` 冻结视频特征、完整 `[16,58]` **实测位置**、16个采样时刻、状态时刻、ROI坐标、有效性及坐标可用时刻。不是把动作命令当实测 state，也不是只保存窗口末帧。单窗口为16帧、10Hz、跨度1.5秒；窗口终点按2Hz推进，默认最多保存16个窗口。读取1/8/16窗口是待验证的候选，不代表已选出最佳历史长度；8/16窗口终点跨度分别为3.5/7.5秒，包含最早窗口的原始覆盖分别为5/9秒。

阶段记录和独立结果不进入该版读出输入。模型结构已存在，尚未完成正式读出训练及收益验证；默认B0、阶段Prompt及机器人执行逻辑不变。读出头的四维输出对应 `pickup/harvest/return/adjust`，不是七维阶段概率。

`mode='online'` 要求当前及过去可用的因果ROI；离线预览使用显式 `mode='offline_diagnostic'`。OpenPI仓库本批 `full-block-more` 的前后帧辅助ROI是离线数据，不能声明为在线自动定位结果；这批数据到Agent Memory的读取适配器仍待接入。本轮只统一代码归属，不把已有离线crop直接接到在线机器人。

## 迁入的早期协同原型

| 内容 | 仓库内位置 |
|---|---|
| 原 `memory.py` | `tools/tomato_cooperative_memory.py` |
| 原11项合成测试 | `tests/test_tomato_cooperative_memory.py` |
| 原缓存回放脚本，改为仓库相对默认路径和显式参数 | `tools/smoke_tomato_cooperative_memory.py` |
| 历史协议及原始源码哈希 | `configs/tomato_cooperative_memory_v1/experiment.json` |
| 2026-11-11历史回放结果 | `experiment_results/tomato_cooperative_memory_v1/historical_smoke_20261111.json` |

该原型保存三路视觉特征和末帧58D位置，复用 `TaskMemory`、`OperationMemory` 与B0的旧七位置流程检索上下文；输入仍是四类current/future概率。其 `reposition_on_plate` 等目标是**旧任务假设**，不能直接套到新批次“持串旋转后继续摘取”的流程。保留它用于回归与对照，不作为新版state＋局部RGB研究的默认入口。

独立结果只有外部审核或注册验证器的证据才能确认；未提供证据时保持Unknown。原型没有经过训练的结果识别器，也不会自动切换Prompt或发送动作。历史2269观测回放只证明与B0输出一致，不能视为Memory增益或实机成功率。

## 同事怎样开始

在自己的工作目录检出同一分支，按本仓库训练与接口说明安装依赖。下面的合成测试不需要本地数据集、缓存、权重、旧原型目录或机器人：

```bash
git clone --branch research-tomato-agent https://github.com/LTT-BIT/vjepa2_training.git
cd vjepa2_training
python -m pytest -q tests/test_tomato_rgb_memory.py tests/test_tomato_cooperative_memory.py
```

已有真实缓存时可另外运行历史接线回放：

```bash
python -m tools.smoke_tomato_cooperative_memory \
  --manifest /path/to/dataset_manifest.json \
  --history-root /path/to/complete100_history_results \
  --predictions-root /path/to/transition_time/seed42 \
  --out /path/to/new_receipt.json
```

manifest旁的 `data_identity.json` 中特征路径、history结果所引用的checkpoint必须在本机实际存在且哈希匹配。默认路径从当前仓库计算，不绑定某位开发者的 `/home/user`；输出必须是新文件。缺少这些外部材料时只运行合成测试，不虚构真实回放结果。

代码、配置、测试及精简结果摘要提交Git；原始数据、特征缓存、checkpoint、密钥留在各自外部存储。变更输入协议时同步修改配置、测试和本文；讨论时注明使用新版RGBStateMemory还是旧七位置原型，避免混合对照。

ROI生成和π₀.₅训练代码位于另一个协作仓库：
<https://github.com/LTT-BIT/tomato-openpi/tree/tomato-subtask-policy>，主要入口为 `scripts/tomato_full_block_roi.py`。Agent内的四路特征提取在本仓库，两个仓库通过明确的图像、时间和坐标协议衔接。
