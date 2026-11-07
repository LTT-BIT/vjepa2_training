# 番茄 Agent 训练适配

`research-tomato-agent` 从 `feat/tomato-four-stage` 的 `63d89e8` 建立。
当前使用 Meta 官方 `facebook/vjepa2-vitl-fpc64-256` 冻结编码器，训练四类 current/future heads。
本地94条严格校验配置、三模式真实小规模试验及独立推理接口见 [训练与接口使用说明](docs/TOMATO_LOCAL_AGENT.md)。
OpenPI 通过 `tomato-stage-prediction-v1` 四类协议对接，不加载本仓库 checkpoint 到旧七分类模型。
下面保留的柠檬说明与已有结果属于历史项目，不能用于描述番茄模型性能。

---

# Lemon V-JEPA 2 阶段判断训练包

这是当前六阶段、三路相机、双臂12维状态的 V-JEPA 2 阶段分类工程。模型输入是 `cam0/cam1/cam2` 的因果视频历史和左右臂12个关节位置，输出六个动作阶段：

0. 建立左手支撑
1. 对准铲子
2. 铲取柠檬片
3. 移向杯口
4. 倒入杯子
5. 任务完成

## 文件

- `tools/train_lemon_vjepa_phase.py`：主训练入口；支持 `prepare` 和 `run`。
- `tools/run_lemon_doubao_vjepa.py`：调用豆包 API 对视频重新标注阶段，再训练并用人工标注测试集评估。
- `tools/annotate_lemon_phase_video.py`：把阶段预测叠加到 cam0 视频上。
- `tools/lemon_phase_model.py`：离线/实时模型辅助和旧阶段模型兼容接口。
- `tools/serve_lemon_phase_agent.py`：JSONL 实时阶段推理服务，不发送机器人命令。
- `annotations/phase_annotations_v3_clean.json`：人工五时间戳标注输入。
- `tests/`：训练、标注、模型接口和数据防泄漏测试。
- `experiment_results/`：已有训练配置、manifest、评估结果、best head 和验证视频；不包含预训练权重和特征缓存。

## 训练

当前脚本默认路径来自原工作区：

```text
数据集：/home/descfly/Convert_data/outputs/lemon_move_wrist_reset_20261008
人工标注：/home/descfly/Downloads/phase_annotations_v3_clean.json
运行目录：/home/descfly/hmj/training/experiments/lemon_vjepa2_phase_20261027
```

准备并检查数据：

```bash
cd /home/descfly/hmj/ur
python3 tools/train_lemon_vjepa_phase.py prepare \
  --dataset /home/descfly/Convert_data/outputs/lemon_move_wrist_reset_20261008 \
  --annotations /home/descfly/Downloads/phase_annotations_v3_clean.json
```

训练（需要 CUDA、PyTorch、Transformers、V-JEPA 2 预训练权重）：

```bash
python3 tools/train_lemon_vjepa_phase.py run \
  --dataset /home/descfly/Convert_data/outputs/lemon_move_wrist_reset_20261008 \
  --annotations /home/descfly/Downloads/phase_annotations_v3_clean.json
```

`run` 会抽取冻结的 V-JEPA 2 视频特征，训练阶段分类头，生成 `best_head.pt`、评估 JSON、预测 CSV 和带阶段叠加的验证视频。脚本按原始录制分组划分 train/val/test，避免同一条录制泄漏到不同集合。

## 豆包重标注版本

需要先准备豆包 API key，并按脚本提示输入：

```bash
python3 tools/run_lemon_doubao_vjepa.py prepare \
  --reference-run /home/descfly/hmj/training/experiments/lemon_vjepa2_phase_20261027

python3 tools/run_lemon_doubao_vjepa.py run \
  --reference-run /home/descfly/hmj/training/experiments/lemon_vjepa2_phase_20261027
```

豆包标注不能替代缺失的人工边界；未解析出完整五个边界的样本会被排除。

## 实时阶段服务

```bash
python3 tools/serve_lemon_phase_agent.py \
  --model /home/descfly/hmj/training/experiments/lemon_vjepa2_phase_20261027/best_head.pt \
  --device cuda
```

服务使用 JSONL stdin/stdout 协议，输入三路 224×224 RGB 图像和12维关节状态，只返回阶段概率，不控制机器人。真机客户端仍需使用工作区的 `robot_control/phase_agent_client.py`。

## 测试

```bash
cd /home/descfly/hmj/ur
pytest -q \
  /home/descfly/Desktop/lemon_vjepa2_training_20261106/tests/test_lemon_vjepa_phase.py \
  /home/descfly/Desktop/lemon_vjepa2_training_20261106/tests/test_lemon_doubao_vjepa.py \
  /home/descfly/Desktop/lemon_vjepa2_training_20261106/tests/test_lemon_phase_model.py \
  /home/descfly/Desktop/lemon_vjepa2_training_20261106/tests/test_annotate_lemon_phase_video.py
```

## 说明

包内没有复制原始 LeRobot、视频、V-JEPA 2 预训练权重和全部特征缓存；这些文件体积大且应从原路径或模型缓存读取。`experiment_results/best_head.pt` 只是已训练的分类头，不是 V-JEPA 2 完整预训练模型。

- `tools/download_pretrained.py`：按固定模型 revision 下载并校验 V-JEPA 2 预训练权重。
- `tools/train_source_snapshot.py`：记录训练源文件快照，便于复现实验。
