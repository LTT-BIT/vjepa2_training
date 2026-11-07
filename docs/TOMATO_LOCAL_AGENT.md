# 本地番茄 Agent 训练与接口

当前优先完成可运行的训练和离线推理。冻结官方视觉编码器，训练 current/future 头；不改 OpenPI 模型、训练损失、归一化或 RTC。首次适配阶段只运行三条smoke，未进行94条全规模训练，也没有连接真实机器人。后续用户授权的正式训练以 [多方案训练说明](TOMATO_CAMPAIGN.md) 为准。

## 环境与官方模型

独立环境 `.venv-tomato`：Python 3.11.16，torch 2.7.1+cu128、torchvision 0.22.1+cu128、transformers 4.56.2、NumPy 1.26.4、PyArrow 19.0.1。完整依赖在 `requirements-tomato-local.lock`。本机 RTX5090 32GB、驱动595.84；未改动原pi05环境。

```bash
uv venv --python python3.11 .venv-tomato
uv pip install --python .venv-tomato/bin/python -r requirements-tomato-local.lock --extra-index-url https://download.pytorch.org/whl/cu128
.venv-tomato/bin/python -m tools.download_pretrained
```

模型来源 [Meta 官方仓库](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)，固定 revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`，下载至 `pretrained/vjepa2-vitl-fpc64-256`。权重1,303,947,864字节，SHA256 `25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b`；config SHA256 `3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9`。下载器验证官方元数据、字节区间和最终哈希，已有不同权重会拒绝覆盖。

## 数据与配置

本地数据：`/home/user/datasets/tomato_subtasks_normal_20261107`。仅显式选择的94条normal、94个原始录制ID，不含speedup和剔除数据。保持元数据和标注原样。

`configs/tomato_four_stage_local94.yaml`：joint、未来1秒、λ current=1/future=.5、seed42、严格来源校验false豁免、batch_windows4。来源分组为76/9/9，索引与split在 `runs/tomato/local94_20261107/`。原local93配置保留，不作为本轮入口。

索引/运行产物不提交Git。新机器提供下载后的显式selection，创建新运行目录和三种模式配置：

```bash
.venv-tomato/bin/python -m tools.setup_tomato_local \
  --selection /absolute/path/selection.json \
  --run-root /absolute/path/new_run \
  --pretrained-dir /absolute/path/vjepa2-vitl-fpc64-256
```

setup拒绝覆盖已有run、锁定严格来源SHA、按原始录制分组，三模式共享split和冻结特征缓存。prepare进一步检查视频全部帧/PTS及状态；有一条失败则整次选择失败。未来标签+30原始帧，尾部屏蔽，输入历史只到当前帧。归一化仅拟合训练特征。

本机完整94条分阶段命令（默认只prepare；extract/train须显式执行）：

```bash
bash tools/run_tomato_local94.sh joint prepare
bash tools/run_tomato_local94.sh joint extract
bash tools/run_tomato_local94.sh joint train
.venv-tomato/bin/python -m tools.eval_tomato_vjepa_phase --config configs/tomato_four_stage_local94.yaml
```

提取可以按episode复用已验证缓存。当前训练器保存best head，不支持优化器级断点恢复/last训练状态；中断训练需新run并复用特征，不得覆盖或假称resume。大规模训练前可另行补此能力。

## 已执行的小规模真实实验

`configs/tomato_four_stage_smoke3.yaml` 指向三条完整录制：从完整划分的train/val/test各取一条七段最短录制，长度2234/2685/2773帧。每模式8epoch、同一split与特征缓存；只用于验证代码和I/O，不能评价真实泛化能力或选择最终方法。

| 模式 | 最佳epoch | Test current Macro-F1 | Test future Macro-F1 |
|---|---:|---:|---:|
| current_only | 5 | .7406 | — |
| future_only | 8 | — | .6193 |
| joint | 5 | .7291 | .6018 |

joint未来头弱于model-current persistence基线(.6468)，不能宣称未来预测有效。三模式checkpoint和原始指标位于 `runs/tomato/smoke3_20261107/{mode}/`。joint生成独立评估、测试回放视频和三窗口离线测速。测速不是机器人端到端延迟，也没有证明并发运行时2Hz。

## 独立推理接口

```bash
.venv-tomato/bin/python -m tools.serve_tomato_agent \
  --checkpoint runs/tomato/smoke3_20261107/joint/best_head.pt \
  --pretrained-dir pretrained/vjepa2-vitl-fpc64-256
```

进程只加载一次模型，从stdin读取JSONL。启动/health响应`status=ready`。本地桥接请求：

```json
{"protocol":"tomato-stage-prediction-v1","observation_id":"obs-1","session_id":"session-1","timestamp":1794048000.0,"observation_npz":"/absolute/path/observation.npz"}
```

timestamp为Unix秒；每次请求使用真实观测时刻。NPZ仅包含rgb:uint8[3,16,256,256,3]、joints:float[16,58]、joint_names:string[58]；相机顺序cam0/cam1/cam2，RGB整图resize至256。当前接口接受已构造因果历史，不负责ROS/30Hz缓存、同步或在线缺帧恢复；调用方不得把未来帧放进历史。历史不足时训练采样器重复第一帧。

响应包含协议、checkpoint SHA、session/observation ID、输入/结果时间戳、固定四类顺序、两头logits/probabilities、mode、horizon、valid/error和模型调用耗时。inactive head为null。四类顺序为pickup/harvest/return/adjust；它们不是七阶段概率。关节按名称重排至checkpoint顺序，不把58维相同当成顺序相同。

过期观测和异常返回valid=false；预测异常保持服务进程可继续请求。JSONL为本地离线进程桥接，尚无网络RPC、请求调度/超时客户端、在线观测缓存或完整机器人控制集成。

真实heldout观测→官方Encoder+训练head→JSONL→OpenPI响应校验/Prompt provider的验证记录位于 `runs/setup/real_agent_bridge_verification.json`。未调用π0.5动作模型。Mock测试只证明状态机接口行为，不作为模型准确率结果。
