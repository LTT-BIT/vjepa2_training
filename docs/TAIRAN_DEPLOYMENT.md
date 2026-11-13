# Tairan：番茄 Joint 实机验证交接（2026-11-13）

本次交接固定为 local93 原始 Joint 四阶段双头模型及 main 已提交的 WAIT_LATE_ACK 机制。按用户要求先在 main 工作区修改，交付提交仅放到 `feat/tairan-deployment-handoff-20261113`；本机最后返回 main，继续跟踪 origin/main，不把该交付提交合并到 main。大权重不进 Git。

## 调取代码

```bash
git fetch origin
git worktree add --detach ../vjepa2-tairan-handoff origin/feat/tairan-deployment-handoff-20261113
cd ../vjepa2-tairan-handoff
git rev-parse HEAD  # 记录实际交付代码提交号，发回部署端日志
```

交付基线：`db4cdad6e87af7c077f5be16a00bc277efeb54d9`。WAIT_LATE_ACK 已在该 main 提交中，不需要寻找已删除的旧分支。该交付分支额外补全 GUI 的 TTL 显示、编辑、重置和参数导出，以及校验工具、配置和本说明；没有新增握手切换规则。机器上候选门控实验对 handshake.py 的未提交重构未纳入交付。`configs/tairan_deployment.json` 记录关键推理/握手源码的 SHA256，校验失败时先核对 checkout，不混用实验工作区。

**训练时 Git 提交号未知：** 原 best_head.pt 和训练配置没有保存这个字段；不能把当前 main 或文件日期当成训练代码提交证明。当前交付代码和模型兼容性已通过加载及真实缓存复算验证。

## 正式模型及配套材料

模型输出目录：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/joint/`。

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| `best_head.pt` | 4144993 | `9324cdb3287c407c259eafbc725ac3bc3d3df6d38646545e8435e0faa9ad03d9` |
| `manifest.json` | 2646155 | `474301f37a8dd65d78fdde4624abacc4d0644a5401598121231198e969f11750` |
| `split.json` | 3771 | `5ccda5a1499c01082c5223fbdb04d780f8cfb02da7146d5c50cc748ab32281de` |
| `training_config.json` | 4594 | `6d704437c838cf50dbf8314fb697db1d01ed21d3c6c9211dfb64bf726cf3a553` |
| `validation_lock.json` | 271 | `78384e6670d800e11336b39362f601b439cfc30fb8b5943932c9c0a05053df5c` |
| `validation_receipt.json` | 9667 | `8c532596fa3e8aea3fc8f53e56a179d6c005f8dba1923e7da8c09f2bfc71c7bb` |

以上六文件必须成组提供。已在本机打包为 `/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/tairan_handoff_20261113/joint_artifacts.tar.gz`（SHA256：`6fc2c3bcbfa9ec79cbe3cb52dccf848dab074dd60a3aebc7ed19ed67709082fc`），不含 Encoder、原始数据或特征缓存。这个包及权重均未提交 Git；当前尚无约定的远端共享存储地址，同事需要复制该包并解压，或直接访问上述机器目录。

冻结 Encoder 目录：`/home/user/pi05/vjepa2_training/pretrained/vjepa2-vitl-fpc64-256/`。

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| `config.json` | 785 | `3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9` |
| `model.safetensors` | 1303947864 | `25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b` |
| `weights_metadata.json` | 211 | `a8ef69890ac611854138c9da48fdf9c7c264f62b0268380943f02d2102189336` |

Encoder 为 `facebook/vjepa2-vitl-fpc64-256`，固定 revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`；三文件一起复制。头权重约 4.1 MB，Encoder 约 1.30 GB（十进制）。现有 loader 使用本地目录并校验官方固定哈希，不会下载或静默替换模型。

## 输入、归一化和输出契约

- `best_head.pt` 为 format_version=2、prediction_mode=joint，best_epoch=7；1,026,312 个头/共享 trunk 参数。Current 预测观测时刻，Future 预测该观测时刻 +1.0 秒。标签顺序严格为 `pickup, harvest, return, adjust`。
- `mean` 和 `std` **均嵌入 best_head.pt**，shape 都是 `[4000]`，已核验有限且 std>0，由训练集拟合；部署不能重新拟合，也不能对关节单独另做一次归一化。
- 三相机固定按 `cam0, cam1, cam2` 顺序；每路 16 帧、256×256 RGB、10Hz 历史采样，原模型预测 2Hz。图像使用既有整帧 resize 和 ImageNet 归一化；共享冻结 Encoder 输出 token 均值，每路 1024 维，按相机顺序拼成 3072 维。
- 其后拼接按时间顺序展平的 16×58 个实测关节位置，合计 4000 维；最终对整个向量用 checkpoint 的 `(x-mean)/std`。58 个字段的准确名称和顺序在 `configs/tairan_deployment.json:input_contract.joint_names` 及同次 manifest/checkpoint 中，必须按名映射，不能只凭 observation.state 下标或长度猜测。
- 历史采样采用 `uniform-integer-causal-v2`，复用 `tools/tomato_dataset.py:sample_causal_indices`；三路图像、关节和时间戳对齐后再调用既有 fuse_features/predict_heads，不使用未来帧或动作命令补输入。
- 实机相机序列号与 cam0/1/2 的物理映射，以及实测关节字段、单位和时钟到录制格式的一致性，需由部署端核验；本工程未提供双设备的实际硬件映射。58 个字段必须都真实存在，缺失时不补零、不合成手指状态。
- `training_config.json` 的 lambda_current=1.0、lambda_future=0.5 是训练/验证选模权重，**不是在线把两个头概率加权混合的系数**。

## 当前完整握手参数

直接读取 `configs/tairan_handshake.json`，纯参数对象可以传入既有 `Handshake(config)`；辅助部署信息另存于 tairan_deployment.json，不混进策略 config。

| 参数 | 本次导出值 |
| --- | --- |
| `margin` | `0` |
| `theta_prepare` | `0.65` |
| `theta_commit` | `0.8` |
| `K` | `1` |
| `ack_timeout_s` | `1.5` |
| `late_ack_enabled` | `true` |
| `late_ack_timeout_s` | `5` |
| `prediction_ttl_s` | `0.75` |
| `min_confirm_span_s` | `0` |
| `theta_ack` | `0.7` |
| `plan_period_s` | `0.5` |
| `plan_offset_s` | `0` |
| `latency_s` | `0.09` |
| `legal_edges` | `true` |
| `ack_enabled` | `true` |

固定机制：Joint 2Hz；planning 模式、defer=0、初始 Active=pickup；合法后继 pickup→harvest→return→adjust→pickup。`K=1` 是候选确认数；迟到 ACK 的连续新观测要求固定为 **2 个**，不与 K 混用。

普通 ACK 窗口为 commit 后 1.5s；超时进入 WAIT_LATE_ACK，保留 Active/version，当前阶段连续两个新 Current 观测置信≥.7 才解除等待。总硬截止为 **commit 起 5s**，不是再额外等 5s；到期仍进入 RECOVERY_REQUIRED，之后只能显式外部恢复。TTL=.75s；重复/HELD 不能当新确认。正常窗口仍只需一次新 Current 确认。ACK 表示感知一致，不表示机械动作完成。

本次参数来自干净 main 基线加本次 UI 修复后的“恢复默认”，2026-11-13 经真实浏览器下载并显式应用，已核验下载 JSON 与后端配置逐字段一致。它不是声称读取了另一位同事的在线会话：历史 review_ui/handshake 旧日志缺少 late_ack 字段、含旧 commit_mode 包装字段，不可冒充本次交接配置。main 只接受 policy JSON 中的 15 个字段，**不要**把辅助文件中的 commit_mode 等说明字段传给 validate。

`latency_s=.09`、规划周期 .5s 和相位 0 是离线回放配置。实机必须使用观测时间和**真实结果可用时间**构造 Evidence，并按实际规划 tick 调用 planning/timer；不要为了复刻模拟把推理结果强行标成观测 +.09s，也不要忙循环复用同一帧作为新证据。无新结果时仍需运行超时计时器。不要把视频结束等同于任务完成。

## Python 和 Encoder 环境

已核验机器解释器：`/home/user/pi05/vjepa2_training/.venv-tomato/bin/python`，Python 3.12.3，include-system-site-packages=true。实际模块版本如下（不是声称另一台机器已安装）：

| 模块 | 版本 |
| --- | --- |
| numpy | 1.26.4 |
| torch | 2.15.0 |
| torchvision | 0.29.2 |
| transformers | 5.22.1 |
| safetensors | 0.8.0 |
| pyarrow | 19.0.1 |
| av | 19.0.2 |
| PyYAML | 6.0.3 |
| pytest | 9.2.0 |
| huggingface_hub | 2.4.0 |
| OpenCV (system cv2) | 4.6.0 |

Torch 实际 build：`2.15.0+cu132`，CUDA build：`13.2`；OpenCV 来自系统模块 `/usr/lib/python3/dist-packages/`，无 pip opencv-python 分发元数据。本次头加载/复算运行于 CPU。Encoder 的现有 CUDA loader 要求可用 CUDA 和 BF16；不支持时显式选择 CPU FP32，不存在静默降级。本次没有测量实机吞吐或用历史 .09s 证明实时性。依赖范围参考 requirements-tomato.txt，精确本机版本与路径记录在部署清单 environment 内。

## 同事校验命令

在干净交付 checkout、材料仍在原位置时：

```bash
/home/user/pi05/vjepa2_training/.venv-tomato/bin/python -m tools.verify_tairan_deployment
```

复制到另一台机器后，可将 joint_artifacts.tar.gz 解压得到 joint/；保持六个 run 文件和三个 Encoder 文件的原文件名，按实际目录传入，不修改锁定 manifest 的路径或字节：

```bash
python -m tools.verify_tairan_deployment --run-dir /实际存储目录/joint --encoder-dir /实际存储目录/vjepa2-vitl-fpc64-256
```

工具只读，成功输出 status=PASS；逐项校验九个文件的大小/哈希、源码、完整策略字段、checkpoint/manifest/split/训练配置/验证锁、归一化及冻结 Encoder 绑定。**不会**按 manifest 原训练路径访问全部数据，也不会运行 Encoder/连接机器人；材料迁移后原训练路径可以保持不变。PASS 表示交接材料一致，不表示实机验收完成。

本次测试可复跑：

```bash
python -m pytest -q tests/test_tairan_deployment.py tests/test_tomato_wait_late_ack.py
python -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py tests/test_tairan_deployment.py
```

## 本次实际验收与剩余工作

- 独立干净交付树全套回归：**170 passed in 11.67s**（新增 9 项交接测试）（包含切柠檬和既有番茄测试）；新增测试覆盖目录迁移、材料/源码/配置篡改、漏 TTL、无效归一化、manifest 绑定及训练/交接契约不一致。
- 现有完整工作区额外验证：284 passed in 12.23s，包含未提交实验测试；这不是本交付分支单独拥有的测试数。
- 九项真实文件验证通过；真实 Joint val 缓存的前 8 个锚点 `[8,4000]` 在 CPU 复算，Current/Future 与原冻结概率最大绝对差分别为 `6.123e-11` / `2.061e-11`。这是推理一致性检查，不是新精度评估；未评分 test、未训练模型。
- 浏览器验证：默认下载包含全部 15 字段；后端应用完全一致；TTL 改为 .9 后旧模拟失效且重放使用新值；恢复默认还原 .75。实际日志见 docs/validation/tairan_deployment_20261113/；截图留在本机交接包旁 ui_export.png。
- 未做在线三相机/关节接入、实机动作验证、双选 UI、飞阶段映射、执行反馈或 Agent 接入。这些由部署同事继续完成。握手机制仍可能在末尾 return 后误提交 adjust；任务继续授权和物理完成守卫仍未实现，不能把这次材料交接当作已修复。
- 原训练 Git 提交无记录、远端大文件共享位置未约定、实机相机/关节/时钟映射和硬件环境未验收，均已明确列出，没有用合成单测代替真实模型或实机指标。
