# 番茄四阶段分类器实施报告（2026-11-06，Asia/Shanghai）

> **历史报告说明**：下文保留首次实施（截至 5dafbe2）的真实审计事实。基线 11d2b57 之后的三模式、P1-1～P1-4 修复、真实 Encoder/三条样本 smoke 及实际评估，见 [本轮三模式报告](TOMATO_CURRENT_FUTURE_JOINT_REPORT.md)。本轮未改变 11d2b57 的元数据豁免策略；目前已有真实训练和回放，不能将下文早期“缺权重/未训练”作为本轮状态。

代码与合成测试已按四阶段实施；**真实番茄训练和模型评估未执行**。所有修改和提交仅在本地 `feat/tomato-four-stage`，没有 push、PR、数据发布或部署。分类固定为 pickup、harvest、return、adjust；没有实现 Agent、Prompt Manager、π0.5、RTC/T3、ROS 或真机控制。

## 基线和分阶段结果

训练仓库本地 main 和 GitHub main 在开始实施时均为 `6a6df2f89e81bd1c47471f3a3eb306d6180ac977`。只读标签参考的远端 `subtasks-data-convert` 仍为 `90ad2663ec645c4b2d6322e9df25284dfa7e0b0a`，与方案一致。另一个本地 convert_data_job-worker checkout 在不同的 main 上，未切换、拉取、修改或当作该参考版本。已读取远端固定版 core/workspace 源码核对保存、审核及来源身份。

| 本地阶段 | 提交/内容 | 实际相关单测及原 V-JEPA 核心回归 |
|---|---|---:|
| 1 | `c351ff8`：显式索引、只读来源校验、58 维白名单、采样、prepare | 32 passed |
| 2 | `e97a9f0`：冻结 Encoder、预处理/融合、安全 checkpoint 和特征缓存 | 37 passed |
| 3 | `29ca601`：分组训练、四分类及时间指标、grouped CV | 43 passed |
| 4 | 离线回放、GPU 测速入口、文档、来源/缓存复核及最终回归 | 54 passed |

原仓库 **21 个已跟踪文件逐字节保持 main 原样**，包括全部 tools/旧文件、tests/旧文件、annotations/、README、SHA256SUMS 以及 experiment_results/ 的分类头和视频。输入导出包自带 SHA256SUMS 中 **27/27** 个文件复核通过。详见 [完整性和环境证据](validation/integrity_and_environment.json)。未修改原始 LeRobot、粗/细人工标签和只读参考仓库。

## 实际数据结构与匹配结果

样本根目录：

```text
/home/user/pi05/sharpa_1103_normal_000000-000002_20261106-160943/1103_620_642_normal_lerobot
```

`meta/info.json`：`codebase_version=v2.1`，`robot_type=dual_fr3_sharpa`，30 FPS，按 data_path/video_path 模板解析；三路相机 `observation.images.cam0/cam1/cam2`。Parquet 为实际观测 state **116 维**、action 58 维，但 **action 完全不作为输入或监督**，task_index/coarse_stage_index 也不用。

state 中的位置白名单 **D=58**：左/右 Franka 臂各 7 个 `.position`；左/右 Sharpa 手各 22 个 `.position`。其余 58 个速度不使用。名称/顺序全部写在 [配置](../configs/tomato_four_stage.yaml)，具体原列与手部关节名见 [训练说明](TOMATO_FOUR_STAGE_TRAINING.md#2-数据审核和关节白名单)。这不是沿用柠檬的 12 维列号。

三条样本均完成实际 Parquet 读取、时间轴/源时间戳、三路 MP4 **全量解码**、FPS、帧数、尺寸、逐帧 PTS 对齐校验。共核对 43,848 个相机视频帧，最大视频 PTS 与 Parquet 时间误差约 `7.12077e-6 s`。选择的 58 个位置通道均有限；Episode 0 的 58 个通道都有数值变化。真实样本读取证据见 [data_audit.json](validation/data_audit.json)。

| Episode | 原录制 source_recording_id | 帧数 | 诊断 2 Hz 锚点 | pickup / harvest / return / adjust 候选锚点数 |
|---:|---|---:|---:|---|
| 0 | `76354615cf9746758c5c7e6e5ab579d4` | 4016 | 268 | 34 / 190 / 18 / 26 |
| 1 | `e826c474e80547d5a3b24e5eb0f30977` | 5350 | 357 | 59 / 201 / 32 / 65 |
| 2 | `7d879f9d5d5a4be09beb55cf88591b0d` | 5250 | 350 | 43 / 234 / 21 / 52 |
| 合计 | 3 个不同录制来源 | 14616 | 975 | 136 / 625 / 71 / 143 |

这些数量是**审计统计，不是获准训练的样本量或 train/val/test 数量**。没有为真实样本锁定训练切分。

三份 coarse 都是正式保存的 revision 1，每份 7 段，全部 reviewed=true，重复阶段合法，标签完整覆盖。粗标签文件哈希：

| Episode | coarse SHA256 | 正式训练处置 |
|---:|---|---|
| 0 | `554d6a272a5d61005784a650a95784bc87f5639ef91dcd3f5fbdd05abb3f97b4` | 拒绝：来源元数据哈希冲突 |
| 1 | `0b1f347687c92deab87027ed2ed08fde44a4414a75a58d971831cb17f7e066f4` | 拒绝：来源元数据哈希冲突 |
| 2 | `df1a34e626cbfee6183e7560a6b111ef2b465f9d9938830c8f783e07d2b96855` | 拒绝：来源元数据哈希冲突 |

具体冲突：已审核 JSON 的 source.sha256 声明仍锁定原 7 条包的 `meta/info.json` 和 `meta/episodes.jsonl`，而此导出包将它们改为三条子集。三份标注中的 Parquet/三路视频哈希 **12/12 匹配**，tasks 元数据也匹配；但两个已声明元数据哈希不符，按方案不能忽略。

导出包 provenance 记录了裁剪行为，但不能充当对已审核 source.sha256 的自动改写授权。程序未采用跳过哈希、伪造新 source、替换审核标记、恢复旧全集元数据等方式绕过。三个 fine 文件是未审核草稿（revision 1/2/1），未用于监督，未自动推导 coarse。

实际 `prepare` 已运行，返回失败，并为三条记录保存拒绝原因：[prepare_rejections.json](validation/prepare_rejections.json)、[原始 prepare 输出](validation/sample-prepare.txt)。因此：**正式训练接收 0 条，未生成真实 manifest/split/checkpoint**。诊断选择文件不是权威 annotation_index。

另对真实 Episode 0、锚点 450 构造了输入，读取实际时间戳作因果 floor：

```text
RGB: uint8 [3,16,256,256,3]
measured joint history: float32 [16,58]，值有限
input_indices: [405,407,411,413,417,420,422,426,428,432,435,437,441,443,447,450]
last_input_frame: 450 <= anchor 450
```

个别索引早于理想 `405,408,…,450`，来自 float32 时间戳舍入后的严格向过去取帧；未使用未来一帧“对齐”。理想精确 30 Hz 的规定序列另有单测验证。此真实 smoke **未运行 Encoder、未生成真实融合特征或预测**。详见 [raw_sample_smoke.json](validation/raw_sample_smoke.json)。

## 模型、训练及可追溯性

固定 Encoder：`facebook/vjepa2-vitl-fpc64-256` @ `b3c1679b7c34d3255ef3547f27c7b226aefab26f`。

- 官方 model.safetensors SHA256：`25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b`。
- 同 revision config.json SHA256：`3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9`。
- 16 个时间点、10 Hz 历史采样、约 1.5 秒窗口、2 Hz 判断；三个视角全图 256×256 RGB，ImageNet 归一化，冻结/评估模式 Encoder，token mean pooling。
- 视觉 3072 + 状态 928 = **4000** 维；`4000→256→4` head，**1,025,284** 个可训练参数。优化器仅接收 head；冻结与反向梯度已用测试验证。
- train-only mean/std、train-only 四类频率权重；AdamW、学习率、batch、early stop 等沿用方案首版值。最佳模型由 val 选择，test 不参与 epoch/参数选择。
- split 按 source_recording_id 整组，复制数据包不会打散；锁定 manifest/split；缺可靠来源、缺类、数据不足明确失败。
- grouped 5-fold 功能已实现并合成测试，外层 test 与内层早停 val 分离；3 个真实来源不能跑 5-fold；不同 seed 和视觉/状态消融没有真实执行。
- 特征 NPZ 无标签；独立标签指纹包含粗标签 SHA/revision；视频/Parquet/state schema/关节顺序/采样/模型变更使特征指纹失效。文件内容 SHA 复核。checkpoint 使用 Tensor/基本类型、`weights_only=True`，包含任务/模型/采样/白名单/归一化/manifest 锁定信息。
- eval 保存四类原始 softmax、argmax、目标真值、实际历史索引和最大输入帧；回放显示三相机、真值、概率、时间和 NEW/HELD，不通过状态机改变分类结果。

## 实际环境、测试与回归

初始系统 Python 没有 torch/pyarrow/transformers；已在工程内创建隔离 `.venv-tomato`，没有更改全局 Python 或其他项目环境。当前 GPU 为 **RTX 4070 Ti SUPER 16 GiB**，CUDA 可用、BF16 支持；CUDA **不是当前阻塞项**。

实际版本：torch 2.15.0+cu132、torchvision 0.29.2、transformers 5.22.1、numpy 1.26.4、pyarrow 19.0.1、PyAV 19.0.2、pytest 9.2.0、PyYAML 6.0.3、safetensors 0.8.0、huggingface_hub 2.4.0、OpenCV 4.6.0。可运行 prepare 的独立子进程单测禁止导入 torch/transformers 仍成功，证明轻量预检不依赖模型环境。

最终相关命令：

```bash
source .venv-tomato/bin/activate
python -m pytest -q \
  tests/test_tomato_dataset.py tests/test_tomato_phase_model.py \
  tests/test_train_tomato_vjepa_phase.py tests/test_tomato_metrics.py \
  tests/test_eval_tomato_vjepa_phase.py tests/test_lemon_vjepa_phase.py
# 54 passed

python -m pytest -q --continue-on-collection-errors tests
# 3 failed, 55 passed, 2 skipped, 1 error
```

原始日志：[阶段 1](validation/phase1-tests.txt)、[阶段 2](validation/phase2-tests.txt)、[阶段 3](validation/phase3-tests.txt)、[阶段 4](validation/phase4-tests.txt)、[最终全量回归](validation/final-regression.txt)。

完整测试的失败与 skip 均在**未修改 main 的 git archive 独立副本**复现：[main 原始测试结果](validation/baseline-tests.txt)，结果为 **3 failed / 8 passed / 2 skipped / 1 error**：

1. `test_lemon_doubao_vjepa.py` 收集失败：原工具导入仓库没有提供的 `tools.annotate_lemon_doubao`。
2. `test_lemon_phase_model.py::test_boundaries_and_causal_alignment`：原 MobileNet 路线函数只接受 4 个边界，测试又要求支持 5 个。
3. `test_lemon_phase_model.py::test_v3_manifest_group_split_and_exclusions`：原脚本没有 `V3_BOUNDARIES`。
4. `test_annotate_lemon_phase_video.py`：原测试寻找不存在的 `tests/annotate_lemon_phase_video.py`，实际脚本在 tools/。
5. 两条 rosbag 参数化测试因 rosbags 未安装 skip；不以 skip 当通过。

原 V-JEPA 2 主链专项测试始终 **7/7 通过**，完整旧测试中另外 1 条可通过，因此最终完整测试的 passed 数比相关命令多 1。未通过 monkeypatch 旧接口、补假模块、改测试路径或删测来掩盖基线问题。

合成测试覆盖：审核/版本、半开区间及重复阶段、source SHA、normal-only/Sharpa-only、同号 Episode、缺来源/缺相机/路径冲突、PTS 偏移、源时间戳间断、关节缺失/非有限值/schema、因果时间采样、无 action/task 输入、冻结/形状/梯度、checkpoint 往返、缓存失效/损坏及标签重建、train-only 统计、来源划分/CV、四类指标/边界/抖动、最小 head 训练和四阶段回放。

合成端到端测试实际训练 2 个 epoch，并编码/解码核验 120 帧三相机回放视频及逐帧 HELD 标记；人工查看了生成面板的 PNG。它仅验证软件路径，**未把合成 accuracy/F1、概率、视频或耗时登记为真实番茄实验结果**。

## 真实模型指标与未解决事项

| 请求指标 | 本次实际情况 |
|---|---|
| 真实 train/val/test 来源范围、样本数 | 未锁定；3 条来源校验失败，0 条获准进入训练 |
| 真实训练 epoch/最佳 head | 未执行 / 无真实 checkpoint |
| 番茄 overall accuracy / Macro-F1 / NLL | 未测量 |
| pickup/harvest/return/adjust Precision、Recall、F1 | 未测量 |
| 4×4 confusion、return↔adjust 错例 | 未测量 |
| 真实边界延迟 p50/p95、漏检、假切换/分钟 | 未测量 |
| 真实 Encoder+head GPU 平均/P95、峰值显存 | 未测量 |
| 真实 held-out 预测视频 | 未生成；仅完成真实输入结构 smoke 和合成回放验证 |

阻塞项：

1. **审核标注与导出元数据的身份不一致**。需要数据提供方提供可验证且相互匹配的正式 LeRobot/审核版本，或另行定义并审核导出子集的来源迁移契约；本实现保持严格失败，未擅自修改源文件和审核真值。
2. **本地缺少完整固定版 HF Encoder 目录**，`pretrained/vjepa2-vitl-fpc64-256/config.json` 首项即不存在，权重也未下载；实际检查见 [pretrained_check.json](validation/pretrained_check.json)。官方 SHA 已核验并固化，未运行替代模型。
3. **这三条仅是结构/功能测试样本**，不足以做方案建议的 5-fold 或稳定泛化结论。正式实验需完整合格来源、类覆盖与来源/场景审查。
4. **原仓库既有旧测试缺件/接口失配**保留未解决，以满足原柠檬代码不变；上述可复现失败不归入番茄精度问题。

研究目标 Macro-F1≥0.90 尚未验证，不能引用柠檬的指标替代。视觉-only、状态-only 和多 seed 对照暂未开展。

## 文件修改清单和后续运行

全部为新增文件；既有 main 跟踪文件无修改：

| 文件 | 用途 |
|---|---|
| `.gitignore` | 排除新虚拟环境、番茄 runs、模型权重/缓存及 Python 临时文件 |
| `requirements-tomato.txt` | 本分类器依赖和已测试的 NumPy/PyArrow 兼容约束 |
| `configs/tomato_four_stage.yaml` | 58 个真实关节名、四类顺序、固定采样/模型/首版参数 |
| `tools/build_tomato_annotation_index.py` | 只读候选审计和显式版本锁定 |
| `tools/tomato_dataset.py` | LeRobot/source/review/media/joint 校验与因果采样 |
| `tools/tomato_phase_model.py` | 冻结模型、共用预处理/融合、特征缓存、安全 checkpoint |
| `tools/train_tomato_vjepa_phase.py` | prepare/extract/train、来源分组、CV 和监督重建 |
| `tools/tomato_metrics.py` | 四类统计、逐来源/Episode、边界延迟和错误切换 |
| `tools/eval_tomato_vjepa_phase.py` | test 回放、预测文件、性能统计与 CV 汇总 |
| `tests/tomato_helpers.py` | 仅测试用的合成 LeRobot/标签 fixture |
| `tests/test_tomato_dataset.py` | 数据/审核/来源/对齐/因果性测试 |
| `tests/test_tomato_phase_model.py` | 模型冻结/融合/checkpoint/缓存测试 |
| `tests/test_train_tomato_vjepa_phase.py` | prepare、分组、CV、统计隔离、合成训练和标签更新 |
| `tests/test_tomato_metrics.py` | 四分类、NLL、延迟、漏检与抖动定义测试 |
| `tests/test_eval_tomato_vjepa_phase.py` | held-out 约束、预测、实际视频读写和失败保留报告 |
| `docs/TOMATO_FOUR_STAGE_TRAINING.md` | 逐步骤环境、数据/权重、运行/CV、产物说明 |
| `docs/TOMATO_IMPLEMENTATION_REPORT.md` | 本报告 |
| `docs/validation/` | 实际审计 JSON 和各阶段/基线测试原始日志；不含模拟精度报告 |

当前真实审计输出目录为 `/home/user/pi05/vjepa2_training/runs/tomato/environment/`，其中包含 `sample_config.yaml`、`sample_selection.json`、`data_audit.json`、`raw_sample_smoke.json`、`sample_prepare/prepare_rejections.json`、`sample_prepare/status.json` 和测试日志。审计证据另复制到 docs/validation 便于代码交付审阅；原始数据/权重/特征未加入 Git。

新机运行步骤及权重校验脚本见 [完整运行说明](TOMATO_FOUR_STAGE_TRAINING.md)。来源问题解决、权重准备和 annotation_index 锁定后使用：

```bash
cd /home/user/pi05/vjepa2_training
source .venv-tomato/bin/activate
python tools/train_tomato_vjepa_phase.py prepare --config configs/tomato_four_stage.yaml
python tools/train_tomato_vjepa_phase.py extract --config configs/tomato_four_stage.yaml --batch-windows 1
python tools/train_tomato_vjepa_phase.py train --config configs/tomato_four_stage.yaml
python tools/eval_tomato_vjepa_phase.py --config configs/tomato_four_stage.yaml
```

未来成功的 manifest/checkpoint 准确默认路径是 `runs/tomato/baseline/manifest.json`、`runs/tomato/baseline/best_head.pt`（**目前尚不存在**），数值指标 `runs/tomato/baseline/evaluation.json`，回放与 GPU 性能 `runs/tomato/baseline/offline_eval/`。本报告没有将这些规划路径描述为已完成产物。
