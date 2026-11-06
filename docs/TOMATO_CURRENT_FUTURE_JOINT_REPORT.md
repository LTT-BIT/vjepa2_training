# Tomato Current / Future / Joint 实施与离线报告

2026-11-06，Asia/Shanghai；基线 `feat/tomato-four-stage` @ `11d2b57358b382aa638264303ebfac582d743029`。本轮代码、提交、模型和报告均仅在本地，没有 push 或部署。

**代码和 P1-1～P1-4 已实施，并实际完成三模式真实 smoke 训练/评估。** 本机固定版 Encoder 已就绪，CUDA/BF16 可用；当前只有用户提供的 3 条 normal 录制，不能把一次 1 train / 1 val / 1 test 的结果当作完整数据集或稳定泛化结论。三模式均使用原定参数，未按 test 指标调整 λ、horizon、split 或最佳 epoch。

## 1. 范围与未变项

- 固定 Sharpa 番茄四类：`pickup, harvest, return, adjust`；三相机历史 + 实测 58 维位置；冻结 V-JEPA 2，MLP 四分类。没有 π0.5、Agent、HFSM、Prompt Manager、RTC/T3 或 ROS/真机代码。
- **`allow_subset_metadata_mismatch` 默认仍为 true，原有两个 metadata 豁免路径、来源 SHA 策略不变。** `load_reviewed_coarse`、`source_hash_report`、`assert_source_unchanged` 与基线 AST 相同；真实 manifest 保留允许差异和 TEST ONLY warning，实际输入 SHA 仍锁定。
- 原 Lemon main 的 21 个文件逐字节未变；现有 tomato 的 `head_model`、预处理、encode_clips、fuse_features 语义和源实现不变。输入包 SHA256SUMS **27/27** 通过，未更改原数据或人工审核标签。
- 原 v1 tomato checkpoint 只解释为 current_only，安全读取、相同特征下预测逐位一致；合法 legacy manifest 可用 `--full-revalidate` 建立旁路验收收据，原 manifest/checkpoint 不重写。旧 Lemon 六分类继续拒绝。

证据：[完整性、策略及真实复现](validation/current_future/integrity-and-real-evidence.json)、[CUDA checkpoint 回读](validation/current_future/checkpoint-cuda-readback.json)。

## 2. 分阶段实施与测试

| 阶段 | 本地提交 / 内容 | 相关测试及原 V-JEPA 核心回归 |
|---|---|---:|
| 1 | `64dd807`：P1-1，整数因果采样、future targets/mask | 64 passed |
| 2 | `3bbb2f7`：P1-2，三模式、独立 λ、v1/v2、安全 CUDA 初始化 | 72 passed |
| 3 | `baa535f`：P1-3，三模式训练、锁定验收收据、共享缓存 | 80 passed |
| 4 | P1-4，稳定/提前/错误持续时间、三模式视频、最终文档 | 89 passed |

完整回归：**90 passed、3 failed、2 skipped、1 collection error**。基线 11d2b57 为 58 passed、相同 3 failed、2 skipped、1 error；原 main 的同名失败也在先前独立 archive 中复现。没有修改老测试或补假模块掩盖失败：缺 `tools.annotate_lemon_doubao`、旧 MobileNet 的五边界/V3 接口缺失、旧视频测试的脚本路径错误；两项 rosbags 测试因依赖缺失跳过。原 V-JEPA 核心仍 7/7 通过。

原始日志位于 [validation/current_future/](validation/current_future/)：phase1/2/3/4-tests、baseline-tests 和 full-regression。合成测试单独使用临时 Encoder/LeRobot/特征，未把其 accuracy、概率、视频或耗时算入本节后的真实实验。

P1 验收重点：

1. `uniform-integer-causal-v2`：30FPS float32 round-trip 的 anchor450 历史严格为 `405,408,…,450`；不足首帧填充；非整除/非均匀轴严格 prefix floor fallback。改变未来时间戳不会改变该锚点的历史选择。新指纹隔离旧缓存，legacy 只读路径保留旧语义。
2. cuBLAS workspace、Python/NumPy/Torch seed 和确定性策略先于任何 CUDA；不兼容或外部提前初始化明确拒绝。CPU、实际 CUDA、分开 extract/train、同进程真实 Encoder→train 均已验证。
3. 首次 prepare 完整解码/PTS/Parquet/身份/审核及覆盖检查，锁定 manifest/split/validation_receipt/validation_lock。日常加载不调用全量媒体验证，仍核验全部相关 SHA 和配置；同大小文件改动也拒绝。full-revalidate 不改写锁定文件。
4. 原始 F1/概率不平滑。Current first-correct 与第 K 次稳定确认分开；K=2/3、短阶段 miss/null、提前误报、错误时长有解析单测。Future 预警提前量、事件漏检、超窗误报、持久性基线、多边界歧义及 masked tail 分开测试；视频和完整时间轴明确时间语义。

## 3. 数据与共同实验契约

实际数据：

```text
/home/user/pi05/sharpa_1103_normal_000000-000002_20261106-160943/1103_620_642_normal_lerobot
```

三份 coarse 均为人工审核 revision 1，每份 7 段；不使用 fine 草稿或 action/task_index。116 维 observation.state 中只取双臂 14 + 双手 44 个 `.position`；D=58。白名单见 [配置](../configs/tomato_four_stage.yaml)。

| split | Episode | source_recording_id | current 锚点/四类数 | valid future 锚点/四类数 | masked |
|---|---:|---|---|---|---:|
| train | 1 | `e826c474e80547d5a3b24e5eb0f30977` | 357 / 59,201,32,65 | 355 / 57,201,32,65 | 2 |
| val | 0 | `76354615cf9746758c5c7e6e5ab579d4` | 268 / 34,190,18,26 | 266 / 32,190,18,26 | 2 |
| test | 2 | `7d879f9d5d5a4be09beb55cf88591b0d` | 350 / 43,234,21,52 | 348 / 41,234,21,52 | 2 |

三个模式实际 `split.json` 内容及 SHA 一致：`ab960cd70310488f71ab8f8ff16f95838b638c705e42afb88512d68de07acf73`；来源集合无交集。B/C 显式使用 A 的 split-file。三份 feature_index 指向**完全相同的三个缓存路径**；B/C 验证后复用，没有重编码或复用旧标签。

- Encoder：`facebook/vjepa2-vitl-fpc64-256` @ `b3c1679b7c34d3255ef3547f27c7b226aefab26f`，实际加载并冻结。
- 权重 SHA：`25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b`；config SHA：`3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9`，本地字节校验通过。
- 三路 16 帧、历史采样 10Hz/约 1.5 秒、判断 2Hz，RGB 256×256，ImageNet/token mean；BF16+SDPA。真实融合特征形状分别为 `[268,4000] / [357,4000] / [350,4000]`，975 个窗口全因果；视觉 3072 + 位置 928。
- Current-only/Future-only 参数 1,025,284，Joint 1,026,312；只训练 head/trunk，Encoder 不训练。Joint 保留同构共享 4000→256 GELU/Dropout(.3)，两个独立 256→4 输出。
- Δ=1.0 秒严格 +30 帧；λc=1.0、λf=0.5，活跃头归一化。每头 train-only sqrt 逆频率类权重独立；mean/std 对三模式使用相同 train 观测，逐位一致。Future-only 不训练尾部无效锚点，Joint 保留全部 Current 监督且每个 batch 有真实有效 Future 监督。
- AdamW lr3e-4 / wd.01、batch128、最多100epoch、patience15、clip1、seed42。验证集按活跃 λ 加权 Macro-F1、NLL 次选；test 不参与选模。日志可逐 epoch 复算双 λ loss 和选择分数。

额外真实复现：`runs/tomato/current_future_validation/actual_single_process_run/` 在新缓存中实际加载 Encoder、提取全部真实特征、再同进程训练 Current-only。与分开 extract/train 的 A：head state、mean/std **逐位相同**，各 split 全部指标相同，最佳 epoch34。这同时检验了 P1-2 的初始化顺序和重置 seed，未用模拟 Encoder 替代。

## 4. 真实 held-out smoke 指标

以下均来自 test Episode2 的真实冻结 V-JEPA 特征、训练后 head 和未参与选模的真值。**只有一条测试来源**；不作为论文级泛化、完整数据或部署成功证明。

| 实验 | Epochs / best | 头 | 有效/总数 | Accuracy | Macro-F1 | NLL |
|---|---|---|---:|---:|---:|---:|
| A | 49 / 34 | current | 350/350 | 0.977143 | 0.947797 | 0.073483 |
| B | 17 / 2 | future | 348/350 | 0.936782 | 0.886184 | 0.198953 |
| C | 16 / 1 | current | 350/350 | 0.951429 | 0.859212 | 0.225912 |
| C | 16 / 1 | future | 348/350 | 0.905172 | 0.779584 | 0.373416 |

| 实验/头 | 类别 | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| A/current | pickup | 0.976190 | 0.953488 | 0.964706 | 43 |
| A/current | harvest | 0.995690 | 0.987179 | 0.991416 | 234 |
| A/current | return | 0.826087 | 0.904762 | 0.863636 | 21 |
| A/current | adjust | 0.962264 | 0.980769 | 0.971429 | 52 |
| B/future | pickup | 0.857143 | 0.878049 | 0.867470 | 41 |
| B/future | harvest | 0.991031 | 0.944444 | 0.967177 | 234 |
| B/future | return | 0.678571 | 0.904762 | 0.775510 | 21 |
| B/future | adjust | 0.909091 | 0.961538 | 0.934579 | 52 |
| C/current | pickup | 0.816327 | 0.930233 | 0.869565 | 43 |
| C/current | harvest | 1.000000 | 0.987179 | 0.993548 | 234 |
| C/current | return | 1.000000 | 0.476190 | 0.645161 | 21 |
| C/current | adjust | 0.866667 | 1.000000 | 0.928571 | 52 |
| C/future | pickup | 0.794118 | 0.658537 | 0.720000 | 41 |
| C/future | harvest | 0.974249 | 0.970085 | 0.972163 | 234 |
| C/future | return | 0.900000 | 0.428571 | 0.580645 | 21 |
| C/future | adjust | 0.732394 | 1.000000 | 0.845528 | 52 |

**A/current**（行是真值、列为预测，pickup/harvest/return/adjust）：

```text
[41, 0, 0, 2]
[0, 231, 3, 0]
[1, 1, 19, 0]
[0, 0, 1, 51]
```

**B/future**（行是真值、列为预测，pickup/harvest/return/adjust）：

```text
[36, 0, 0, 5]
[6, 221, 7, 0]
[0, 2, 19, 0]
[0, 0, 2, 50]
```

**C/current**（行是真值、列为预测，pickup/harvest/return/adjust）：

```text
[40, 0, 0, 3]
[3, 231, 0, 0]
[6, 0, 10, 5]
[0, 0, 0, 52]
```

**C/future**（行是真值、列为预测，pickup/harvest/return/adjust）：

```text
[27, 0, 0, 14]
[6, 227, 1, 0]
[1, 6, 9, 5]
[0, 0, 0, 52]
```

- A/Current：first-correct P50/P95 0.4667/1.0500s；stable K=2 P50/P95 0.9667/1.5500s，stable miss=0.0000；false switches/min=2.7429；early false bursts=2、1.5000s；wrong-stage duration=4.0000s。
- B/Future：事件 1/6，miss=5，ambiguous=0；已检出事件 lead P50/P95=0.6000/0.6000s；超窗误报 8 次、6.0000s；forecast error duration=11.0000s；对 oracle persistence F1 增益=-0.044570。
- C/Current：first-correct P50/P95 0.5000/1.5917s；stable K=2 P50/P95 1.0000/2.0917s，stable miss=0.0000；false switches/min=3.7714；early false bursts=1、1.9667s；wrong-stage duration=8.5000s。
- C/Future：事件 1/6，miss=5，ambiguous=0；已检出事件 lead P50/P95=0.9667/0.9667s；超窗误报 11 次、11.5000s；forecast error duration=16.5000s；对 oracle persistence F1 增益=-0.151170。
  Joint model-current persistence F1=0.828320，Joint Future 对其增益=-0.048736。

**真实 GPU 性能**：RTX 4070 Ti SUPER，PyTorch 2.15.0+cu132 / CUDA13.2，BF16+SDPA；每模式预热2、测量20个真实窗口，同步计 Encoder+head。下表显存是本进程 PyTorch allocator peak，含模型与中间张量，不是整卡占用。RGB/resize 另列，解码/I/O/ROS及其他并发控制不在时间范围内，不能作为真机部署延迟。

| 实验 | Mean ms | P50 ms | P95 ms | P99 ms | Peak allocated GiB | 三路 RGB/resize Mean ms/源帧 |
|---|---:|---:|---:|---:|---:|---:|
| A | 87.217 | 87.107 | 87.596 | 88.486 | 0.7288 | 2.055 |
| B | 87.080 | 87.057 | 87.265 | 87.487 | 0.7281 | 1.982 |
| C | 87.252 | 87.243 | 87.500 | 87.883 | 0.7281 | 1.992 |

Current-only 的 Macro-F1=0.9478；Future-only=0.8862；Joint 的 Current/Future=0.8592/0.7796。Joint Current 比 A 下降约 0.0886，Joint Future 比 B 下降约 0.1066。本次没有调参消除退化。

Future oracle-current persistence Macro-F1=0.930754（真实当前标签保持，诊断用、不可部署）。B/C 相对它均为负增益；两者各只提前预报 6 次转场中的 1 次。Joint 另保存 model-current persistence。不能据整体 Future F1 宣称具有可靠提前切换能力。

稳定延迟、提前量与误报的定义：

- Current first-correct 在真实边界后该段内首次预测正确；stable K=2 在第 2 个连续正确锚点确认，不回填首次时间。短段不足 K 或从未稳定计 missed，延迟为 null。
- Future 事件窗口 `[boundary-1s,boundary)`；多个真实边界相距小于 horizon 时显式 ambiguous，不计成功/不混入 recall 分母。本测试 6 个事件无歧义。
- Current early false switch 是前一个真实段中对后继阶段的错误预测 burst；Future 超窗误报是未被 horizon 内同类真实转场支持的非当前阶段预测 burst。分类错误区间与预警误报区间分别保留，不能互相替代。
- 错误持续时间是 2Hz anchor-label 矩形估计；Future 的时间轴为预测发出锚点，同时提供对应未来目标帧，避免当成当前识别延迟。
- 长/短阶段以预设 1.5 秒分层；此真实测试粗阶段均较长，短层 samples=0、metrics=null，未伪造短阶段成绩。

每个模式的原始独立报告、逐 Episode/来源、混淆矩阵、P/R/F1/support、NLL、边界/错判区间及持久性对照均见 [validation/current_future/](validation/current_future/) 的 A/B/C.offline-evaluation.json。真实预测/CSV、完整时间轴、事件和视频位于各自 `offline_eval/`，没有状态机后处理。

## 5. 模式目录、Checkpoint 与实际产物

| 模式 | 实际 run 目录（相对工程根） |
|---|---|
| A | `runs/tomato/experiments/current_only_seed42/` |
| B | `runs/tomato/experiments/future_only_h1s_seed42/` |
| C | `runs/tomato/experiments/joint_h1s_lc1_lf0p5_seed42/` |

每个目录都有：annotation_index、manifest、split、validation_receipt、validation_lock、feature_index、独立 labels cache、training_config、metrics.jsonl、best_head.pt、evaluation、predictions.jsonl/csv、status。

各自 `offline_eval/` 都有 evaluation、predictions.jsonl/csv、latency、errors（空）、replay.json，以及测试视频 `9efc7aeb8216f65ff27d467e.mp4`、完整 `*.frames.jsonl`、`*.events.json` 和 PNG 预览。视频均实际解码核验 **5250 帧**。人工查看了 Joint 中段和 Future-only 尾部面板，确认双头、四概率、NEW/HELD、锚点+Δ目标及 TARGET_UNAVAILABLE 正确显示。

v2 checkpoint 固定类别、mode/horizon/独立 λ/K、相机/关节/采样、Encoder/manifest/split SHA、归一化、最佳 epoch 和两头验证指标。CUDA 回读三种模型，活跃输出均为有限 `[N,4]`，非活跃头为 None。

## 6. 运行命令

以下是本次实际路径；已完成 run 不允许再次训练覆盖。重跑请改新 output-dir，仍可复用共享缓存和固定 split。

```bash
cd /home/user/pi05/vjepa2_training
source .venv-tomato/bin/activate
TOMATO_DATA=/home/user/pi05/sharpa_1103_normal_000000-000002_20261106-160943/1103_620_642_normal_lerobot

# A；也已分别实际执行 prepare、extract、train 并与同进程 run 复现比较
python tools/train_tomato_vjepa_phase.py run --config configs/tomato_four_stage.yaml \
  --dataset "$TOMATO_DATA" --prediction-mode current_only \
  --output-dir runs/tomato/experiments/current_only_seed42 \
  --feature-cache-dir runs/tomato/experiments/shared_feature_cache

# B
python tools/train_tomato_vjepa_phase.py run --config configs/tomato_four_stage.yaml \
  --dataset "$TOMATO_DATA" --prediction-mode future_only --future-horizon-s 1.0 \
  --output-dir runs/tomato/experiments/future_only_h1s_seed42 \
  --split-file runs/tomato/experiments/current_only_seed42/split.json \
  --feature-cache-dir runs/tomato/experiments/shared_feature_cache

# C
python tools/train_tomato_vjepa_phase.py run --config configs/tomato_four_stage.yaml \
  --dataset "$TOMATO_DATA" --prediction-mode joint --future-horizon-s 1.0 \
  --lambda-current 1.0 --lambda-future 0.5 \
  --output-dir runs/tomato/experiments/joint_h1s_lc1_lf0p5_seed42 \
  --split-file runs/tomato/experiments/current_only_seed42/split.json \
  --feature-cache-dir runs/tomato/experiments/shared_feature_cache

python tools/eval_tomato_vjepa_phase.py --config configs/tomato_four_stage.yaml \
  --prediction-mode current_only --output-dir runs/tomato/experiments/current_only_seed42
python tools/eval_tomato_vjepa_phase.py --config configs/tomato_four_stage.yaml \
  --prediction-mode future_only --future-horizon-s 1.0 --output-dir runs/tomato/experiments/future_only_h1s_seed42
python tools/eval_tomato_vjepa_phase.py --config configs/tomato_four_stage.yaml \
  --prediction-mode joint --future-horizon-s 1.0 --lambda-current 1.0 --lambda-future 0.5 \
  --output-dir runs/tomato/experiments/joint_h1s_lc1_lf0p5_seed42

# 显式重新进行全量媒体验收；三个模式均可添加此参数
python tools/train_tomato_vjepa_phase.py prepare --config configs/tomato_four_stage.yaml \
  --prediction-mode current_only --output-dir runs/tomato/experiments/current_only_seed42 --full-revalidate

python -m pytest -q tests/test_tomato*.py tests/test_train_tomato*.py \
  tests/test_eval_tomato*.py tests/test_lemon_vjepa_phase.py
python -m pytest -q --continue-on-collection-errors tests
```

`--stable-transition-k 3`、horizon、λ、seed 的不同设置均须新的隔离 run，不能覆盖既有 manifest/checkpoint 或用 test 选参。分开命令使用相同 mode/horizon/λ/output-dir；首次 prepare 可加 dataset，后续从 run 自动恢复已锁定 index。未来五折数据准备好后，原 grouped CV 入口保留，三模式按头独立汇总，不混合为一个准确率。

## 7. 文件级修改

- `configs/tomato_four_stage.yaml`：增加五项模式/未来监督参数，原元数据设置、白名单和 Encoder 不变。
- `tools/tomato_dataset.py`：P1-1 采样版本、future target/mask、配置及 JSON null 契约；来源校验策略函数未修改。
- `tools/tomato_phase_model.py`：独立双头、模式输出、活跃 λ loss、v2 checkpoint 与 legacy v1、显式 CUDA 初始化；原 Current head/视觉/融合函数不变。
- **新增** `tools/tomato_runtime.py`：在 CUDA 前设置 cuBLAS/seed/确定性并记录环境，拒绝过晚或不兼容上下文。
- `tools/train_tomato_vjepa_phase.py`：P1-3 验收收据、快验、共享缓存；三模式类权重/损失/验证选模/独立报告/CLI；不覆盖已训练 run。
- `tools/tomato_metrics.py`：Current/Future 命名空间、mask、persistence、稳定/提前/误报/错误时长、长短阶段分层。
- `tools/eval_tomato_vjepa_phase.py`：三模式时间明确的回放、完整时间轴/事件、模式化 CUDA benchmark、CV 按头汇总。
- `tests/tomato_helpers.py` 及 tomato dataset/model/train/eval tests：保留原当前路径回归并增加所需 fixture/断言；新增 joint_model、runtime、joint_training、joint_metrics 四个测试文件。**未修改 Lemon 测试。**
- 两份既有 TOMATO 文档保留首轮审计，新增本轮入口；本报告及 `docs/validation/current_future/` 保存本轮实际证据。模型大权重、head、视频和特征仍在被忽略的 runs/pretrained 中，没有加入 Git。

## 8. 未解决问题和结论边界

代码、三模式合成验收、真实 GPU 前向、真实三条 smoke 训练/评估/回放和同进程复现均完成。当前不足：完整正式 normal 数据尚未提供；仅三来源且测试一条，未做五折/多 seed/视觉与状态消融；本次 Future 未优于 oracle persistence，Joint Current 相对 A 退化。已有 metadata 测试豁免及原 Lemon 非核心测试缺件/失配保持原状并记录，没有擅自处理。

这些结果不能证明未来头可用于稳定提前切换，更不代表任何 π0.5/HFSM/RTC/T3 部署已经可用。本轮不实现这些部署代码，也不把当前粗阶段识别等同于下一操作决策。
