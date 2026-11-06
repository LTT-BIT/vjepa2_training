# Sharpa 番茄四阶段分类器

此分支只实现 `pickup / harvest / return / adjust` 分类、冻结 V-JEPA 2 特征提取、分类头训练和离线评估。没有 Agent、Prompt Manager、π0.5、RTC/T3 或控制接口。原柠檬脚本、标注、测试和 `experiment_results/` 保持基线字节不变。

本次实测状态、数据问题及测试结果见 [TOMATO_IMPLEMENTATION_REPORT.md](TOMATO_IMPLEMENTATION_REPORT.md)。**当前样本来源校验未通过，真实番茄训练未执行。** 单元测试生成的模型、概率和视频均为合成测试产物。

## 1. 环境

工程目录：`/home/user/pi05/vjepa2_training`，分支 `feat/tomato-four-stage`。路径示例均从该目录执行：

```bash
cd /home/user/pi05/vjepa2_training
source .venv-tomato/bin/activate
```

新机器可创建独立环境（不更改其他项目）：

```bash
uv venv --python python3.12 .venv-tomato
uv pip install --python .venv-tomato/bin/python -r requirements-tomato.txt
source .venv-tomato/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

需按机器驱动选择合适的 PyTorch/CUDA wheel；明确选择 `--device cpu` 时 Encoder 使用 FP32，CUDA 使用 BF16+SDPA，不会偷偷换模型/精度或下载备用权重。`prepare` 不导入 torch/transformers，不需要 CUDA 或模型权重，只需 NumPy、PyYAML、PyArrow、PyAV；RGB resize/回放另需 OpenCV。`ffmpeg` 不是程序运行的硬依赖，使用 PyAV 检查全部帧及 PTS。

实测依赖版本在交付报告中列出。`pyarrow==19.0.1` 和 NumPy 1.x 的组合经过实际验证；没有沿用安装时不兼容的 PyArrow 26/NumPy 1.x 组合。

## 2. 数据、审核和关节白名单

输入只读：LeRobot v2.1 `meta/info.json`、`meta/episodes.jsonl`、`meta/tasks.jsonl`、三相机 MP4、状态 Parquet 和显式选择的 coarse JSON。支持不同数据包有相同 Episode 编号。按元数据模板及 `chunks_size` 定位文件，不硬编码 chunk。只接受 `dual_fr3_sharpa` 和显式 `variant: normal`；发现 speedup 来源映射/时间轴会拒绝。

配置：[../configs/tomato_four_stage.yaml](../configs/tomato_four_stage.yaml)。路径全部相对于 **YAML 所在目录**解析，包含可选 `feature_cache_dir`、`split_file`。复制配置时建议把输入/输出路径改成绝对路径。

本批 schema：116 个状态字段，含 58 个位置和 58 个速度。明确白名单选择：

| 观测 | 原始 state 列（零基） | 位置维度 |
|---|---|---:|
| 左臂 `left_fr3_joint1.position` 至 `left_fr3_joint7.position` | 0–6 | 7 |
| 右臂 `right_fr3_joint1.position` 至 `right_fr3_joint7.position` | 14–20 | 7 |
| 左手位置 | 28–49 | 22 |
| 右手位置 | 72–93 | 22 |

双手每侧实际字段（前缀 `left_` 或 `right_`，后缀 `.position`）：

```text
thumb_CMC_FE, thumb_CMC_AA, thumb_MCP_FE, thumb_MCP_AA, thumb_IP,
index_MCP_FE, index_MCP_AA, index_PIP, index_DIP,
middle_MCP_FE, middle_MCP_AA, middle_PIP, middle_DIP,
ring_MCP_FE, ring_MCP_AA, ring_PIP, ring_DIP,
pinky_CMC, pinky_MCP_FE, pinky_MCP_AA, pinky_PIP, pinky_DIP
```

配置逐个列出全部 58 个名称，按名称解析实际列。检查重复、缺失、非 position、整套 schema 顺序变化、非有限值和时间轴不连续。不会使用整个 state、速度或 action。缺少手部实测数据时必须显式建立 `joint_mode: arms-only` 且仅列 14 个臂关节的新实验配置，不自动降级。

### 只读审计本批三个样本

```bash
python tools/tomato_dataset.py \
  --dataset /home/user/pi05/sharpa_1103_normal_000000-000002_20261106-160943/1103_620_642_normal_lerobot \
  --config configs/tomato_four_stage.yaml \
  --output runs/tomato/environment/data_audit.json
```

此命令完整检查视频/状态，同时报告不合格标注来源和候选类数量；**诊断通过不等于获准训练**。本批审计报告中三个 coarse 的 `eligible` 都是 false，原因是两个导出元数据文件的哈希与已审核 JSON 不符。12 个 MP4/Parquet 输入哈希匹配，不能因此忽略其余声明。不能重写粗标签里的 SHA、替换审核状态或用旧元数据覆盖本子集。

### 显式选择版本

先列候选，不自动选择：

```bash
python tools/build_tomato_annotation_index.py \
  --dataset /absolute/path/to/normal_lerobot \
  --report runs/tomato/candidates.json
```

候选包含路径、SHA、revision、saved_at、审核段数、同 Episode 候选个数和校验错误。人工明确选择后写只读索引，例如 `runs/tomato/selection.json`（这不是新标注真值）：

```json
[
  {
    "dataset_root": "/absolute/path/to/normal_lerobot",
    "episode_index": 0,
    "coarse_annotation": "/absolute/path/to/normal_lerobot/annotations/subtasks/coarse/episode_000000.json",
    "variant": "normal"
  }
]
```

以此锁定版本：

```bash
python tools/build_tomato_annotation_index.py \
  --selection runs/tomato/selection.json \
  --output runs/tomato/annotation_index.json
```

只接受 `schema_version=2`、`stage=coarse`、`revision>=1`、有 `saved_at`、`inherited=false` 的正式保存/发布记录；每段 reviewed 必须严格为 true。自动推导、草稿、显式未发布、unknown、排除/失败记录、缺来源 ID、重复选择、源 SHA 冲突均拒绝。正式保存且全部审核的 workspace coarse 不要求虚构一个上游没有的 `published=true` 字段。原上游定义参见固定版 [workspace.py](https://github.com/ma39nong/convert_data_job-worker/blob/90ad2663ec645c4b2d6322e9df25284dfa7e0b0a/src/convest/subtasks/workspace.py)。

多个候选时须在 selection 中明确路径，锁定器记录该版本 SHA/revision，不覆盖已有索引。行为审核的 note/outcome/review_events/training_review 原样记录进 manifest；粗阶段不代表摘取成功。细阶段草稿完全不作为训练监督。

## 3. 固定模型及因果采样

- `labels = [pickup, harvest, return, adjust]`，ID 0–3，同一类可以重复出现。区间 `[start_frame,end_frame)` 连续覆盖全长，标签只取当前预测帧。
- 固定 2 Hz 时间网格，不读取标签边界决定取样；每个锚点向过去以 0.1 秒间隔取 16 个时间点，约 1.5 秒历史。只用查询时刻已观测的帧，初始不足时重复第 0 帧/状态。
- 理想精确 30 Hz 时 `t=450` 得到 `405,408,…,450`。真实 Parquet float32 时间戳存在微小舍入，floor 到真实时间轴时个别索引可以更早一帧；不会为凑齐理想帧号使用晚于查询时间的帧。逐样本保存实际 `input_indices`。
- RGB 为 `[B,3,16,256,256,3]` uint8；全图 RGB resize，除 255，ImageNet mean/std，转成 `[B*3,16,3,256,256]`，共享冻结 Encoder。`last_hidden_state.mean(dim=1)` 每视角 1024 维。
- 视觉 3072 + 关节 16×58 = **4000** 维。训练集 mean/std（std 下限 .01）→ `Linear(4000,256) → GELU → Dropout(.3) → Linear(256,4)`。仅 head **1,025,284** 个参数可训练。
- 沿用加权交叉熵、sqrt 逆频率权重且均值归一为 1；AdamW 3e-4/0.01、batch 128、最多 100 epoch、patience 15、clip grad norm 1、seed 42；按 val Macro-F1、再 val NLL 选择。训练集缺任一类直接失败。

Encoder 固定 `facebook/vjepa2-vitl-fpc64-256`，revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`。

| 固定资源 | SHA256 |
|---|---|
| model.safetensors（1,303,947,864 bytes） | `25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b` |
| config.json | `3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9` |

哈希来源：[固定 revision 的官方文件元数据](https://huggingface.co/api/models/facebook/vjepa2-vitl-fpc64-256/revision/b3c1679b7c34d3255ef3547f27c7b226aefab26f?blobs=true)。config 的 SHA 按同一 revision 的原始字节计算。程序不把原柠檬 best_head 当作番茄模型，也不加载其他 revision 备用。

## 4. 准备模型权重（本次未下载）

后续在需要真实提特征时准备完整 HF 目录：

```bash
hf download facebook/vjepa2-vitl-fpc64-256 config.json model.safetensors \
  --revision b3c1679b7c34d3255ef3547f27c7b226aefab26f \
  --local-dir pretrained/vjepa2-vitl-fpc64-256
python - <<'PY'
from pathlib import Path
from tools import tomato_dataset as ds, tomato_phase_model as model
p = Path('pretrained/vjepa2-vitl-fpc64-256')
assert ds.digest(p/'model.safetensors') == model.WEIGHTS_SHA256
assert ds.digest(p/'config.json') == model.CONFIG_SHA256
ds.write_json(p/'weights_metadata.json', dict(model_id=ds.MODEL_ID, revision=ds.REVISION,
    sha256=model.WEIGHTS_SHA256, size=(p/'model.safetensors').stat().st_size))
print(model.verify_pretrained(p))
PY
```

`pretrained/`、运行产物和特征缓存在 `.gitignore` 中，未提交。缺 config、权重、metadata，字节哈希错误或 revision 不符都会报出具体错误。

## 5. 分阶段运行

先解决来源冲突并锁定合格索引，然后运行：

```bash
python tools/train_tomato_vjepa_phase.py prepare --config configs/tomato_four_stage.yaml
python tools/train_tomato_vjepa_phase.py extract --config configs/tomato_four_stage.yaml --batch-windows 1
python tools/train_tomato_vjepa_phase.py train --config configs/tomato_four_stage.yaml
python tools/eval_tomato_vjepa_phase.py --config configs/tomato_four_stage.yaml
```

配置默认隔离输出是 `/home/user/pi05/vjepa2_training/runs/tomato/baseline`，默认共享特征缓存是 `runs/tomato/feature_cache`。显存不足可减小提取 batch（默认已是每批一个锚点、三个视角）；明确的 CPU 路径是给 extract/train/eval 传 `--device cpu`，不会自动选择其他模型。

`prepare` 严格核验所有显式输入；任一 Episode 不合格均使该次选择失败，写 `prepare_rejections.json`，不能悄悄跳过。最少需要 3 个不同原始录制来源，来源 ID 相同的副本统一 split；小于 10 个来源会注明仅适合 smoke。train/val/test 各自必须有全部四类锚点。默认 grouped 80/10/10、seed 42；可选 `split_file` 是包含 train/val/test 来源 ID 列表的 JSON，需完整且不重叠。

`manifest.json`、`split.json` 一旦存在，输入或分组变化不能覆盖它们，须新 output_dir。修改粗标注时应新建 index/run，特征缓存可共享；源视频、Parquet、state schema、采样、相机顺序、关节顺序或模型发生变化会改变特征指纹。NPZ **只含特征、锚点、时间、历史索引和指纹，没有 y**；每次从当前锁定粗标注生成独立标签缓存。读取 NPZ 同时检查文件内容 SHA、维度、时间轴及有限值。

提取可按 Episode 安全重跑，已验证缓存直接复用，损坏缓存明确失败。分类头训练采用新 run，不自动覆盖已有 best_head/evaluation 或重新调试已看过的 test。训练中断如已产生 checkpoint，保留旧目录并用新 run 重跑，可复用特征。

默认 eval 选择 manifest 内按稳定 Episode ID 排序的第一条 test 回放，也可用 `--episode-id` 明确选择另一条已锁定 test。拒绝 train/val 来源泄漏。评估所有 test 锚点，回放展示三相机、四类概率、预测、当前真值、帧号和 `NEW/HELD`。不做状态机纠错。视频失败或测速缺权重时保留数值评估，`errors.json` / `status.json` 明示错误，CLI 返回 2；`--no-video`、`--skip-benchmark` 是显式跳过选项。

## 6. 可选 5-fold grouped CV

本批只有 3 个来源，**不能运行 5-fold**。将来有足够且每个 split 四类覆盖的数据时可逐折运行：

```bash
for fold in 0 1 2 3 4; do
  python tools/train_tomato_vjepa_phase.py prepare --config configs/tomato_four_stage.yaml --cv-folds 5 --cv-fold-index "$fold"
  python tools/train_tomato_vjepa_phase.py extract --config configs/tomato_four_stage.yaml --cv-folds 5 --cv-fold-index "$fold"
  python tools/train_tomato_vjepa_phase.py train --config configs/tomato_four_stage.yaml --cv-folds 5 --cv-fold-index "$fold"
done
python tools/eval_tomato_vjepa_phase.py --summarize-cv runs/tomato/baseline/cv5
```

各折锁定于 `runs/tomato/baseline/cv5/fold_N/`；外层 test 来源只在其中一折出现，早停的 val 从其余来源中另划，不使用外层 test。每折保留原始指标及 manifest，`cv_summary.json` 汇总 accuracy、Macro-F1、NLL 的 Mean/Std，并检查全部折的输入一致、来源无重复、外层覆盖完整。不能评价的折明确失败。不同 seed 需复制配置并使用新输出目录。视觉-only / 状态-only 消融尚未实施，作为方案允许的后续实验。

## 7. 产物与指标

成功真实训练后，run 下有：

```text
manifest.json / split.json / training_config.json / status.json
feature_index.json / labels/<annotation-fingerprint>.json
metrics.jsonl / best_head.pt / evaluation.json
predictions.jsonl / predictions.csv
offline_eval/evaluation.json / predictions.jsonl / predictions.csv
offline_eval/<held-out-episode-id>.mp4 / *.frames.jsonl / *.png
offline_eval/replay.json / latency.json / errors.json
```

checkpoint 仅含 Tensor/基本类型，以 `weights_only=True` 加载；锁定类别顺序、关节名/维度、相机、采样、模型 ID/revision/SHA、mean/std、head、最佳 epoch、val 指标和 manifest SHA。原六分类 checkpoint 明确拒绝。

报告 train/val/test 的 accuracy、Macro-F1、逐类 P/R/F1/support、4×4 confusion、稳定 log-softmax NLL、逐 Episode/来源统计、`return↔adjust` 混淆；边界从真实切换开始到当前段结束内第一条正确锚点的延迟/漏检，p50/p95、2 Hz 分辨率；同一真实阶段内预测变化计为 false switch，保存每分钟次数和错判区间。未观测的短段在 manifest 记录，不伪造锚点；真实困难边界不从测试删掉。

测速在加载真实权重后，对固定测试锚点预热并同步测 Encoder+head，记录均值/P95、三路 RGB/resize 时间、峰值显存、GPU/PyTorch/CUDA 信息。它是离线计算成本，不能当真机控制延迟。**本次没有真实训练 checkpoint、真实预测视频、真实精度或 GPU Encoder 延迟报告。**

## 8. 测试和基线回归

```bash
python -m pytest -q \
  tests/test_tomato_dataset.py tests/test_tomato_phase_model.py \
  tests/test_train_tomato_vjepa_phase.py tests/test_tomato_metrics.py \
  tests/test_eval_tomato_vjepa_phase.py tests/test_lemon_vjepa_phase.py
python -m pytest -q --continue-on-collection-errors tests
git diff main -- tools/train_lemon_vjepa_phase.py tools/lemon_phase_model.py \
  tools/run_lemon_doubao_vjepa.py tools/serve_lemon_phase_agent.py \
  experiment_results annotations tests/test_lemon_vjepa_phase.py \
  tests/test_lemon_doubao_vjepa.py tests/test_lemon_phase_model.py tests/test_annotate_lemon_phase_video.py
```

最后一项应为空。完整旧测试的既有 3 fail / 1 collection error / 2 skip 不被隐藏，详见报告中 main 副本对照及原始测试日志。番茄测试用合成 Encoder/特征验证程序行为，不是以替代模型估计真实精度。
