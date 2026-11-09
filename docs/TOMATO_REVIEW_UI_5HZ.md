# UI：2 Hz / 5 Hz 按需预测

2026-11-09，修改保留在当前工作树，未提交、未切换分支。

## 使用

在“识别模式”旁的“预测频率”选择 **5 Hz**，然后点击“识别此样本”。默认仍为2 Hz。频率变化清空旧结果和导出按钮，不自动启动任何计算。

本次独立验证服务位于 **http://127.0.0.1:8771**，没有重启主线程原8770服务。启动新版本：

```bash
.venv-tomato/bin/python -m tools.tomato_review_ui.server \
  --run-root "$PWD/runs/tomato/local93_20261107" \
  --output-dir "$PWD/runs/tomato/local93_20261107/review_ui_5hz" \
  --port 8771
```

5Hz首次提取默认使用CUDA/BF16；缺少GPU或固定权重时作业明确失败，不假造5Hz结果。`--encoder-device cpu`可显式选择CPU提取（FP32、会慢很多）；不同精度可能有轻微数值差异，若切换提取设备建议使用独立output-dir。仅访问页面或2Hz冻结预测不需要GPU。后台并行任务数仍为1。

## 语义和缓存

- 视频仍为源30 FPS。2Hz每15帧一个锚点；5Hz每6帧一个锚点，生成真实的新增特征与概率，不插值旧概率。
- 历史窗口仍16帧@10Hz，未来时距仍为原anchor+1秒。使用原checkpoint/归一化，保留其训练2Hz契约，通过UI专用推理频率字段区分，没有伪改checkpoint或训练manifest。
- 仅复用来源、fingerprint和历史索引完全一致的旧2Hz特征，其他锚点实际运行冻结Encoder。5Hz缓存位于UI专属`features_5hz`，跨三模式复用；每个模式单独运行自己的分类头。
- 2Hz和5Hz结果ID、任务去重键不同；5Hz结果绑定独立缓存SHA。旧2Hz缓存/预测和原始数据不覆盖。导出使用所选result的锚点和轨道，标注预测Hz及原anchor/target。
- 非锚点帧使用HELD，例如frame8使用frame6/0.2s的预测，目标仍为1.2s；尾部不可评分预测照常显示。
- 原Parquet时间戳为float32。5Hz采用整数6帧步长，验证原时间戳在float32量级容差内贴合注册帧轴，再保留原时间戳。避免时间查找将32.4秒等锚点错到前一帧。该容差仅用于UI 5Hz，原2Hz校验和训练采样未改。

## 实际验证

训练来源示例`1df7c31ae9a58dc60d4e57d6`，1065帧、35.5秒：

- 2Hz为71锚点；5Hz为178锚点，索引`0,6,...,1062`。
- 复用36个历史一致的原特征，实际计算142个新锚点。GPU首次提取及Future分类头整次请求约18.10秒。
- Joint和Current-only随后命中同一5Hz缓存，只运行对应CPU分类头；三模式均为178锚点。
- 浏览器真实验证三模式轨数、frame8显示anchor6、固定future target=1.2s、频率切回2Hz恢复71锚点、没有JS错误。
- 全部130项相关测试通过（含11项新增5Hz测试）；原321个训练输出SHA不变。新测试包含float32时间轴边界、真实新增窗口计算的合成可控对照、跨头缓存复用、损坏缓存和失败Encoder不发布成品。

证据：`runs/tomato/local93_20261107/review_ui_5hz/validation.json`、`browser_validation.json`及三模式截图。回归命令：

```bash
.venv-tomato/bin/python -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py
```

没有重训、没有评估或调优93条模型精度、没有批量提取/导出视频，也没有据5Hz结果宣称精度改善。本轮只用一条训练来源验证真实计算和UI流程，未改测试集配置。
