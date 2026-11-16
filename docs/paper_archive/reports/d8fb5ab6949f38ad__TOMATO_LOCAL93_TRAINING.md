# 本地 93 条数据：三种模式训练

原始数据：`/home/user/pi05/tomato_sharpa_remote93/A` 和 `B`。
配置：`/home/user/pi05/vjepa2_training/configs/tomato_four_stage_local93.yaml`。
索引：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/annotation_index.json`，沿用远程 93 条的标注 SHA256 和 revision，只替换本地路径。
划分：`configs/tomato_remote93/split.json`，75 train / 9 val / 9 test，三个模式共同使用；所有训练超参数与原三样本实验一致，未来 1.0 秒，lambda_current=1.0、lambda_future=0.5，不依据测试集调参。

## 用户启动

```bash
cd /home/user/pi05/vjepa2_training
bash tools/run_tomato_local93.sh all
```

依次执行 Current-only、Future-only（未来 1 秒）、Joint 的准备、特征提取、分类头训练与离线评估。纯本地读取，不需要 SSHFS 或服务器连接。三种模式共享 `runs/tomato/local93_20261107/shared_feature_cache`。本地 Current-only 已完成的数据准备结果会按原程序规则校验复用，其他模式仍会生成各自的 manifest；不跳过源文件/标注检查。

也可单独运行：

```bash
bash tools/run_tomato_local93.sh current_only
bash tools/run_tomato_local93.sh future_only
bash tools/run_tomato_local93.sh joint
```

结果目录：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/` 下的 `current_only`、`future_only`、`joint`。各模式保存 `launcher_run_*.log`、`metrics.jsonl`、`training_config.json`、`evaluation.json`、`best_head.pt`、预测文件和锁定划分。已完成的模式不要重复启动；原程序会拒绝覆盖训练结果。若其中一个已完成，后续单独启动其他模式。

## 启动前核验

完整数据准备日志：`runs/tomato/local93_20261107/prepare_validation.log`。使用原 prepare 函数，仅增加控制台进度，不修改原始数据或训练实现。预训练 model.safetensors 和 config.json 已通过固定 SHA256 校验。实际验收结果见同目录 `local_preflight_summary.json`（全部通过后生成）。

本次准备不启动 Encoder 特征提取或分类头训练，也不修改此前的远程实验与三条本地样本实验。
