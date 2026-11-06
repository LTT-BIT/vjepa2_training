# 新腕部视角 V-JEPA 2 六阶段识别实验（2026-10-27）

数据：`/home/descfly/Convert_data/outputs/lemon_move_wrist_reset_20261008`。
标注：`/home/descfly/Downloads/phase_annotations_v3_clean.json`，209条完整五时间戳。
原始数据共225条；没有完整标注的记录不参与本实验。
核对三路视频长度、10fps、标注cam0 SHA-256、状态时间轴和五个边界帧。
按原始rosbag录制来源隔离：167训练、21验证、21测试，种子42。
测试演示预先固定为episode58，其原始录制不会出现在训练集或验证集。

## 模型与训练

预训练：facebook/vjepa2-vitl-fpc64-256，固定revision见manifest.json。
使用冻结的ViT-L视频编码器（BF16/SDPA），不训练预测器。
每路最近16帧（10Hz，1.5秒历史），三路共享编码器；RGB全图缩放256x256，ImageNet归一化。
起始帧不足时只重复已观测的第一帧；严禁读取未来帧。
逐视角token均值特征与相同窗口16x12双臂实测关节角拼接；不使用专家动作、时间戳值或真实阶段作为输入。
固定每5帧提取一次（2Hz），采样时刻不依赖人工阶段边界。
特征只按训练集均值、标准差归一化。
分类头：输入→256→6，GELU，Dropout0.3。
AdamW：lr3e-4，weight decay0.01；batch128；类别平方根倒数加权交叉熵；梯度范数限制1。
最多100epoch，每epoch验证；按val Macro F1优先、val NLL次优选择best_head.pt；15epoch不改善提前停止。
测试集只作最终评估，不参与模型选择；metrics.jsonl和evaluation.json保留指标。

## 运行与进度

启动/从缓存继续：
```bash
bash /home/descfly/hmj/training/experiments/lemon_vjepa2_phase_20261027/run.sh
```
同一时间只运行一个实例。继续时复用已完成的逐episode特征，分类头会重新训练。
```bash
watch -n 5 cat /home/descfly/hmj/training/experiments/lemon_vjepa2_phase_20261027/status.json
tail -f /home/descfly/hmj/training/experiments/lemon_vjepa2_phase_20261027/run.log
```
源码：`/home/descfly/hmj/ur/tools/train_lemon_vjepa_phase.py`，运行时复制train_source_snapshot.py。

## 产物

manifest.json：数据哈希、录制来源、分组、预训练revision、预先固定演示episode。
pretrained/：预训练权重、官方SHA-256；best_head.pt：分类头、训练集归一化和manifest哈希。
features/：逐episode特征缓存，原数据不修改。
test_episode_000058_phases.mp4：cam0画面，叠加三路+状态模型预测、人工标注、六阶段概率；每个判断只使用历史观测。
demo_predictions.json：逐帧预测和最后一个已用输入帧编号；不使用强制单向切换隐藏错误。
latency.json：同步CUDA测速，三路16帧输入、关节状态、分类头；源图像转RGB/缩放另外测量。
测速暂不包含ROS传输、Pi05并行GPU竞争；2Hz判断不等于10Hz或动作控制频率。
只有最近历史窗口，启动不需要等待整段任务；初始不足16帧使用重复首帧。
本实验只训练、离线演示和测速，不连接或控制机器人。
