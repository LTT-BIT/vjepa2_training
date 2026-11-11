# 因果概率历史修正：六次微型GPU训练与分组对照

本轮实际训练了6个148参数的修正器，冻结V-JEPA2和B0双头，没有重复训练已有分类头。一个对照出现小幅开发集收益，但未通过预注册门槛，保留B0默认研究配置。没有修改OpenPI、RTC、Prompt映射或机器人执行。

## 方法与身份

起点`34b2967`，`research-tomato-agent`工作区干净。100条完整七阶段录制仍为84训练/8开发验证/8历史测试，manifest SHA256为`95679c2f4457701c3898e538b83883c27106aa3aa44c647ad046b92dd20410e1`。沿用原有效Subtask和历史临时边界版本，不新增或修订人工标签，不再要求重复结果标注。

原B0为独立Current/Future1.5秒双头，种子42/43/44。新模块仅输入两头在当前及过去0.5/1.0/1.5秒的四分类概率：32个裁剪log概率及4个可用性掩码，经一个36→4零初始化线性层产生中心化残差，加到原Future log概率。共148个可训练参数。不可用历史填零并标记，绝不用较新的观测填旧时刻；录制独立初始化，超过0.75秒的观测间断清空历史。

动机是让Current/Future联合证据学习微小时间相关修正，避免统一平滑的固定迟延。它仍预测t+1.5秒类别，不是假装能直接输出准确转换时间。Current输出和七阶段Prompt规则保持不变，逐观测轨迹均与冻结B0一致。

只使用现有冻结4000D特征重新运行原分类头获得训练录制概率，编码器调用0次。修正器归一化仅拟合内部训练部分；Adam0.01、未来无权CE+0.01残差平方惩罚、batch512、最大100epoch/patience15固定。内部留出NLL选epoch，允许保留epoch0；不按开发集事件分数选择epoch，不降低K、不调整阈值。每个配置均先保存JSON再训练。

## 两个可解释的训练划分

A：从84条训练录制固定分组留出16条，68条训练修正器。初次结果后核查发现其中15条曾被原B0训练过。因此留出只独立于修正器，不独立于完整模型。

B：在发现上述问题后，仅改变内部划分，74条旧训练录制拟合修正器，10条原B0未训练过的新增录制选择epoch。网络、损失、优化器和决策不变。已核对三种子B0的split SHA与原94条manifest一致。该10条曾参与其他历史候选训练/探索，不是新的独立盲测。此补充对照是适应性研究步骤，其收益不能描述为预注册独立验证成功。

两组均不refit；最终评估仍是相同8条开发验证。历史测试数组未加载。旧9条39/50与本轮新8条37/48严格分开。

## 开发集统一事件比较

严格同from/to、一对一`[b−1.5,b)`，Future0.5/K2，实际第二次确认时间计事件；迟到和过早预警仍计未匹配。表中误报为次数/分钟。

|Seed|B0命中/误报|A命中/误报|B命中/误报|B0→B adjust→pickup|
|---|---|---|---|---|
|42|37/48 · 0.695|36/48 · 0.748|37/48 · 0.802|2/8→2/8|
|43|34/48 · 0.962|32/48 · 1.069|35/48 · 0.909|1/8→2/8|
|44|35/48 · 0.802|36/48 · 0.748|38/48 · 0.748|3/8→4/8|

三种子平均召回：B0为73.61%，A为72.22%，B为76.39%；B平均误报0.8196/分钟，与B0相同。B对种子42增加2次误报，种子43/44各减少1次，所以平均持平不能掩盖代表种子的退化。固定采用门槛要求均值改善且seed42改善并不增误报，两组均未通过。不因seed44最高而改选代表种子。

Future Macro-F1：

|Seed|B0|A|B|
|---|---:|---:|---:|
|42|0.937368|0.930763|0.935119|
|43|0.928515|0.927591|0.930060|
|44|0.938317|0.938233|0.941180|

Current分类概率完全相同，seed42 Current Macro-F1仍0.971598。Prompt和七阶段位置逐时刻一致，所以本实验没有改善提前Prompt或Current迟确认，不把Future指标当作这些问题的修复成果。

## 逐事件收益与代价

A种子42没救回旧漏检，反而将`27d8bc04...`第3次return→adjust从提前0.467秒推到迟到1.033秒；种子43救回2次但丢4次，种子44救回3次丢2次。A全部种子的窗口NLL降低，却没有整体事件收益，说明概率误差目标不足以保证时机正确。

![A配置丢失一次原正确预警](/home/user/vjepa2_training/docs/validation/tomato_probability_residual_20261111/27d8bc0413a42341c7e8c3e5_b3.png)

B在三个种子分别救回0/1/3次，未丢失各自原命中；但seed42新增两个不匹配预警。seed43救回`27d8bc04...`的adjust→pickup，提前0.600秒。seed44救回该录制及`a8b23394...`的pickup→harvest，并救回`5b82f4bb...`的adjust→pickup，提前0.300秒。不能把跨种子的这些事件当作4个独立新增物理事件。

![同K2条件下救回一次adjust到pickup](/home/user/vjepa2_training/docs/validation/tomato_probability_residual_20261111/unseen_control/seed44_recovered_adjust_pickup.png)

按录制整体配对bootstrap（4000次，先在同一录制内平均三种子）得到B相对B0召回差95%区间约+0.69～+5.56个百分点，误报差区间−0.132～+0.210次/分钟。该区间只描述暴露开发集的重采样，不纠正本轮适应性对照及多轮模型探索，不是泛化显著性证明。8条开发录制太少，误报不退化尚无可靠保证。

## 更重要的内部留出结果

|Seed|B0在10条上的命中/误报数|B修正后命中/误报数|
|---|---|---|
|42|37/60 · 31|36/60 · 32|
|43|39/60 · 30|37/60 · 32|
|44|33/60 · 34|33/60 · 32|

这里也没有稳定事件收益。尽管B的留出NLL均改善，事件命中不升；1/2/1个epoch就选中最优，不能从开发集的偶然增益推断需要继续加大训练。该结果说明后处理概率修正的泛化依据仍弱，不证明原视觉表征必然不足。

## 资源、测试与产物

6次GPU微型训练合计约3.45秒（不含缓存读取、冻结头推理、评估及进程启动）；A选中epoch22/8/24，B选中1/2/1。PyTorch训练峰值allocated约69.1～69.5MiB，不含CUDA context。单观测、GPU驻留输入的修正器P95约0.046～0.130ms，不包含视频、编码器或传输，不作为机器人端到端时延。

244项番茄测试通过，包括4项新增因果历史/间断/非法输入/残差初始化测试。6个checkpoint在CPU逐条重载，48个录制×种子×配置回放的概率最大差小于2e-6，类别及全部事件时间与保存GPU输出一致。每个run保存源代码快照并与执行时SHA核对；旧模型、数据和权重只读。

- 新代码：`tools/tomato_probability_residual.py`、`tools/train_tomato_probability_residual.py`、`tools/report_tomato_probability_residual.py`。
- 配置：`configs/tomato_probability_residual_20261111/protocol.json`和`unseen_control.json`。
- A完整本地产物：`runs/tomato/probability_residual_20261111_v1`。
- B完整本地产物：`runs/tomato/probability_residual_unseen_20261111_v1`。
- 每组`seed{42,43,44}/residual.pt`包含参数、归一化、输入Schema、原模型路径/SHA、数据SHA、划分SHA、seed和epoch，均不纳入Git。
- 可审计摘要、逐事件对照、每录制结果、checkpoint重放及bootstrap在`docs/validation/tomato_probability_residual_20261111/{random_holdout,unseen_control}`。

复现必须使用新的输出目录：

```bash
cd /home/user/vjepa2_training
.venv-tomato/bin/python -m tools.train_tomato_probability_residual --out runs/tomato/NEW_A
.venv-tomato/bin/python -m tools.train_tomato_probability_residual --config configs/tomato_probability_residual_20261111/unseen_control.json --out runs/tomato/NEW_B
.venv-tomato/bin/python -m pytest -q tests/test_tomato*.py
```

结论：有小幅可追踪信号，尚不足以替换B0。接下来优先补足对原B0真正未见、模型及阈值冻结的新录制上的验证，并设计训练组内部的事件时机目标对照；不在当前8条上继续搜修正强度，不直接更换seed，不要求重复标Subtask，也没有证据支持马上扩大编码器或长期记忆网络。
