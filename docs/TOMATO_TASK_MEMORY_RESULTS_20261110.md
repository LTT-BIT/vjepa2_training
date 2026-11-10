# 第五阶段：长时程任务记忆与事件驱动状态估计（2026-11-10）

## 结论

实际实现了独立TaskMemory/EventRecord、候选确认/拒绝/过期、事件证据账本、历史冲突检测、Shadow隔离、完整录制回放和7种故障注入。**目前记忆有审计与诊断价值，没有证明比B0更准确。保留B0，不进入M3自动恢复，也不训练新记忆网络。**

正常验证三种子M0/M1/M2逐窗口七位置完全相同。seed42仍为Current0.97156、Future1.5秒0.93808、严格预警39/50、13误报即0.674/分钟、46/50按时Prompt、40秒总状态不一致、adjust→pickup2/8。没有把“不改变B0的Shadow”宣称为预测改善。

连续6次分类损坏使M0/M2状态不一致同时升到156.5秒、按时Prompt降到37/50、提前升到11次、级联失配从1升到8。记忆在8/9条录制中留下冲突，但未阻止错误阶段确认。它仍可能忠实保存一个错误决策的历史；决策已经发生与物理操作已经成功必须分开。

## 审计与版本冻结

本轮起点research-tomato-agent / 9a63961，工作区干净。完整阅读并核对状态恢复、转换审计、人工边界复核、时间头与非硬门控报告。原B2的12个受影响事件源自7个硬门控偏移，**不是B0的12个同类故障**；本轮没有把旧多假设状态机当作默认记忆。

读取的开源论文/代码、精确commit和借鉴取舍见[方法阅读记录](TOMATO_TASK_MEMORY_REFERENCES_20261110.md)，代码文件SHA见configs/tomato_task_memory_20261110/reference_sources.json。外部代码只在/tmp只读研究快照中审阅，未安装其运行环境、未引入新VLM、未移植完整框架。

固定9条开发录制、50个真实转换、14项人工边界**复核提案**版本。原标注未写回；原始标签、提案与已批准训练修订仍保持区分。输入/概率/模型SHA及先前基线receipt校验见validation/tomato_task_memory_20261110/receipt.json；另外重新核验9条当前原标注、9个状态Parquet和9个冻结特征缓存与原身份一致，详见source_audit.json。B0是原历史分类器与归一化（并非此次重训的74/84条分类头）；74/84条新训练版本及旧Joint稳定默认均保留，10个既有导出/基线checkpoint SHA核对未变。

101条中的新增10条已用于训练/开发，未再当盲测。本轮0次训练、0次编码器前向、0次历史测试推断，未修改OpenPI、RTC、π₀.₅或机器人接口，未推送远程。

## 已完成范围

P0完成资料、代码、模型/数据版本审计；P1/P2完成数据契约和事件更新；P3完成Shadow隔离；P4完成M0/M1/M2独立离线位置对照；P5完成正常与故障完整录制回放；P6依据结果停止新网络与M3；P7保存逐事件指标、资源和失败案例。

实际90个完整Episode流：正常3种子×9条=27个，seed42的7种故障×9条=63个。每条流均同时运行M0、M1、M2。故障未做三种子穷举，也未把合成故障当作新独立数据。完整运行目录：`/home/user/vjepa2_training/runs/tomato/task_memory_20261110_final`。

## 数据契约和更新规则

`tools/tomato_task_memory.py`提供Observation、EventRecord、TaskMemory；版本tomato-task-memory-v1。完整序列化契约在configs/tomato_task_memory_20261110/memory.schema.json，模型与58D命名输入绑定在model_bindings.json，独立规则在protocol.json。

Observation包含Session、严格递增sequence与timestamp、规范化session:sequence观测ID、实际到达时间、模型版本、固定四类顺序和Current/Future概率。可选58D位置必须匹配完整关节名顺序；状态取最后一个不晚于当前观测的真实样本，禁止未来插值。运动摘要为相邻有效观测的因果位置差分，按名称分14臂/44手；它不是抓取成功传感器，也不是30Hz瞬时速度。

短期记忆默认16样本，可配置，保存概率、年龄、有效性和轻量运动摘要。近期长期事件FIFO上限256，可配置；六个已确认有序转换另存紧凑账本，裁剪近期审计不丢失关键进度。完整事件修订流写每Episode JSONL，离线完整轨迹存在磁盘，不属于在线常驻记忆。显式reset清空阶段事实/观测/候选并更换generation，旧Session消息不能进入新任务事实。

EventRecord记录唯一ID、类型、事件迹象时间、更新时刻、确认观测时刻、from/to位置、来源、置信度、支持观测ID、candidate/confirmed/rejected/expired/uncertain状态、修订原因和版本。candidate首次迹象时间与confirmation_time分开，不能拿候选开始当确认时刻。

四种语义严格分开：model_stage_recognized是四分类预测记录；transition/phase_enter/phase_exit是阶段决策证据；operation_command表示未来可能接收的真实指令日志；physical_outcome需要独立结果证据。此次没有后两类事件，physical_success一律unobserved。进入return_stem时decision_fraction可到1，但task_complete始终false；它表示已进入最后位置，绝不表示返回或采摘物理成功。

当前候选只针对账本允许的相邻位置，Current≥0.8、连续两次新鲜观测才由M2确认；候选TTL1.5秒，观测间隔>0.75秒或无效消息打断证据，超过1秒的数据年龄拒绝。Future仅记candidate advisory，永不据此确认当前阶段或更新Prompt。稳定确认历史不因短暂分类反转而删除；冲突另存uncertain事件，观察恢复兼容后记录conflict_resolved，不篡改原确认时间。

M1只观察B0决策快照，B0对象、Prompt、计数器均不提供给记忆写入。M2只按自身有效观测形成事件前缀，再从前缀检查任务进度；没有GT位置输入、按固定Episode时间推进或自动重同步。重复四类不会触发七位置argmax重置。无论M1/M2，版本化接口`suggests_prompt_update=false`。

## 统一评价口径

分类分数是未修改Current/Future头的原始四分类结果；记忆没有生成新的分类概率。严格预警保持原Future0.5/K2事件、一对一精确from/to、[b−1.5,b)半开窗口、原尾掩码和19.275分钟分母。M1/M2不改变预警，因此该指标相同是隔离/设计结果，不能算新能力。

阶段/Prompt按完整Current时间轴评价，按有序目标位置在[b,min(b+2,该段结束))确认；M2列仅为离线假设Prompt时机，不发送Prompt。人工七位置与边界只进入scorer和图表，在线初始化pickup_1来自任务协议。状态不一致包括正常确认等待；持续偏移另计连续>2秒；级联数为同一连续偏移区间内跨过的后续真实边界。

## 正常录制三种子对照

Prompt列是按时/提前/延迟，均无缺失。下表全部采用相同数据/标签/概率；M2与B0的位置信息相同，但事件证据更多。

|seed|方案|Current F1|Future F1|严格预警|误报/min|Prompt按时/提前/迟|总不一致秒|持续偏移段/秒|级联数|
|---|---|---|---|---|---|---|---|---|---|
|42|M0|0.97156|0.93808|39/50|0.674|46/2/2|40.0|3 / 10.5|1|
|42|M1|0.97156|0.93808|39/50|0.674|46/2/2|40.0|3 / 10.5|1|
|42|M2|0.97156|0.93808|39/50|0.674|46/2/2|40.0|3 / 10.5|1|
|43|M0|0.96742|0.92865|35/50|0.986|46/2/2|46.5|3 / 13.0|1|
|43|M1|0.96742|0.92865|35/50|0.986|46/2/2|46.5|3 / 13.0|1|
|43|M2|0.96742|0.92865|35/50|0.986|46/2/2|46.5|3 / 13.0|1|
|44|M0|0.97048|0.93814|36/50|0.830|45/4/1|37.0|2 / 8.0|1|
|44|M1|0.97048|0.93814|36/50|0.830|45/4/1|37.0|2 / 8.0|1|
|44|M2|0.97048|0.93814|36/50|0.830|45/4/1|37.0|2 / 8.0|1|

逐Episode、七位置混淆、后半段表现见summary/results；1500条逐事件/方法/场景比较见validation/tomato_task_memory_20261110/event_comparisons.csv。三种子一致结果证明Shadow隔离与规则复现，没有证明记忆改善性能。9条开发录制已多轮使用，不能替代独立泛化验收。

## 重复阶段与adjust→pickup

seed42七位置采样正确率如下，M0/M1/M2相同。位置错误包含正常确认延迟，不能与四类Macro-F1混用；初始pickup_1由任务协议已知，不能把100%当成纯视觉模型能力。

|有序位置|正确窗口/总窗口|位置正确率|
|---|---|---|
|pickup_1|138/138|100.000%|
|harvest_1|878/889|98.763%|
|return_truss|76/87|87.356%|
|adjust|325/340|95.588%|
|pickup_2|120/144|83.333%|
|harvest_2|672/681|98.678%|
|return_stem|51/61|83.607%|

事件前缀能解释同一pickup/harvest为什么处于第一次或第二次，也保留进入该位置的证据；但原B0的七阶段位置已编码同样的线性任务进度，所以没有新增可区分信息。缺少adjust事件证据时，不允许跳到pickup_2。若历史本来就错，结构化存储不会自动使它变真。

全部8次困难转换（最近Future同类型事件偏移仅作诊断，不替代严格事件匹配）：

|Episode前缀|提案边界秒|严格Future|最近Future偏移秒|Current确认结果/偏移秒|
|---|---|---|---|---|
|3e645bf4|74.233|漏检|-4.233|early -2.233|
|27d8bc04|109.600|命中|-0.600|on_time +0.400|
|0b12ac0f|99.133|漏检|-2.633|late +3.867|
|61c42084|90.400|命中|-0.400|on_time +1.100|
|5b82f4bb|90.300|漏检|+0.200|on_time +1.200|
|f53f0b6f|94.867|漏检|+4.133|late +4.633|
|a8b23394|65.267|漏检|+0.233|on_time +0.733|
|d0f8727d|65.600|漏检|+0.400|on_time +1.900|

仍为2次命中、2次过早、4次迟报。三种方法逐事件一致。典型提前Prompt案例仍在72.0秒确认，较原边界73.3秒提前1.3秒，较提案74.233秒提前2.233秒。记录历史没有消除Current的提前强证据。现有结果支持优先补充稳定可观测锚点与标注规范；仍不足以证明视觉特征一定缺信息，也没有理由为这8例硬编码。

## 故障注入结果

注入时刻由max(8,窗口数//3)固定观测索引决定，完全不看阶段边界。单次错误改变一个Current；连续错误改变六个Current；Future保持原概率。另测试缺失4个观测、4个过期结果、重复消息、乱序消息和错误下一阶段候选。错误类以原模型argmax的下一个类别确定，未按GT挑最坏结果；wrong_candidate仅用在线B0下一合法类。

M1始终镜像M0，不在下表重复。严格Future预警在这些既定注入位置仍为39/50、0.674/分钟；**这不能掩盖Current决策已经严重退化**。

|场景|方案|Prompt按时/提前/迟|总不一致秒|持续偏移段|级联数|
|---|---|---|---|---|---|
|normal|M0|46/2/2|40.0|3|1|
|normal|M2|46/2/2|40.0|3|1|
|single_class_error|M0|46/2/2|40.0|3|1|
|single_class_error|M2|46/2/2|40.0|3|1|
|consecutive_class_errors|M0|37/11/2|156.5|12|8|
|consecutive_class_errors|M2|37/11/2|156.5|12|8|
|missing_observations|M0|46/2/2|40.0|3|1|
|missing_observations|M2|46/2/2|40.0|3|1|
|stale_results|M0|46/2/2|40.0|3|1|
|stale_results|M2|46/2/2|40.0|3|1|
|duplicate_message|M0|46/2/2|40.0|3|1|
|duplicate_message|M2|46/2/2|40.0|3|1|
|out_of_order|M0|46/2/2|40.0|3|1|
|out_of_order|M2|46/2/2|40.0|3|1|
|wrong_candidate|M0|46/2/2|40.0|3|1|
|wrong_candidate|M2|46/2/2|40.0|3|1|

单帧错误与单候选错误未形成新增确认；丢失/过期/重复/乱序消息没有在这些注入位置改变最终事件指标。不能据此声称任意边界附近丢帧都无害：固定索引多数落在稳定阶段，边界证据断裂由单元测试另行覆盖，尚需独立压力录制。

连续错误造成9条录制的提前确认，8条出现持续历史/Current冲突。回到同一方法的干净回放轨迹，中位恢复12.0秒、范围1.0～22.0秒；M0/M1/M2一致。这个“恢复”只是状态重新与干净轨迹一致，通常是实际任务后来走到已误提前的位置，并非记忆纠正旧事件或自动回退。GT恢复时间另列，不把原有边界误差算作故障修复。

正常录制M2没有强历史冲突告警，注入连续错误后199个0.5秒窗口带有冲突标记。无告警不等于状态正确：一阶段落后或重复类别仍可能看似合法。长期账本保留错误确认和冲突两种证据，没有在看到GT后删除错误事件。

## 成功与失败回放产物

`/home/user/vjepa2_training/runs/tomato/task_memory_20261110_final/seed42/<场景>/<Episode>/`含M1/M2_events.jsonl与final快照；每场景replay.json含收到的真实/注入概率、观测时钟、阶段、版本、候选、运动摘要和冲突。正常三种子与7种故障均有完整Episode记录。

26张图位于`/home/user/vjepa2_training/runs/tomato/task_memory_20261110_final/plots`：9张正常全程、9张连续错误全程、8张adjust→pickup概率与确认局部图。图中GT仅为离线参考。典型单帧错误成功拒绝与六帧错误失败样本已摘录到single_class_error_example.json和consecutive_class_errors_example.json；完整记录仍保留。

最小可读回放：

```bash
.venv-tomato/bin/python -m tools.inspect_tomato_task_memory --run runs/tomato/task_memory_20261110_final --episode 3e645bf4e137d0f836c9d7ff --scenario consecutive_class_errors --start 42 --end 58
```

该例44.5秒产生return候选、45.0秒误确认、48.0秒记录历史冲突。原真实return边界55.8秒；模型输入错误是注入的，不能当作原模型真实误判率。单帧注入则候选被拒绝，没有写入确认历史。

## 开销与2Hz条件

|模块|P50毫秒|P95毫秒|最大毫秒|
|---|---|---|---|
|M0|0.01314|0.01854|0.05819|
|M1|0.03640|0.07809|0.17905|
|M2|0.03234|0.06812|0.15526|

每模块7020个正常完整时间轴更新（三种子），M1/M2计时包含轻量运动摘要与事件JSONL缓冲写入。最大常驻对象约82.1KiB（递归Python对象估计，排除外部文件缓冲）；最大JSON快照约44.1KiB。不是完整Python进程RSS，也不把磁盘回放文件算作在线记忆。长事件流通过有界FIFO与单独确认账本避免无限驻留。

没有新增GPU模型/CUDA张量。额外P95不足0.1毫秒，与2Hz的500毫秒预算相容。此前编码器链路实测见转换/时间头报告；本轮只测缓存输入后的记忆增量，**未重新实测摄像头、编码器、传输和记忆组合的完整在线链路**，不作硬实时保证。

## 测试与复现

26项新增记忆测试，全部197项番茄专项测试通过，日志保存在validation/tomato_task_memory_20261110/tomato_tests.log。包括因果前缀、无GT API、只读Shadow、事件不重复确认、过期/拒绝、非有限/过期/重复/乱序消息、跨Episode隔离、schema字段/状态语义、有界缓存、58D名称及时间因果、深层快照防篡改、账本损坏拒绝静默重置、恢复后清除当前传输告警但保留审计。

这里是番茄专项通过，不宣称整个仓库所有旧任务测试通过；旧柠檬测试缺少annotate_lemon_doubao导入的问题未在此任务修复。JSON Schema保存并做了字段/枚举/序列化契约测试；本环境未安装完整JSON Schema验证器，没有宣称运行了第三方完整schema validator。

在仓库根目录复现（输出必须全新）：

```bash
.venv-tomato/bin/python -m pytest tests/test_tomato_task_memory.py -q
.venv-tomato/bin/python -m tools.evaluate_tomato_task_memory --out runs/tomato/task_memory_reproduce_unique
PYTHONPATH=/tmp/tomato-transition-plotdeps .venv-tomato/bin/python -m tools.render_tomato_task_memory --run runs/tomato/task_memory_reproduce_unique --out runs/tomato/task_memory_reproduce_unique/plots
```

绘图需要既有隔离Matplotlib依赖，评价不依赖绘图库。不需要下载权重、启动GPU训练、调用VLM或连接机器人。

新增代码：tomato_task_memory.py、evaluate_tomato_task_memory.py、render_tomato_task_memory.py、inspect_tomato_task_memory.py及对应测试。配置/契约/资料快照在configs/tomato_task_memory_20261110，结果摘要在docs/validation/tomato_task_memory_20261110；原checkpoint和所有历史run未覆盖。

## 下一步判断

1. **历史事件比一个位置整数更便于审计**：可以追溯何时、用哪两个观测确认，候选何时过期，物理结果是否有证据。但本次未增加判别信息或降低错误。
2. **没有减少状态不一致或级联**：正常三种子均零差值，连续错误下同样受损。不能把新增冲突日志当成已实现恢复。
3. **能解释重复阶段，未提升重复阶段正确率**：其作用依赖正确历史前缀，与原七位置任务图同样受错误前缀影响。
4. **adjust→pickup未改善**：主要还需检验观测锚点、边界语义和局部分类时机，记忆本身不能制造新证据。是否需Token读出仍需新的单因素表征对照，当前不启动。
5. **增量开销支持继续2Hz研究**，全链路验收仍需后续实际组合测量。
6. **继续只服务Agent的只读诊断**：可以输出版本化进度与候选供未来接口讨论，尚不把它写入π₀.₅ Prompt或当作抓取/摘取完成事实。

当前不进入M3，不训练新GRU/TCN或其他记忆网络。下一步先补充独立开发压力录制：在非GT触发的自然观测缺失、长时间类别混淆、失败抓握/重复调整中，验证冲突与物理锚点的关系。需新证据时优先加入可观测结果而非更长的错误历史。新增独立盲测必须与现有101条分原始录制/日期/场次，先锁模型、规范和规则再一次性验收；不能再用已训练的新10条选方法。
