# 番茄 Agent 阶段转换审计：2026-11-09

本轮没有训练新模型，没有读取测试集特征或重新推断测试集，没有修改标注、OpenPI、RTC或机器人控制。基于 `research-tomato-agent` / `1d15065bc3a5492a030f801a49dffcb378ba7a2b` 的干净工作区开展。OpenPI保持 `33ce417344b22ecd66e362e5daab6946fab439ae`。原76/10/8划分及历史结果不变。

## 结论

保留独立1.5秒双头、2Hz研究候选。相对Joint seed42，其严格预警40/56对21/56，误报0.962对1.626次/分钟；限制为已反复使用的验证集证据。当前模型仍漏检16/56，未达到≥80%严格预警召回的既定门槛，不能认为已可靠解决转换。

本轮没有验证出比现有候选更可靠的总体方案。4Hz算力充足、部分条件检测更快；但在匹配确认时长后，Future只净增1个命中，Current丢失2个命中，接口错误预警及提前Prompt均增加。继续2Hz，不因4Hz某个指标变高自动替换推荐模型。

## 冻结规则和指标分层

- 分类指标：逐窗口四类Current标签；Future监督严格取t+h人工标签，不能与转换事件召回混用。
- 严格事件：置信度≥0.5，连续确认；首次稳定标签仅初始化。精确匹配from/to，按时间先后进行一对一匹配；Future在[b−h,b)，Current在[b,min(b+2s,新阶段结束))。任何未匹配预测算误报，未匹配真实事件算漏检。原2Hz实现已逐项复现。
- 原始允许提前窗：1秒模型h=1，1.5秒模型h=1.5；另报告共同1秒事件窗，避免单纯扩大窗口带来不公平增益。Future分类目标仍是各自真实训练时距。
- 2Hz K2样本跨度0.5秒；4Hz K2约0.233/0.267秒，4Hz K3跨度0.5秒。两种4Hz均事先登记，不进行阈值扫描。
- OpenPI层：原Provider阈值0.8，Current连续2次确认下一个有序阶段；Future仅对下一阶段发单帧level warning，不能更新Prompt。4Hz K3回放仅把Provider构造参数确认次数设为3，不改其源码。
- 宽松预警覆盖：允许在真实边界前有效窗内出现任何一次正确下一阶段warning；不要求发生严格四类accepted-change事件。不能替代严格召回。
- 抖动：原始四类稳定切换后2秒内反向切回的次数。七阶段Provider单向前进，零回退不表示没有提前误切换。
- 主表按输入观测时刻计时；另在paired_rates.json给出加测得P95计算时延的可用时刻敏感性分析。接口回放使用+40ms结果/+50ms消费的合成传输时钟，不声称测得真实传输。

## 三模型与2/4Hz统一结果（全部seed42）

|方案|Current F1|Future F1|Current事件命中|Current P50/P95延迟s|Future事件命中|Future误报/min|Future中位提前s|Current/Future抖动|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|joint_2hz_k2|0.97268|0.93380|52/56|0.833/1.215|21/56|1.626|0.400|2/2|
|independent_h10_2hz_k2|0.97481|0.94445|52/56|0.867/1.200|25/56|1.376|0.233|0/1|
|independent_h15_2hz_k2|0.97481|0.92356|52/56|0.867/1.200|40/56|0.962|0.567|0/2|
|joint_4hz_k2|0.96903|0.93570|50/56|0.400/0.918|30/56|1.627|0.450|5/7|
|independent_h10_4hz_k2|0.97242|0.94143|51/56|0.433/0.917|37/56|1.293|0.500|5/9|
|independent_h15_4hz_k2|0.97242|0.92272|51/56|0.433/0.917|38/56|1.256|0.867|5/7|
|joint_4hz_k3|0.96903|0.93570|51/56|0.633/1.100|24/56|1.585|0.417|1/3|
|independent_h10_4hz_k3|0.97242|0.94143|50/56|0.700/1.103|32/56|1.084|0.367|0/1|
|independent_h15_4hz_k3|0.97242|0.92272|50/56|0.700/1.103|41/56|0.879|0.700|0/2|

共同1秒事件窗下，2Hz Joint / 独立1秒 / 独立1.5秒分别为21/56、25/56、35/56。1.5秒原生窗40/56中，5次来自额外0.5秒允许提前窗。独立1.5秒依然更好，但不能把所有增益都归因架构。三模型的损失/训练预算差异沿用冻结历史训练；本轮是现成checkpoint比较，不是重新控制训练预算的因果架构实验。

所有表格都是seed42：71.43%是40/56；历史约72.02%是三种种子平均严格召回。本轮实际接口宽松覆盖47/56=83.93%；历史约81.55%是三种种子平均宽松覆盖，二者不能和72%互换。本轮没有新增训练种子或将窗口当成独立录制来宣称显著性。

## 56个真实转换与错误归因

验证集10条录制，9条七阶段、1条仅pickup/harvest/return，56个类别转换。阶段重复出现时保留其有序位置；不把第二次pickup与第一次pickup视为同一事件。完整列表见下文事件目录及events.json；每项含原始录制UUID、原标注SHA/帧/时间、两头概率、首个/稳定目标预警、严格命中、Current延迟、实际阶段确认时间、邻近误报、58D实测位置派生速度和视频。

|实际转移|数量|Joint命中|独立1秒命中|独立1.5秒命中|
|---|---:|---:|---:|---:|
|pickup->harvest|19|4|9|16|
|harvest->return|19|14|13|17|
|return->adjust|9|1|1|5|
|adjust->pickup|9|2|2|2|

16次严格漏检的互斥时间归因：10次最近同from/to事件在边界之后，5次过早发生在b−1.5秒以前，1次没有匹配的类别路径。这是离线误差描述，不将真值输入模型。另一个证据维度：8次没有窗内稳定预警但附近存在窗外转换；5次窗内已有稳定目标概率却没有新的精确from/to转换；3次只有短暂窗内预警。这两套统计口径不同，不相加。

最薄弱的是adjust→pickup，仅2/9；pickup→harvest为16/19、harvest→return为17/19。优先研究边界语义和预警时间校准，而非扩大分类头。某个事件曾经有正确概率，并不意味着在线确认时刻正确。

证据与待验证假设分开：时间偏差、错误路径、短预警和提前Prompt由冻结预测序列可直接复核；遮挡导致漏检、人工标注错误、不同动作定义导致系统偏差，目前不能仅凭曲线确证。已查看下列代表性曲线与边界前/中/后三视角帧，抓取和调整常共享相似手部姿态，部分关键接触点被手掌遮住；这些只是复核线索。全部56个视频已生成，但未逐帧人工审核全部视频，不宣称完成专业人工重标注。

|样本|直接证据|需人工复核|
|---|---|---|
|3e645…_b3 return→adjust|严格Future确认晚0.567秒；手部速度在边界附近变化|return结束、松手与adjust开始是否用同一准则|
|3e645…_b4 adjust→pickup|Future事件早3.3秒；Current使Prompt早1.3秒更新|开始接近/接触/抓稳的边界定义；两头都提前不能证明原标注错|
|6d46…_b4 adjust→pickup|Future事件早2.0秒；有效窗内概率已稳定但无新切换|重复pickup的语义与调整结束时刻|
|6d46…_b2 harvest→return|Future事件晚2.233秒，Current更早识别return|冻结两头缺少时序一致性；视觉接触遮挡和动作停顿是否相关|
|3e645…_b5 pickup→harvest|窗内仅一个高概率样本，K2确认落在边界后|4Hz是否增加有效预警跨度，而非仅减少等待|

状态输入按58个实测位置名称读取，速度使用真实相邻时间戳差分，双臂14D/双手44D分别RMS。只作为诊断，不增加模型输入或用速度阈值改标注。局部最大状态时间间隔保存在每个事件motion字段；未见这些已读取片段的明显观测缺失证据，不能据此排除所有原始采集问题。重复操作仍使用七个序号处理。人工复核清单保留原始帧、建议帧=null、空修订历史；没有自动修改任何标签。

## 实际OpenPI离线接口

|方案|宽松预警覆盖|真实Prompt确认≤2s|错误warning上升沿/min|提前Prompt次数|Prompt确认P50/P95延迟s|
|---|---:|---:|---:|---:|---:|
|joint_2hz_k2|38/56|52/56|1.167|1|0.900/1.530|
|independent_h10_2hz_k2|37/56|51/56|1.126|1|0.900/1.333|
|independent_h15_2hz_k2|47/56|51/56|0.962|1|0.900/1.333|
|joint_4hz_k2|41/56|49/56|1.543|4|0.500/1.040|
|independent_h10_4hz_k2|39/56|50/56|1.168|3|0.533/1.080|
|independent_h15_4hz_k2|49/56|50/56|1.507|3|0.533/1.080|
|joint_4hz_k3|41/56|50/56|1.543|3|0.717/1.285|
|independent_h10_4hz_k3|39/56|50/56|1.251|3|0.783/1.328|
|independent_h15_4hz_k3|49/56|50/56|1.507|3|0.783/1.328|

2Hz独立1.5秒：207个warning窗口中95个不在正确预警窗内，75个在边界之后、20个超过允许提前量；这不是95个独立误报事件。严格误报23次和接口错误warning上升沿23次在这个模型恰好相等，定义仍然不同。

反事实检查覆盖43464个回放观测：保留Current，替换Future为均匀概率并同步重算logits，Prompt/phase/version完全不变。每个Chunk绑定session/model/Prompt版本，转换后旧Chunk匹配失败，新Chunk匹配成功。七个阶段及英文Prompt由已有接口提供，四类概率始终四维；仅三阶段录制停在return_truss，task_complete始终false。不调用动作模型、硬件或RTC。

Future错误预警本身不会提前更新Prompt；Current仍能错误提前确认。这是本轮真实发现的限制，不能用接口设计正确来替代决策可靠性。

## 2Hz/4Hz真实观测与性能

完整验证集新鲜4Hz窗口5795个，原30Hz时间轴采用向下取最近观测，间隔交替7/8帧；16帧×10Hz历史因果采样，最新帧保留，机器人状态同帧。未用旧预测重复生成4Hz。完整提取GPU路径累计270.31秒、三视频解码134.28秒；缓存独立身份含源文件SHA、关节名称、相机顺序、锚点、采样规则和预训练来源。

2Hz复用原冻结logits。4Hz重新编码与2Hz重合的2898个时刻用于数值控制：位置完全相同，BF16视觉最大绝对差0.03125、logit最大差0.002862；所有四类argmax、0.5/0.8置信门、严格事件和实际Provider结果一致。数值非逐位相等，不能把BF16差异隐藏；这些控制支持本轮差异来自新观察时刻/确认规则。

|节拍|真实独立输入|P50/P95/max ms|观测年龄P95/max ms|超时|峰值分配/保留MiB|
|---|---:|---:|---:|---:|---:|
|2Hz|60|42.38/42.54/43.31|42.60/43.39|0|794.6/2374.0|
|4Hz|120|42.51/43.24/43.55|59.55/59.85|0|794.6/2374.0|

实测是每条验证录制首个边界附近的新鲜窗口，共2Hz60次、4Hz120次，warmup后按墙钟节拍运行，包含CPU窗口堆叠、H2D、预处理、编码器和双头分类；分类头P95约0.55/0.72ms。报告另存reserved显存。三模型对照头驻留GPU，生产单候选可略省资源。

观测年龄=离线重放调度等待+源帧向下取样量化+实测推理，4Hz量化通常约16.67ms；不包含真实摄像头采集、历史帧在线解码、网络或ROS。原始解码计时为离线逐录制处理，并非在线逐请求解码。不能把这里43ms称为机器人全链路延迟。

配对比较：4Hz K2在共同命中的49个Current事件上，中位检测提前0.267秒，但严格Future命中从40降38、Current抖动从0升5。4Hz K3在共同命中的50个Current事件上，中位检测时间差为0，平均减少0.107秒；Future修复3个漏检同时引入2个漏检，仅净增1。其原始四类Future误报较少，但实际有序Provider错误warning升至1.507次/分钟，提前Prompt从1升3。现有瓶颈不是GPU吞吐，不能只凭更快推理或41/56就切4Hz。

## 录制级、分类混淆与边界邻域

summary.json保存全部9方案的固定四类混淆矩阵、各类Precision/Recall/F1/支持数、分转移事件统计及接口录制级结果。neighborhoods.json给出边界前/后0.25、0.5、1秒分类质量；2Hz下0.25秒区间可能只有部分事件有样本，必须看支持数，不能当56个事件全覆盖。

|Episode|真实转换|2Hz独立1.5严格Future命中|未匹配事件|Current F1|Future1.5 F1|
|---|---:|---:|---:|---:|---:|
|3e645bf4e137d0f836c9d7ff|6|3|3|0.95276|0.90436|
|27d8bc0413a42341c7e8c3e5|6|5|1|0.98303|0.94786|
|0b12ac0fda9d7cd77dee6e60|6|3|5|0.98170|0.90757|
|61c420843f11ca54bfe804ee|6|6|0|0.98557|0.97573|
|5b82f4bb76e70d8bf12c0798|6|6|0|0.97034|0.96488|
|f53f0b6fb7f3888f87a0f8ff|6|3|3|0.96421|0.89062|
|ca784332e44d979f99018a52|2|2|0|0.72834|0.71643|
|a8b23394b642f0bb769a0391|6|5|1|0.99166|0.92829|
|6d46f3c8c5c61bd2c6d22973|6|2|9|0.97228|0.87823|
|d0f8727d35eced697dcf79fe|6|5|1|0.96338|0.91503|

6d46…贡献9/23个严格误报，不能只看所有窗口平均。ca784…只有三类，固定四类Macro-F1对缺失adjust记0，不能仅凭低Macro-F1认定它最差。

## 推荐配置与下一步

候选checkpoint：`/home/user/vjepa2_training/runs/tomato/temporal_final_20261108/selected_research_bundle/agent.pt`；SHA `7b9275832dc559ad5f9b17faa6380bb5b7571897bcee8588cf91ad5e3c260b00`。推荐配置保存于 `configs/tomato_transition_20261109/recommended_inference.json`，引用既有归一化与完整输入Schema及SHA。编码器/归一化/模型版本与旧候选相同，2,050,568个双头参数，303,885,312个冻结编码器参数。

研究事件分析0.5/K2与接口0.8/Current K2、Future单次预警分别保存，不能把0.5分析阈值写入Prompt接口。稳定默认Joint seed42不自动替换，本候选依旧research_only。没有调参以消除已发现错误。

下一步优先复核adjust→pickup边界语义、Current提前Prompt样本及Future滞后于Current的难例；保留原标注并记录合理时间区间。再用新的开发录制验证轻量预警时间校准/Current-Future一致性规则及困难负例，进行单因素试验。若提出持续状态式预警，应另立事件定义和消融，不能改口径把现有漏检抹掉。

新增录制计划见 `docs/TOMATO_NEW_RECORDINGS_20261109.md`：12条开发、20条封存独立测试，覆盖困难转换、遮挡、不同速度、重复/缺失阶段和边界模糊，目标独立测试≥100转换；模型/阈值/规则锁定之前不开放测试用于选择。未实际采集新数据，现有验证改善不能作为独立泛化证明。

## 检查、产物与复现

110项单元/回归测试通过；实际OpenPI回放、2898时刻数值/门限/事件一致性控制通过。旧产物SHA核查1197项、原标注SHA核查94项均未变化。训练数0，新增测试集推断0，硬件连接0。

最初绘图因训练环境缺少matplotlib失败；随后通过uv安装到独立 `/tmp/tomato-transition-plotdeps`，没有修改训练环境依赖。首次尝试pip不可用，也未安装pip进训练环境。最终媒体及全部比较曲线生成通过。绘图依赖固定在configs目录单独requirements文件。

完整本地产物：`/home/user/vjepa2_training/runs/tomato/transition_audit_20261109`。56个8秒左右三路视频、56个双头/速度曲线、56个三模型×两频率比较图、56个接触帧对照，另有23个未匹配事件的三路视频和23个概率/状态曲线。全部误报列表见 `docs/validation/tomato_transition_20261109/FALSE_EVENTS.md`；其中6d46…_fp8在262秒预警return、距实际转换还有8.1秒，Current仍稳定harvest，是远离边界的典型错误预警。视频可见主要剩余果梗，是否因外观相似或动作停顿触发仍需新数据验证。详见 [全部事件目录](/home/user/vjepa2_training/runs/tomato/transition_audit_20261109/EVENT_CATALOGUE.md)、[逐事件JSON](/home/user/vjepa2_training/runs/tomato/transition_audit_20261109/events.json)、[人工复核清单](/home/user/vjepa2_training/runs/tomato/transition_audit_20261109/human_review.json)。`nearby_false_events`按±3秒归属，可跨邻近事件重叠，不能相加替代唯一误报计数。

原始缓存、视频、权重不加入Git；版本化代码、冻结规则、推荐配置、事件证据和结果摘要。重新运行应使用独立工作根目录并锁定原manifest；不要覆盖旧训练目录。当前脚本默认本轮专用根目录，已有完整extract只校验并复用，不重写测量报告。

```bash
.venv-tomato/bin/python -m tools.audit_tomato_transitions extract
.venv-tomato/bin/python -m tools.audit_tomato_transitions analyze
PYTHONPATH=/tmp/tomato-transition-plotdeps .venv-tomato/bin/python -m tools.audit_tomato_transitions media
.venv-tomato/bin/python -m tools.report_tomato_transitions
PYTHONPATH=/tmp/tomato-transition-plotdeps .venv-tomato/bin/python -m tools.plot_tomato_transition_comparison
.venv-tomato/bin/python -m tools.compare_tomato_transition_rates
.venv-tomato/bin/python -m tools.check_tomato_transition_parity
PYTHONPATH=/tmp/tomato-transition-plotdeps .venv-tomato/bin/python -m tools.audit_tomato_false_events
.venv-tomato/bin/python -m tools.write_tomato_transition_report
```
