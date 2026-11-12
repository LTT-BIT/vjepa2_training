# B0冻结、独立评价入口与长时程链路离线验收

本轮完成实际代码、输入验证、真实编码器合成冒烟、三种子历史完整回放及OpenPI隔离接口验收。B0保持不变，没有训练、重算归一化或调阈值。当前仍100条完整七阶段录制：84训练/8开发/8历史测试。本轮没有新的真实独立录制入库，所有真实数据指标均来自已暴露的开发8条。

## 1. 正式锁与仓库状态

开发起点`research-tomato-agent / 561c28a`，工作区干净。正式机器锁为`configs/tomato_frozen_independent_20261112/b0.lock.v3.json`，SHA256：

`f8ba41f61224756b1531cf5a39dde57846b6aca92bb9a6a1c3ab1ab3c5f237d8`

同目录v1/v2是构建过程的准备锁，保留不覆盖；正式v3补全流水线源码与接口源码夹具，模型、数据、阈值和Future/Prompt匹配规则相同，尚未用于任何新真实独立集合。新命令默认v3。

|种子|独立双头checkpoint|SHA256|
|---|---|---|
|42，固定代表种子|runs/tomato/optimization_20261108/bundles/independent_h15_s42.pt|7b9275832dc559ad5f9b17faa6380bb5b7571897bcee8588cf91ad5e3c260b00|
|43，稳定性复验|runs/tomato/optimization_20261108/bundles/independent_h15_s43.pt|794a42e17dad0b4e6f028c39497b029f56f6aa3f410ed40eb72accec89a2bea6|
|44，稳定性复验|runs/tomato/optimization_20261108/bundles/independent_h15_s44.pt|b5989cddcad1ee1c391723660a547060cbb97efc8cdb44553ee8894194014362|

正式锁包含三模型的完整归一化统计身份SHA、原训练split SHA、四分类顺序、58个实测关节名称、三相机、16帧×10Hz严格因果历史、官方编码器身份、代码逐文件SHA，以及100条逐录制身份、原标注SHA和有效边界版本SHA。当前100条manifest SHA为`95679c2f4457701c3898e538b83883c27106aa3aa44c647ad046b92dd20410e1`。历史14项临时验证边界提案保持原身份，不描述成新批准标签。

B0仍用其原训练checkpoint与归一化；100条84/8/8是当前研究清单，不能据此声称B0已重新用84条训练。模型/处理代码/原标注发生变化时verify会拒绝，不能静默换权重。主指标、代表seed42和误报统计不从新集合重新选择。

冻结口径：Future1.5秒=30Hz后的45帧；Agent2Hz；严格Future0.5/K2，首次稳定标签只初始化；同from/to一对一`[b−1.5,b)`，无未来尾部标签则屏蔽。Current Prompt为0.8/K2、有序七位置、显式从pickup_1初始化；按序号计提前、按时`[b,min(b+2,end))`、迟到、缺失。观测时间是原时间轴，不将第二次确认回填第一次候选。

## 2. 新独立录制入口已实际实现

一条命令导入、验证、编码和回放：

```bash
cd /home/user/vjepa2_training
.venv-tomato/bin/python -m tools.run_tomato_frozen_evaluation \
  --index /ABSOLUTE/PATH/new_catalog.json \
  --out runs/tomato/new_independent_COLLECTION_ID
```

输入示例与schema位于正式锁同目录；采集、标注及使用说明见`docs/TOMATO_NEW_INDEPENDENT_RECORDINGS_20261112.md`。可用`--validate-only`只做输入检查。输出目录不覆盖，原数据只读。

入口不仅检查source ID，还检查Parquet/视频内容哈希、原始目录隔离和重复输入。历史隔离表包含104个曾暴露source ID及416个内容哈希：有效集合仍100条，多出的隔离身份是已剔除或历史暴露样本，不是新增有效数据。全部候选通过完整三相机解码、PTS/状态时间轴、命名关节及已发布七段原Subtask检查后，才开始评分，不自动删掉验证失败项来跑剩余易例。

真实集合使用固定全局登记`runs/tomato/frozen_independent_registry_v1`；无训练、校准、阈值或持续时间先验拟合入口。失败评价保留登记和现场，不自动释放为“未见”。哈希不能识别所有重编码或虚假的来源声明，需要保留真实采集来源；任何改用训练/开发都会取消独立资格。

合成输入只能显式进入smoke模式，输出标为`synthetic_smoke_only`，使用独立临时登记。已实际将120帧×三路合成MP4、真实Parquet和七段测试标签走完完整入口，调用本地官方冻结V-JEPA2，得到8×4000特征、严格因果索引，再运行三种子双头及全部Agent评分/Memory记录。此测试验证软件路径与张量/身份约束，不作为真实数据质量或实机性能成绩。证明文件见验证目录的`synthetic_gpu_smoke.json`。

## 3. 已暴露开发8条上的完整Agent结果

|Seed|Current Macro-F1|Future Macro-F1|严格Future命中|误报/分钟|七位置准确率|状态不一致秒|
|---|---:|---:|---:|---:|---:|---:|
|42|0.971598|0.937368|37/48|0.695|0.966064|38.5|
|43|0.966352|0.928515|34/48|0.962|0.959894|45.5|
|44|0.969552|0.938317|35/48|0.802|0.968709|35.5|

|Seed|Prompt按时/提前/迟到|六次Prompt全按时的Episode|走完有序状态序列|>2秒持续失配累计|
|---|---|---|---|---:|
|42|44/2/2|5/8|8/8|10.5s|
|43|44/2/2|4/8|8/8|13.0s|
|44|43/4/1|5/8|8/8|8.0s|

走完状态序列不等于顺利完成动作：首个指标只检查`0→1→2→3→4→5→6`，并不要求每次按时，也不意味着当前Agent驱动的VLA完成了物理任务。未连接机器人、未运行π₀.₅动作模型，没有实机成功率。

本表Future提前量和Prompt时间沿用**观测参考时刻**的离线口径，不是消息到达或硬件应用时刻的真实反应时延。回放中的后续录像也不会因错误Prompt改变，所以8/8到达末位置不能外推为闭环完成率。

四类各自Precision/Recall、7×7状态矩阵、重复pickup_1/pickup_2、harvest_1/harvest_2、return_truss/return_stem的独立统计均保存在`episode_scores.json`及完整本地逐Episode结果中。位置来自在线历史维护，不用真实阶段纠正或按argmax重置。

### Future漏检究竟影响了多少Prompt

|Seed|Future漏检|其中Prompt仍按时|同时Prompt提前/迟到|
|---|---:|---:|---|
|42|11|8|1/2|
|43|14|10|2/2|
|44|13|11|1/1|

三个种子合计38个漏检实例中29个Prompt按时、9个同时提前或迟到；这是同一48事件在三种子下的重复，不是144个独立物理事件。9个共现不能解释成Future导致Prompt错误。

在真实概率回放中将Future替换为均匀分布，6,807次观测的Prompt、七阶段位置、版本和文本完全不变：**锁定桥接接口的Prompt由Current确认驱动，Future错误的直接因果影响为0**。Future命中的106个实例中也有4次Prompt提前，进一步说明两类指标必须分开。

### 级联计数的修正解释

旧汇总把持续失配区间中经过的真实边界都叫“级联”。三个种子各有1例其实是`3e645bf4...`提前进入pickup_2后，自己的adjust→pickup边界随后到来，并未继续跨过下一个真实阶段边界。

新结果保留旧计数`legacy_boundaries_crossed_during_persistent_mismatch=1`，另外以失配起点的共同阶段前缀推断根边界，单列其后的边界重叠；本次三个种子均为0。它只是离线时间归因，不证明系统不会级联，也不能把单个较长失配自动说成后续多阶段错误。Future/Prompt既有评分未改。

## 4. OpenPI隔离联调：通过部分与明确缺口

本地OpenPI主仓库为`tomato-subtask-policy / b9e367f95376690d383851962a8a50d1c3a6205d`，初次核对工作区干净；交付核验时出现了本轮之外的ROI脚本、测试和配置改动，原样保留且不纳入本轮提交。原`/home/user/tomato-openpi-agent`不在本地，当前checkout没有`src/openpi/subtasks/agent_bridge.py`。没有切换或修改它。

使用已保存Git对象`e5036843ac26e612e78bf8a289e2c6145b15d433`中的三个原始接口文件进行隔离回放；原字节归档在配置目录，锁定哈希，避免以后Git对象回收影响复现。它是只读测试夹具，不是把Agent装进当前OpenPI，更没有更换默认模型。

24个Episode×seed回放共6,807次观测：

- Current七阶段及Prompt版本与B0逐时刻一致，Future反事实不改Prompt。
- 144次阶段变化，旧Chunk的Prompt/阶段/模型/会话版本比对均不匹配，Mock拒绝误用；重复pickup即使文本相同仍以不同有序位置和版本区分。
- 原始观测58D按真实关节名转至OpenPI顺序，再经现有适配器转回Agent顺序；168个起点/切换历史窗口逐值一致，没有仅凭维数推断兼容。
- 48个注入的过期或错误模型消息被拒绝；单元测试另覆盖请求时间/observation ID、非法Chunk、重复结果、同Prompt过期结果及Mock安全重规划边界。
- 记录每次请求参考时刻、模拟完成/到达时刻、PromptContext、Chunk绑定、Memory版本及完整事件日志。模拟时延0.04/0.05秒只是测试安排，不是现场测量。

Mock以50步Chunk/15步前缀复现历史研究用法，只做元数据准入与数值连续性测量，没有调用真实执行或修改RTC。故意构造相邻Chunk起点差0.1的测试，正确报告跳变而不改动作值：版本正确并不能保证VLA动作连续。另记录切换时的实测关节差，来源是旧录像中的状态，不是新策略闭环输出。实际π₀.₅动作连续性及实机成功率未验证。

严格Future0.5/K2事件、bridge单帧0.8显示提示和Memory的候选记录是不同字段，不能混为同一预警指标。所有七位置/四类英文Prompt映射均来自锁定源码。

### 可复现的传输层缺口

原bridge最大年龄2秒，而现有TaskMemory M1仅接受年龄≤1秒。独立诊断实际复现：

|模拟年龄|bridge|Shadow|阶段是否一致|
|---|---|---|---|
|0.05s|接受两次Current证据并确认|接受并记录|一致|
|1.5s|接受并确认|拒绝迟到观测|不一致|
|2.5s|拒绝|拒绝|均保持原阶段|

这是接口/观测准入约束不一致，不是VLA执行误差。没有偷偷收紧B0或放宽Memory；本轮保持默认。未来接通真实链路前，需显式统一观测年龄策略，或把“控制器已接受的阶段事实”与“过期感知证据”分开记账，并回放完整延迟序列。当前只可宣称正常新鲜回放时Shadow忠实，不可宣称任意传输延迟都可靠。

## 5. Memory与长时程瓶颈

正常回放下，Shadow的6,807次状态快照与B0一致，24条有序确认前缀均可从事件日志重建；版本重复/乱序、过期、重复类别跳阶段有测试约束。Memory不写Prompt、不自动确认操作成功，也未显示学习性能增益。完整日志与紧凑确认历史都保留，不因FIFO截断丢掉六次确认前缀。

这是对**算法决策历史的忠实记录**，不是对真实阶段的自动纠错。若B0早判，Memory也会忠实记录该早判。离线脚本在Episode结束落盘，未做进程崩溃/断电持久性验收；1～2秒迟到还有上述明确不一致。原先操作结果补标保持暂停，不训练新的成功验证器。

当前已观察到的Agent问题是Current过早/过晚、高置信度证据与人工边界错位，以及七阶段确认等待；不能将它们归咎于RTC。相反，物体是否抓稳、动作误差、接触失败或策略输出跳变，需要新Agent+VLA实际闭环轨迹，本轮没有证据分配这些执行责任。

下一轮按新采集说明锁定后采16～20条独立录制，覆盖不同速度、较长adjust、pickup_2前停顿与实际连续变化。用原Subtask标准，不重做旧100条。若Current在新数据同样分错才研究表征；若当前识别对而未来启动不具可观测先兆，研究转换监督/不确定性；若证据对而版本或历史错，先修决策和传输；若只有真实执行失败再研究VLA。Time Head、S1/S2和Memory保留研究身份，没有证据自动替代B0或扩大神经记忆。

## 6. 产物与验证

新增工具：`tomato_frozen_protocol.py`、`tomato_independent_import.py`、`run_tomato_frozen_evaluation.py`、`tomato_full_chain.py`、`replay_tomato_frozen_history.py`、`tomato_offline_contract.py`、`check_tomato_frozen_openpi.py`、`check_tomato_transport_age.py`。均在V-JEPA仓库，不改OpenPI工作区。

273项番茄Python测试通过（包含真实GPU编码器合成冒烟）；模型或处理代码字节变化、重复/更名旧数据、无效Subtask、PTS/状态错位、向原始目录写输出、改用空登记绕过独立身份等均有检查或回归测试。模型和100条原标注哈希最终复核不变。环境版本记录在验证目录的environment.json。

- 历史完整结果、24份逐Episode Markdown/JSON、概率和Memory JSONL：`runs/tomato/frozen_chain_20261112_release`。
- OpenPI隔离接口与Chunk时间线、完整Memory日志：`runs/tomato/frozen_openpi_20261112_release`。
- 延迟边界复现：`runs/tomato/frozen_transport_age_20261112_v1`。
- 验证摘要及每Episode指标：`docs/validation/tomato_frozen_evaluation_20261112`。
- 原始数据、模型权重、完整运行缓存不进Git；只提交代码、锁定身份、被校验的接口夹具、摘要及说明。

复现：

```bash
cd /home/user/vjepa2_training
.venv-tomato/bin/python -m tools.replay_tomato_frozen_history --out runs/tomato/NEW_CHAIN
.venv-tomato/bin/python -m tools.check_tomato_frozen_openpi --out runs/tomato/NEW_BRIDGE
.venv-tomato/bin/python -m tools.check_tomato_transport_age --out runs/tomato/NEW_AGE
TOMATO_FROZEN_GPU_SMOKE=1 .venv-tomato/bin/python -m pytest -q tests/test_tomato*.py
```

本轮达到“锁定并可复现离线评价”的目标；没有新的独立泛化结论，没有部署、训练或实机成功声明。
