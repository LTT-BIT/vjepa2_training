# 独立操作证据与记忆接口：实际结果（2026-11-10）

本轮完成P0–P5的代码、独立审核工具、训练入口和跨仓库离线联调。**没有训练出新的操作结果模型，没有证明Memory改善阶段或动作表现。** 原因是101条录制中没有已注册、独立审核的操作结果监督；不能把阶段标签改名当成功标签。B0继续作为研究基线，OpenPI原Joint配置保留，Memory默认关闭。

2026-11-11后续选择：用户要求完整任务实验排除唯一三阶段录制，当前100条为84训练/8验证/8历史测试；选择身份见`configs/tomato_complete100_20261111/selection.json`。本文以下101条审计与39/50回放保留为历史结果，不代表新100条选择的重评结果。审核页面切换为92条训练/验证录制、736候选，其他已保存意见按原身份迁移；原数据、标签与checkpoint不删除。

## 版本与边界

- Agent：`/home/user/vjepa2_training`，`research-tomato-agent`，起点`686f68a`。本轮修改未覆盖历史模型、数据、划分或标注。
- OpenPI主目录由其他工作使用，实际处于`tomato-subtasks-ab-train`，本轮未修改或切换该目录。独立工作树`/home/user/tomato-openpi-agent`使用指定`tomato-coarse-subtasks-agent`，起点`33ce417`。
- 用户最后明确授权合理修改自动保存、提交、推送，覆盖附件较早的逐次确认条款。Agent推送origin同名分支；OpenPI推送github同名分支，不推GitLab origin、不强推。
- 没有机器人连接、RTC改动、Action Expert替换、正式GPU训练或大型权重下载。实际交付提交由Git历史与`docs/validation/tomato_operation_memory_20261110/receipt.json`源码SHA核对。

## P0：原TaskMemory为何未增益

核对`docs/TOMATO_TASK_MEMORY_RESULTS_20261110.md`、原回放产物及本轮真实缓存重放：B0为39/50，13次未匹配预警，0.67445次/分钟，Prompt按时46/50，提前2、延迟2，状态不一致40秒，adjust→pickup为2/8。原M2正常回放与B0相同。

原账本主要重复分类与七位置确认历史，缺少独立物体结果。M1仅Shadow；M2虽然检查历史约束，没有新物理锚点，因此错误前缀仍可传播。原8/9冲突检测和连续6次错误分类是注入故障诊断，不能当物理结果验证准确率；连续6次错误仍会误确认。原CPU P95约0.068–0.078ms是旧记忆增量，不是完整编码器链路时延。

## P1/P2：101条真实数据审计与独立审核

冻结清单：`runs/tomato/refresh101_20261109_v2/finalized_exports/refresh101/dataset_manifest.json`，SHA256 `d268001110c737e77bdd511f86ee1651b9d669798c166249938c1bc7addc9880`。

|核查项|实际结果|
|---|---|
|录制及划分|101个不同source ID；84训练、9开发验证、8历史测试；100条七阶段、1条三阶段|
|相机|三路RGB cam0/1/2；存档元数据未证明物理相机角色，暂不擅自映射左右腕|
|状态|原始116D：58位置+58速度；现有Agent按关节名称选58D位置，不可混同为116D策略输入|
|动作|58D目标指令；不能当实际执行成功反馈|
|时间|30Hz、timestamp、source_timestamp_ns；审核按原始帧号，证据可用时间独立记录|
|其他字段|含arm_engaged，不代表抓取成功；未记录触觉、深度、力、独立抓握结果字段|
|在线信号|既有接口支持RGB/位置，速度可由因果差分构造；真实硬件其他信号可用性本轮未核实|
|独立结果标签|已注册批准标签0，未发现可直接作为结果监督的归档标注；原coarse明确仅表示阶段|

审计只读所有录制的Schema/元数据；历史测试未读数值用于新模型或规则选择。训练/开发93条生成741个**未审核候选**，阶段仅用于定位。测试8条不进入审核选模队列。候选不穷尽每次摘取，不能据此计算摘取数。

独立规范见`docs/TOMATO_OPERATION_LABEL_STANDARD_V1.md`。八类事件区分开始尝试与成功结果；可见失败为negative、遮挡不可判定为Unknown，未审核不产生训练标签。两位审核者独立导出后，`merge_reviews`只批准事件、区间、证据一致的意见；不自动平均争议边界，不改原coarse标签。

本地审核页面：`http://127.0.0.1:8770/`。三路原视频、同步播放、逐帧定位、开始/结束/证据可用帧、独立证据说明、正/负/Unknown、个人本地保存与导出均已验证。服务仅loopback，媒体路径按目录白名单，支持Range。原8768页面不变。此次没有代替人填写任何审核。

**另外发现跨项目数据问题：** OpenPI主工作目录`normal93_20261109_selection.json`的93条中，74/8/8条分别覆盖Agent训练/验证/历史测试，另3条来自已被Agent剔除的`B/1101_572_601_normal_lerobot`（episode 0、1、2）。清单SHA `fcf1a1b9e4db11d2b187f95c119711a709ad44debcb38d094fbde23109f1fc2c`。未修改该旧清单或AB checkpoint；今后不能把Agent验证/测试直接称作该动作模型的独立留出集，Memory动作实验须重新冻结质量清单和来源分组。

## P3/P4：操作验证器与有来源的记忆

`tomato_operation_memory.py`新增EvidenceRecord及OperationMemory，四种kind独立：观测、阶段预测、外部报告执行指令、已验证操作结果。提供append/update/confirm/reject/expire/snapshot/reset；完整Episode重放由sidecar逐帧更新。

只有独立双人审核或显式注册、已合格的结果验证器才能确认物理结果。当前合格验证器注册表为空，阶段/命令不能提升为成功。记录来源、观测、置信度、有效期、冲突、版本、更新原因；重复提交幂等，重复确认拒绝。冲突保留原确认记录，但对外暂不输出争议事实。当前物体状态超时变Unknown，过去结果仍可保留。picked_count恒为Unknown，不虚构Tomato ID。未接入执行回执时command_history_status为not_observed。

`tomato_outcome_verifier.py`与`train_tomato_outcome_verifier.py`实现冻结缓存、小MLP、八输出、Unknown掩码BCE及同预算消融入口：

|组|输入维度|含义|
|---|---:|---|
|A|928|已有16×58因果机器人历史|
|B|3072|已有三相机池化视觉特征|
|C|4000|视觉+机器人历史|
|D|8000|C及同Episode前一特征差分，不跨Episode、不读未来|

学习率0.001、10epochs、batch256，归一化仅拟合有监督训练窗口；验证用于开发，拒绝历史测试监督。初始Smoke门槛是某事件至少10正、10负且跨6录制，**不是最终充分性保证**。已实际调用A组训练入口，真实标签0，保存`not_trained/insufficient_independent_outcome_labels`后停止；没有虚构训练Loss。CPU测试使用合成张量验证前后向、Unknown掩码和研究checkpoint加载，不计为真实数据训练。

训练产物即使存在也标为未校准研究候选，不能自动确认结果；仍需独立校准、事件级评估后另行注册。当前验证器八类均返回概率null、insufficient_evidence。

## P5：OpenPI可选Memory接口

新增`src/openpi/subtasks/memory_bridge.py`，两个仓库共享`tomato-memory-input-v1`Schema。none模式返回原PromptContext对象；oracle只接受独立审核的过去物理事实，predicted要求操作验证器来源。当前七阶段仍来自在线B0，不注入人工真实位置。

- 保留四类原英文Prompt及两个return共用文本；追加紧凑事实，不覆盖原指令。
- 训练与推理共用condition_prompt；训练适配器校验Episode、观测、**精确时间戳**、阶段及过去证据。没有自动接入旧训练流水线，默认无记忆样本为原样复制。
- 用实际SentencePiece计算完整Task+Memory+58D State+Action输入，优先近期完整事实；放不下明确记录丢弃项，不截断状态/原指令、不偷偷扩到512。
- 同语义不增加Prompt版本；语义改变有新digest/cache key；当前sample_actions每次原本就重建prefix KV，本轮未虚称改了运行时KV缓存。
- 新Chunk携带版本、digest及原始有效期。刷新相同内容不能延长旧Chunk有效期；只在调用方既有安全重规划边界检查，不改变执行中的Chunk/RTC队列。
- 过期、未来、乱序、跨Session/Episode、非法来源、重复事件拒绝使用。Memory配置默认关闭，不注册为可直接正式训练的recipe。

实际官方tokenizer SHA `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`。四种阶段原Prompt最坏token分别254、255、253、252。257种均匀状态bin及每Prompt200组混合状态，共1828组：最大256；1204组保留至少一个事实、624组全部记忆省略。**这是人工状态bin预算分布，不是真实录制省略率。** 256预算存在严重容量限制，不能据接口完成就断言记忆可被模型有效利用。

## P6–P8：实际联调与未完成实验

`replay_tomato_operation_memory.py`用9条清理后验证录制、2340个真实已保存Current/Future预测，经过真实OpenPI PredictionPromptProvider、OperationSidecar及none/oracle/predicted接口逐条回放。延用14项复核提案下的历史比较边界，不将提案冒充新批准原始标签。预警一对一匹配窗口为[边界−1.5s,边界)，Prompt确认窗口为[边界,min(边界+2s,录制结束))，与原39/50口径一致。

|结果|B0与本轮|
|---|---:|
|严格事件预警|39/50，78%；Precision75%，F1约76.47%|
|未匹配预警|13，0.67445次/分钟|
|Prompt按时/提前/延迟|46/2/2|
|状态不一致|40秒|
|adjust→pickup|2/8|
|本轮额外物理事实|0，均Unknown|
|相对B0改进|零；未替换默认配置|

三种模式均未改变原分类、阶段、Prompt文本和版本。**空Oracle只是无新增信息的接口回归，不是Oracle有效性实验；不能据三者相同断言预测记忆与Oracle无差距。** Tokenizer联调使用显式58个零bin样例，不是实际归一化策略状态。sidecar CPU P50/P95/最大时间见receipt，属于缓存后增量，不含视频、GPU编码器和动作推理，不是完整2Hz验收。

218项番茄专项测试、OpenPI43项相关测试通过。覆盖物理证据分离、冲突、TTL、Reset、重复、因果、训练/推理同构、过期Chunk、跨Episode及No Memory回归。UI JavaScript语法检查与真实浏览器三路视频/逐帧检查通过。未宣称全仓所有旧任务测试通过，也未运行完整JAX动作策略。

E0/E1/E2动作效果、E3记忆条件微调、Action MAE/RMSE、错误记忆动作传播、真实物理结果准确率、真实冲突Precision/Recall均**未测**。没有可信结果标签、没有独立Oracle增量，本轮不以伪造监督启动GPU或π₀.₅微调。

## 七个问题与下一步

1. **Memory有无真正新增信息？** Schema和独立证据通道具备新增能力，当前填入的真实独立结果为0，尚未证明新增判别信息。
2. **三相机和58D能否验证结果？** 存在视觉与运动线索，关节闭合/抬臂不等于成功。需要独立审核和A/B/C/D验证，当前不能报告可靠性。
3. **Oracle是否改善离线表现？** 尚不可测；先获得独立过去结果，排除仅重复阶段的“Oracle”。
4. **Predicted与Oracle差距？** 未知，不能填0；当前没有合格结果模型。
5. **π₀.₅能否利用历史？** 尚未验证；接口、预算、版本可用不等于模型学会利用。需要相同初始化/数据/步数E0/E3及合法历史交换对照。
6. **真实与模拟边界？** 真实101录制元数据审计、真实2340缓存概率跨仓回放、真实tokenizer；训练前后向、故障/过期/重复等为CPU合成测试。动作模型调用0、真实新训练0、硬件执行0。
7. **是否完整记忆微调？** 暂不值得直接启动。先独立审核训练/开发录制（覆盖成功、空抓、失败摘取、遮挡、重抓与放回），评估一致性与可观察性；再小预算A/B/C/D、Oracle增量与token容量实验。动作实验须重新排除质量问题并冻结来源分组，锁规则后另录独立盲测。保持B0与Joint默认，不预设Memory必有收益。

## 最小复现

```bash
cd /home/user/vjepa2_training
.venv-tomato/bin/python -m pytest -q tests/test_*tomato*.py
.venv-tomato/bin/python -m tools.prepare_tomato_operation_review --out runs/tomato/new_review_unique
.venv-tomato/bin/python -m tools.serve_tomato_operation_review --directory runs/tomato/new_review_unique --port 8771
.venv-tomato/bin/python -m tools.train_tomato_outcome_verifier --labels runs/tomato/operation_memory_20261110/approved.json --mode A --out runs/tomato/new_readiness_unique
PYTHONPATH=/tmp/tomato-memory-tokenizer-deps .venv-tomato/bin/python -m tools.replay_tomato_operation_memory --out runs/tomato/new_memory_replay_unique --openpi /home/user/tomato-openpi-agent --tokenizer /home/user/.cache/openpi/big_vision/paligemma_tokenizer.model
cd /home/user/tomato-openpi-agent
PYTHONPATH=src:tests /home/user/vjepa2_training/.venv-tomato/bin/python -m pytest -q tests/test_tomato_agent_bridge.py tests/test_tomato_memory_bridge.py tests/test_tomato_subtasks.py
```

SentencePiece本轮仅装入`/tmp/tomato-memory-tokenizer-deps`隔离目录，未改既有训练环境。复现其他机器须自备同版本依赖和同SHA本地数据/tokenizer，输出目录必须全新。完整本地缓存位于`runs/tomato/operation_memory_20261110`，Git仅保存源码、Schema、配置、规范及可核对摘要，不上传视频、标签工作目录、权重或完整缓存。
