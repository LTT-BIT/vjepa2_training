# 双头握手决策受控实验报告（V2，2026-11-12）

实验：`/home/user/pi05/vjepa2_training/runs/tomato/learned_handshake_v2_20261112_212907_36ff73`。基线 `db4cdad6e87af7c077f5be16a00bc277efeb54d9`；分支 `feat/tomato-learned-handshake-v2-20261112`。未提交/推送，原 GUI/分类器/握手未修改。

**验证集锁定结论：KEEP_H0。** 选型先看非法实际commit、硬超时、过早、漏切及额外提交，再看严格命中、Wrong Active与完整顺序。不能以Wrong Active下降宣称更安全。

## 结论解读：当前优先保留 H0，R1作为明确取舍候选

本轮要选的是两个冻结head之间的决策机制，不是重新选分类器。按预注册“窄窗提交优先且不增加过早/硬超时/漏切/额外提交”的条件，结论为 KEEP_H0。没有学习器同时满足准入约束，也没有用历史test反过来挑网络或阈值。

|机制|窄窗命中（54边界）|Wrong Active(s)|过早≤-.5s|漏切|未匹配commit|硬超时|
|---|---:|---:|---:|---:|---:|---:|
|H0|23|98.233|4|11|11|3|
|R1|21|76.300|3|9|9|2|
|R2|23|98.233|4|11|11|3|
|L0（三seed均值）|24.33|94.600|5.00|10.67|10.67|2.67|
|L1a（三seed均值）|24.00|95.878|7.67|11.00|11.00|3.00|
|L1b（三seed均值）|9.33|602.178|9.67|27.67|12.67|6.33|
|L2（三seed均值）|8.67|610.178|9.33|29.33|13.67|6.33|
|L3a（三seed均值）|20.33|352.444|6.33|19.33|11.00|3.00|

**R1的取舍最清楚**：错误Active减少21.933秒（约22.3%），明显过早4→3、硬超时3→2、漏切/额外提交各11→9，但窄窗正确23→21。它不是全面优于H0，也不是因有些指标下降就完全无用；若下一轮用户愿意用两次窄窗命中交换这些收益，应另立评价目标和独立验证，本轮不改默认。R2在这些93数据的val回放中没有增加收益。

**学习器中线性L0最值得保留为研究对照**：三seed各24/24/25窄窗，Wrong略降，但明显过早均5（H0为4），因此全部被既定风险条件拒绝。不能为了采用线性网络而事后容忍多一次过早。L1a门控窄窗24，但过早7/8/8；同时学习目标的L1b和GRU更差，不能把监督loss较低当作Active更好。

**显式时间hazard没有在本轮胜出**：L3a seed43为26个窄窗，但Wrong Active297.767秒、漏切16、过早5；seed44仅13个窄窗。L3a与L1b同MLP干线，输出头仅多165参数，不能把时序差异解释为更大网络。CDE质量阈值消融也没有改善，结果完整保留。

三种网络seed与9来源都不是新增独立数据。本轮看到的负结果仅适用于这些固定输入、监督、预算及执行外壳；不能推出任何可学习策略都无效。


## 实验身份与比较含义

93条原normal来源，75/9/9固定划分。仅原B0 Joint seed42双头概率；三种子只改变策略网络训练，绝不混用上游三种子。三路完整RGB/58D/+1s/2Hz契约不变，本轮不解码/编码视频，也未训练任何视觉、Memory或Joint teacher。五折OOF复用上一轮已验证54 train/6 inner-val/15 held教师，val/test为固定全75训练B0。

主概率使用上一轮B0-frozen保存值，来自同一个原B0 checkpoint；其先前与原分split分批概率最大浮点误差5.64e-7。本轮H0与这些历史事件、时间戳、Active逐值相等；val必须复现23/54窄窗与98.233秒。任何性能结果均由本轮实际回放产生。

H0原码直接调用；R1/R2及学习器复用原Handshake的ACK、timer和planning，通过独立REQUEST注入点替换候选逻辑。ACK输入的Future暂时中和为Active以抑制旧候选旁路，随后恢复真实Evidence；不改变原概率源或Current ACK判断。任务mask是外部四类图，单测覆盖非NEXT图和空图。Active只能原planner原子提交，阻塞/TTL/身份约束全部保留。

## 验证集主表（9来源、54边界）

|策略|窄窗|宽窗|漏切|未匹配commit|提前≤-.5s|晚于.5s|Wrong Active(s)|顺序正确|硬超时|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|H0|23/54|43|11|11|4|12|98.233|9/9|3|
|R1|21/54|45|9|9|3|19|76.300|9/9|2|
|R2|23/54|43|11|11|4|12|98.233|9/9|3|
|L0_s42|24/54|43|11|11|5|10|96.767|9/9|3|
|L0_s43|24/54|43|11|11|5|10|96.767|9/9|3|
|L0_s44|25/54|44|10|10|5|10|90.267|9/9|2|
|L1a_s42|24/54|43|11|11|7|9|95.767|9/9|3|
|L1a_s43|24/54|43|11|11|8|8|95.933|9/9|3|
|L1a_s44|24/54|43|11|11|8|8|95.933|9/9|3|
|L1b_s42|9/54|28|26|13|10|3|547.700|3/9|6|
|L1b_s43|9/54|25|29|14|9|2|581.933|3/9|6|
|L1b_s44|10/54|26|28|11|10|1|676.900|2/9|7|
|L2_s42|9/54|25|29|14|9|2|566.433|3/9|6|
|L2_s43|8/54|25|29|14|10|2|582.200|3/9|6|
|L2_s44|9/54|24|30|13|9|1|681.900|2/9|7|
|L3a_s42|22/54|38|16|12|8|4|263.500|7/9|4|
|L3a_s42_mass|14/54|27|27|18|7|2|388.200|2/9|6|
|L3a_s43|26/54|38|16|11|5|5|297.767|7/9|3|
|L3a_s43_mass|9/54|27|27|14|10|2|542.967|4/9|6|
|L3a_s44|13/54|28|26|10|6|6|496.067|3/9|2|
|L3a_s44_mass|9/54|15|39|20|4|0|738.533|2/9|9|

提前/迟到仅按宽窗[-1,+2]一对一真实from/to匹配事件计数，窄窗是其中[-.2,+.5]的子集；未匹配提交单列。Wrong Active以完整Episode与GT四粗类交叠积分，逐Episode等权比例另见CSV。三个seed不是独立来源。

## 模型、训练与状态分布

L0为56维显式L2 Logistic（BCE+.5*.01*||W||²、不惩罚bias，优化器decay0）；L1a为候选MLP门控；L1b为92→64→32→5动作；L2为masked GRU32加相同状态后小读出；L3a与L1b同MLP trunk，next4/time6输出。L0/L1a仅候选受Future≥.65限定；L1b/L2/L3a可主动提出目标，不在旧H0已请求之后才veto。

模型AdamW 3e-4（L0用显式L2、optimizer decay0；其余WD.01）、batch256、最多80epoch、patience10、dropout.1、clip1；train-only归一化和事件/来源/时间桶加权。checkpoint仅按固定val-H0状态监督损失早停，之后才进行自身Active完整回放。自回放训练状态若相对H0 Active时间偏移>.10，只做一次预锁定train-only轨迹聚合，再同seed重训；触发则直接使用v1，不按val Active挑v0/v1。

## 核心问题逐项回答

1. 固定B0输入，最终按风险条件选择 KEEP_H0；完整准入审计在selection.lock.json。所有分类Current/Future F1保持完全一致，变化只来自决策。
2. 简单规则：R1窄窗21、Wrong 76.300s；R2窄窗23、Wrong 98.233s。相对H0须连同表中过早/硬超时判断，不能因神经网络更复杂而优先选择。
3. L1a只学时机：三seed窄窗均值24.00，Wrong 95.878s；L1b同时选目标：9.33、602.178s。非法目标提议被外部mask拒绝，真实非法commit须为0，拒绝率和动作混淆另列。
4. L2 GRU窄窗均值8.67、Wrong 610.178s，对照L1b；只读4个2Hz概率锚点（跨度1.5s），没有读取原视觉/关节Memory，也没有验证更长上下文。
5. L3a时间hazard窄窗均值20.33、Wrong 352.444s。同backbone L1b对照；time概率好看不代表commit更准。*_mass仅为锁定CDE质量映射消融，不参与主选型。
6. 归因按policy_proposal→accepted request→commit→ACK保存：提前误切需同时看原Future、提议时刻与planner边界；漏切需区分HOLD、task_mask、ACK/RECOVERY阻断。summary/policy_proposals_and_commit_attribution.jsonl与casebook保留原始证据；不凭GT复位。
7. K1仅在本轮主策略锁定后，以现有seed42 Future做固定输入替换；结果或NOT_RUN理由见summary/k1_future_swap.json，不用它再选策略。
8. 参数与单次CPU P50/P95/P99见latency.json；只加载一个被选择策略，不是集成15个网络。Encoder+π0.5并发、实机物理完成均NOT_MEASURED/NOT_EVALUABLE。离线回放不是机器人闭环测试；若风险未通过，不建议改变默认进入控制执行。

## 监督、hazard与边界不确定性

监督在实际下一planner提交tick计算，严格窗commit−GT∈[-.2,+.5]；已迟到但GT当前仍为合法候选时保留late_catchup正例。hazard A/B/C/D/E/F严格按协议分桶；迟到超过1秒但仍可追赶的样本time loss被mask，行动监督仍为正，绝不冒充F负例。末尾下一个边界右删失时屏蔽next/time；commit tick仍有当前GT则保留HOLD行动监督；录制范围外没有监督，不填F负例，更不代表物理FINISH。

L3固定cost：switch=[4,3,2,0,.3,4]，hold=[0,.25,1,3,4,0]，next≥.5且期望switch<hold才请求。没有依据val/test调整cost。time混淆、Brier/ECE、near-boundary MAE在hazard_time_bins.json；纯监督分数不能代替Active评价。

只对train标签做±1帧和±.1秒扰动审计，变化数量见train_labels_audit.json；原标注未修改。OOF与全75训练概率质量、置信度直方图、分歧、边界附近质量见teacher_quality.json；teacher质量/分布漂移可能限制learned策略，不把负结果解释为神经策略普遍无效。

## 历史保留集诊断

val选型和权重SHA锁定后，仅H0与唯一候选评估。旧test已多次曝光，不是新盲测，未用于更改架构、阈值、代价或默认策略。

|策略|窄窗|漏切|未匹配|提前≤-.5s|Wrong(s)|硬超时|
|---|---:|---:|---:|---:|---:|---:|
|H0|26/54|8|8|4|47.233|1|

## 跨来源不确定性、最坏情况和局限

5000次来源聚类配对bootstrap固定seed；val只有9来源且被多轮观察，区间只是描述，不构成显著泛化/安全认证。所有seed和失败方案都保留，最差seed/来源不删除。各模型HOLD比例、正例召回、Action混淆/PR-AUC和hazard bins详见JSON；直接行为克隆若塌缩为HOLD也如实记入。

R3：NOT_RUN，旧V1的预注册累积阈值文件缺失，没有临时自造阈值。L3b和长度8：NOT_RUN_BUDGET，初批五组已覆盖受控最小矩阵，不扩大架构搜索。任务末尾return→adjust仍无物理终止守卫，本轮不能宣称FINISH成功。

## 文件、复现与验收

新增代码仅在tools/tomato_review_ui/learned_handshake/，新增配置configs/tomato_learned_handshake_v2_20261112.yaml、专项测试和本报告。此前Memory成果保留，没有修改原Current-only/Future-only/Joint训练、GUI、Handshake、原标签或数据。

```bash
PY=.venv-tomato/bin/python
EXP=runs/tomato/learned_handshake_v2_NEW_UNIQUE
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --check-only
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --phase-a
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --smoke
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --run
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --verify
$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --report-only
$PY -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py
```

输出拒绝覆盖既有训练或回放。完整模型保留decision_models/*/seed*/v*/best_policy.pt，SHA见对应manifest/checkpoint.sha256；所有概率、状态和决策事件保留replays/与predictions/，没有打包视频、Encoder、全部特征或大权重。

## 为什么直接动作网络可能更差

主任务图每个Active只有一个允许后继。L1b/L2多类别选择在此没有多合法技能之间的收益空间，却可能选到当前类或非法类，被外壳拒绝，或者过早提出唯一合法后继。主实验是top1提议后做task mask拒绝，没有“选最高合法logit”的额外搜索或回退。比较混合了输出空间/归纳偏置，不可宣称已经验证通用多技能Agent的选项选择。

L1b与L2都读同样4个概率锚点，跨度1.5秒；因此本轮否定的是额外GRU结构收益，不是证明概率上下文不需要。N=1/N=8与GRU-hazard均未追加。

OOF概率在54源teacher和75源完整teacher间存在质量/置信度漂移；加上错误commit改变Active/ACK，可造成状态分布偏移。L1b/L2触发一次train-only聚合后仍有坏轨迹；v1按预注册规则使用，没有再挑更好看的v0。

## 提前与漏切归因（描述性审计）

|策略|提交数|Future对目标<.65的提交|已匹配过早事件|这些过早中Current对目标<.7|
|---|---:|---:|---:|---:|
|H0|54|0|4|4|
|R1|54|0|3|3|
|R2|54|0|4|4|
|L0_s42|54|0|5|5|
|L1a_s42|54|0|7|7|
|L1b_s42|41|25|10|10|
|L2_s42|39|28|9|9|
|L3a_s42|50|13|8|8|

原Future是t+1s类别，不是“现在该提交”。相同Future提前证据在不同策略下被更早/更晚接受；planner仅量化到0.5s，ACK在commit后检查并可能阻挡后续，不能撤回这次过早commit。学习器无需Future达到原.8，直接动作策略甚至无需.65，因此应看commit_evidence.csv，而不能把错切全部归咎于编码器。

missed_attribution.csv按事后已保存状态区分hard recovery、ACK阻断、HOLD/拒绝、时机及重复同类配对歧义。它是粗略证据分类，不冒充已完成因果干预；所有真正匹配、顺序和错误秒仍由固定scorer给出。

## 最坏seed与正常来源回归

|组|最坏seed（Wrong Active）|Wrong(s)|窄窗|
|---|---:|---:|---:|
|L0|42|96.767|24|
|L1a|43|95.933|24|
|L1b|44|676.900|10|
|L2|44|681.900|9|
|L3a|44|496.067|13|

baseline_normal为H0完整粗阶段顺序正确且末态没有RECOVERY_REQUIRED的事后分层，只供审计，绝不进入策略。

|策略|基线正常来源|错误秒改善/退化/不变|丢失完整顺序|
|---|---:|---:|---:|
|R1|6|2/4/0|0|
|R2|6|0/0/6|0|
|L0_s42|6|2/2/2|0|
|L1a_s42|6|3/3/0|0|
|L1b_s42|6|2/4/0|3|
|L2_s42|6|1/5/0|3|
|L3a_s42|6|3/3/0|1|

全部三seed和9个来源的变化见normal_source_regression.csv，不能只看当前最坏案例或仅比较总体秒数。

## K1 Future替换（主选型之后，val-only敏感性）

主候选为KEEP_H0，所以K2a/K3a在定义上与H0的两组输入重复；没有趁此再挑一个神经策略。

|对照|窄窗|Wrong(s)|提前≤-.5s|漏切|硬超时|
|---|---:|---:|---:|---:|---:|
|K0|23|98.233|4|11|3|
|K1a|20|48.733|5|7|1|
|K2a|23|98.233|4|11|3|
|K3a|20|48.733|5|7|1|

使用固定K1 seed42 Future，额外读出587272参数及4锚点特征/运动上下文，不能当成免费输入。没有重训，Current仍为原B0。仅作分布替换敏感性，不反馈主排名。

## 实验修正与未解决项

正式学习器之前的烟雾审计发现尾部删失监督问题：保留初始smoke及协议/源码，修正为无可执行commit tick不监督、未知下一边界不填hazard负例，随后重新锁协议并重跑全部烟雾。未改架构、cost、风险阈值或根据网络val Active调标签。训练标签±1帧/±.1秒仅审计变化数，没有额外重训扰动组，不宣称对标注噪声稳健。

策略超时在本离线协议中按消息可用年龄/TTL处理；没有实现或验证实机wall-clock watchdog，也没有真实动作队列。没有收集物理成功标签，末尾误return→adjust/FINISH问题仍需独立完成/任务继续信号，不能以此次决策网络替代。

R3旧阈值来源缺失、L3b/N8未运行；OOF teacher与完整B0的分布漂移、小样本/多轮可见val/test、受NEXT单后继图限制，都是本轮结论的实际边界。

下一轮最小建议：保留H0；如要继续，先在新的来源上验证R1的“延迟vs过早”取舍，以及L0的额外过早来自哪些概率/时间模式。暂不移植直接动作GRU或hazard到默认Agent，也不因这次负面结果继续在历史test调cost。

## K1替换的跨来源审计

九个val来源的Wrong Active变化：5改善、2退化、2不变。总变化-49.500秒，来源配对95%描述区间[-125.8, 6.433]；窄窗总变化-3.0。仅固定K1 seed42的输入替换，不能据此证明稳健泛化，更不据此放松主风险条件。

离线策略改变自己的Active，但未来输入仍来自同一份已录制轨迹；真实机器人换prompt后可能产生不同动作和观察。本轮无法反事实评估真实闭环行为，也未验证实机wall-clock watchdog。

## 计算、复现与实现验收

原B0固定val Current/Future Macro-F1为0.95266/0.91873，所有主策略完全相同；本轮任何Action指标都不应写成V-JEPA识别准确率提升。

15个最终策略实际执行24次正式fit（含9次预锁定train-only聚合重训），另有5组最终smoke及保留的5组初次尾部修正前smoke。各模型样本/updates/最佳epoch/墙钟/参数与SHA见training_budget.csv。存在条件聚合预算差异，不能宣称完全逐更新等价的架构因果隔离。

|组 seed42|参数量|完整CPU提议P50/P95/P99(ms)|GPU驻留forward P95(ms)|新Session上下文reset P95(ms)|
|---|---:|---:|---:|---:|
|L0|57|0.0977/0.1043/0.1073|0.0223|0.0028|
|L1a|8065|0.1303/0.1566/0.1721|0.0421|0.0028|
|L1b|8197|0.1255/0.1413/0.1480|0.0375|0.0028|
|L2|14149|0.2626/0.2779/0.2905|0.2328|0.0028|
|L3a|8362|0.1346/0.1422/0.1513|0.0418|0.0027|

CPU完整提议包括输入校验、构造/归一化、网络和proposal；不包括原ACK/planner、视频编码器或π0.5。GPU仅forward含同步，不含传输/归一化。每组预热20次、测200次；只选择一个策略不是集成15网络。原始上下文特征368字节另加权重/统计/对象开销，不是进程RSS。最终KEEP_H0不新增决策NN；真实Encoder+π0.5并发与实机端到端仍未测。

最终畸形输出审计补强了“proposal早于证据available_at”和非有限分数/时间的HOLD日志处理；这是验证保护，不是策略参数修改。全部385555个已保存实际提议均满足新增守卫，180条重载val回放事件与Active仍逐值相同；该守卫不改特征、标签、cost和选型；另有L0显式L2修正单独补跑，其余12个学习器权重不变，修改前后SHA见runtime_validation_amendment.json。

R1的K沿用旧candidate支持口径（Future≥.65的不同新锚点累计），当前提交需≥.8，当前≥.95则K1；未要求历史每个支持都≥.8。R2的加速和“不以Current硬否决”的路径已用合成fixture验证；val与H0相同不表示该路径未实现。

专项测试最终42项（包括R2两条行为和显式L2数学测试）；全套番茄及切柠檬回归231项通过（以logs/tests.log实际耗时为准），实际输出见logs/tests.log。合成fixture只验证代码，全部策略效果来自真实概率与已审核GT。1,332条已保存回放指标复算、6,733次commit的planner原子性/单调版本/合法边校验通过；180条checkpoint重载与rule回放逐事件一致。原文件10,490个SHA均未变。

## 文件修改与交付

新增configs/tomato_learned_handshake_v2_20261112.yaml、tools/tomato_review_ui/learned_handshake/研究包及README、tests/test_tomato_learned_handshake_v2.py（42项专项）、本报告与docs/validation/tomato_learned_handshake_v2_20261112/摘要。原训练、Joint权重、标注、GUI、Handshake和上一轮Memory源码/成果未修改，未git commit/push。

主命令通过CLI执行；补充实际执行命令为：`.venv-tomato/bin/python supplemental/k1_swap.py EXP`、`supplemental/verify_saved.py EXP`、`supplemental/reload_parity.py EXP`、`supplemental/latency.py EXP`、`supplemental/detailed_analysis.py EXP`（在仓库根目录，以实验目录下真实脚本路径调用）。记录位于commands.json、logs/及源SHA清单，完整大输出仅本机，handoff.zip包含报告、摘要、所有val/test逐事件、训练预算、源码和测试日志。

## L0显式L2数学实现修正

初轮L0使用AdamW解耦WD.01，不严格等同规格的L2 Logistic。最终审计按规格补正为`BCE + .5*.01*sum(W²)`（不惩罚bias、optimizer decay0），val早停只看BCE；λ、种子、预算、输入、分组、标签、阈值和风险条件均未搜索或改变。修正前先保存L0_L2_correction.lock.json，再补跑3个seed，旧源码/模型/回放/最初selection/test receipt保留。修正发生在初轮结果之后，属于显式实现修正，不能将它说成未经任何修正的单一预注册实验。

正确L2版本依然为24/24/25个窄窗、5/5/5次过早，主选择仍KEEP_H0。其余12个学习器权重SHA完全相同。历史H0诊断直接复用，没有重开任何test调参；新增selection.corrected.lock.json与test_lock_receipt.corrected.json保留再次锁定证据。正式预算共27次fit：最初24次（含9次预定聚合）+3次数学修正；不是27个独立模型或数据集。
