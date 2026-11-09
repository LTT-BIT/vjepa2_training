# 新录制逐条清单与盲测方案

计划12条开发、20条主封存测试，另列4条条件启用的备用测试；共6个开发场次组、10个主测试场次组、2个备用场次组。尚未采集。全部使用真实三相机/实测58D位置与原时间戳，保留接触、闭合、带动物体、离盘、返程、触盘、释放等人工辅助锚点。

|ID|用途/分组|速度|重点|具体录制要求|
|---|---|---|---|---|
|DEV_01|development/DEV_G01|slow|adjust_pickup|adjust后静止5-8秒，再闭合、带动物体、离盘分别留可见证据|
|DEV_02|development/DEV_G01|fast|adjust_pickup|连续调整后快速再次拿起，禁止为了对齐模型刻意停顿|
|DEV_03|development/DEV_G02|normal|adjust_pickup|重复盘内调整3次，只接近/闭合后撤2次，最后才真正拿起|
|DEV_04|development/DEV_G02|normal|adjust_pickup|抓握后握持但不抬起5秒；标明确pickup尝试或继续adjust，不能预设负例|
|DEV_05|development/DEV_G03|normal|adjust_pickup|手掌遮挡局部接触，至少一路相机能看到物体与盘的分离|
|DEV_06|development/DEV_G03|slow|adjust_pickup|手腕视角部分离画，远景保留物体和盘，标缺失观测而不补帧|
|DEV_07|development/DEV_G04|normal|return_adjust|返程、触盘、释放、盘内重新定位四个锚点分开|
|DEV_08|development/DEV_G04|fast|return_adjust|触盘与手指调整连贯重叠，保留合理边界区间|
|DEV_09|development/DEV_G05|slow|harvest_return|剩裸梗后继续等待8秒，随后才返程，覆盖提前8.1秒误报类型|
|DEV_10|development/DEV_G05|normal|pickup_harvest|拿起持稳后左手失败尝试/重复摘取，记录阶段内子事件|
|DEV_11|development/DEV_G06|normal|full_sequence|新果串形态、照明、取放速度组合；完整七阶段|
|DEV_12|development/DEV_G06|normal|noncanonical|只前三阶段/缺失adjust/任务中止，不补齐七阶段|
|BLIND_01|sealed_independent_test/BLIND_G01|slow|adjust_pickup|调整完成后随机长度停顿，区分闭合与离盘；组内独立录制1|
|BLIND_02|sealed_independent_test/BLIND_G01|slow|adjust_pickup|调整完成后随机长度停顿，区分闭合与离盘；组内独立录制2|
|BLIND_03|sealed_independent_test/BLIND_G02|fast|adjust_pickup|快速连续调整再拿起；组内独立录制1|
|BLIND_04|sealed_independent_test/BLIND_G02|fast|adjust_pickup|快速连续调整再拿起；组内独立录制2|
|BLIND_05|sealed_independent_test/BLIND_G03|normal|adjust_pickup|接近后撤、重复调整、握住不拿起等困难条件；按可见动作区分真实pickup尝试与负例；组内独立录制1|
|BLIND_06|sealed_independent_test/BLIND_G03|normal|adjust_pickup|接近后撤、重复调整、握住不拿起等困难条件；按可见动作区分真实pickup尝试与负例；组内独立录制2|
|BLIND_07|sealed_independent_test/BLIND_G04|normal|adjust_pickup|局部手指遮挡，三路中至少一路可见关键物体运动；组内独立录制1|
|BLIND_08|sealed_independent_test/BLIND_G04|normal|adjust_pickup|局部手指遮挡，三路中至少一路可见关键物体运动；组内独立录制2|
|BLIND_09|sealed_independent_test/BLIND_G05|slow|harvest_return|裸梗/末果状态长停顿，正常返程对照；组内独立录制1|
|BLIND_10|sealed_independent_test/BLIND_G05|slow|harvest_return|裸梗/末果状态长停顿，正常返程对照；组内独立录制2|
|BLIND_11|sealed_independent_test/BLIND_G06|normal|return_adjust|放下/松手/重定位重叠与明确分离各一条；组内独立录制1|
|BLIND_12|sealed_independent_test/BLIND_G06|normal|return_adjust|放下/松手/重定位重叠与明确分离各一条；组内独立录制2|
|BLIND_13|sealed_independent_test/BLIND_G07|normal|pickup_harvest|左手接近/失败摘取/持稳等待，阶段内多次尝试；组内独立录制1|
|BLIND_14|sealed_independent_test/BLIND_G07|normal|pickup_harvest|左手接近/失败摘取/持稳等待，阶段内多次尝试；组内独立录制2|
|BLIND_15|sealed_independent_test/BLIND_G08|normal|full_sequence|新的物体形态与光照，完整七阶段；组内独立录制1|
|BLIND_16|sealed_independent_test/BLIND_G08|normal|full_sequence|新的物体形态与光照，完整七阶段；组内独立录制2|
|BLIND_17|sealed_independent_test/BLIND_G09|normal|partial_sequence|三阶段终止或缺失adjust，按实际序列评估；组内独立录制1|
|BLIND_18|sealed_independent_test/BLIND_G09|normal|partial_sequence|三阶段终止或缺失adjust，按实际序列评估；组内独立录制2|
|BLIND_19|sealed_independent_test/BLIND_G10|normal|noncanonical|返程中止/回退/异常重复，标真实序列并单报适配器失配；组内独立录制1|
|BLIND_20|sealed_independent_test/BLIND_G10|normal|noncanonical|返程中止/回退/异常重复，标真实序列并单报适配器失配；组内独立录制2|
|BLIND_RESERVE_01|sealed_test_reserve/BLIND_G11|slow|adjust_pickup|仅由保管者按事件数量缺口或技术失败启用；新植株完整七阶段；绝不依据模型分数追加|
|BLIND_RESERVE_02|sealed_test_reserve/BLIND_G11|normal|adjust_pickup|仅由保管者按事件数量缺口或技术失败启用；新植株完整七阶段；绝不依据模型分数追加|
|BLIND_RESERVE_03|sealed_test_reserve/BLIND_G12|slow|adjust_pickup|仅由保管者按事件数量缺口或技术失败启用；新植株完整七阶段；绝不依据模型分数追加|
|BLIND_RESERVE_04|sealed_test_reserve/BLIND_G12|normal|adjust_pickup|仅由保管者按事件数量缺口或技术失败启用；新植株完整七阶段；绝不依据模型分数追加|

## 锁定与保管

Before release, lock model SHA, source/schema/norm SHA, annotation convention, thresholds .5/K2 analysis and .8/K2 Current provider, horizon1.5 and2Hz, one-to-one matching, exclusions and statistical analysis. Collector/custodian retains new test videos/labels; developers cannot access before lock. No model-containing HTML used for true blind test.

Aim>=100 real blind transitions, with>=20 adjust->pickup if naturally available in complete sequences; current20-row plan includes4 partial/noncanonical clips, so exact counts may be lower. Add prespecified whole NEW blind groups based on missing event counts/technical failures only, never on model scores.

No shared raw recording/crop/reencoding/plant/session between development and blind. Prefer different operators and days; if unavailable report overlap rather than claim unknown-operator generalization. Group uncertainty by10 session/plant groups, not video windows.

Primary canonical seven-stage events reported separately from incomplete/backtracking sequences. Never force labels into seven phases. Report abstention/review need and premature prompts for out-of-contract sequences.

After one locked primary evaluation, any subsequent tuning makes that test exposed; a new independent test is then needed. Do not pick a different threshold/model based on highest test score.

主要缺口证据：训练转换pickup→harvest/harvest→return各152次，而return→adjust/adjust→pickup各76次；验证后两类各9次。仅凭次数不能确认遮挡/停顿/重复动作的覆盖，故必须补条件标签。adjust→pickup7次失败全部可由过早3次/迟到4次描述，优先补这些条件，而非复制稳定harvest长片段。

真正独立测试不能使用含模型曲线但CSS隐藏的复核网页。先发无模型输出的原始标注包；两位盲标者各自留存意见，再形成共识。开发者先锁定候选、Joint参考及所有规则，保管者再释放测试数据给锁定脚本一次评价；保存完整身份清单与首次访问记录。

规范v1先在开发录制试标。若与历史规范的动作语义存在系统差别，须预先声明评价版本，不能在看到测试分数后选标注定义。保留原标注、辅助锚点、分歧区间和修订历史。20条录制并非20个完全独立场景；按10个场次/植株组报告不确定性和每类支持数。
