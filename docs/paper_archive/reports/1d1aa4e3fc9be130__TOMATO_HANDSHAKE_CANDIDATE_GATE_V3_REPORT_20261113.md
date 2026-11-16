# 番茄同候选空间握手门控补充消融报告（2026-11-13）

## 结论：KEEP_H0

本轮实际完成 H0、R1 和 G0/G1/G2/G3×3 seed。验证集固定9来源/54转换；正式版本 `v1`。选型要求三seed分别满足全部风险门槛，代表seed固定42；没有根据test调参。默认握手机制保持 H0。

运行目录：`/home/user/pi05/vjepa2_training/runs/tomato/handshake_candidate_gate_v3_20261113_112707_tailmask`。分支 `feat/tomato-candidate-gate-v3-20261113`，基线 `db4cdad6e87af7c077f5be16a00bc277efeb54d9`。本轮未提交、未推送。

## 1. 实验身份与受控差异

原LOCAL93清洗normal、source_recording_id分组75/9/9不变；单一官方B0 Joint seed42双头、+1秒/2Hz。训练仅复用5折OOF（每折54train/6内val/15held）概率，val/test使用全75训练官方B0。三seed只改变门控初始化/批次，不能当成27条验证录像。未调用Encoder，未重新提取视觉或训练Memory/K1。

协议SHA：`17cb48bde3be42ce741d39276e770f7ad52c7eeb1afd890ac03d4fbf78fd0bec`。输入SHA、93来源、标注、模型/normalization、五折成员和概率逐文件身份分别见source_identity.json / split.lock.json / oof_teacher_manifest.json / protected_sha256.json。校验保护文件 18861 个，差异 0 个。

修正审计：首个输出handshake_candidate_gate_v3_20261113_110116_4ee71f保留全部初跑结果，train尾部核验发现未知主gate标签误记HOLD负例，已显式标记SUPERSEDED。只修正主gate mask并对所有模型统一重跑，未改阈值/超参数/准入规则；正式结果以本运行目录为准。初跑仅H0进入历史test，没有失败学习器test筛选。

上一轮全量成果和源码均只读复用。本轮唯一生产代码变更：handshake.py提取receive_evidence（只处理ACK/TTL）与propose_default_candidate；原prediction仍顺序调用二者。新CandidateRequestPort只复用旧apply的请求安全检查，覆盖ingest并传入真实Future，从不调用旧neutralize路径。H0/R1完整事件/Active/Pending/summary与历史逐值一致（H0额外84条train/val instrumented-vs-pure）。原GUI/四分类训练/柠檬/π0.5/Agent/RTC未改。

统一候选：Future top1≠自身Active、外部图合法、p≥.65，至多一个；所有模型只输出HOLD/REQUEST（≥.5），不能输出另一目标。G0/G1/G3输入同一240维（4×59+4mask），G2读取同一4×59和mask；4锚点仅概率/自身在线状态，非视觉Memory。G3的6桶仅辅助监督，不参与动作。

## 2. 验证集最终 Active 主结果

|策略|窄窗/54|宽窗|漏切|未匹配|过早≤−.5s|迟到>.5s|Wrong(s)|完整顺序/9|硬超时|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|H0|23|43|11|11|4|12|98.233|9|3|
|R1|21|45|9|9|3|19|76.300|9|2|
|G0_s42|23|50|4|4|4|22|42.100|9|0|
|G0_s43|15|42|12|12|0|25|68.067|9|0|
|G0_s44|22|48|6|6|2|23|49.200|9|0|
|G1_s42|19|32|22|7|2|10|451.700|5|1|
|G1_s43|21|48|6|6|4|22|60.467|9|0|
|G1_s44|21|49|5|5|3|24|53.833|9|0|
|G2_s42|22|49|5|5|2|24|46.000|9|0|
|G2_s43|20|49|5|5|2|25|45.667|9|0|
|G2_s44|21|49|5|5|2|25|46.500|9|0|
|G3_s42|24|49|5|5|2|22|44.700|9|0|
|G3_s43|20|39|15|7|3|15|238.933|7|0|
|G3_s44|21|50|4|4|3|25|47.333|9|0|

所有指标均为本轮真实冻结概率回放，不是合成测试精度。使用旧一对一from→to匹配器：宽窗[-1,+2]、严格[-.2,+.5]，边界保留帧时间。提前/迟到统计限定已宽窗匹配，额外提交另列；Wrong按完整Active和GT交叠积分。

### 三seed准入审计

|组|各seed通过全部风险|均值strict|均值Wrong(s)|最坏过早|最坏硬超时|达到收益门槛|最终准入|
|---|---|---:|---:|---:|---:|---|---|
|G0|[True, False, False]|20.00|53.122|4|0|True|False|
|G1|[False, False, False]|20.33|188.667|4|1|False|False|
|G2|[False, False, False]|21.00|46.056|2|0|True|False|
|G3|[True, False, False]|21.67|110.322|3|0|False|False|

预锁：各seed illegal/cross-session/nonplanning均0，early≤4、hard≤3、miss≤11、extra≤11、strict≥23、exact order≥9、Wrong≤98.233333s；所有seed通过后，组均值strict≥25或Wrong≤88.410s才有收益。没有best-seed选择、没有放宽门槛。

## 3. 候选覆盖与候选→请求→提交→ACK归因

GT-aligned Active的oracle候选诊断：42/54边界在严格窗可获得合法新鲜Future候选，缺失12个。H0自身状态下可执行覆盖 27，仅阻塞条件覆盖 0（两集合可重叠）。oracle只衡量候选供给，不是在线输入；H0条件覆盖也不是其他策略的绝对上界。

|策略|无候选锚点|有候选HOLD|REQUEST被拒锚点|请求事件|实际commit|普通ACK|迟到ACK|硬超时|最长阻塞(s)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|H0|2935|0|0|54|54|47|4|3|41.033|
|R1|2910|41|0|54|54|49|3|2|23.033|
|G0_s42|2899|52|0|54|54|51|3|0|3.590|
|G1_s42|2776|175|12|39|39|37|1|1|78.000|
|G2_s42|2892|59|0|54|54|52|2|0|4.590|
|G3_s42|2895|56|0|54|54|52|2|0|4.590|

此表“被拒锚点”仅统计真正REQUEST且被壳拒绝，ACK阻塞下的日志HOLD另记；精确REQUEST/validated/request-event/commit/ACK关联保存在summary/candidate_request_commit_ack_attribution.jsonl.gz。候选提供时刻（first_observed/available）与壳内Pending开始不同，均记录，不能把网络REQUEST直接算成功。未决请求、ACK阻塞、错误预测、尾部删失分开。

## 4. 监督、统一纠偏与实际训练

门控只在新鲜合法候选且可请求状态形成样本，以下一实际planner tick标定。严格窗主正例；若GT当前仍是候选且前一真值类是自身Active，则保留相邻late-catchup正例（不能据此复位Active），其他错配HOLD。每来源/事件/占用时间桶等权，train加权neg/pos（上限10）；故输出是重加权研究分数，未经自然先验校准，不是物理安全概率。

G0 BCE+.5*.01*||非biasW||²，AdamW decay0；其他decay.01。G3 BCE+.25×有效hazard样本加权平均censored NLL；右删失将兼容桶概率求和，不把末尾当无转换负例。6桶delta=b−tick：<−.5、[−.5,0]、(0,.2]、(.2,.5]、(.5,1]、>1，边界浮点按1e−9处理。无法确定候选事件的hazard mask，不伪造边界/FINISH。尾部若end−tick<.2秒，下一真实边界是否在正窗内未知，主gate BCE也mask（hazard仍按右删失学习）。

H0-OOF初始训练可执行tick 581（其中主gate未知屏蔽1个），正384/负196，覆盖来源 75/75；observed 574、censored 7、hazard masked 0。无可执行候选来源仍完整回放，未删除。每来源/转换/桶数量、最多21例确定性边界审计与±1帧/±.1s标签敏感性见train_labels_audit.json，未修改标注。

全局纠偏触发：True。先完成全部12个v0，仅train OOF比较自身Active累积偏移>10s或未见状态组合>5%；任一触发则所有12个一律聚合一次、重初始化并使用v1。每个v1使用相同baseN+ownN样本预算（不足有放回并留SHA）、相同种子批次顺序；共享统计只池化train base和全部12个own来源。早停统一val-H0监督gate BCE，最多80epoch/patience10，不看val Active选epoch/v0v1。实际更新数因预定早停不同，未额外扩预算。

|模型|参数|best/实际epoch|updates|train tick|正/负|训练(s)|GPU峰值MiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|G0_s42|241|76/80|400|1162|755/405|0.35|33.61|
|G0_s43|241|1/11|55|1162|792/369|0.05|33.61|
|G0_s44|241|80/80|400|1162|863/298|0.35|33.61|
|G1_s42|17537|39/49|245|1162|816/344|0.26|34.02|
|G1_s43|17537|30/40|200|1162|763/397|0.21|34.02|
|G1_s44|17537|33/43|215|1162|765/396|0.25|34.02|
|G2_s42|9345|36/46|230|1162|757/403|0.46|35.37|
|G2_s43|9345|37/47|235|1162|761/400|0.46|35.37|
|G2_s44|9345|53/63|315|1162|761/399|0.61|35.37|
|G3_s42|17735|35/45|225|1162|803/358|0.35|34.02|
|G3_s43|17735|51/61|305|1162|789/372|0.48|34.02|
|G3_s44|17735|47/57|285|1162|763/397|0.47|34.02|

### 门控与辅助头（最终自身轨迹，仅可请求tick）

|模型|正例/n|PR-AUC|Precision|Recall|Brier|ECE|
|---|---:|---:|---:|---:|---:|---:|
|G0_s42|73/106|0.889|0.870|0.644|0.210|0.230|
|G0_s43|136/172|0.877|0.926|0.368|0.276|0.336|
|G0_s44|88/123|0.893|0.907|0.557|0.222|0.225|
|G1_s42|180/214|0.807|0.846|0.183|0.491|0.561|
|G1_s43|102/135|0.901|0.870|0.461|0.218|0.281|
|G1_s44|104/139|0.918|0.907|0.471|0.211|0.257|
|G2_s42|80/113|0.902|0.907|0.613|0.206|0.217|
|G2_s43|78/111|0.897|0.889|0.615|0.203|0.205|
|G2_s44|82/116|0.912|0.907|0.598|0.200|0.198|
|G3_s42|77/110|0.920|0.907|0.636|0.192|0.199|
|G3_s43|204/237|0.862|0.870|0.196|0.414|0.517|
|G3_s44|91/126|0.903|0.907|0.538|0.213|0.234|

各组自身轨迹的可请求集合不同，因此门控分类分数不宜脱离Active横排。val监督早停集合相同；训练log保留BCE/hazard两项。G3各seed的观察桶confusion、bin-MAE、Brier/ECE和censor-compatible NLL见对应gate_metrics.json；尾部桶无界，未伪造连续秒MAE。

## 5. OOF→full-B0分布漂移

|概率来源|Current准确率/ECE|Future准确率/ECE|双头分歧|候选比例|可请求比例|
|---|---:|---:|---:|---:|---:|
|held-source OOF|0.966/0.017|0.950/0.022|0.046|0.056|0.027|
|full75 B0 frozen|0.979/0.012|0.963/0.016|0.038|0.023|0.023|

概率分位数、边界±1秒质量、NLL/Brier及fold身份详见teacher_shift_report.json。来源数量不同且val仅9条，描述差异不能唯一归因为teacher漂移；没有在val拟合校准器。

## 6. 配对统计、正常样本回归与边界类型

来源为bootstrap单位，5000次固定rng20261113；seed先在每来源平均，绝不将seed当独立录像。区间为总量差，对反复研究的val只作探索性描述。

|差值(后者−前者)|strict总量差 [95%CI]|Wrong秒差 [95%CI]|
|---|---:|---:|
|G0_vs_H0_seedmean|-3.00 [-7.33,0.33]|-45.111 [-120.489,7.456]|
|G1_vs_H0_seedmean|-2.67 [-9.00,2.67]|90.433 [-58.043,234.181]|
|G2_vs_H0_seedmean|-2.00 [-7.67,2.67]|-52.178 [-123.310,0.745]|
|G3_vs_H0_seedmean|-1.33 [-7.67,3.00]|12.089 [-76.357,102.786]|
|G1_minus_G0_sourcepaired_seedmean|0.33 [-2.33,3.00]|135.544 [38.178,245.490]|
|G2_minus_G1_sourcepaired_seedmean|0.67 [-3.67,4.00]|-142.611 [-255.482,-44.264]|
|G3_minus_G1_sourcepaired_seedmean|1.33 [-1.00,3.67]|-78.344 [-197.425,12.900]|

**G0_s42正常来源对照H0**：未变0/9；按Wrong秒排序（并列看strict），改善/退化各至多3例：
- improved_top3: `a5dfe3c0e42ec1640001cd5d` strictΔ+0 / WrongΔ-34.667s / earlyΔ+0 / hardΔ-1; `7c99ac6ac7bf9ef487aae3de` strictΔ+0 / WrongΔ-16.300s / earlyΔ+0 / hardΔ-1; `796aae799456b09dfd16c507` strictΔ-1 / WrongΔ-4.700s / earlyΔ+0 / hardΔ-1
- degraded_top3: `faf6b8f0b07f427a262ba0b5` strictΔ-2 / WrongΔ+1.500s / earlyΔ+0 / hardΔ+0; `11ce818df3007fe3f5ebbcd8` strictΔ+0 / WrongΔ+1.300s / earlyΔ+0 / hardΔ+0; `847d6e2f57d27893ec98e6a6` strictΔ+0 / WrongΔ+0.200s / earlyΔ-1 / hardΔ+0

**G1_s42正常来源对照H0**：未变0/9；按Wrong秒排序（并列看strict），改善/退化各至多3例：
- improved_top3: `a5dfe3c0e42ec1640001cd5d` strictΔ+1 / WrongΔ-32.867s / earlyΔ-2 / hardΔ-1; `7c99ac6ac7bf9ef487aae3de` strictΔ+0 / WrongΔ-16.300s / earlyΔ+0 / hardΔ-1; `32cdea7e7dcc6d46b66b7d2b` strictΔ+0 / WrongΔ-2.700s / earlyΔ+0 / hardΔ+0
- degraded_top3: `796aae799456b09dfd16c507` strictΔ-1 / WrongΔ+156.800s / earlyΔ+0 / hardΔ-1; `11ce818df3007fe3f5ebbcd8` strictΔ-1 / WrongΔ+97.800s / earlyΔ+0 / hardΔ+1; `faf6b8f0b07f427a262ba0b5` strictΔ-4 / WrongΔ+71.900s / earlyΔ+0 / hardΔ+0

**G2_s42正常来源对照H0**：未变0/9；按Wrong秒排序（并列看strict），改善/退化各至多3例：
- improved_top3: `a5dfe3c0e42ec1640001cd5d` strictΔ-1 / WrongΔ-31.867s / earlyΔ-2 / hardΔ-1; `7c99ac6ac7bf9ef487aae3de` strictΔ+0 / WrongΔ-16.800s / earlyΔ+0 / hardΔ-1; `796aae799456b09dfd16c507` strictΔ+0 / WrongΔ-5.200s / earlyΔ+0 / hardΔ-1
- degraded_top3: `faf6b8f0b07f427a262ba0b5` strictΔ-2 / WrongΔ+2.000s / earlyΔ+0 / hardΔ+0; `847d6e2f57d27893ec98e6a6` strictΔ+0 / WrongΔ+1.633s / earlyΔ-1 / hardΔ+0; `11ce818df3007fe3f5ebbcd8` strictΔ+0 / WrongΔ+0.967s / earlyΔ+0 / hardΔ+0

**G3_s42正常来源对照H0**：未变0/9；按Wrong秒排序（并列看strict），改善/退化各至多3例：
- improved_top3: `a5dfe3c0e42ec1640001cd5d` strictΔ+1 / WrongΔ-32.867s / earlyΔ-2 / hardΔ-1; `7c99ac6ac7bf9ef487aae3de` strictΔ+0 / WrongΔ-16.300s / earlyΔ+0 / hardΔ-1; `796aae799456b09dfd16c507` strictΔ+0 / WrongΔ-4.700s / earlyΔ+0 / hardΔ-1
- degraded_top3: `faf6b8f0b07f427a262ba0b5` strictΔ-2 / WrongΔ+1.500s / earlyΔ+0 / hardΔ+0; `847d6e2f57d27893ec98e6a6` strictΔ+0 / WrongΔ+1.133s / earlyΔ-1 / hardΔ+0; `11ce818df3007fe3f5ebbcd8` strictΔ+0 / WrongΔ+0.667s / earlyΔ+0 / hardΔ+0

所有9来源、全部3seed包括最坏样本均保留在casebook与episode_scores；不能只举原问题样本。

|策略seed42|harvest→return strict/总|return→adjust|adjust→pickup|末次return|短return≤1.5s|
|---|---:|---:|---:|---:|---:|
|H0|2/18|3/9|2/9|1/9|0/0|
|R1|6/18|2/9|1/9|3/9|0/0|
|G0_s42|5/18|2/9|1/9|1/9|0/0|
|G1_s42|5/18|1/9|1/9|0/9|0/0|
|G2_s42|6/18|1/9|3/9|1/9|0/0|
|G3_s42|6/18|2/9|2/9|1/9|0/0|

|策略|边界±.5s Wrong/时长(s)|边界±1s|稳定>1s|
|---|---:|---:|---:|
|H0|19.767/54.000|29.833/108.000|68.400/1394.100|
|R1|21.100/54.000|32.400/108.000|43.900/1394.100|
|G0_s42|21.033/54.000|30.733/108.000|11.367/1394.100|
|G1_s42|26.667/54.000|46.433/108.000|405.267/1394.100|
|G2_s42|22.500/54.000|33.500/108.000|12.500/1394.100|
|G3_s42|21.433/54.000|31.500/108.000|13.200/1394.100|

以上窗口是时间并集，±.5包含于±1；稳定>1与±1互补，不将重叠窗口重复计总Wrong。Pending/ACK状态时长与延迟窗口Brier见temporal_strata.json。

重复return按发生次序分列，短段与末尾分列：transition_cohorts.json。最终return之后没有已观察边界时不能制造return→adjust/FINISH；当前概率预测与已发生的物理完成不是同一监督。

## 7. 固定故障压力测试

H0/R1/G0–G3 seed42均完成13种×9来源=702条注入回放。注入仅按源锚点索引模运算，或因果首次commit后5.5s无更新，不按GT边界挑时机；+0.1/+0.3s、长gap、重复、倒序、跨session/model、HELD、单/连续错Future、Current翻转/丢失、直到硬超时均覆盖。

|注入|H0 strict/Wrong/hard|R1|G0|G1|G2|G3|
|---|---|---|---|---|---|---|
|delay_010|23/98.2/3|21/76.3/2|23/42.1/0|17/592.2/1|29/39.1/0|22/58.3/0|
|delay_030|23/98.2/3|21/76.3/2|23/42.1/0|15/662.5/1|24/97.4/3|20/383.8/1|
|long_gap|22/103.2/3|20/63.8/1|21/46.9/0|19/302.7/1|20/51.0/0|22/50.0/0|
|duplicate|23/98.2/3|21/76.3/2|23/42.1/0|19/451.7/1|22/46.0/0|24/44.7/0|
|reorder|23/98.7/3|20/77.3/2|23/42.1/0|19/128.7/0|24/45.5/0|19/47.7/0|
|cross_session|23/98.7/3|20/77.3/2|23/42.1/0|19/128.7/0|24/45.5/0|19/47.7/0|
|wrong_model|23/98.7/3|20/77.3/2|23/42.1/0|19/128.7/0|24/45.5/0|19/47.7/0|
|held|23/98.7/3|20/77.3/2|23/42.1/0|19/128.7/0|24/45.5/0|19/47.7/0|
|future_single|1/1209.3/9|0/1209.9/9|21/55.8/1|15/412.3/2|13/83.1/2|20/160.3/1|
|future_run|9/1273.5/9|7/1274.7/9|18/63.0/2|8/1269.7/9|19/65.8/2|21/62.0/1|
|current_flip|23/98.2/2|21/76.3/2|24/41.6/0|20/280.7/1|25/44.1/0|21/46.2/0|
|current_loss|22/99.7/3|20/78.3/2|22/43.6/0|19/132.2/0|24/46.1/0|20/48.7/0|
|blackout_to_hard|9/371.4/9|7/372.6/9|8/371.7/9|8/372.1/9|7/372.6/9|8/372.1/9|

非法/跨会话/非规划commit均0，全部版本单调且TTL受限。错误但格式合法的概率可能改变合法提交，协议无法自动识别语义错类；因此上表不是“故障下性能不变”。所有记录见stress/，不用于选型。

## 8. 锁定后的 historical test

先写selection/test_lock，再只运行H0及唯一winner；本轮实际评估 ['H0']。该9来源历史已反复曝光，不是新盲测；没有用它搜索模型或阈值。

|策略|窄窗/54|漏切|额外|过早|Wrong(s)|硬超时|
|---|---:|---:|---:|---:|---:|---:|
|H0|26|8|8|4|47.233|1|

## 9. 验收与运行成本

实际pytest：**275 passed in 11.95s**，完整输出logs/pytest.log。模型重新加载、14策略×val9完整事件/分数重算一致，batch-vs-stream误差≤1.79e-07，删除GT字段推理输入/决策完全一致。共审计2637个回放、14311次实际commit协议约束。

|模型|CPU前向P50/P95(ms)|GPU前向P50/P95(ms)|
|---|---:|---:|
|G0_s42|0.0082/0.0084|0.0227/0.0244|
|G1_s42|0.0275/0.0282|0.0461/0.0484|
|G2_s42|0.1110/0.1133|0.1685/0.1761|
|G3_s42|0.0310/0.0315|0.0532/0.0557|

batch1纯决策网络、200次测量、预热20、单CPU线程，GPU同步；不包含Encoder/视频/π0.5并发。4锚点输入占944+4字节有效数值缓存，另有Python/状态开销；每episode重新创建特征/候选/执行状态，GRU不跨4锚点保存hidden。

## 10. 文件与复现

新增：configs/tomato_candidate_gate_v3_20261113.yaml；tools/tomato_review_ui/candidate_gate_v3/*.py及README；tests/test_tomato_candidate_gate_v3.py；本报告和docs/validation/tomato_candidate_gate_v3_20261113。生产文件仅handshake.py兼容接口提取，原始文件副本original_handshake.py和before/after SHA保留。所有旧实验未覆盖。

```bash
PY=.venv-tomato/bin/python
EXP=runs/tomato/handshake_candidate_gate_v3_NEW_UNIQUE
$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --check-only
$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --smoke
$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --run
$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --verify
$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --report-only
$PY -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py
```

输出目录拒绝覆盖。完整权重、训练曲线、归一化、所有决策序列/事件、逐来源/逐转换CSV、bootstrap/压力测试均在run目录。handoff.zip仅报告、代码、配置、审计/汇总/训练日志/失败个例，不含视频、原始特征、Encoder或完整模型权重。

## 11. 最终研究意见与未验证项

- G0：三seed strict均值20.00、Wrong均值53.122s；准入False。风险失败与平均收益应分开看，不能拿更低BCE取代最终Active。
- G1：三seed strict均值20.33、Wrong均值188.667s；准入False。风险失败与平均收益应分开看，不能拿更低BCE取代最终Active。
- G2：三seed strict均值21.00、Wrong均值46.056s；准入False。风险失败与平均收益应分开看，不能拿更低BCE取代最终Active。
- G3：三seed strict均值21.67、Wrong均值110.322s；准入False。风险失败与平均收益应分开看，不能拿更低BCE取代最终Active。

本轮只支持“同候选/同监督/同历史”的局部结构对照；旧五动作GRU/Hazard退化不能直接解释为结构无效，同样也不能从9条val证明新结构普适有效。候选缺口、OOF到full-B0概率漂移、粗边界监督、ACK阻塞都需要分别处理。

若未通过全部准入，建议继续保留H0，将后续精力放在候选供给、边界监督与独立实时Shadow；不在本轮结果上继续搜阈值/模型。即使通过也只作为Shadow研究候选。K1敏感性NOT_RUN（可选，非主线）；新独立录像、人工逐帧物理完成核验、真机闭环、Encoder+π0.5并发、生产延迟/成功率均未验证。

## 12. 供下一轮讨论的具体诊断（仅解释，不再选型）

1. **GRU gate在本轮条件下更稳定，但尚不能替换H0。** 三seed Wrong为46.000/45.667/46.500秒，过早均2、硬超时均0；strict为22/20/21，全部低于23。对同候选MLP的来源配对Wrong差区间不跨0，但这只有9个重复研究来源的探索性对照，不能外推为普遍优势。

2. **总错误秒改善与边界精度是两个不同结果。** 代表seed42：G2在GT边界±1秒内Wrong为33.500秒，H0为29.833秒；稳定段>1秒Wrong为12.500秒，H0为68.400秒。减少的主要是长期错误Active停留，精确边界处反而略差；这解释了Wrong大降而strict仍不过关。

3. **MLP及时间辅助头仍有种子级失败。** G1_s42 Wrong451.700秒、可请求正例召回0.183、完整顺序仅5/9；G3_s43 Wrong238.933秒、召回0.196、顺序7/9，且硬超时为0。由此不能把所有退化都归因为ACK硬超时；在自己的在线状态里长期HOLD和任务候选可达性也需要看。G3_s42单个seed的24 strict/44.700秒不满足三seed选择规则，不能据此宣称Hazard成功。

4. **监督代理与严格窗口存在张力，训练有效样本也少。** H0-OOF只有581个可请求tick；384正例中，181个（47.1%）是提交tick已落后边界超过.5秒、仍处于相邻候选阶段的追赶正例。该预锁监督兼顾恢复，但不等同“提高strict命中”的单一目标。±.1秒边界扰动使18/21个gate标签、81/99个时间桶改变。以上是后验诊断，不是因果消融结论；本轮没有据此修改监督或再调权重。

5. **更复杂门控不能补回原Future缺席的候选。** 12个oracle窄窗缺口中，10个没有正确Future top1，2个虽top1正确但p<.65。完整列表见candidate_supply_failure_causes.json。H0/R1在自身Active相同的锚点上候选stage始终一致，候选差异来自各自状态轨迹；没有偷偷给学习器额外目标空间。

6. **延迟故障暴露了时钟特征分布外敏感性。** 训练延迟固定90ms，prediction_age/until_plan近似常量，标准化std均触及.05下限。+100ms分别约移动+2.667/−4个标准差；+300ms约+8/−12个标准差。虽然很多可执行planner tick未改变，网络输入已经明显偏离训练分布，G1/G3压力回放大幅波动。常量输入相关方向缺乏训练变化是一个有证据支持的可能原因，但尚未通过特征消融确认；不能把某个延迟下注入后的分数上升当真实改进。

7. **下一步建议仍为保留H0。** 若继续研究，先另立协议明确边界精度与追赶/长期错误的优先级、采集独立来源，并考虑train-only时钟扰动与候选/边界监督质量；不要在已曝光val/test继续搜索阈值。现阶段不扩大视觉Memory或直接接入真实控制。

本节生成脚本为run目录中的finalize_analysis.py；它只读取已锁定结果和训练统计。重建报告后可再次执行此脚本恢复解释补充，随后刷新handoff清单；不影响模型、选择或测试结果。
