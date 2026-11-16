# 握手机制完善措施：跨样本配对验证与最终修改意见（2026-11-12）

## 结论

不能只延长超时，也不能一刀切禁止return→adjust。推荐分开实现：普通ACK等待、迟到ACK观察窗口、真正需要授权的硬异常恢复；再以明确任务契约/外部继续信号约束是否开启下一轮。所有自动迟到ACK结果只是离线/Shadow候选，不是可直接驱动π0.5的安全授权。生产默认与UI未改变，代码未提交。

## 样本与对照边界

共85个来源：原9条val为主要验证，75条train作为回归覆盖（不是独立泛化精度），再加用户当前外部样本81902aa3f04224836c044c78/source e9f631dd459a4699b51e8a210747d693。原9条test没有评分。所有策略复用同一份Joint2Hz概率，θ_prepare=.65、θ_commit=.8、K1、θ_ACK=.7、延迟.09、规划周期.5、defer0、合法边开，初始pickup不从GT读取。

“正常样本”采用基线事后分层：提交顺序与人工序列完全一致、录制结束不处于RECOVERY_REQUIRED，共4条val+72条train=76条。这个分层只供分析，不参与策略决策。全部85条均保留，没有删除难例。

首先预声明7种对照，3秒迟到窗口在val无收益；之后仅在val/回归上追加一次5秒总上限敏感性与对应任务契约对照，属于明确的探索性追加，不是预注册无偏结论。原初实验目录/协议完整保留，没有继续扩大网格或对test寻优。最终9策略×85来源=765回放；基线85条与现有生产引擎逐事件完全一致，当前样本还与截图对应的旧simulation逐事件相等。

## 策略定义

| ID | 规则 |
|---|---|
| B_original | 当前1.5s超时后要求显式恢复 |
| T_timeout2 | 仅普通超时延至2秒 |
| L_lateACK1 | 1.5s软告警后接受一次新Current一致；总上限3秒 |
| S_lateACK2 | 1.5s软告警后需两个独立连续新Current一致；总上限3秒 |
| S5_lateACK2_hard5 | 同上，总上限5秒；超出后必须人工/外部授权 |
| C_lateACK2_contract2cycles / C5... | 3/5秒迟到窗口＋明确两轮任务假设，只授权一次return→adjust |
| TC_timeout2_contract2cycles | 2秒普通窗口＋同样两轮任务假设 |
| F_lateACK2_failclosed | 3秒迟到窗口＋没有外部授权则禁止所有return→adjust，刻意验证其代价 |

迟到ACK要求observed_at晚于commit、Current top1=当前Active、置信≥.7；K2必须不同锚点，间隔超过TTL会重新计数。仅解除对当前Active的感知不一致，不将Active改到Current或Future；下一阶段仍走原候选/提交逻辑。研究包装器为兼容旧状态复用RECOVERY_REQUIRED表示软告警观察窗口，以hard_alarm区分硬异常；下一版正式实现应新增WAIT_LATE_ACK，不能把真正的硬RECOVERY_REQUIRED设为可自动解锁。

两轮契约是统一预先设置的假设参数（包括三阶段样本也使用同一假设以暴露局限），从实际commit历史计数，不读取GT段数、未来标签、录像剩余时长或7阶段索引。只有任务启动时确实已知两轮才可使用；本轮没有获得真实物理完成/继续授权信号，不能把条件成立当成已验证事实。

## 当前截图样本：改善和剩余错误

| 策略 | 匹配 | 错误Active秒 | 完整顺序 | 正常ACK/迟到ACK | 软timeout/硬timeout | 改善/退化/不变来源 |
|---|---|---|---|---|---|---|
| B_original | 4/6 | 8.367 | 0/1 | 4/0 | 1/0 | 0/0/1 |
| T_timeout2 | 5/6 | 3.367 | 0/1 | 7/0 | 0/0 | 1/0/0 |
| L_lateACK1 | 5/6 | 3.367 | 0/1 | 6/1 | 1/0 | 1/0/0 |
| S_lateACK2 | 5/6 | 3.367 | 0/1 | 6/1 | 1/0 | 1/0/0 |
| C_lateACK2_contract2cycles | 5/6 | 2.700 | 1/1 | 5/1 | 1/0 | 1/0/0 |
| F_lateACK2_failclosed | 2/6 | 72.000 | 0/1 | 2/0 | 0/0 | 0/1/0 |
| TC_timeout2_contract2cycles | 5/6 | 2.700 | 1/1 | 6/0 | 0/0 | 1/0/0 |
| S5_lateACK2_hard5 | 5/6 | 3.367 | 0/1 | 6/1 | 1/0 | 1/0/0 |
| C5_lateACK2_hard5_contract2cycles | 5/6 | 2.700 | 1/1 | 5/1 | 1/0 | 1/0/0 |

原机制73.5秒提交harvest，75秒timeout；75.09秒Current才高置信追上。2秒或迟到ACK均能解开，随后113秒提交return，但118秒又误提交adjust，录制结束118.667秒仍为adjust。两轮任务契约能拦住这次多余继续动作，保持return，错误Active从8.367秒降至2.700秒，顺序从不完整变为正确七段。

这并非全部时序正确：harvest仍在73.5秒提交，人工边界74.9秒，提前1.4秒，超出[-1,+2]主匹配窗；因此即使顺序正确也只有5/6匹配。完成守卫仍未验证，不能用新的“完整顺序正确”替代物理安全。

## 原9条val：没有把单样本改善当整体结论

| 策略 | 匹配 | 错误Active秒 | 完整顺序 | 正常ACK/迟到ACK | 软timeout/硬timeout | 改善/退化/不变来源 |
|---|---|---|---|---|---|---|
| B_original | 37/54 | 273.733 | 6/9 | 41/0 | 5/0 | 0/0/9 |
| T_timeout2 | 37/54 | 273.733 | 6/9 | 41/0 | 5/0 | 0/0/9 |
| L_lateACK1 | 37/54 | 273.733 | 6/9 | 41/0 | 5/5 | 0/0/9 |
| S_lateACK2 | 37/54 | 273.733 | 6/9 | 41/0 | 5/5 | 0/0/9 |
| C_lateACK2_contract2cycles | 37/54 | 273.733 | 6/9 | 41/0 | 5/5 | 0/0/9 |
| F_lateACK2_failclosed | 16/54 | 884.700 | 0/9 | 17/0 | 1/1 | 0/8/1 |
| TC_timeout2_contract2cycles | 37/54 | 273.733 | 6/9 | 41/0 | 5/0 | 0/0/9 |
| S5_lateACK2_hard5 | 43/54 | 98.233 | 9/9 | 47/4 | 7/3 | 3/0/6 |
| C5_lateACK2_hard5_contract2cycles | 43/54 | 98.233 | 9/9 | 47/4 | 7/3 | 3/0/6 |

2秒方案与3秒迟到方案在9条val的实际Active完全不变。进一步核对5个timeout后的首次高置信Current支持距commit约3.09、4.09、3.09、7.59、16.09秒，解释了为何2秒/3秒均无效；长期不一致不能当作稍微迟到，更不能无限等待后自动推进。

5秒/Klate2候选：3个来源错误时长下降、6个不变，合计273.733→98.233秒，减少175.500秒；匹配37→43/54，完整顺序6→9/9。但未匹配提交从9增加到11，严格窗口只从19到23/54，提前≥.5秒仍为4个，时序问题未消失。软timeout计数5→7是恢复后继续执行产生了更多提交机会（46→54次），不能孤立解读为更差/更好；硬timeout为3次。

| val来源 | 基线分层 | S5错误秒变化 | 匹配变化 | 未匹配变化 | 顺序：原→新 |
|---|---|---|---|---|---|
| 6b8eb1ba3f3843e2b952e7a2228dedd1 | baseline_normal | +0.000 | 0 | 0 | True→True |
| 2c30455ce7794b87af8b0a60af0a0dab | baseline_problem | -65.867 | 2 | 0 | False→True |
| 9a90a945e77e49aabb7a3d4e44246155 | baseline_problem | -70.433 | 3 | 1 | False→True |
| 97b0a0e7b27140cc933c1c6f8bc7ca1f | baseline_problem | -39.200 | 1 | 1 | False→True |
| 7bedea6b1c034982bf1cec5050ad2dc3 | baseline_problem | +0.000 | 0 | 0 | True→True |
| c4735706f30843569cc2693c5802fae4 | baseline_normal | +0.000 | 0 | 0 | True→True |
| a2d3cd24f74e4ec1ba30f1a6abc81193 | baseline_normal | +0.000 | 0 | 0 | True→True |
| c0298a852584430885235869cde14f02 | baseline_problem | +0.000 | 0 | 0 | True→True |
| 45ece5143b704a3fb609c2e4a8c5c8d2 | baseline_normal | +0.000 | 0 | 0 | True→True |

配对来源bootstrap仅作探索（9来源、5000次、seed20261112），S5−基线错误秒总量95%百分位区间约[-351.000, 0.000]。数据少且5秒为追加敏感性，不能据此宣称稳定优势。

## 正常样本会不会退化

76条基线正常来源中，T2、S5、C5的Active区间逐条逐值完全相同：未观察到新增错误、顺序退化或切换时间变化。这是此次样本上的回归结论，不是“任何正常任务都不会受影响”的证明。

75条train整体：2秒/S5/C5均1条改善、74条不变；错误时长252.733→176.600秒，匹配439→442/446、顺序74→75/75。未匹配提交3→4，仍有时序代价；训练来源只作回归，不可拿75/75当新数据模型精度。

| 策略 | 匹配 | 错误Active秒 | 完整顺序 | 正常ACK/迟到ACK | 软timeout/硬timeout | 改善/退化/不变来源 |
|---|---|---|---|---|---|---|
| B_original | 439/446 | 252.733 | 74/75 | 439/0 | 3/0 | 0/0/75 |
| T_timeout2 | 442/446 | 176.600 | 75/75 | 445/0 | 1/0 | 1/0/74 |
| L_lateACK1 | 442/446 | 176.600 | 75/75 | 442/3 | 4/1 | 1/0/74 |
| S_lateACK2 | 442/446 | 176.600 | 75/75 | 442/3 | 4/1 | 1/0/74 |
| C_lateACK2_contract2cycles | 442/446 | 176.600 | 75/75 | 442/3 | 4/1 | 1/0/74 |
| F_lateACK2_failclosed | 150/446 | 5795.000 | 1/75 | 149/1 | 1/0 | 0/73/2 |
| TC_timeout2_contract2cycles | 442/446 | 176.600 | 75/75 | 445/0 | 1/0 | 1/0/74 |
| S5_lateACK2_hard5 | 442/446 | 176.600 | 75/75 | 442/3 | 4/1 | 1/0/74 |
| C5_lateACK2_hard5_contract2cycles | 442/446 | 176.600 | 75/75 | 442/3 | 4/1 | 1/0/74 |

一刀切禁止return→adjust明确失败：val有8/9来源更差，完整顺序6→0；train有73/75来源更差，完整顺序74→1。76条原本正常来源中75条被破坏（另1条是三阶段、本来就无需adjust）。它会挡住第一轮return之后本应存在的下一轮，而非只挡末尾误切。

两轮契约在本次val/train没有进一步改善：这些来源没有发生超出该预算的return→adjust，守卫基本未被触发，其额外收益主要来自当前案例；不能据此证明契约守卫的广泛有效性。不同轮数任务、错误早期commit消耗预算、三阶段/未知任务均需独立验证。

## 最终完善修改意见

1. 不把“改成2秒”作为最终修复。保留其为低成本候选，但val没有收益，不能解决长期感知不一致与尾部误切。

2. 下一版优先引入明确的WAIT_LATE_ACK：普通窗口1.5秒后告警并继续保留Active；仅在新观察的Current连续2次高置信支持当前Active时，恢复后续候选资格。本次5秒总上限是值得继续Shadow验证的实验配置，不是物理安全上限或已确定最优参数；超过硬上限仍RECOVERY_REQUIRED并要求外部授权，不无限自动等到某一瞬时一致。不要把Current≠Active作为提交前硬门控。

3. 将“是否继续下一轮”独立于四类感知标签。只有明确task contract才可按commit历史配置最大继续次数；未知任务应请求任务意图/外部授权，不默认为“两轮”，更不能一刀切禁止return→adjust。即使已到最后return，也只保持return并等待完成确认，不凭视频结束合成FINISH/STOP。

4. 为pickup→harvest、harvest→return保留阶段完成守卫接口。本轮无真实完成信号，未验证；Current/Future一致仍可能同时错误，不能用两头一致替代物理完成。不建议为了当前案例继续提高置信度阈值或降低ACK阈值。

5. 界面持续区分正常ACK、迟到ACK、硬异常、需要继续任务授权，显示上次commit、最早可用证据、窗口截止、迟到支持计数及拦截原因。恢复后额外提交也必须纳入评估。新策略先可选Shadow/离线，不立即替换真实控制默认值；下一阶段需要独立保留来源而非反复使用原test。

## 实施与复现

生产handshake.py/GUI默认与已有用户修改未动。本轮仅新增handshake_improvement_study.py、对应4个单元测试和本报告/证据。当前分支保留全部未提交工作，未执行commit/push。没有模型训练/Encoder/CUDA、视频转码或test评分。

最终产物：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/handshake_improvement_sensitivity_20261112`。每来源/策略JSON包含完整事件、Active/Pending和指标；per_episode.csv、paired_changes.csv、summary.json、normal_sample_audit.json、final_integrity.json用于核算。原阶段目录`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/handshake_improvement_study_20261112`保留3秒初始结果与原源码快照。

```bash
cd /home/user/pi05/vjepa2_training
.venv-tomato/bin/python -m tools.tomato_review_ui.handshake_improvement_study --run-root runs/tomato/local93_20261107 --output-dir runs/tomato/local93_20261107/improvement_NEW
.venv-tomato/bin/python -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py
```

已有目录拒绝覆盖。当前脚本复现包含3秒与5秒的完整9策略；初始7策略源码保存在第一次输出目录policy_source.py并与当时协议SHA一致。

```text
........................................................................ [ 40%]
........................................................................ [ 81%]
.................................                                        [100%]
177 passed in 11.38s
```

原321训练输出、2645个旧P1–P6产物SHA保持不变。基线精确复现85来源，当前案例与截图对应历史事件也完全一致；研究脚本主进程未导入torch。真正的物理安全、完成信号、实机Shadow/5090并发、动作块恢复、5Hz均NOT_RUN。
