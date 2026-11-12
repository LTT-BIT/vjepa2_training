# 我们的B0与同事离线Agent分支对照

核验日期2026-11-12。只读获取同事分支，未切换工作分支、修改模型或默认决策。

## 比较身份与范围

- 我们：`research-tomato-agent`，比较起点`855f68ec0d8984f96c3e822f4e2ee181fc323bea`，B0锁`configs/tomato_frozen_independent_20261112/b0.lock.v3.json`。
- 同事：远端`feat/tomato-offline-review-ui`，`07f93115d5427b3ebaaf1735e9bd18a1e1240aa5`。独立归档读取，没有合并。
- 截图：Joint、2Hz、Future+1s、prepare .65、commit .8、K1、ACK .7/1.5s、规划周期.5s、相位0、模拟延迟.09s、TTL .75s、margin0、minspan0、ACK和合法边启用。
- 截图有`commit_mode=planning`及提交时机选项；远端上述提交的handshake.py/DEFAULT并没有这个字段或选项，只有规划时钟提交。不能据此声称已审计截图所在机器的全部当前代码。截图也未提供当前Episode身份、模型SHA和全量数据SHA。

## 参数对照

| 项目 | 我们冻结B0 | 同事分支可核验内容 |
|---|---|---|
| 编码器 | 冻结V-JEPA2 ViT-L，256像素 | 同系列，facebook/vjepa2-vitl-fpc64-256，revision b3c1679… |
| 输入 | cam0/1/2；16帧×10Hz；58D命名位置历史 | 同样配置；均4000D融合输入 |
| 模型 | Current/Future分别优化的独立MLP；编码器共享，只运行一次 | 截图Joint；JointHead共享256D可训练trunk，两个四分类输出 |
| 分类头 | 每头4000→256→4，GELU，Dropout .3；合计2,050,568参数 | Joint 1,026,312参数；同样GELU/Dropout .3 |
| 未来目标 | t+1.5s | t+1.0s；catalog校验固定1s |
| 在线频率 | 2Hz | 截图2Hz；另有新鲜5Hz提特征工具，握手实验5Hz明确NOT_RUN |
| 优化配置 | AdamW，LR .0003，WD .01，batch128，最多100epochs，patience15 | 对应local93配置相同 |
| 多任务权重 | 独立训练；配置仍保留1:.5字段，不能理解为共享trunk上的任务竞争 | Joint Current:Future=1:.5 |
| 随机种子 | 42、43、44 | local93配置42；未找到握手三种子对照 |
| 当前数据清单 | 100完整七阶段，84/8/8；不是宣称B0已用100条重训 | local93配置75/9/9，UI报告允许七阶段与三阶段 |
| B0训练来源 | 历史checkpoint保留原训练身份；历史94清单76/10/8，与当前清单分开 | 握手锁有checkpoint SHA，但完整local93模型产物未随分支提供，无法本机核验权重 |
| Future预警 | argmax置信度≥.5，连续K2，只报告预警 | prepare≥.65，commit≥.8，截图K1，符合条件后请求切换 |
| 实际阶段确认 | Current预期下一类别≥.8且连续K2 | Future提出并提交；Current≥.7只用于提交之后ACK |
| 时钟 | 2Hz两个连续新观测相隔.5s；不是必须额外等1秒 | .09s模拟结果延迟；等下个.5s规划时钟提交；不等Future目标时刻 |
| ACK | B0无这一套后置ACK状态 | 提交后1.5s无ACK锁RECOVERY_REQUIRED，保留已切换Active，需external_recovery |
| 状态身份 | 七位置，终点return_stem；明确区分重复pickup/harvest/return | 四类NEXT循环pickup→harvest→return→adjust→pickup；version记录提交次数，但无七阶段终点约束 |
| 过期 | 独立评价/归档bridge上限2s；Memory1s，已记录差异 | prediction TTL .75s（实现依据available_at间隔/age），不能直接视为相同观测年龄指标 |

依据：我们`tools/tomato_phase_model.py`、`tomato_independent_heads.py`、`tomato_state_recovery.py`、冻结锁；同事`configs/tomato_four_stage_local93.yaml`、`configs/tomato_remote93/split.json`、`tools/tomato_review_ui/{catalog,handshake,handshake_experiment}.py`。

## 两边在解决什么问题

我们把“预警”和“Prompt确认”分开，降低Future误报直接改变Prompt的风险。但Current自己也会提前，现有实例72.00s确认、人工74.23s，提前2.23s；仍有adjust→pickup漏预警。我们仍有阶段偏移、独立泛化证据不足、真实链路延迟未闭环等问题，不能因比较对象有缺口而忽略。

同事明确区分propose/request/commit/ack/timeout并绑定模拟时钟，适合观察接口交互与候选状态。这值得借鉴。但Current≠Active不会阻止commit，后置ACK是事后感知一致性，既不是切换前确认也不是物理完成。Future预测t+1s不表示现在应切换，规划时钟等待也不保证对齐人工边界。ACK超时会持续阻断后续，可能形成级联失配。

四类合法边不是完整七阶段状态约束。`NEXT['return']='adjust'`也作用于最终return_stem；若没有外层终止管理，最终放回后仍可能进入adjust。不能仅凭七段彩条或prompt_version推断重复操作定位已解决。

## 已保存结果为什么不能直接排名

我们新8条dev的seed42 Future严格预警37/48、13误报，匹配窗口[b−1.5,b)；Prompt另算，44按时/2提前/2滞后，按时窗[b,b+2)。

同事`docs/validation/handshake_v2/val_summary.json`按54边界汇总，属于9条验证录制的离线握手探索；`handshake_experiment.score`先在[b−1,b+2]匹配提交，再把[-.2,+.5]记为strict_hits。这不是我们的Future严格预警定义。

| 同事策略 | 宽窗matched/54 | 窄窗strict_hits/54 | wrong_active秒 | ACK超时 |
|---|---:|---:|---:|---:|
| K1、不启用ACK、commit .8 | 43 | 23 | 98.23 | 0 |
| K1、启用ACK、commit .8（截图接近此配置） | 37 | 19 | 273.73 | 5 |
| K2、不启用ACK、commit .8 | 49 | 13 | 66.93 | 0 |
| K2、启用ACK、commit .8 | 46 | 12 | 188.70 | 3 |

这些是其仓库已有汇总的读取，未在本机重新运行真实local93评价。它们显示ACK可能以保守锁定换来后续失配，但不能以37/54对我们的37/48做胜负结论。其wrong_active按四类计算，不能直接对我们七位置不一致时间排名。

另一个易混点：`docs/validation/current_future/A/B/C.*`是每个split各1条录制的早期Smoke证据，不是93条正式训练性能。不要用其中test Macro-F1作为同事正式模型成绩。

## 本轮机制验证

`tools/check_tomato_colleague_mechanisms.py`对同事归档handshake.py做隔离合成输入验证，不含真实模型推断或录制精度：

1. Current持续pickup、Future持续harvest：同事默认配置.5s提交harvest、2.0s超时；我们的B0维持pickup_1。证明其ACK不是切换前门控。
2. 按七位置类别依次输入后继续提供adjust，关闭ACK以隔离拓扑：同事第7次提交进入adjust；我们的B0停在七位置末尾。证明四类循环本身没有七位置终止约束，不能推断所有实际会话都会发生此错误。

结果：`docs/validation/tomato_colleague_comparison_20261112/mechanism_probes.json`。

复现（先将固定同事提交git archive到独立临时目录）：

```bash
.venv-tomato/bin/python -m tools.check_tomato_colleague_mechanisms \
  --colleague-source /tmp/tomato-colleague-07f9311/tools/tomato_review_ui/handshake.py \
  --out /tmp/tomato-mechanism-probes.json
```

## 建议

保留B0，借鉴其只读三机回放、propose/commit/ACK事件可视化和配置身份绑定。不要直接复制Future驱动Prompt切换或四类循环。

公平对照应分别控制两件事：先在同一冻结概率序列上交换决策器，统一七阶段状态与事件定义，比较决策；再在同一隔离录制、同一标注SHA上比较Joint1s和独立1.5s，统一提前窗口，比较模型。当前先锁算法，不据反复使用的dev集重新挑阈值。截图当前模型SHA、Episode身份、未推送源码和完整local93评估身份未齐之前，不声称已完成模型胜负验证。
