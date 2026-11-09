# Current–Future Handshake V2：主方案与补充说明联合实施报告

日期2026-11-09；当前分支feat/tomato-offline-review-ui，基线HEAD `d40757b2ee15f9eb7fd37e95d882ccd80cebc35b`。本轮修改未提交。主方案后补到达，与V2合并审查，冲突以V2为准。

## 状态与范围

| 项目 | 状态 |
|---|---|
| 纯策略、时钟回放、val评分 | PASS |
| GUI阈值/K/ACK超时/高级设置、独立Active/Pending/ACK轴 | PASS |
| 9来源val消融、逐转换/来源CSV与事件JSON | PASS · EXPLORATORY |
| 七阶段/三阶段自动时间轴与MP4样例 | PASS |
| test新评分/调参 | NOT_RUN |
| 握手5Hz、相机30→10/5Hz退化、实时质量遥测 | NOT_RUN：V2本轮限定冻结2Hz |
| 5090+π0.5并发、ROS/RTC/T3、真实Shadow部署 | NOT_RUN |
| Completion/FINISH | NOT_SUPPORTED |

已有真实5Hz识别功能保留，不修改其实现；握手API只接受Joint 2Hz。Future-only没有Current，不伪造ACK，本轮双模型Current-only配对对照未做。主方案H0～H5命名与V2不同，以下H1/H2/H3/H4严格按V2分别表示K1无ACK/K2无ACK/K1有ACK/K2有ACK；另增BASE_H0_raw与LEGAL_OFF_K2_ACK独立参考。

## 状态与时间语义

初始pickup为启动先验；READY→CANDIDATE_PENDING→SWITCH_REQUESTED→WAIT_CURRENT_ACK→READY，超时RECOVERY_REQUIRED保留Active，要求明确外部recover事件才解锁。Current仅做提交后ACK，绝不以Current≠Active拒绝提交。候选记录anticipatory/catch_up/disagreement；无GT纠偏，无未训练FINISH。

默认prepare=.65、commit=.8、margin=0、K=1可选2、最小确认跨度0、ACK阈值.7、ACK超时1.5s、合法后继开；defer恒为0。连续K条不同新锚点均满足prepare，只有最新一条要求commit/margin，不是K次都超过commit。K2在2Hz天然跨0.5秒：10.0/10.5s观察，90ms延迟后第二条10.59s可用，规划相位0提交11.0s；相位.25提交10.75s。没有额外固定0.5秒等待。

ACK仅取observed_at严格晚于commit的Current，且在超时前可用；同刻先结果、再timeout、再规划。超时1–2s自实际commit起计，绝不自动进入下一Prompt/重置GT。PromptSnapshot为frozen dataclass，包含active_prompt/version/committed_at；只提供Shadow接口，不驱动动作块，旧chunk排空与真实安全守卫尚未接入。

策略Evidence只含观测/可用时间、两头概率和显式有效标志，不接收GT、future_valid、真实边界、Episode长度。录制终止仅由回放驱动器安排，不传入策略。invalid_observation撤回候选，stale与倒序/重复锚点明确处理；真实camera_age/skew/joint_age/独立历史帧质量阈值未测，不能宣称已有完整传感器质量模块。

## val预注册与结果

全部冻结Joint 2Hz，prepare=.65，commit=.80/.90/.95，K1/K2、ACK开关，ACK超时默认1.5；另在commit=.8比较1/2秒。主方案补到后按要求添加raw与关闭合法边参考，共18配置。配置与输入/权重/策略源码SHA先写protocol.lock.json，再跑9条val，没有test评价或根据test选择。这里只做EXPLORATORY对照，不锁定上线最优策略。

| 配置 | 匹配/54 | 未匹配提交 | 提前>.2/≥.5 | 严格窗口命中 | Δt P50/P95秒 | 错误Active秒 | 完整顺序/9 | ACK/提交 | timeout |
|---|---|---|---|---|---|---|---|---|---|

| H1_t0.8_ack1.5 | 43/54 | 11 | 8/4 | 23/54 | 0.267/1.153 | 98.233 | 9/9 | N/A | 0 |

| H3_t0.8_ack1.5 | 37/54 | 9 | 8/4 | 19/54 | 0.233/1.060 | 273.733 | 6/9 | 41/46 | 5 |

| H2_t0.8_ack1.5 | 49/54 | 5 | 8/5 | 13/54 | 0.600/1.393 | 66.933 | 9/9 | N/A | 0 |

| H4_t0.8_ack1.5 | 46/54 | 4 | 8/5 | 12/54 | 0.583/1.333 | 188.700 | 7/9 | 47/50 | 3 |

| H1_t0.9_ack1.5 | 45/54 | 9 | 5/3 | 22/54 | 0.400/1.313 | 60.133 | 9/9 | N/A | 0 |

| H3_t0.9_ack1.5 | 43/54 | 9 | 5/3 | 21/54 | 0.400/1.323 | 126.000 | 8/9 | 49/52 | 3 |

| H2_t0.9_ack1.5 | 50/54 | 4 | 8/5 | 13/54 | 0.617/1.562 | 52.567 | 9/9 | N/A | 0 |

| H4_t0.9_ack1.5 | 48/54 | 4 | 8/5 | 12/54 | 0.617/1.550 | 117.633 | 8/9 | 50/52 | 2 |

| H1_t0.95_ack1.5 | 46/54 | 8 | 5/3 | 20/54 | 0.467/1.633 | 68.500 | 9/9 | N/A | 0 |

| H3_t0.95_ack1.5 | 45/54 | 7 | 5/3 | 19/54 | 0.500/1.640 | 96.300 | 8/9 | 50/52 | 2 |

| H2_t0.95_ack1.5 | 50/54 | 4 | 8/5 | 12/54 | 0.633/1.703 | 60.633 | 9/9 | N/A | 0 |

| H4_t0.95_ack1.5 | 48/54 | 4 | 7/4 | 11/54 | 0.650/1.710 | 88.500 | 8/9 | 51/52 | 1 |

| H3_t0.8_ack1.0 | 33/54 | 5 | 8/4 | 16/54 | 0.233/1.087 | 411.567 | 4/9 | 32/38 | 6 |

| H3_t0.8_ack2.0 | 37/54 | 9 | 8/4 | 19/54 | 0.233/1.060 | 273.733 | 6/9 | 41/46 | 5 |

| H4_t0.8_ack1.0 | 46/54 | 4 | 8/5 | 12/54 | 0.583/1.333 | 188.700 | 7/9 | 47/50 | 3 |

| H4_t0.8_ack2.0 | 46/54 | 4 | 8/5 | 12/54 | 0.583/1.333 | 188.700 | 7/9 | 47/50 | 3 |

| BASE_H0_raw | 46/54 | 45 | 13/8 | 25/54 | 0.050/1.108 | 54.567 | 2/9 | N/A | 0 |

| LEGAL_OFF_K2_ACK | 46/54 | 7 | 8/5 | 12/54 | 0.583/1.333 | 192.100 | 5/9 | 49/53 | 3 |

### 结论边界

K2增加确认时间，但此val网格中无ACK的K2在三个阈值上均提升匹配、降低错误Active时长。例如.8时K1→K2：43→49/54，错误98.2→66.9秒，中位偏差.267→.600秒。提升来自确认/规则交互，不能外推新来源或物理安全。

两个阈值职责不同；.728→.876在prepare=.65/commit=.8/K2下第二条即Request，单元测试证明；.728→.60撤销。网格固定prepare，没有单独prepare消融，不能从本轮推断最优prepare。无ACK K2在.90时错误Active约52.6秒，.95为60.6秒且匹配同为50/54；有ACK K2在.95时较少超时，但不据此推荐部署或对test复选。

ACK会造成真实的离线中断：.8/K1时1秒timeout为6/38（15.8%），1.5秒5/46（10.9%），2秒仍5/46；K2在这三种timeout均3/50（6%）。超时后保持Active导致后续流程不完整、错误时长增加，这些是显式RECOVERY_REQUIRED，不是静默卡住或自动恢复。恢复处理次数均0：离线不注入人工/GT复位。

旧P3_d50仅历史参考：先前Joint val46/54、错误77秒、Δt P50约1.083秒；新K2/noACK/.8是49/54、66.9秒/.600秒，有ACK则46/54、188.7秒/.583秒。合法边、ACK及状态逻辑不同，不能把差异单独归因于去掉defer；V2明确要求不将旧P3当公平单因素对照，本轮没有重选旧策略。

## 逐事件与来源/阶段评分

每配置每Episode JSON含完整propose/support/cancel/request/request_withdrawn/commit/ack/timeout/blocked事件、Active/Pending区间和metrics。独立GT scorer在完整回放后按精确阶段对、单调一对一匹配[-1,+2]秒内先最大化数量、再最小化绝对偏差；严格窗口[-.2,+.5]按主匹配δ统计。

per_transition.csv保留gt_boundary、first_future_candidate、pending_ready、request、commit、ACK、δ、首次/最终return、短return标记；per_episode.csv保留9来源的错误时长/顺序/ACK/timeout。JSON waits独立记录prepare→request、request→commit及首/末观察可用时间；ack_waits独立记录commit→ACK。failure_cases.jsonl保留漏检/过早及timeout/撤回/非法候选。短return阈值≤1.5秒，若某来源没有短return不能将0漏检解读为100%召回。所有GT序号只用于事后评分，不进入状态机。

## GUI/API与导出

启动现有server后选择Joint和2Hz、点击识别，再点击“应用并重新模拟当前Episode”。最小必需阈值有滑条+数字框，K与ACK超时直观可选；高级设置有margin/跨度/ACK阈值/周期/相位/延迟/合法边/ACK开关。修改后标为待应用并清除旧Active，手动应用才从开头重放，保持播放光标。恢复默认、下载配置JSON可用。

原GT/Current/Future轨保留；新增Pending、Simulated Active（仅commit_at变化）、Request/Commit/ACK/Timeout标记。未运行显示“未生成”，不以GT代替。详情包含配置、当前Active/Pending状态、证据K和请求/提交/ACK时间；RECOVERY_REQUIRED可见。事件按钮可seek。缓存键绑定原识别result、概率、全部配置及策略源码SHA，生成独立handshake目录。

POST /api/handshakes/simulate接受本会话result_id+config；GET /api/handshakes/{id}按会话授权。POST /api/exports/video可选simulation_id，严格要求与本会话result_id一致；修改参数须重新应用才可叠加，旧模型识别与导出兼容。导出使用独立key，800px高画布保留GT/Current/Future/Active/Pending和状态事件，旁存handshake.json；普通不叠加仍680px。

## 真实样例与浏览器验证

产物根目录：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/handshake_v2_combined_20261109_v2`。选择最短七阶段和特殊三阶段仅作功能示例，默认配置生成两份带握手视频，不作独立精度证据。

- 7段：`runs/tomato/local93_20261107/handshake_v2_combined_20261109_v2/ack_overlay_exports/ui_exports/7ed8a2815bb81e0be7bb89f8/joint/a86803b4121ba1fe085a0ed8c700aece7b426fbb489f52e59d18355522474ade/annotated.mp4`，帧数2234，耗时13.91s；配置和状态逐帧绑定sample_7_timeline.json。

- 3段：`runs/tomato/local93_20261107/handshake_v2_combined_20261109_v2/ack_overlay_exports/ui_exports/1df7c31ae9a58dc60d4e57d6/joint/6e0b122f6e36c2205ba33b68c97a5986378dd912160169636330dd5538f451a6/annotated.mp4`，帧数1065，耗时5.32s；配置和状态逐帧绑定sample_3_timeline.json。

自动浏览器在七段/三段上实际识别Joint2Hz并模拟，frame37光标不变；margin改动清旧结果，应用后新值生效；非法prepare>commit被拒绝，默认恢复可用。截图是高级margin=.1的交互示例，MP4是默认margin0，二者未宣称为同一配置。

![七段GUI](/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/handshake_v2_combined_20261109_v2/gui_7.png)

![三段GUI](/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/handshake_v2_combined_20261109_v2/gui_3.png)

## 测试、保护及复现

```text
........................................................................ [ 48%]
........................................................................ [ 96%]
.....                                                                    [100%]
149 passed in 11.28s
```

原始训练/预测321个文件SHA保持不变，实验主进程torch_imported=False。没有运行Encoder、CUDA、训练或修改GT。既有5Hz识别相关回归也保留执行。旧P1–P6源码本工作树缺失，保留其产物，不覆盖其策略或报告。

```bash
.venv-tomato/bin/python -m tools.tomato_review_ui.server --run-root runs/tomato/local93_20261107 --port 8781
.venv-tomato/bin/python -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py
.venv-tomato/bin/python -m tools.tomato_review_ui.handshake_experiment --run-root runs/tomato/local93_20261107 --output-dir runs/tomato/local93_20261107/handshake_NEW
```

当前验证页面http://127.0.0.1:8781；原8770如已在运行需重启才加载新增后端。新实验目录必须不存在。18配置完整CSV/JSON保存在runs，概要与协议/回归日志复制到docs/validation/handshake_v2。实现为新增handshake.py、handshake_service.py、handshake_experiment.py、HANDSHAKE_README.md与tests/test_tomato_handshake.py，扩展原server/HTML/JS/video_export；全部当前分支未提交。

主方案后补到达后补齐margin/关系分类/快照/CSV；一次追加失败索引时因candidate字段名重复导致val编排中止，修复序列化，在新目录完整重跑，未更改模型或测试集策略。旧输出保留。浏览器高级设置测试初次未展开details导致测试等待，展开后验证通过，非UI参数失效。

## 下一步与尚未解决问题

尚未实现完整传感器质量遥测/降采样重推理，未运行本握手5Hz或5090+π0.5并发。既有5Hz能力不是本轮握手5Hz证据。Evidence.valid只提供上游质量拒绝接口，不代表已验证camera_age/skew/joint_age阈值。

Current ACK只表明感知一致性，不能证明摘完、抓稳或放回。FINISH仍需外部完成机制；两个return只在事后分段统计中区分。真实接入前需要测量规划边界、动作chunk版本/排空契约、安全守卫、急停、通信超时及明确恢复授权。本轮只有不可变快照与外部recover接口，没有真实控制路径或自动解锁。

视频另以ACK_at/timeout_at持久标识复核导出，最终metadata见sample_3_video_ack_overlay.json与sample_7_video_ack_overlay.json；普通识别结果、旧视频及训练文件不覆盖。最终全回归仍149通过。
