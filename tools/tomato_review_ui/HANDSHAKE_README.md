# Current–Future Handshake V2

仅离线/Shadow预备，不接机器人。2Hz两个新锚点跨0.5s，不必等待1s：10.0/10.5观察、90ms后第二条在10.59可用，规划10.5/11.0时提交11.0；相位改成10.25/10.75则提交10.75。K确认不是固定defer，默认min_confirm_span=0、defer=0。5Hz必须新推理，不能复制HOLD；本轮握手5Hz NOT_RUN，保留现有5Hz识别功能但拒绝其进入本握手实验。

Current≠Active不会阻止提交；Current仅在commit后ACK。ACK是感知一致性，不是物理完成。普通ACK窗口1–2s（默认1.5）。启用迟到ACK后软超时进入WAIT_LATE_ACK；commit后总时长5s为默认硬截止。硬超时仍进入RECOVERY_REQUIRED，仅明确external_recovery可解锁，不用GT复位。Future概率不是安全切换概率。默认θ_prepare=.65，θ_commit=.8，K=1（GUI可选2），θ_ack=.7。

启动原review_ui.server；识别Joint 2Hz后“应用并重新模拟当前Episode”。改配置不触发推理/编码，点击应用才从pickup开头重放，保留视频光标。API POST /api/handshakes/simulate只接受本会话result_id+config；GET /api/handshakes/{id}；导出可附simulation_id，必须与result_id绑定。未生成Active绝不显示GT替代。未来单模型不伪造ACK。本轮无双模型ACK配对实验、无EMA、无隐藏额外等待。完整events/config/Active分段写到UI专属handshake目录，键绑定原结果、概率、全部参数和策略版本。


## WAIT_LATE_ACK（2026-11-12）

GUI默认开启迟到ACK：普通窗口1.5秒后只告警，Active/version保持；在从实际commit起5秒内，连续两个不同的新Current观测top1为Active且置信≥theta_ack，才记录late_ack并恢复READY。预测间隔超过TTL、无效输入、Current不支持Active都清空迟到计数；HELD/倒序拒绝。正常窗口内ACK依旧只需一次新观测。

软/硬截止同刻先消费已完成预测，因此硬截止恰好到达的第二个支持可确认；超过硬截止后任何预测都不能自动解锁。没有新预测时计时器仍在硬截止告警；新commit会重置窗口和支持计数。等待时不接受下一阶段请求。

新字段late_ack_enabled、late_ack_timeout_s（commit起的总时长，非软超时后再加5秒）随配置锁保存；GUI应用/恢复默认开启，旧配置缺省late_ack_enabled=false维持原行为，可取消勾选作对照。启用后summary区分正常acks、late_acks、软timeouts和hard_timeouts，事件/时间轴展示迟到支持、迟到ACK、硬异常与截止时刻。

策略缓存已有源码SHA，因此重新应用自动生成新结果，不覆盖旧日志。不改任务轮数、return→adjust规则或物理完成条件；延长观察不等于安全切换，尾部adjust误切依然可能发生。仅离线/Shadow预备，不控制机器人。
