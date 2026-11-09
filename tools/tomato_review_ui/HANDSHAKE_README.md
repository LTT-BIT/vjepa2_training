# Current–Future Handshake V2

仅离线/Shadow预备，不接机器人。2Hz两个新锚点跨0.5s，不必等待1s：10.0/10.5观察、90ms后第二条在10.59可用，规划10.5/11.0时提交11.0；相位改成10.25/10.75则提交10.75。K确认不是固定defer，默认min_confirm_span=0、defer=0。5Hz必须新推理，不能复制HOLD；本轮握手5Hz NOT_RUN，保留现有5Hz识别功能但拒绝其进入本握手实验。

Current≠Active不会阻止提交；Current仅在commit后ACK。ACK是感知一致性，不是物理完成。超时1–2s（默认1.5）保留Active并锁定RECOVERY_REQUIRED，只有明确external_recovery可解锁，不用GT复位。Future概率不是安全切换概率。默认θ_prepare=.65，θ_commit=.8，K=1（GUI可选2），θ_ack=.7。

启动原review_ui.server；识别Joint 2Hz后“应用并重新模拟当前Episode”。改配置不触发推理/编码，点击应用才从pickup开头重放，保留视频光标。API POST /api/handshakes/simulate只接受本会话result_id+config；GET /api/handshakes/{id}；导出可附simulation_id，必须与result_id绑定。未生成Active绝不显示GT替代。未来单模型不伪造ACK。本轮无双模型ACK配对实验、无EMA、无隐藏额外等待。完整events/config/Active分段写到UI专属handshake目录，键绑定原结果、概率、全部参数和策略版本。
