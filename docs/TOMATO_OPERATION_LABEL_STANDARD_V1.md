# 番茄独立操作结果标签规范v1

此规范是新标签任务，不等于已有pickup/harvest/return/adjust。阶段只定位候选区间；阶段边界、文件normal/C分组、关节闭合、手臂抬升都不能直接生成成功标签。初始全部未审核，Unknown也必须是实际审核后的结论；不能把未审核自动当Unknown或negative来训练。

|事件|Positive独立可见条件|Negative可见反例|Unknown|
|---|---|---|---|
|right_grasp_started|右手新的接触/闭合抓握序列可见|明确仅在移动或张手，没有新的抓握尝试|接触被遮挡、无法分清调整/抓握|
|truss_stably_lifted|番茄串离开支撑并随右手持续共同运动|右臂空抬、目标仍在盘上或明确滑落|盘面/物体遮挡，无法确认共同运动|
|left_pick_started|左手新的摘取接近/接触可见|仅返程/空手经过，明确未接近目标|目标不可见或接触关系不清|
|tomato_detached|果实从果梗分离的直接视觉证据，且后续可见状态支持|果实可见且仍连在梗上，或明显失败尝试|接触瞬间遮挡、离开画面，无法确认分离|
|truss_returned|物体受盘面稳定支撑并有释放/卸载证据|物体仍被举持，或返程但目标未落盘|放置和释放被遮挡|
|adjustment_ended|可见调整序列终止的动作锚点，并能与短暂停顿区分|明确仍在重新定位或继续调整|仅静止或手指姿态不足以确定结束|
|right_regrasp_started|新的第二次接触/闭合序列可见|仍是原调整或未开始抓握|缺少历史上下文或接触不可见|
|second_grasp_verified|新的物体-右手稳定共同运动/持有证据|空抓、滑落或目标仍在盘面|物体遮挡，不能判断是否真正持有|

“开始尝试”和“结果验证”是两类不同语义。即使两者都为positive，开始尝试不能作为结果成功的训练替代。Negative必须具备可观察条件；看不见不是negative。

帧区间为原始30Hz的[start_frame,end_frame)，不得按裁剪后视频起点写回。evidence_end_frame是作出当前结论最早所需的最后观测帧，必须位于区间内。Oracle历史只有在这个帧已到达后才释放；不能用区间开始提前释放。模型只读当前及过去的特征。标签区间应覆盖物理条件真实可判断的时间，不为凑2Hz样本随意延长。

两名审核者独立填写直接证据说明、可见相机和帧区间。每个导出保留源标注SHA和审核者身份；完全一致才由merge工具批准，分歧交人工裁决，不自动平均边界。相机角色尚无归档映射证据，先保留cam0/1/2，不擅自称其右腕/左腕。工具不展示模型分数，阶段标签仅用于后台寻找候选。

首批741个候选来自84条训练和9条开发验证，历史8条测试不进入审核选模队列。一个harvest候选不能代表所有摘取；当前页面不提供新增候选功能，后续必须扩充候选目录、逐次独立审核真正可见的分离事件后才可能讨论计数。当前版本不分配Tomato ID、不把阶段次数当摘取数，picked_count一直为Unknown。

打开`http://127.0.0.1:8770/`：选项目→查看三路视频→逐帧定位→填写正/负/Unknown及证据→保存个人意见→分别导出两位审核文件。未审核项不会导出为标签。两名审核者应使用独立浏览器配置/机器，不能在同一LocalStorage共享意见后声称独立。

```bash
.venv-tomato/bin/python -m tools.serve_tomato_operation_review --directory runs/tomato/operation_memory_20261110 --port 8770
.venv-tomato/bin/python -m tools.prepare_tomato_operation_review --merge /path/reviewer_a.json /path/reviewer_b.json --out /new/path/consensus.json
```

当前候选区间不是穷尽标注。首轮先审核一组训练/开发录制的正例、明确失败和遮挡，检查标签一致性与有效样本支持，再开启A/B/C/D低成本训练。已有101条均已曝光或参与开发，新的独立盲测需另行录制并在规则锁定后一次验收。
