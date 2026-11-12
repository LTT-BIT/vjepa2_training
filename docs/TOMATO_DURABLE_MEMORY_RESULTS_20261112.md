# TaskMemory持久化与新鲜度接线验收

2026-11-12；research-tomato-agent；起点cbd94dc。保留原冻结B0、旧Memory、标签和checkpoint，不修改OpenPI/RTC，不连接机器人。

## 本轮完成

1. 新增独立DurableTaskMemoryShadow包装器；没有修改锁定的tomato_task_memory.py或旧bridge阈值。
2. 新鲜预测与控制器已确认事实分离：观测仍须≤1s；超过1s的感知不能参与新鲜证据，但合法、相邻有序、版本连续的历史控制器事实可以迟到入账。它只证明控制器做过该决策，不证明动作成功，也不授权新的Prompt。
3. SQLite WAL/synchronous FULL持久化，每条命令、处理结果、连续序号、哈希链和提交尾部在同一事务中写入。单会话单写者；幂等command_id支持重试。
4. 重启按命令恢复原Memory，逐项校验结果；会话、Episode、模型SHA、数据SHA、joint_names、代码SHA不一致时拒绝接入，不自动清空旧数据库。
5. 新增紧凑Shadow上下文：默认最多4份近期预测、至多6条控制器有序转换事实，最大8192字节规范JSON。字节不是LLM token；没有把完整日志或机器人向量自动塞入π₀.₅ Prompt。
6. 读取必须给出同一时间域的当前时刻，禁止用更早查询时刻读取后来历史。重启本身不恢复新鲜性；过期后fresh_current为空，确认事实仍保留。

## 真实数据与中断验收

- 当前完整100条84/8/8身份保留；真实回放为开发8条×三个种子，6807次完整TaskMemory快照与原M1逐项一致，保存144条控制器确认事实（同一录制跨种子重复）。
- 24个会话各在1/3、2/3处重开数据库，结束后再重开，共72次恢复校验通过。
- 另用一条真实seed42录制，在回放一半后os._exit(77)，故意不关闭SQLite或执行析构；父进程从WAL恢复，重试最后已提交命令，再续接剩余录制。最终六条有序事实无丢失、无重复，完整Memory与不中断参考一致。
- 单元测试另覆盖事务提交前退出、提交后退出、重复消息、错误身份、跳阶段、Future冒充确认、并发写者、日志内容损坏和提交尾记录丢失。未进行真实断电或硬盘故障测试，不能称为硬件级容灾验收。
- 主回放总时间17.69s；没有训练模型、提取新视觉特征或执行ROI。

### B0行为保持一致

|种子|Future严格预警|未匹配预警|Prompt按时/提前/滞后|
|---|---|---:|---|
|42|37/48|13|44/2/2|
|43|34/48|18|44/2/2|
|44|35/48|15|43/4/1|

这表示Shadow没有改变B0决策，不代表Memory纠正了提前切换。之前adjust→pickup的严重提前仍会被忠实记录，physical_success始终unknown。当前模块没有独立学习或状态恢复决策能力。

## 状态映射与ROI边界

- 对84训练+8开发共92条录制，按真实关节名称从源116D实测state取58个position：与原数据读取器、全部缓存窗口的16×58历史逐值一致。历史测试8条未参与评价。
- 只读解析当前OpenPI ROI源码的实际状态切片，未执行该定位器。其58D顺序为左臂7+左手22+右臂7+右手22；Agent是左臂7+右臂7+左手22+右手22。
- ROI输出转Agent的索引排列为0..6、29..35、7..28、36..57。实现使用精确名称，不以这组裸下标作为一般输入假设，也不把velocity或action当作position。
- OpenPI只读版本：6baa8e6c8b88c2fb1cece6b54d298187a90a1f77。名称映射验收通过，不等于ROI物体身份、因果框可用时间或图像像素链路验收通过。
- 原四路RGBStateMemory/RGBMemoryReader保留，Reader仍未训练；完整100条Agent四路ROI缓存仍未准备好，未用人工框、复制全图或上采样旧预览冒充自动局部观测。

## 时效原则与限制

- 新鲜感知：仍是原TaskMemory的1s门槛；旧bridge若允许2s，不因此放宽Memory。
- 控制器事实：独立历史账本保留真实接收的控制器确认时间、观测时间、Prompt版本；不能把Future预警写成事实，不能跳过重复类别对应的有序位置。
- perception字段是旧M1新鲜输入流的记录视图，并非新的独立状态估计器。迟到事实可能使它落后于controller_reported_phase；用显式disagree标识，绝不以真实阶段或静默改状态消除差异。
- 新鲜度测试使用既有离线观测时间+0.04s到达fixture，并额外注入1.5s迟到案例；不是ROS/网络时延实测。
- journal用于恢复审计，不作为模型输入。哈希链用于一致性与损坏检测，不代表外部控制器身份的密码学认证。

## 上下文预算与资源

- 24份恢复后紧凑上下文验收通过；最大规范UTF-8 JSON为3375字节，低于8192字节上限。过期读取移除fresh_current，但六条确认事实仍可追溯。
- 预算不够时先减少近期概率历史；若身份与完整已确认前缀仍放不下，拒绝输出，不偷偷截断事实。未测π₀.₅ tokenizer token数，且没有接入其Prompt。
- 单条持久命令P95跨会话范围0.677–0.758ms，最大单条5.030ms；包括本机SQLite提交，不含采集、网络、编码或机器人控制。恢复采用顺序重放，当前只验收单Episode规模，不宣称无界日志恢复耗时固定。

## 测试与下一步

新增15项Memory持久化/因果上下文测试通过；全部番茄专项339 passed/1 skipped。B0锁、数据、原标签与checkpoint身份核验通过。没有重新启动已知缺失模块的柠檬全仓库测试。

**本轮完成的是可持久恢复、时效清晰、命名state对齐的Shadow记忆底座，不是学习型Memory效果提升。** 下一步若训练视觉Memory，先独立验收因果ROI的原图来源、可用时间和缓存身份；随后固定同一个Agent做无Memory/只记录/历史参与读出的单因素对照。原Subtask复用，8770补标继续暂停，没有成功标签就不训练操作成功验证器。

## 复现与产物

```bash
.venv-tomato/bin/python -m pytest -q tests/test_tomato_durable_memory.py
.venv-tomato/bin/python -m tools.audit_tomato_memory_state --out runs/tomato/memory_state_audit_NEW
.venv-tomato/bin/python -m tools.replay_tomato_durable_memory --out runs/tomato/durable_memory_NEW
.venv-tomato/bin/python -m tools.report_tomato_durable_memory --run runs/tomato/durable_memory_NEW
```

回放当前显式引用已验收的`runs/tomato/memory_state_audit_20261112_v1/state_mapping.json`，先核对该文件及receipt内SHA；不覆盖旧run。完整数据库、真实回放和紧凑上下文保留本地`runs/tomato/durable_memory_20261112_v2/`；代码、配置与可审查摘要提交仓库，不提交数据库或大缓存。
