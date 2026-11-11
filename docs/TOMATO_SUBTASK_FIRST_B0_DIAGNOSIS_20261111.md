# 回到现有Subtask：暂停重复补标与B0漏检诊断

2026-11-11。核验实际起点为dbfe4ed，research-tomato-agent，工作区原本干净；最新RGB+完整状态Memory模块保留。本轮仅调整补标工作流、复用现有B0缓存诊断，不改其模型结构、权重、RTC或机器人执行。

## 调整已经落实

8770主页已改为“你现在不需要再标一遍”。16条/128项不再作为必须完成的任务，原工具、页面版本、原Subtask和已保存意见保留；暂停页只提供可选的旧意见只读导出，不会保存新标注。默认配置标记paused_no_duplicate_annotation，操作结果监督实验不再作为Agent研究前置。

此前只因文件里没有独立result字段，就默认用户的Subtask缺少完成信息，这个推断不充分。用户确认原标准清晰；现在以现有清晰Subtask区间和完成判据作为阶段训练、进度历史及评估依据。先复用，不要求重复标注。也不把阶段名称自动重编码为一批新的物理成功标签。

已有原始区间、14项历史复核提案及其版本分别保留，未新增边界建议、未自动修标签。100条仍84训练/8开发验证/8历史测试，manifest SHA256 `95679c2f4457701c3898e538b83883c27106aa3aa44c647ad046b92dd20410e1`；新8条B0仍37/48、13次误报、0.69488次/分钟，旧9条39/50不混用。

## 11次严格Future漏检，不等于11次当前阶段确认失败

重新执行固定B0 .5/K2 Future接受事件、.8/K2 Current确认：每条七阶段轨迹和预警事件与缓存逐项相符。严格预警仍要求相同from/to的一对一事件在[b−1.5,b)，不改变门槛和时距，不降低K，不将事后确认回填到首次候选时间。

|直接机制|次数|含义|
|---|---:|---|
|预警过早|4|在窗口开始前已接受下一类；窗口内虽持续预测正确类别，也不重复产生转换事件|
|K2确认跨边界|4|窗口最后一次观测才出现达标下一类，第二次连续达标已经在真实边界后|
|候选闪现后不稳定|1|窗口内短暂出现下一类，随后回落，直到边界后约4.13秒才稳定发出事件|
|窗口内无达标候选|2|未来头在允许预警窗口内仍未给出满足现有条件的下一类|

分类型漏检：adjust→pickup 6次、pickup→harvest 2次、return→adjust 2次、harvest→return 1次。adjust→pickup的6次中：过早2次、K2跨界3次、短暂候选不稳定1次。

这11次中Current Prompt按时8次、提前1次、迟到2次。两次迟到的首次独立Current0.8/K2候选就在+3.867和+4.633秒，与实际Prompt确认时刻一致；在这11次相关首次Current候选处未发现七阶段顺序阻挡。

**B0的Future预警没有被七阶段状态过滤。** 因此这些严格Future漏检不能归咎于在线阶段状态机阻断，历史B2硬门控/状态偏移结论不能直接套到当前B0。正确过去锚点可以改善Current确认，但H1/H2已证明不会改变这一条Future预警流。

## 全部漏检清单

下面时间均相对原有真实边界。相关预警事件可有多次，逐事件原始列表和窗口概率保存在analysis.json；表格不是重新匹配后的得分。

|录制|转换|Future机制|相关Future事件偏移秒|Prompt偏移秒|Prompt结果|
|---|---|---|---|---:|---|
|A/1103_620_642_normal_lerobot/episode_000000|adjust->pickup|预警事件过早|-4.233|-2.233|early|
|A/1103_620_642_normal_lerobot/episode_000000|pickup->harvest|窗口内没有达标候选|+0.500|+1.000|on_time|
|A/1104_652_697_normal_lerobot/episode_000002|return->adjust|预警事件过早|-2.667|+0.833|on_time|
|A/1104_652_697_normal_lerobot/episode_000002|adjust->pickup|预警事件过早|-2.633, +4.367|+3.867|late|
|A/1104_652_697_normal_lerobot/episode_000002|pickup->harvest|预警事件过早|-2.767|+0.733|on_time|
|A/1104_652_697_normal_lerobot/episode_000009|adjust->pickup|K2确认跨过边界|+0.200|+1.200|on_time|
|A/1106_702_739_normal_lerobot/episode_000004|return->adjust|窗口内没有达标候选|+0.800|+1.300|on_time|
|A/1106_702_739_normal_lerobot/episode_000004|adjust->pickup|候选闪现后不稳定|+4.133|+4.633|late|
|A/1106_702_739_normal_lerobot/episode_000004|harvest->return|K2确认跨过边界|+0.067|+0.567|on_time|
|A/1107_783_783_normal/episode_000000|adjust->pickup|K2确认跨过边界|+0.233|+0.733|on_time|
|B/1106_750_791_normal/episode_000005|adjust->pickup|K2确认跨过边界|+0.400|+1.900|on_time|

## 典型证据

黑线为固定Subtask边界，绿色是严格提前1.5秒窗口，橙线是实际发出的Future事件，紫线是实际Prompt确认。这里只看冻结概率，不据此声称发生了遮挡、标签错误或视觉证据缺失。

![过早预警与过早Current确认](/home/user/vjepa2_training/docs/validation/tomato_subtask_first_20261111/3e645bf4e137d0f836c9d7ff_4.png)

第一例adjust→pickup：Future提前4.233秒接受pickup，Current也提前2.233秒确认。评价窗口内Future已持续输出pickup，无法再次发出新的接受事件。优先问题是时间校准，不能要求人重新划阶段来迁就预测。

![K2确认跨过边界](/home/user/vjepa2_training/docs/validation/tomato_subtask_first_20261111/5b82f4bb76e70d8bf12c0798_4.png)

第二例窗口内最后一次观测（边界前0.3秒）才达标，K2事件在边界后0.2秒。2Hz确认会增加0.5秒等待，但不能据此直接降低K或回填预警时间；必须另行公平验证误报代价。

![短暂候选与迟确认](/home/user/vjepa2_training/docs/validation/tomato_subtask_first_20261111/f53f0b6fb7f3888f87a0f8ff_4.png)

第三例adjust→pickup：边界前有一次候选，后面回落，Future边界后4.133秒才稳定切换，Current确认晚4.633秒。这里应分析模型持续证据，而非新增同义人工标签。

## 下一步研究优先级

1. 维持100条、原Subtask和固定B0，Memory先保存有来源的预测阶段历史；不把自己的预测当Oracle或确认成功。
2. 优先研究Future预测时机与Current持续置信度。训练监督仍用现有Subtask的未来标签与边界，先检查训练分组上的时序校准/确认策略，不从历史测试集选阈值。预注册误报约束，不能用更宽窗口或事后时间回填虚增37/48。
3. 自动阶段状态估计仍有研究价值，但这11例的证据不支持把首要原因统称状态约束卡住；也不支持立即扩大长期记忆或训练操作成功验证器。
4. 只有某个具体错误在现有标注和观测下确实无法解释，才明确指出所缺信息、检查原标准并讨论局部补充；不预设要补满16条/128项。

## 复现与边界

```bash
cd /home/user/vjepa2_training
PYTHONPATH=/tmp/tomato-transition-plotdeps .venv-tomato/bin/python -m tools.analyze_tomato_b0_misses --out runs/tomato/new_subtask_first_unique --plots
.venv-tomato/bin/python -m pytest -q tests/test_tomato_b0_misses.py
/home/user/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node tests/test_tomato_review_paused_ui.cjs
```

所有11例曲线与局部概率在runs/tomato/b0_subtask_first_20261111_release，Git只保存可核查摘要和3张典型曲线。诊断工具不读取测试数组、不拟合参数、不更换模型、不改动任何原始标签；没有新增训练或真实机器人操作。
