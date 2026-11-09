# 新上传10条番茄录制：冻结模型独立评估（2026-11-09）


**结论：未知录制上的阶段转换泛化明显弱于旧验证参考。** 主测Current Macro-F1为0.94845、Future1.5秒为0.91219；严格预警37/60（61.7%），Precision54.4%、事件F1为0.578、未匹配预警31次（1.61次/分钟），命中事件中位提前0.70秒。旧验证39/50是在另一组录制和部分修订提案边界下取得，不能直接把差值视为同一评价集合上的模型退化；这批独立录制证实原78%不能直接外推。

三种子严格预警召回60.6%±5.1个百分点，误报1.64±0.11次/分钟。按录制bootstrap的主测召回95%区间48.3%—75.0%，样本量仍小。新adjust→pickup为4/10，与旧2/8属于不同录制，不代表模型经过本轮改善。下一轮必须继续保留本次固定结果；如果使用这10条做误差开发，需要另录新的封存盲测。

未匹配预警中12次过早、17次迟报、2次错误转移类型；23个漏检都能在其他时刻找到同类转换（最近事件11次过早、12次迟报），提示时间错位仍是主要数值现象，不是证明视频原因。Current分类事件52/60，中位确认延迟0.733秒；这不等于真实七阶段Prompt决策成功率。

时间头固定诊断B1为28/60、6次误报；B2为13/60、3次误报，均无adjust→pickup命中。它们依然不能替代B0，不按新测试结果改阈值，也不改选seed43。

## 测试身份与边界

数据来自28服务器的A/1109_783_818_normal（episode1—7）和B/1109_696_811_normal（episode1—3）。包内两个未标注编号0条目未纳入。10条均有已保存、reviewed且非inherited的七段粗标注。主测前锁定模型SHA、阈值0.5/K2、2Hz和匹配规则；新数据全部只用于评估，没有训练、选epoch、改阈值、重拟合归一化或选择新模型。

按source_recording_id与旧94条全量对照无交集；全部选中Parquet与视频SHA和旧数据、彼此不重复。远端和本地每文件SHA一致。三路完整解码的帧数/PTS、58D按名称对齐、因果16帧×10Hz历史均核验。粗阶段标注不等于逐条物理采摘成功已人工复核；没有凭模型分数排除新测试样本。

这是此前未用于训练/验证的录制，仍可能共享场景/操作者。只能支持未知录制泛化，不等于跨场景/跨机器人的泛化。现在评分已揭示，这批数据若以后用于调参就不再是新的盲测；本轮不反复搜索。

## 固定模型结果

|模型|Current Macro-F1|Future Macro-F1|Current事件|Future事件|Future Precision|Future误报/分钟|Future中位提前秒|共同1秒窗命中|
|---|---:|---:|---|---|---:|---:|---:|---:|
|independent_h15_s42|0.94845|0.91219|52/60|37/60|0.544|1.610|0.700|30|
|independent_h15_s43|0.95185|0.90884|54/60|39/60|0.565|1.558|0.700|31|
|independent_h15_s44|0.95342|0.91067|52/60|33/60|0.493|1.765|0.833|26|
|joint_s42|0.94003|0.92009|51/60|21/60|0.318|2.327|0.467|21|
|independent_h10_s42|0.94845|0.93361|52/60|26/60|0.406|1.965|0.433|26|

不同Future时距分别评估固定1秒/1.5秒目标，不能直接按Future F1排名；共同1秒事件窗作为额外公平对照。Current事件指分类流确认，不冒充七阶段Agent/Prompt实际确认。所有模型保留原训练统计；没有用新测试进行归一化适配。

## 主测逐录制

|原数据|Current F1|Future F1|预警命中|未匹配预警|
|---|---:|---:|---|---:|
|A/1109_783_818_normal/episode_000001|0.9535|0.9191|2/6|4|
|A/1109_783_818_normal/episode_000002|0.9310|0.9024|5/6|3|
|A/1109_783_818_normal/episode_000003|0.9199|0.9209|5/6|1|
|A/1109_783_818_normal/episode_000004|0.9538|0.9204|5/6|3|
|A/1109_783_818_normal/episode_000005|0.9902|0.9461|3/6|3|
|A/1109_783_818_normal/episode_000006|0.9523|0.8703|2/6|5|
|A/1109_783_818_normal/episode_000007|0.9081|0.8555|4/6|4|
|B/1109_696_811_normal/episode_000001|0.9878|0.9697|5/6|1|
|B/1109_696_811_normal/episode_000002|0.9501|0.9183|4/6|2|
|B/1109_696_811_normal/episode_000003|0.9316|0.8684|2/6|5|

## 转换类型与不确定性

{'pickup->harvest': {'total': 20, 'detected': 15}, 'harvest->return': {'total': 20, 'detected': 13}, 'return->adjust': {'total': 10, 'detected': 5}, 'adjust->pickup': {'total': 10, 'detected': 4}}

{'recall': {'mean': 0.6055555555555555, 'sample_sd': 0.05091750772173154}, 'false_per_minute': {'mean': 1.6443098225876245, 'sample_sd': 0.10809170053480567}}

{'primary_future_recall_recording_bootstrap_ci95': [0.48333333333333334, 0.75], 'recordings': 10, 'draws': 3000, 'scope': 'same scene/operator may limit independence beyond recording'}

主测漏检数值归因：{'too_early': 11, 'late': 12}；未匹配预警归因：{'too_early': 12, 'late': 17, 'wrong_transition_type': 2}。这是时序证据，不凭它推断视觉遮挡或篡改人工边界。逐事件清单见missed_events.json/false_events.json。

## 新时间头固定诊断

{'B0': {'total': 60, 'detected': 37, 'missed': 23, 'precision': 0.5441176470588235, 'recall': 0.6166666666666667, 'f1': 0.578125, 'false_events': 31, 'false_per_minute': 1.609692773691043, 'minutes': 19.258333333333333, 'lead_p50_s': 0.7000000000000028, 'lead_p05_s': 0.1, 'by_transition': {'pickup->harvest': {'total': 20, 'detected': 15}, 'harvest->return': {'total': 20, 'detected': 13}, 'return->adjust': {'total': 10, 'detected': 5}, 'adjust->pickup': {'total': 10, 'detected': 4}}}, 'B1_gate': {'total': 60, 'detected': 28, 'missed': 32, 'precision': 0.8235294117647058, 'recall': 0.4666666666666667, 'f1': 0.5957446808510638, 'false_events': 6, 'false_per_minute': 0.3115534400692341, 'minutes': 19.258333333333333, 'lead_p50_s': 0.7666666666666608, 'lead_p05_s': 0.13500000000000192, 'by_transition': {'pickup->harvest': {'total': 20, 'detected': 14}, 'harvest->return': {'total': 20, 'detected': 12}, 'return->adjust': {'total': 10, 'detected': 2}, 'adjust->pickup': {'total': 10, 'detected': 0}}}, 'B2': {'total': 60, 'detected': 13, 'missed': 47, 'precision': 0.8125, 'recall': 0.21666666666666667, 'f1': 0.34210526315789475, 'false_events': 3, 'false_per_minute': 0.15577672003461704, 'minutes': 19.258333333333333, 'lead_p50_s': 0.4666666666666668, 'lead_p05_s': 0.273333333333332, 'by_transition': {'pickup->harvest': {'total': 20, 'detected': 8}, 'harvest->return': {'total': 20, 'detected': 5}, 'return->adjust': {'total': 10, 'detected': 0}, 'adjust->pickup': {'total': 10, 'detected': 0}}}}

这是第四阶段已失败模型的预先锁定阈值0.5诊断，未在新数据重选阈值。分类分数不变，事件决策另计；不因测试集某个均值更高就替换现有模型。

## 产物

原始数据仅下载到/home/user/datasets/tomato_independent10_20261109，不回写服务器。完整manifest、独立新特征缓存、五组模型真实logits、协议与结果位于/home/user/vjepa2_training/runs/tomato/independent10_20261109。版本化文件仅包含代码、摘要和可核对身份，不上传原数据/权重。没有操作机器人或修改OpenPI/RTC。本轮沿用不推送远程的用户约束。

本轮新增2341个因果观测窗口，Future1.5秒有效2311个；使用新特征缓存，不是复用旧录制预测。原始录制时间区间与旧94条也未发现交集。133项番茄回归测试通过，所有模型SHA、58D名称顺序及预训练身份复核通过。逐录制事件图位于运行目录per_recording_events.png。
