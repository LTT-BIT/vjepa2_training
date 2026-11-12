# Episode 3e645bf4e137d0f836c9d7ff

source_recording_id: 76354615cf9746758c5c7e6e5ab579d4

Current Macro-F1: 0.918003; Future1.5s Macro-F1: 0.916091

严格Future事件: 4/6; 未匹配预警: 2

七阶段位置准确率: 0.958955; 状态不一致: 5.500s

有序位置序列: [0, 1, 2, 3, 4, 5, 6]; 六次Prompt均按时: False

上述为Agent离线决策；未运行VLA或机器人，物理成功未知。Memory仅忠实记录决策。

|序号|转换|Future命中|Prompt结果|Prompt相对边界秒|
|---|---|---|---|---|
|1|pickup->harvest|True|on_time|0.6666666666666661|
|2|harvest->return|True|on_time|0.7000000000000028|
|3|return->adjust|True|early|-0.2666666666666657|
|4|adjust->pickup|False|early|-2.2333333333333343|
|5|pickup->harvest|False|on_time|1.0|
|6|harvest->return|True|on_time|0.8000000000000114|
