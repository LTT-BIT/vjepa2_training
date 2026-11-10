# 第五阶段：开源方法阅读与最小设计取舍

本文用于工程设计记录，不移植或运行外部框架，不把其论文实验效果当作番茄结果。

## ProTAS

已阅读[CVPR 2024论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf)的方法与[作者代码](https://github.com/Yuhan-Shen/ProTAS)。因果动作分割结合动作内进度预测和任务图；代码SingleStageModel包含因果TCN和单向GRU进度分支，TaskGraphLearner使用前驱/后继矩阵和累计进度约束类别输出。

本工程只借鉴“进度必须有明确语义”和“有序任务图”的思想。phase_index/6仅表示决策走到哪个位置，不是动作内进度，更不是抓取/摘取成功率。没有把人工阶段结束时间、真实时长归一化进度或论文的监督进度作为在线输入。重复pickup/harvest/return必须拆成不同图位置。

实际阅读快照：`0f04ee6b3c4370ce22f9230bca42e9aabe47b428`；关键位置：model.py的SingleStageModel、TaskGraphLearner及utils/write_progress_values.py。其训练标签生成依赖人工段长，是训练监督；不能偷带到番茄在线记忆。

## HeLM

已阅读[论文第3节](https://arxiv.org/html/2611.02660v1#S3)与[作者代码](https://github.com/kimspacehuni/HeLM)。HeLM分离事件检测与事件触发的语言记忆更新，并区分工作记忆与跨任务情境。代码eval_real_time_qwen.py的DETECT_SYSTEM与UPDATE_SYSTEM分别限定检测和更新职责。

本工程采用独立候选检测、确认后写入结构化事件的边界，保留原证据和修订原因。没有引入VLM、语言生成、低层动作命令或跨Episode持久情境；本任务在每个Episode显式重置，防止泄漏。论文的“完成事件”在此不能直接由阶段分类代替，操作物理结果一律标记unobserved。

实际阅读快照：`50b13bd5d772603be44bb1b90465dcc1e0624de5`；关键文件evaluate/eval_HLP_LLP/eval_real_time_qwen.py，以及evaluate/eval_helm下分别评价detect/update的函数。

## HiMe

已阅读[论文第3节](https://arxiv.org/html/2607.03449v1#S3)与[作者代码](https://github.com/HappyWaterXP/HiMe)。HiMe将执行、工作记忆监测与长期策略分层；结构化长期存储支持显式增、改、删操作。代码MultiTagMemory提供create/update/delete、容量裁剪和快照。

本工程采用有限观测FIFO和紧凑事件账本，事件发生才更新长期语义。由于确认事件在这里是决策证据而非物理事实，不照搬“删除旧信念”来掩盖错误；保留确认、冲突与过期的版本化审计。近期事件可FIFO裁剪，但六个已确认有序转换另行保留，完整事件修订流落盘。无图像记忆库、检索向量或Planner调用。

实际阅读快照：`261766e3b067db8cf92fe2d34cc7da32034d142a`；关键位置memory/multitag_recorder.py的MultiTagMemory，agent/observer.py的ObserverAgent.run，以及server/task_state.py。

## 最小方案与证据门槛

M0是未改动的B0；M1记录B0转换与模型识别事件，不能改变B0；M2根据自身连续、有效观测形成有序确认事件，用账本检查进度及历史冲突，只作离线位置估计。它不会把历史存在本身当成推进条件，也不自动重同步。此账本在固定线性任务图下可能与原B0七位置有相同的判别信息；其审计价值与预测增益必须分开检验。

只有正常验证和故障回放显示M2减少错误、没有增加偏移时，才考虑M3。缺少这一证据就停止，不把旧GRU/TCN换名重训。即使未来研究可学习记忆，也应以跨阶段事件为输入并进行历史移除/乱序消融，证明收益来自任务历史。
