# 番茄 V-JEPA 2 交互式离线验证 UI 实施报告

日期：2026-11-09（Asia/Shanghai）。基线 HEAD：`63d89e8700cfb9444e8d319ada34b5ba8f071a2e`；新增分支：`feat/tomato-offline-review-ui`。本报告与实现一起提交，不推送远端。

## 实施结果

新增本机只读页面：样本检索、三路视频同步播放/seek/逐帧/倍速、只读GT、三个模型模式按需识别、观测轴原始预测轨、HELD概率详情、单条带标签MP4导出。没有训练、阈值调整、GT修改、Prompt调度、Agent或机器人控制。

| 验收项 | 状态与实测证据 |
|---|---|
| M1 数据与播放 | PASS：93来源，七阶段与三阶段样本实测，三机seek时间一致 |
| M2 按需识别 | PASS：三模式命中冻结预测，一次显式CPU分类头fallback |
| M3 时间轴 | PASS：GT七/三段保留，Future两轨、Joint三轨，HELD固定原target |
| M4 单条视频导出 | PASS：同一真实七阶段样本Future/Joint各一份MP4，逐帧解码通过 |
| M5 负面与回归 | PASS：119测试；历史训练输出、数据及已保存仿真产物SHA核验通过 |
| 旧P1～P6源码回归 | NOT_RUN：现场该源码不在工作树，仅核验已保存产物 |
| 目标时间平移视图 | DEFERRED：可选功能，首版仅观测时刻轴 |
| 实时Encoder/真机验证 | NOT_RUN：本轮不涉及 |

## 现场基线与参考源码

开始时git工作树干净，HEAD为63d89e8；此前受控实验、Prompt仿真的未提交源码及文档在当前工作树已不可见，而runs下真实实验/仿真结果仍存在。本轮没有删除或恢复它们，不声称对缺失源码执行回归；原训练模型模块逐文件未改动。

参考 [convert_data_job-worker subtasks-data-convert](https://github.com/ma39nong/convert_data_job-worker/tree/subtasks-data-convert) 的index.html/app.js/style.css/server.py/docs/subtasks.md，通过raw分支只读获取并阅读样本选择、seek/tick、时间条比例与Range行为。本机转换仓库没有该分支，本轮未向其fetch/写入。参考仓库根LICENSE请求返回404，未独立核验计划所述4fe797c提交。因此只借鉴交互原则，全部UI/服务独立编写，不复制代码、logo、图片，不import convest/Workspace/Store或标注/豆包模块。

## 启动与依赖

```bash
cd /home/user/pi05/vjepa2_training
.venv-tomato/bin/python -m tools.tomato_review_ui.server \
  --run-root "$PWD/runs/tomato/local93_20261107" \
  --host 127.0.0.1 --port 8770
```

地址：http://127.0.0.1:8770 。当前验收服务已启动。默认只允许loopback；如需SSH转发，使用同名本地端口8770，不能绑定0.0.0.0。可用`--output-dir`指定新的专属UI输出目录，不能放入原始数据或任一训练模式目录。服务不依赖Flask/FastAPI；播放/冻结预测使用标准库，CPU fallback使用现有torch/numpy，导出使用现有PyAV/OpenCV/libx264。打开页面不启动Torch/Encoder。

自动浏览器验收是可选开发依赖：本轮通过`uv pip install --python .venv-tomato/bin/python playwright`安装playwright 1.64.1，使用本机`/usr/bin/google-chrome`，不作为UI运行依赖。

## 数据身份、模型绑定与缓存

目录从三模式manifest构建，以episode_id+source_recording_id匹配，不以重复的episode_index作全局主键。三模式契约、来源和split一致才启动；标注SHA/连续覆盖/reviewed检查，注册相机存在及PTS契约检查。列表支持数据包/split/编号/来源搜索，不扫描或重编号数据。

选中样本识别时核验当前manifest/split、checkpoint、原数据与标注。保存预测仅在已有analysis/original_result_sha256.json收据中checkpoint/manifest/predictions均匹配时复用。缺失、不完整或没有可信收据时，才单独读取该Episode的NPZ，通过原load_cache/load_checkpoint/predict_heads作CPU分类头推理；无Encoder提取/训练。fallback沿用训练mean/std和原模型pin，拒绝错mode/split/损坏缓存。

派生key绑定episode、mode、checkpoint/manifest/split/predictions/NPZ SHA；结果写入新的review_ui/predictions，不覆盖训练JSONL。重复点击合并任务/复用结果；checkpoint或输入变化使作业key失效。ThreadPoolExecutor最大1个后台模型/视频作业，HTTP媒体线程独立。更换样本/模式增加前端版本号，旧作业结果不能覆盖新选择。

## API与安全

| Endpoint | 行为 |
|---|---|
| GET /api/session | 创建本机会话及CSRF token |
| GET /api/episodes?split=&q=&package= | 列表和过滤 |
| GET /api/episodes/{episode_id} | 注册三机URL、GT、fps/length、来源修订 |
| GET /api/media/{episode_id}/{cam0..2} | 原MP4 Range流 |
| POST /api/predictions/recognize | episode_id、prediction_mode，可选regenerate布尔值 |
| GET /api/jobs/{job_id} | 自己提交的作业状态 |
| GET /api/predictions/{result_id} | 本会话授权的原始锚点/概率/来源 |
| POST /api/exports/video | 只接受result_id |
| GET /api/exports/{job_id} | 导出状态与metadata |
| GET /api/exports/{job_id}/download | 本会话成品MP4预览/下载 |

不接受任意路径、输出目录、shell或checkpoint。相机严格白名单，拒绝路径越权/符号链接；HTTP Host只接受本机地址，POST要求同Origin、HttpOnly SameSite cookie与token，作业及result按会话授权。Range实测200/206/416，支持suffix及open-ended，拒绝multi-range。

## 真实样本与时间语义

七阶段：`7ed8a2815bb81e0be7bb89f8`，source `34a4b430af2c4fd6a1912ff03628ef3f`，train，2234帧 / 74.466667秒。

三阶段：`1df7c31ae9a58dc60d4e57d6`，source `ef866ed4c8f949d7b1a7d7fb3ef8ffbb`，train，1065帧，未添加adjust。

三个模式的首/中/末真实锚点四类概率与原JSONL逐值相等；类别使用raw argmax，无阈值、平滑或控制仿真Active/Pending层。GT保留原始段序号；Future-only仅Future轨，Joint同时Current/Future，Current-only仅Current。

浏览器seek至frame37时，最近锚点为frame30/1.0秒，Future target固定2.0秒，绝不改为当前1.233秒+1。三机currentTime均为1.234999秒，±1帧后frame38；恢复播放约1秒后frame69/70，浏览器无JS错误。尾部future_valid=false仍保留预测，明确显示目标GT不可评分。

![Joint界面](/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/review_ui_acceptance_20261109_final/7ed8a2815bb81e0be7bb89f8_joint.png)

![三阶段Future界面](/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/review_ui_acceptance_20261109_final/1df7c31ae9a58dc60d4e57d6_future_only.png)

## 缓存命中与按需计算

| 模式 | 首次实际路径 | 七阶段锚点数 | 初次接口耗时秒 |
|---|---|---|---|

| future_only | frozen_predictions | 149 | 0.202 |

| joint | frozen_predictions | 149 | 0.202 |

| current_only | frozen_predictions | 149 | 0.413 |

显式三阶段Future重算：CPU分类头，71锚点，耗时1.027秒；feature hash/fingerprint保存在acceptance.json，encoder_executed=false。最终完整复跑复用既有派生结果/视频，未重复生成全量实验。

## 两份真实带标签视频

原FPS/帧序逐帧同步流式解码，三机等比例letterbox，1152×680 H.264/yuv420p，无音轨；每帧显示GT、对应模式全部头、四类概率、原anchor/target、NEW/HELD和共享游标时间条。生成partial后完整解码核验并原子rename，metadata最后落盘；失败不标记完成。

| 模式 | 帧数 / FPS / 时长秒 | 首次实际渲染耗时秒 | SHA256 |
|---|---|---|---|

| future_only | 2234 / 30 / 74.466667 | 10.343 | `865c1d379b94c59eb47a0e65090744b55c34b14251cbe3c6fb5dec37f45e273b` |

| joint | 2234 / 30 / 74.466667 | 10.087 | `e0d09e5d49e831d60ff8d04acf9d79c6ad00f7eae8c7debe18a1adfa313f4c6d` |

- future_only 原成品：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/review_ui/ui_exports/7ed8a2815bb81e0be7bb89f8/future_only/74d6cb2442d77bb83542da48491c0735ee02f832b730392d68d1296e30a93b3d/annotated.mp4`；验收副本：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/review_ui_acceptance_20261109_final/future_only.mp4`。

- joint 原成品：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/review_ui/ui_exports/7ed8a2815bb81e0be7bb89f8/joint/1cbe761d7d7b684bfa8fd1fd6311f1a47ee94bd1e5051c981d6318c7cf8267a1/annotated.mp4`；验收副本：`/home/user/pi05/vjepa2_training/runs/tomato/local93_20261107/review_ui_acceptance_20261109_final/joint.mp4`。

两份成品均全帧重新解码，帧数2234、时长74.466667秒；抽取frame0/37/2233检查画面内标注，非外挂字幕。输出metadata包含模型/标注/数据/结果SHA，旁有逐锚点JSONL。只有验收脚本的显式两次导出请求才生成，页面打开/切模式不会生成MP4，没有全量视频转码。

## 负面测试、回归与保护审计

30项新增测试覆盖：尾部/HELD时间语义、缺失/重复/错来源/错模式/NaN概率/未来输入、Range边界、目录/符号链接越权、GT空洞/重叠/缺相机、跨包同episode_index、缺失checkpoint、导出解码中断、会话与job隔离、checkpoint变化使去重失效。真实浏览器另验证模式变化使旧请求不能覆盖新显示。原checkpoint兼容性/错格式/训练与评估回归沿用现有测试。

```text
........................................................................ [ 60%]
...............................................                          [100%]
119 passed in 11.38s
```

原321个训练输出SHA均不变。逐项核对全部93条原始Parquet/视频及审核标注共465项SHA，与锁定manifest一致。另核对已保存Prompt仿真2645个产物SHA不变；因其源码当前缺失，未宣称可重新执行P1～P6测试。证据副本位于docs/validation/review_ui。

## 文件修改清单与复现

全部为新增：tools/tomato_review_ui/{__init__,catalog,timeline,prediction_store,recognize,server,video_export,acceptance}.py，static/{index.html,app.js,style.css}，tests/test_tomato_review_ui.py，本报告与docs/validation/review_ui中的JSON/测试日志。原训练脚本、权重、预测、标注和conversion工程均不修改。

```bash
.venv-tomato/bin/python -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py
# 下列是显式验收，会请求当前一条七阶段样本的两种模式导出：
.venv-tomato/bin/python -m tools.tomato_review_ui.acceptance \
  --run-root "$PWD/runs/tomato/local93_20261107" \
  --output-dir "$PWD/runs/tomato/local93_20261107/review_ui_acceptance_NEW"
```

已存在验收目录会拒绝覆盖。UI专用结果/视频按内容key复用；原321文件保护审计与浏览器详细测量、源文件核验记录已交付。初次验收脚本曾错误地检查Joint的Current尾部是否不可评分，修正为检查Future后在新目录完整重跑；此问题只在验收断言，不涉及模型/预测语义。

## 已知限制与未做项

只实现观测时刻轴，目标对齐视图可选延后；浏览器播放同步采用主视频时钟及60ms漂移校正，逐帧seek有实测证据，不声称硬件级三机同步。CPU fallback依赖已有共享NPZ，不会隐式运行Encoder；缺失特征需先用原工具提取。模型/视频后台作业最大1，暂不提供用户取消长作业。仅绑定本机/同Origin，不提供公网账号体系。

本轮没有重新计算93条模型精度、训练、选模、调阈值或GT重审；自动浏览器/解码验收不等于机器人安全或物理完成确认。大MP4和截图保留在runs，不提交到Git。参考源码许可不明确，因此未复制其实现与静态资源。
