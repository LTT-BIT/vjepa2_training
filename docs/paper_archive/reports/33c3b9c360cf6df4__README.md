# 远程读取实测：2026-11-07

使用 SSHFS 只读挂载 A/B，复用工程实际来源哈希、状态/视频核验、冻结 V-JEPA 2 Encoder 和完整窗口特征提取函数。未训练分类头、未修改项目源码或服务器文件、未复制原始视频/Parquet/标注到本地持久存储。操作系统可能在 RAM 中缓存读取内容。

| 样本 | 长度 s | 来源哈希 s | 视频核验 s | 提取总计 s | Encoder/融合 s | 特征形状 | 压缩缓存 MB |
|---|---:|---:|---:|---:|---:|---|---:|
| A/1106_702_739_normal_lerobot / episode 0 | 152.5 | 8.34 | 9.16 | 65.58 | 27.61 | [305, 4000] | 2.21 |
| B/1105_703_713_normal_lerobot / episode 0 | 156.2 | 9.06 | 9.65 | 65.44 | 28.16 | [313, 4000] | 2.28 |

提取总计包含提取前后两遍来源哈希、再次远程解码/缩放、GPU 编码/关节融合、压缩写缓存，不能与其内部 Encoder 时间重复相加。模型首次载入 1.63 秒。每条含核验/提取/缓存重读约 85–86 秒。两条各 7 个来源哈希全部一致，全部三路视频帧数/FPS/尺寸/PTS 核验通过，58 维关节位置有限，最终 618×4000 特征全部有限，缓存重读核验通过。

## 阻塞项

- 两条标注缺失 inherited 字段；现有严格加载器要求 inherited is False，故严格标注验收失败。本次单独执行来源身份/段覆盖/哈希比对后提取无标签特征；不代表允许进入正式训练，未填补或修改标注。
- A/B 共 93 个粗阶段 JSON，当前 SSH 账号无权读取其中 68 个；仅清点标注读权限，未证明其余媒体全部可读。详见 permission_inventory.json。

## 性能解释

首次来源哈希逻辑吞吐 4.60–4.90 MB/s，不能当作网卡带宽；不是清除缓存后的基准，也未做本地副本对照。按同等视频长度和当前吞吐，100 条粗估 2.4 小时，实际受视频长度、服务器负载、网络影响。两条压缩缓存合计约 4.49 MB，按此样本规模 100 条约 225 MB，不含额外产物。

## 重现

在仓库根目录执行（需先恢复同一路径的只读挂载；原脚本重跑会覆盖本次 report.json，并可能复用缓存；全新耗时测量应换独立输出目录）：

```bash
./runs/tomato/remote_io_probe_20261107/client/unpacked/usr/bin/sshfs tairan28:/mnt/workspace/train_platform/datasets/tomato_sharpa/normal/no-block-more runs/tomato/remote_io_probe_20261107/mount -o ro,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,BatchMode=yes
PYTHONDONTWRITEBYTECODE=1 .venv-tomato/bin/python -u runs/tomato/remote_io_probe_20261107/probe.py
fusermount3 -u runs/tomato/remote_io_probe_20261107/mount
```

原始实测日志 probe.log；结构化结果 report.json；临时客户端来自 Ubuntu sshfs 软件包，仅解包在此目录，未安装系统包。
