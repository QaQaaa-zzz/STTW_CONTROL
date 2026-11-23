# PCRP 同槽配对判决实验

2026-11-23 09:01 UTC，E2 与 E1a 已启动并收到 Bark 启动回执。E1b 等待 E2 和 E1a 完成及平地阳性对照通过，E1c 随后检验 E1 各案例的赢家。当前可查看第 0 步的真实 XY 对照及原 672 槽复验的全部 112 组轨迹。[启动快照](launch_snapshot.json) · [第 0 步 XY](../decision_replay/README.md) · [原 672 槽 XY](../decision_replay/original672/README.md)

E1a 的首个 H75 α=0 基准批已经返回。401 组平地走满 10 秒，末秒保持且通过任务 C；4.2 m 和 4.5 m 脊分别在 2.775 s 和 2.135 s 达到侧倾终止线。采样 159.56 s，写盘 2.57 s。这三条属于搜索前的基准。E1c 已固定到提交 bae0fdfd，按两项前置阶段与阳性对照依赖排队。 [首批实际与参考 XY](first_batch/xy.png) · [逐案例指标](first_batch/episodes.csv) · [计时](first_batch/timing.json) · [E1c 运行版本](paired_services/E1c_pin.json)

## 冻结配置与预算

沿用原 S2–S4 的 H75 中层、Local300 底层及 90° 单弯参考。目标速度 2.6 m/s，参考半径 1.8 m，中层 / 底层 / 物理频率为 50 / 200 / 5,000 Hz，每回合 10 s。物理组 401–408 属于已研究的 DEV 组，训练更新为 0。五个 α 取 0、0.25、0.5、0.75、1。Zero 保留 Local300 和 ECBC/ESO，关闭 H75 中层修正。[原登记](registration.json) · [模型、初态和资产哈希](../decision_replay/protocol.json)

| 阶段 | 物理槽 × 批次 | 物理回合 | 用途 |
|---|---|---:|---|
| 第 0 步 | 36 × 2 | 72 | 两次小批重复的 36 条 XY 逐位一致，与历史大批中的对应 36 条比较，整轨通过 0/36。 |
| E2 | 24 × 35 + 16 × 25 | 1,240 | 脊位轴 7 档 × 五 α。初速轴 5 档 × 五 α，包括在该轴自身槽位重测的 80 条零扰动基准。原定 1,160 个主工况均保留。 |
| E1a | 3 × 326 | 978 | 401 组的平地、4.2 m 脊和 4.5 m 脊。平地赢家须通过任务 C。 |
| E1b | 21 × 326 | 6,846 | 再测 21 个案例。与 E1a 的 3 个案例合并，覆盖 6 种布局 × 401–404 四个组。 |
| E1c | 3 × 4 + 21 × 4 | 96 | 16 个单脊案例分别调整脊中心 −2、−1、+1、+2 cm。4 个平地与 4 个碎石案例各在原地形回放赢家四次。单脊位移 64 条，名义对照 32 条，分别报告。 |
| **主实验预约** | | **9,232 / 9,500** | 余量 268；可选 E1d/E1e 未预约。 |
| 单列的原批复验 | 672 × 1 | 672 | 已完成，672/672 条复现归档，满窗及任务 C 均无翻转。 |

每个 E1 案例独占一个固定物理槽。先实测五档 H75 与 Zero，从五档中选择 V 最小者作为 CEM 的 α。每代完成全部 32 个候选后更新，10 代共 320 个候选，每例连热启动合计 326 条。按附件分段步长得到 27 个节点、54 个变量。E2 在每个扰动轴内保持槽位、排列与世界坐标一致。[E1a 合同](e1/E1a/protocol.json) · [E1b 合同](e1/E1b/protocol.json) · [E2 合同](e2/protocol.json) · [E2 分析口径](e2/analysis_protocol.json) · [生成几何与输入核查](e2/verification.json)

**决议顺序。** 附件与原登记的 SHA-256 一致，为 `cf7fad6464ee1f91988c069602a21f1aee69680ffe8e6b65a5c47f5a96312c72`。第 0 步触发原停止条件，原 672 槽复验完成后，用户批准将 672 条单列，并以同槽配对继续 E2/E1。该授权记录仍保留当时对 E1c 平地及碎石的待决文字。随后用户授权继续推进，E1c 采用上述 64 条单脊位移与 32 条原地形对照。[原始附件](decision_experiments_plan.md) · [同槽授权](paired_authorization.json) · [后续 E1c 规范](e1c_scope.json)

## 运行和报告入口

本机运行根目录为 `/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_decision_20261122`。可在此查看实时服务状态、每批原始轨迹和分析输出。下列均为相对该目录的路径。[本机运行目录](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_decision_20261122)

| 阶段 | 进度与原始轨迹 | 阶段末分析入口 |
|---|---|---|
| E2 | `e2/status.json`，`e2/paired_lanes.csv` 逐槽索引，`e2/matched/` 各批 | `e2/analysis/README.md` |
| E1a、E1b | `e1/<阶段>/status.json`，`warm/w*/`、`g*/c*/` 内各案例原始轨迹，`candidates.csv` | `e1/<阶段>/analysis/README.md` |
| E1c | `e1/E1c/status.json` | `e1/E1c/README.md` |

编排进度与通知回执保存在 `paired_services/`。下列命令在运行主机只读取进度和日志。

```sh
systemctl --user show pcrp-matched-{e2,e1a,e1b,e1c}.service -p Id -p ActiveState -p SubState
journalctl --user -u pcrp-matched-e2.service -u pcrp-matched-e1a.service -n 40 --no-pager
```

## 复现入口与墙钟

E2 冻结到 `41a68418468c05fa2d8aa99453d03e3fc5908a95`，E1a 和 E1b 到 `583865efe408891263259e150f2eeeadaa81a69e`。服务从本机外盘根目录读取调度脚本，以各自 pin 指定的 Git 工作树运行采样和分析。需具备本机原 OAT、H75/Local300 转储、资产、冻结环境和用户级 systemd。查看本包及已完成的 XY 图无需运行仿真。[E2 pin](paired_services/E2_pin.json) · [E1a pin](paired_services/E1a_pin.json) · [E1b pin](paired_services/E1b_pin.json)

四份服务文件保存了实际的 `ExecStart`、`ExecStopPost` 和阶段依赖。两个 CPU 编排可并行提交；每个完整的 Isaac 子批单独通过 `gpu_queue.sh` 取得公用 GPU 锁 `research-hub/.gpu-job.lock`，子批退出后交接。调度器从第 0 步入口导入持锁记录函数 `record`；通知程序在运行主机读取私有 Bark 配置。阶段启动、完成、错误、中断及对照异常触发相应通知；从进入队列起 30 分钟仍运行则发送一次等候通知。用户交互在报告长等待后暂停，后台阶段按依赖继续。[实际调度器](dispatch_paired.py) · [台账记录函数](dispatch_step0.py) · [E2 服务](paired_services/pcrp-matched-e2.service) · [E1a](paired_services/pcrp-matched-e1a.service) · [E1b](paired_services/pcrp-matched-e1b.service) · [E1c](paired_services/pcrp-matched-e1c.service) · [队列脚本](helpers/gpu_queue.sh) · [Bark 程序](helpers/notify_sttw_job.py)

第 0 步两个 36 槽批实测从采样到导出为 173.64 s、173.40 s。若以 173.52 s/批估算，E2 的 60 批约 2.9 h，E1a、E1b 各 326 批约 15.7 h/阶段。正式各批为 3、16、21 或 24 槽，实际用时会随槽数变化，另需计入 GPU 排队、Isaac 初始化和退出及分析绘图。最终以各阶段记录的批次墙钟、采样、导出和分析耗时为准。[原始计时摘录](launch_snapshot.json) · [E1a 启动回执](paired_services/E1a_started_receipt.json) · [E2 启动回执](paired_services/E2_started_receipt.json) · [包内 SHA-256](SHA256SUMS)
