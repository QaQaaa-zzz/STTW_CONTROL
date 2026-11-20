# STTW_CONTROL

双轮机器人控制与连续骑行—跳跃—骑行研究。

> **当前第三章高台实测。** 粗糙图八个开发初态、两种冻结策略共 32/32 条接近均真跳、台面承重、锁摆交回。96 条真实驶离尚未给出同根双向任务反转。[实测 XY、逐例结果和整程录像](docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/native_platform/approach_alignment/paired_tasks/RESULTS.md)

**此前的宽沟协议（11 月 18 日）。** 0.55米沟已核实原始Source主动腾空、落地交回并完成0.505秒终点保持；首次1/36，之后两批复跑各0/36。完整RB已实测飞越，整程仍待完成。

[P1 宽沟快照复验](docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p1_snapshots/wide_gap/RESULTS.md)已完成，12/12 条阶段轨迹逐项复现至终点。[P2 配对录像与矩阵](docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p2_wide_gap/RESULTS.md)记录了同一个完整初态下的两条实测飞越。0.55 米沟上，右弯接近方案两次完成避障和终点保持，左弯接近方案在远岸承重前失败。[成功录像](docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p2_wide_gap/source_055_goal.mp4) · [失败录像](docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p2_wide_gap/source_055_left_failure.mp4)。

## 从这里阅读

|入口|内容|
|---|---|
|[第三章结果总览](docs/evidence/chapter3_playground_sttw_rb_sttw/README.md)|当前结论、成功与失败、P0–P7进度|
|[STTW越野验收](docs/evidence/chapter3_playground_isaac_recovery/README.md)|锁摆、4cm脊坡、1.8m/s下96/96与连续8/8|
|[宽沟重验录像与实测数据](docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p0_active_jump/RESULTS.md)|完整RB主动飞越；原始Source有一条真实飞越、落地交回及终点保持的连续录像|
|[阶段α实际运行](docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_alpha_grid/README.md)|旧终点标签，适配shared225已知DEV399/648，固定α0→0为72/72|
|[原source与完整RB配对](docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_source_full_rb/RESULTS.md)|原冻结权重各0/36最终完成，保留全部XY|
|[论文状态](docs/chapter3_paper/README.md)|当前证据与旧稿边界，尚未达到提交条件|

## 代码与数据

- `learning/src/sttw_control/`：控制、环境、训练和预测模块。
- `learning/cli/`、`learning/configs/`：运行入口与配置。
- `docs/evidence/`：有范围声明的实验报告；历史记录不代表当前采用结论。
- `mujoco_ros_ws20260313/`：ROS与原始物理资产。
- `PROJECT.md`与旧方法日志保留工程历史，历史“正在运行”不可当实时状态。

## 精简交付范围

保留代码、逐例数值和关键图；已核验跳沟录像合计约3.2MiB，逐步原始轨迹及权重保存在数据盘。详见[清单](docs/evidence/chapter3_playground_sttw_rb_sttw/compact_delivery_manifest.json)。此包不是含全部模型与数据的独立复现发行版。

不同模型与数据角色不得混用：shared225为PPO-only分支适配；原完整RB是另一组冻结权重。接触远岸、完成交回与最终任务成功分别统计。
