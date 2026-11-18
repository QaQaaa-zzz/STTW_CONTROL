# STTW_CONTROL

双轮机器人控制与连续骑行—跳跃—骑行研究。

> **2026-11-18：第三章从P0重新验收。** 0.55米沟已核实主动腾空和一条落地交回后骑入目标区的记录；该例在10秒截止时保持0.32秒，要求0.50秒。

## 从这里阅读

|入口|内容|
|---|---|
|[第三章结果总览](docs/evidence/chapter3_playground_sttw_rb_sttw/README.md)|当前结论、成功与失败、P0–P7进度|
|[STTW越野验收](docs/evidence/chapter3_playground_isaac_recovery/README.md)|锁摆、4cm脊坡、1.8m/s下96/96与连续8/8|
|[宽沟重验录像与实测数据](docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p0_active_jump/RESULTS.md)|完整RB主动飞越；原始Source另有真实落地交回，所有失败保留|
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

保留代码、逐例数值和关键图；新增三段跳沟录像共约1.33MiB，逐步原始轨迹及权重保存在数据盘。详见[清单](docs/evidence/chapter3_playground_sttw_rb_sttw/compact_delivery_manifest.json)。此包不是含全部模型与数据的独立复现发行版。

不同模型与数据角色不得混用：shared225为PPO-only分支适配；原完整RB是另一组冻结权重。接触远岸、完成交回与最终任务成功分别统计。
