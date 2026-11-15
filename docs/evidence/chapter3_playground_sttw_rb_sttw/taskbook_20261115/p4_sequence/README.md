# P4 全链预测：当前结论与证据索引

**当前仍未采用可靠规划器。** 最新[阶段顺序诊断与匹配续训](phase_order/README.md)修正预测器回退阶段的问题，DEV准入69/72、机制图原始包络选路2/2；仍有误准入，标定后C/I全拒18，P4未验收。完整best/last与退化种子均保留，未认定收敛。
- [阶段顺序、匹配续训与全部XY](phase_order/README.md)
- [当前RB与门槛下直接/残差比较](direct_current/README.md)
- [余弦续训、全部XY与失败](cosine/README.md)
- [此前配对训练、全部XY与失败](handoff_focus/README.md)
- [物理状态/历史互换与事件时刻诊断](departure_interchange/README.md)
- [中心误差与条件标定归因](low_lr/geometry_audit/README.md)
- [低学习率续训与全存档筛选](low_lr/README.md)
- [原G/I/C/Oracle和强回溯比较](comparison/README.md)、[直接/残差结构对照](direct/README.md)
- [历史重置](history_reset/README.md)、[关节门槛](joint_gate/README.md)、[积分表示](trace_audit/README.md)、[状态条件事件](state_event/README.md)

当前因果查询使用真实起点，其后传递预测状态/历史；离线互换诊断明确使用真实未来，不能混入因果规划成绩。各报告的门槛、模型及数据角色以各自合同为准；历史未检查锁定/远岸支持的成绩不当作当前有效交接。

## 原始首轮记录（历史快照，非当前结论）

下面保留首轮诊断及当时缺项。G/I/Oracle、标定和结构对照后来已实现，参见上面的索引；原始数组和数值未改写。

### First C-only full-chain diagnostic

[All18 actual/reference/predicted XY](xy.png).

True admit2, false admit1, false reject2, true reject13;18conditions total, four actually feasible.

First C-only raw-envelope DEV diagnostic; no I/G/strong-backtracking comparison or new margin calibration; full approach/departure tracking-domain and all-stage posture/obstacle gates included

Only common first-frame physical/controller/RB histories enter the predictor. Ground states and estimated history feed RB; predicted RB event state/history feed departure. Real future trajectories/status are used only for offline comparison. Initial RB support assumes the declared flat approach zone; STTW locked hip/knee rates are zero in the ground model.

This is an engineering/DEV result, not deployment or independent TEST. No new physical episodes: actual outcomes reuse the full-snapshot Oracle. G/I/Oracle, same-candidate strong backtracking, calibration and actual selected-plan verification remain required.

## Error diagnosis and next gate

The false admission is root1002 / approach0 / departure1: predicted minimum raw clearance +0.00739m, but the actual approach hits a forbidden obstacle before jumping. This is not a successful jump followed by a failed departure. Both false rejections are root1001 / approach1 / departure0–1: the predicted goal hold is missing; departure0 also has negative predicted clearance. Original 10s/0.5s hold criteria remain unchanged.

History bridge: 72 TRAIN and 24 DEV approach sequences, three seeds with 600 updates each; best updates575/375/600 and normalized entry RMSE0.1730/0.1874/0.1745. Seed0 is used here. These are estimated controller/RB histories, not measured future states. Training completion does not establish convergence or safe planning. Raw logs/checkpoints reside in `taskbook_20261115_p4_history_bridge`; [all seed metrics](history_bridge_summary.json).

P4 remains open. Next: compare G/I/C/Oracle with the same candidates, predictor and constraints, including strong backtracking; diagnose stage occupancy errors and calibrate using declared DEV rules; verify selected-plan outcomes. P5/P6 do not start before this stage is assessed.

**Trigger-contract correction:** these historical predictions used hardcoded8.60m; the executed frozen configuration uses8.55m. Corrected prediction results are in [trigger_fix](trigger_fix/README.md). Original artifacts remain unchanged as evidence of the error.

[Current departure-input diagnosis](departure_interchange/README.md): true physical exits reduce downstream lateral error substantially; actual-history-only swaps do little. Timing accounts for most longitudinal exit error but leaves lateral bias. No model adopted; P4 remains incomplete.

[Matched handoff-focus continuation](handoff_focus/README.md):6x1000 supervised updates,12 best/last candidates and same-rule calibration completed. Focus improves selected exit-state errors but not task decisions; neither arm is adopted, last models regress, P4 remains open.
