# Playground带摆车锁定修复

[近岸XY与全部条件](near_locked_comparison/RESULTS.md) · [远岸XY与全部条件](far_locked_comparison/RESULTS.md)

STTW已用hip/knee关节约束锁定摆机构；RB期间解除，回交按实测姿态重新锁定。保留原生有限位置伺服，未修改力矩上限、质量与惯量。

原Isaac第三章保存网格221502顶点、440326三角形与本轮源网格逐数组一致；512点编译MuJoCo表面射线最大高度差4.49e-8m。原沟位置、粗糙区及2cm/8cm脊坡均保留。

位置伺服+implicitfast+摩擦0.8+锁摆：近岸3/3到达，远岸2/3到达、路线61越廊。两个独立地面分段，不是完整跨沟或独立TEST。对应速度伺服候选6/6物理失败，未采用。未训练；越野恢复尚未完成，不将5/6写成整个第三章恢复成功。

近岸最大锁定数值漂移：hip0.0001974rad、knee4.05e-6rad。CPU制动释放测试通过；完整RB切换未验证。

原始状态、配置和全部逐步轨迹：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_isaac_recovery/`。入口`learning/cli/evaluate_playground_handoff.py`，复核`audit_playground_terrain.py`和`analyze_playground_ground.py --compare-runs`。
