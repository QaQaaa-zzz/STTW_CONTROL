# 越野弯道几何条件与阶段结果

两批分别为120条探索DEV，共用种子、来车初态与地形；更缓条件仅改变S弯与90°圆角几何。不同批次重复初态不计作独立样本；不合并成成功率显著性证据。

- [chapter3_offroad_task_map全部实际/参考XY与逐候选结果](../chapter3_offroad_task_map/RESULTS.md)
- [chapter3_offroad_task_map_gentle全部实际/参考XY与逐候选结果](../chapter3_offroad_task_map_gentle/RESULTS.md)

|条件|路线|完成|进入跳跃|交回STTW|曾到达x≥5.5|曾到达x≥13|终止码分布|
|---|---|---:|---:|---:|---:|---:|---|
|chapter3_offroad_task_map|straight_jump|4/30|22|21|25|8|{3: 17, 13: 4, 4: 1, 15: 5, 18: 3}|
|chapter3_offroad_task_map|s_bend_jump|0/30|1|0|8|1|{15: 27, 3: 2, 9: 1}|
|chapter3_offroad_task_map|right_angle_jump|0/30|0|0|1|0|{15: 30}|
|chapter3_offroad_task_map|ground_detour|3/30|0|0|28|22|{18: 6, 15: 4, 9: 8, 3: 9, 13: 3}|
|chapter3_offroad_task_map_gentle|straight_jump|3/30|22|21|25|6|{15: 6, 13: 3, 4: 3, 3: 16, 18: 1, 9: 1}|
|chapter3_offroad_task_map_gentle|s_bend_jump|0/30|17|6|27|4|{15: 9, 3: 16, 18: 1, 9: 3, 4: 1}|
|chapter3_offroad_task_map_gentle|right_angle_jump|0/30|0|0|0|0|{15: 29, 3: 1}|
|chapter3_offroad_task_map_gentle|ground_detour|5/30|0|0|28|22|{13: 5, 15: 3, 9: 8, 3: 9, 18: 5}|

进入跳跃是控制阶段切换，不等于真实离地或成功清障；交回STTW由首次远岸接触触发，不等于整程完成。地面绕行不要求进入跳跃。x阈值只记录实际到达位置，不能单独证明跨越脊坡稳定。终止码15为运行域退出，3/4为侧倾/俯仰阈值，18为到目标但曾越廊，9为观察窗口结束未成功，13为完成。

更缓S弯6条交接后：3条侧倾超限、1条俯仰超限、1条地面运行域退出、1条到目标但曾越廊；其中5条整程曾越廊。需区分跨沟物理失败与道路任务失败，不能把到目标那条当作合格完成。物理评测和图表已完成；systemd最后通知命令缺Python PATH导致127退出，已改绝对路径并单独补发通知，未重跑仿真。
