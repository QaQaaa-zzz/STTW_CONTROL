# 越野任务条件与阶段结果

本报告比较已完成的探索DEV，共用种子、来车初态与地形；仅将名义转向参考变化速率0.3改为0.6rad/s；执行器与角度限幅、H75、道路和物理不变。各组分母见表，不同批次重复初态不计作独立样本；不合并成成功率显著性证据。

- [chapter3_offroad_task_map_gentle全部实际/参考XY与逐候选结果](../chapter3_offroad_task_map_gentle/RESULTS.md)
- [chapter3_offroad_steer_slew全部实际/参考XY与逐候选结果](../chapter3_offroad_steer_slew/RESULTS.md)

|条件|路线|完成|进入跳跃|交回STTW|曾到达x≥5.5|曾到达x≥13|终止码分布|
|---|---|---:|---:|---:|---:|---:|---|
|chapter3_offroad_task_map_gentle|straight_jump|3/30|22|21|25|6|{15: 6, 13: 3, 4: 3, 3: 16, 18: 1, 9: 1}|
|chapter3_offroad_task_map_gentle|s_bend_jump|0/30|17|6|27|4|{15: 9, 3: 16, 18: 1, 9: 3, 4: 1}|
|chapter3_offroad_task_map_gentle|right_angle_jump|0/30|0|0|0|0|{15: 29, 3: 1}|
|chapter3_offroad_task_map_gentle|ground_detour|5/30|0|0|28|22|{13: 5, 15: 3, 9: 8, 3: 9, 18: 5}|
|chapter3_offroad_steer_slew|straight_jump|2/30|19|19|27|5|{3: 22, 12: 1, 13: 2, 18: 3, 4: 1, 15: 1}|
|chapter3_offroad_steer_slew|s_bend_jump|1/30|28|21|29|8|{18: 6, 15: 1, 3: 15, 12: 5, 4: 1, 9: 1, 13: 1}|
|chapter3_offroad_steer_slew|right_angle_jump|0/30|0|0|2|0|{15: 27, 12: 1, 3: 2}|
|chapter3_offroad_steer_slew|ground_detour|7/30|0|0|26|21|{13: 7, 18: 6, 9: 6, 3: 9, 15: 2}|

进入跳跃是控制阶段切换，不等于真实离地或成功清障；交回STTW由首次远岸接触触发，不等于整程完成。地面绕行不要求进入跳跃。x阈值只记录实际到达位置，不能单独证明跨越脊坡稳定。终止码15为运行域退出，3/4为侧倾/俯仰阈值，18为到目标但曾越廊，9为观察窗口结束未成功，13为完成。
