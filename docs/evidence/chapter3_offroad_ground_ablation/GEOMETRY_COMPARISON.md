# 越野任务条件与阶段结果

本报告比较已完成的探索DEV，共用种子、来车初态与地形；仅关闭H75学习修正；保留原跟随器、ECBC/ESO、同一路线、跳跃权重、目标与道路约束。各组分母见表，不同批次重复初态不计作独立样本；不合并成成功率显著性证据。

- [chapter3_offroad_task_map_gentle全部实际/参考XY与逐候选结果](../chapter3_offroad_task_map_gentle/RESULTS.md)
- [chapter3_offroad_ground_ablation全部实际/参考XY与逐候选结果](../chapter3_offroad_ground_ablation/RESULTS.md)

|条件|路线|完成|进入跳跃|交回STTW|曾到达x≥5.5|曾到达x≥13|终止码分布|
|---|---|---:|---:|---:|---:|---:|---|
|chapter3_offroad_task_map_gentle|straight_jump|3/30|22|21|25|6|{15: 6, 13: 3, 4: 3, 3: 16, 18: 1, 9: 1}|
|chapter3_offroad_task_map_gentle|s_bend_jump|0/30|17|6|27|4|{15: 9, 3: 16, 18: 1, 9: 3, 4: 1}|
|chapter3_offroad_task_map_gentle|right_angle_jump|0/30|0|0|0|0|{15: 29, 3: 1}|
|chapter3_offroad_task_map_gentle|ground_detour|5/30|0|0|28|22|{13: 5, 15: 3, 9: 8, 3: 9, 18: 5}|
|chapter3_offroad_ground_ablation|straight_jump|2/30|23|20|26|6|{12: 3, 15: 7, 3: 18, 13: 2}|
|chapter3_offroad_ground_ablation|s_bend_jump|0/30|1|0|22|0|{15: 25, 3: 5}|
|chapter3_offroad_ground_ablation|right_angle_jump|0/30|0|0|0|0|{15: 30}|
|chapter3_offroad_ground_ablation|ground_detour|0/30|0|0|25|3|{15: 21, 3: 8, 9: 1}|

进入跳跃是控制阶段切换，不等于真实离地或成功清障；交回STTW由首次远岸接触触发，不等于整程完成。地面绕行不要求进入跳跃。x阈值只记录实际到达位置，不能单独证明跨越脊坡稳定。终止码15为运行域退出，3/4为侧倾/俯仰阈值，18为到目标但曾越廊，9为观察窗口结束未成功，13为完成。

结论：H75开8/120、关闭2/120；S弯入口17→1、交接6→0，绕行5→0。保留H75作为主线，拒绝将本轮失败简单归于学习修正。仍是单训练模型与2来车初态的开发诊断，不是独立统计证明。下一步检查名义转向时序/预瞄与实际执行，原物理和道路判据保持。code12为地形失败终止。
