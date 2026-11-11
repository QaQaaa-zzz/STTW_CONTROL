# 新越野任务中的RB技能替换对照

固定同地图、路线、速度、α和来车初态；仅替换跳跃权重。原模型已完成数据复用，列出的候选均使用已完成物理轨迹；各组分母见表。是开发条件，不是独立测试。不同进程的物理重复差异单独列出，不能把所有得失归因于跳跃技能。

- [chapter3_offroad_task_map_gentle全部实际/参考XY与奖励](../chapter3_offroad_task_map_gentle/RESULTS.md)，权重 `learning/deployment/chapter3/policies/heading125`
- [rb400全部实际/参考XY与奖励](../chapter3_offroad_rb400/RESULTS.md)，权重 `/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_rb_jump_scheduled/pipeline/train/training/model_update_00400.pt`
- [rb500全部实际/参考XY与奖励](../chapter3_offroad_rb500/RESULTS.md)，权重 `/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_rb_jump_scheduled/pipeline/train/training/model_update_00500.pt`

|模型|路线|完成|新增/丢失|进入跳跃|交回STTW|首次越廊在接近|接管前最大位置差/m|
|---|---|---:|---:|---:|---:|---:|---:|
|chapter3_offroad_task_map_gentle|straight_jump|3/30|0/0|22|21|3|0|
|chapter3_offroad_task_map_gentle|s_bend_jump|0/30|0/0|17|6|25|0|
|chapter3_offroad_task_map_gentle|right_angle_jump|0/30|0/0|0|0|24|0|
|chapter3_offroad_task_map_gentle|ground_detour|5/30|0/0|0|0|15|0|
|rb400|straight_jump|5/30|5/3|22|21|3|0.00816215|
|rb400|s_bend_jump|0/30|0/0|17|6|25|0.00544772|
|rb400|right_angle_jump|0/30|0/0|0|0|24|0|
|rb400|ground_detour|3/30|0/2|0|0|17|1.18933|
|rb500|straight_jump|3/30|2/2|22|20|3|0.00700091|
|rb500|s_bend_jump|0/30|0/0|17|5|25|0.00544772|
|rb500|right_angle_jump|0/30|0/0|0|0|24|0|
|rb500|ground_detour|3/30|0/2|0|0|16|1.83561|

接管前差异比较相同root/candidate的共同有效前缀，截止双方较早跳跃接管时间之前；地面绕行比较共同完整有效窗口。此统计检测非跳跃阶段差异，不证明相同前缀以外的因果效应。首次违规由20ms记录的累积标志确定。目标、道路、物理终止均未放宽；进入跳跃不等于真实离地，交接不等于到达目标。

结论：原模型8/120、RB400 8/120、RB500 6/120；S弯和90°弯均未新增整程成功。原模型保留为当前执行基线，RB候选保留对照，不宣称增强。S弯25/30起跳前越廊、90°弯始终无跳跃入口，当前首要瓶颈为地面弯道接近；应在原道路约束下修复或适配地面跟随，不继续用跳跃checkpoint筛选掩盖前段失败。绕行控制组存在1.19/1.84m共同观察窗口位置差，跨进程重复性限制单次得失归因。
