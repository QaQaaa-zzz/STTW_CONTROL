# 新越野任务中的RB技能替换对照

固定同地图、路线、速度、α和来车初态；仅替换跳跃权重。原模型已完成数据复用，列出的候选均使用已完成物理轨迹；各组分母见表。是开发条件，不是独立测试。不同进程的物理重复差异单独列出，不能把所有得失归因于跳跃技能。

- [chapter3_offroad_steer_slew全部实际/参考XY与奖励](../chapter3_offroad_steer_slew/RESULTS.md)，权重 `learning/deployment/chapter3/policies/heading125`
- [chapter3_offroad_slew_rb400全部实际/参考XY与奖励](../chapter3_offroad_slew_rb400/RESULTS.md)，权重 `/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_rb_jump_scheduled/pipeline/train/training/model_update_00400.pt`

|模型|路线|完成|新增/丢失|进入跳跃|交回STTW|首次越廊在接近|接管前最大位置差/m|
|---|---|---:|---:|---:|---:|---:|---:|
|chapter3_offroad_steer_slew|straight_jump|2/30|0/0|19|19|1|0|
|chapter3_offroad_steer_slew|s_bend_jump|1/30|0/0|28|21|7|0|
|chapter3_offroad_steer_slew|right_angle_jump|0/30|0/0|0|0|28|0|
|chapter3_offroad_steer_slew|ground_detour|7/30|0/0|0|0|7|0|
|chapter3_offroad_slew_rb400|straight_jump|8/30|7/1|19|18|1|0.0031898|
|chapter3_offroad_slew_rb400|s_bend_jump|3/30|3/1|28|22|7|0.0111981|
|chapter3_offroad_slew_rb400|right_angle_jump|0/30|0/0|0|0|28|0|
|chapter3_offroad_slew_rb400|ground_detour|8/30|1/0|0|0|7|0.771627|

接管前差异比较相同root/candidate的共同有效前缀，截止双方较早跳跃接管时间之前；地面绕行比较共同完整有效窗口。此统计检测非跳跃阶段差异，不证明相同前缀以外的因果效应。首次违规由20ms记录的累积标志确定。目标、道路、物理终止均未放宽；进入跳跃不等于真实离地，交接不等于到达目标。

总完成10→19/120；直线2→8、S弯1→3、直角弯0→0、绕行7→8（各30条）。跳跃路线合计新增10、丢失2；S弯原成功丢失1、另新增3，不能称无损增强。非跳跃绕行也新增1且共同窗口位置差0.772m，跨进程物理差异限制因果归因。RB400＋0.6rad/s参考保留为后续开发候选，不覆盖原基线或宣称独立泛化提升。下一步冻结组合在新来车初态比较，并将完整接近、跳跃、着陆续行轨迹用于事件模型；90°弯前段不可执行仍单独保留，不以删去该路线提高完成率。
