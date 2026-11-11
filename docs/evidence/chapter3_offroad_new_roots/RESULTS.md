# 新越野任务中的RB技能替换对照

固定同地图、路线、速度、α和来车初态；仅替换跳跃权重。原模型已完成数据复用，列出的候选均使用已完成物理轨迹；各组分母见表。数据角色按各组冻结协议列出，不将同一初态的多候选当作独立样本。不同进程的物理重复差异单独列出，不能把所有得失归因于跳跃技能。

- [source全部实际/参考XY与奖励](../chapter3_offroad_new_source/RESULTS.md)，权重 `learning/deployment/chapter3/policies/heading125`；角色：Frozen same-map new-root skill validation; 4 unseen initial states, no model selection or tuning; not unseen terrain generalization
- [rb400全部实际/参考XY与奖励](../chapter3_offroad_new_rb400/RESULTS.md)，权重 `/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_rb_jump_scheduled/pipeline/train/training/model_update_00400.pt`；角色：Frozen same-map new-root skill validation; 4 unseen initial states, no model selection or tuning; not unseen terrain generalization

|模型|路线|完成|新增/丢失|进入跳跃|交回STTW|首次越廊在接近|接管前最大位置差/m|
|---|---|---:|---:|---:|---:|---:|---:|
|source|straight_jump|11/60|0/0|41|38|5|0|
|source|s_bend_jump|5/60|0/0|51|41|11|0|
|source|right_angle_jump|0/60|0/0|1|0|58|0|
|source|ground_detour|18/60|0/0|0|0|15|0|
|rb400|straight_jump|12/60|9/8|41|37|5|0.0107215|
|rb400|s_bend_jump|5/60|4/4|51|43|11|0.0272643|
|rb400|right_angle_jump|0/60|0/0|1|0|58|0|
|rb400|ground_detour|18/60|0/0|0|0|15|1.37247|

接管前差异比较相同root/candidate的共同有效前缀，截止双方较早跳跃接管时间之前；地面绕行比较共同完整有效窗口。此统计检测非跳跃阶段差异，不证明相同前缀以外的因果效应。首次违规由20ms记录的累积标志确定。目标、道路、物理终止均未放宽；进入跳跃不等于真实离地，交接不等于到达目标。

[逐根结果与采用决定](ANALYSIS.md)：原模型34/240、RB40035/240；新增13、丢失12，不支持稳定净增益。
