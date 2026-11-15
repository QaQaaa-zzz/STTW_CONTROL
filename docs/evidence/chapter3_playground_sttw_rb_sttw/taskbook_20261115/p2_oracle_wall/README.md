# P2 开发地图跨阶段候选案例（冻结重复完成）

[全部实际/参考 XY](oracle_xy.png) · [逐例矩阵](oracle_matrix.csv)

地图48、初态1001/1002、每初态3接近×3驶离：18格中4完成、6跳前失败、8跳后失败。左右接近均跨沟交回；左接近的直行/左驶离可完成，右接近的三个允许驶离候选均失败。

这是基于上一轮负结果进行障碍布局调整后的 DEV 案例。新增障碍 x=[12.65,12.75]、y=[-1.35,0.125] m，高0.5m，位于沟后2m平地之外。控制技能、1.8m/s、alpha0、物理参数及10秒目标保持标准不变。只支持有限三候选库内的差异，不代表所有可能控制均无解，也不代表独立测试泛化。

上一轮 map00/47 的36格负结果完整保留于 ../p2_oracle/。本轮6条接近与12条实际后缀构成矩阵，另2条初态采集；不把18格称作18条独立完整运行。净空为5ms保存位姿的原生几何测距，非连续时间保证。

冻结重复18格的跨沟交回、成功、失败阶段和终止分类均与首轮一致。重复沿用相同初态来源，属于技术重复，不将其当作两个新独立样本。见 [重复矩阵](repeat_oracle_matrix.csv) 与 [对比及视频元数据](repeat_comparison.json)。

- [初态1001状态](states_root1001.png) / [占据局部图](occupancy_root1001.png)
- [初态1002状态](states_root1002.png) / [占据局部图](occupancy_root1002.png)
- [配对成功视频](/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p2_oracle_wall/paired_media/up1_down0_root1001.mp4)
- [配对失败视频](/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p2_oracle_wall/paired_media/up2_down0_root1001.mp4)

视频按实际qpos回放，0.5倍速与1秒终态保持；成功576帧/19.2秒，失败409帧/13.633秒，ffprobe核对帧数。完整快照边界qpos/qvel一致后拼接真实前缀和后缀，不生成新物理。状态图不平滑，失败处停止；占据局部图使用各碰撞几何体的投影AABB，保守包围框不是精确mesh或连续安全证明。完整轨迹见首图，局部图只显示x10–14m。

P2所需机制案例、冻结重复、完整失败矩阵、两条配对视频及状态/占据图已具备，可进入P3。现证据仅支持开发任务上有限候选的跨阶段可行性差异；尚未验证学习预测或连续状态传递的收益。P3必须新增全关节/接触RB分支，不能将已有地面模型改名替代。

制图审查参考本地 scientific-visualization 技能；无平滑、无删除失败、无不确定性区间推断，源数据与变换见 state_figure_sources.json。
