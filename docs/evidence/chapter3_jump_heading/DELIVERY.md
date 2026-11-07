# 第三章当前精简交付

先看轨迹：[完整路线root000](bounded_chain_root000.png)、[root003](bounded_chain_root003.png)、[六模型入口对比](bounded_entry_xy.png)。详细尝试和反例见[RESULTS](RESULTS.md)。

## 当前目的与结论

用户最新要求：优先走通技能，不追求所有工况100%完成。已有模型用于地面接近→跳跃→落地回交→续行；接续规划负责选择可执行的速度和α。失败保留，不改成功定义来制造通过。

|已完成测试|父模型150|heading125|新bounded1|
|---|---:|---:|---:|
|同入口20ms协议，分母21|14|13|14|
|完整链20ms协议，分母12|5|7|5|
|完整链原严格协议，分母12|0|1|1|

heading125为当前继续接续工作的工程基线，不称完全可靠。新bounded候选不采用。heading125成功7条的最终|y|≤.077m、|yaw|≤4°，但一条过程中横偏.823m，另有失败例偏移1.074m、末航向128°。不能只凭末端或成功子集宣布偏航解决。

## 模型与频率

- 原RB带摆整车5.091kg，轮半径.07m、宽.05m、21圆柱轮廓，名义摩擦.8；质量/执行器未为本轮改变。
- 物理dt=.0025s（400Hz）；地面控制/传感.005s（200Hz）；H75及跳跃策略.02s（50Hz）。上层规划按事件触发，不另冒称固定频率。
- 地面：H75 actor_0077，351→128→128→64→2 ELU；ECBC+ESO；zero_lower=true，即当前带摆车无学习残差。后轮反馈增益1、补偿限幅±60rad/s。行驶锁摆，跳跃释放，落地回交锁当前摆姿态。
- 跳跃heading125：Actor80→256→256→256→8、Critic106→256→256→256→1，SiLU；4动作，确定性用均值经tanh映射，原动作接口保留。新增4输入为平台横偏、sin航向、cos航向−1、横向速度，需定位/状态估计。
- 平台高.16m，前后沿3.6/7.6m；起伏→R2右弯→5.6m接近；交权x2，平台前1.6m；22秒全窗，原走廊半宽1m、目标(10.6,0)，前方路面延至35m避免离开场地。两个来车根、每根两速1.8/2×三α0/.5/1；不更改原STTW五档α定义。
- 原严格落地判据与额外20ms接触保持判据分别统计。在线交回持续支撑1s，随后1.5m/s、加速度.8m/s²，实速≥1.2m/s回交。20ms结果不代替原严格结果。

详细[车辆配置](../../../learning/configs/rb_sttw_platform.json)、[当前权重身份及SHA256](delivery_config/heading125_identity.json)、[完整链配置](delivery_config/complete_chain_root000.json)。部分配置路径指向工作站资产，远端不含大权重/原始轨迹，不能仅凭此精简包在新机器完整重跑物理。

## 正在推进，尚未作为结果

1. 接触保持150更新：从heading125权重、fresh Adam、lr5e-5；4096环境×24步，Go2 PPO/域随机化；仅启用已有上一策略步接触并集开关。训练采样50Hz，与执行200Hz的接触检测并不完全相同。原严格评测不变，不自动追加训练。
2. 当前模型在线C6：两根×固定2/.5或真实出口选档=4条完整任务、0优化。复用高斯核；查询排除整根，仅3/5条可用异根证据，非校准概率、非独立TEST。离线建议root000为1.8/.5、root003为2/.5，不等于已执行成功。
3. 修复单车在线入口遗漏的20ms接触保持与H75回交进度初始化，复用批量使用的platform_support_mask及resume_ground_lanes，不新写控制器。

见[带时间戳运行快照](delivery_config/running_snapshot.json)、[选档配置](delivery_config/selector_config.json)、[在线配置](delivery_config/online_root000_C6.json)。运行状态可能已变化，快照不作为实时完成证据。

## 代码、配置和复查入口

- `learning/cli/evaluate_isaac_pendulum_jump.py`：冻结技能执行、在线选择与回交。
- `learning/src/sttw_control/jump_connection.py`：可观测特征、整根排除、核选档。
- `learning/cli/export_jump_connection_records.py`：原严格/20ms分开导出，物理/策略/控制接口身份。
- `learning/cli/analyze_jump_heading.py`：同模型跨α全程XY和航向。
- `learning/configs/isaac_rb_jump_heading_contact150.json`与`isaac_phase_u_task_heading_contact.json`：当前训练实际覆盖配置；继承source元数据中的历史频率/接线状态不能替代runner覆盖后的实际配置。
- Go2原实现`legged_gym/envs/base/legged_robot.py:_reward_feet_air_time`使用当前/上一帧接触OR；本次复用项目已有相同思想开关，没有移植四足动作或物理模型。

原始数据根位于`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_jump_heading_bounded150`与`chapter3_heading_online`，训练在`chapter3_jump_heading_contact150`。本包仅保留必要图、逐例摘要、配置和身份，不包含pt权重、body_contact、缓存或密钥。
