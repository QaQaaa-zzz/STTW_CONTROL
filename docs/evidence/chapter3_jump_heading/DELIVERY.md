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

## 在线接续与补充训练均已完成

1. 接触保持150更新已结束（28.19分钟），126入口候选评测完成、无新候选采用。从heading125权重、fresh Adam、lr5e-5；4096环境×24步，Go2 PPO/域随机化；仅启用已有上一策略步接触并集开关。训练采样50Hz，与执行200Hz的接触检测并不完全相同。原严格评测不变，不自动追加训练。
2. 当前模型在线C6四条已完成：20ms协议固定1/2、在线2/2，原严格协议均0/2。root000实际改α0并回交续行，root003保持α.5且与固定全程trace一致。此为两开发根的实际接口打通证据，详细[轨迹与判据](RESULTS.md#当前heading125的真实在线技能接续四条完成)。
3. 修复单车在线入口遗漏的20ms接触保持与H75回交进度初始化，复用批量使用的platform_support_mask及resume_ground_lanes，不新写控制器。

见[带时间戳运行快照](delivery_config/running_snapshot.json)、[选档配置](delivery_config/selector_config.json)、[在线配置](delivery_config/online_root000_C6.json)。旧快照保留当时状态；在线完成以四条结果和online_stage_summary.json为准；补充训练完成及候选表见RESULTS收尾节，旧快照不作为实时状态。

## 任务书工作索引（策略版本分别保留）

|任务|已有证据与判断|
|---|---|
|地面G1–G3|[24根/288名义后缀与窄道](../chapter3_ground_connection_20261106/RESULTS.md)、[已完成队列](../chapter3_completed_queue_20261107/RESULTS.md)；宽路固定档已适用，窄道逐根结果保留。|
|地面G4|[8根/384外扰后缀](../chapter3_ground_force_20261107/RESULTS.md)；六档均可用，事后最优对最佳固定只余.005秒，不因此新增监督训练。|
|距离与接近先导|[18诊断](../chapter3_jump_approach_pilot_20261106/RESULTS.md)、[六新根108后缀](../chapter3_jump_six_roots_20261106/RESULTS.md)；属于旧冻结技能版本，不能冒充本轮heading125成绩。|
|中点R6与36序列|[6根216序列及12条真实在线C6/R6](../chapter3_completed_queue_20261107/RESULTS.md)；旧版本C6 4/6、R6 6/6，最佳固定也6/6，不宣称超越固定。|
|当前技能与选档接口|本轮两根四条；在线实际选α并完成跳跃回交续行；20ms 2/2、原严格0/2，不要求全工况成功。|
|原source/RB同域对照|[原对照](../chapter3_cross_skill_20261106/RESULTS.md)、[同预算25更新适配](../chapter3_skill_adaptation25_20261107/RESULTS.md)；保留失败，当前heading125不替代论文source/RB身份。|
|E5前向支路与重规划|[已有规则/能力分流](../chapter3_forward_route_roots_20261106/README.md)、[队列负结果](../chapter3_completed_queue_20261107/RESULTS.md)；未验证可用的支路不称可靠备用，强制跳跃任务的绕行不算完成。|

以上索引定位既有版本证据，不代表本次重跑全部历史。当前已实测的是两轮各126入口、36链候选、4条在线集成，并补24分支形成同域六根36分支能力表。独立TEST、跨技能归因与适用范围仍须按版本和分母报告；本轮不宣布整份任务书或论文研究全部完成。

## 代码、配置和复查入口

- `learning/cli/evaluate_isaac_pendulum_jump.py`：冻结技能执行、在线选择与回交。
- `learning/src/sttw_control/jump_connection.py`：可观测特征、整根排除、核选档。
- `learning/cli/export_jump_connection_records.py`：原严格/20ms分开导出，物理/策略/控制接口身份。
- `learning/cli/analyze_jump_heading.py`：同模型跨α全程XY和航向。
- `learning/configs/isaac_rb_jump_heading_contact150.json`与`isaac_phase_u_task_heading_contact.json`：当前训练实际覆盖配置；继承source元数据中的历史频率/接线状态不能替代runner覆盖后的实际配置。
- Go2原实现`legged_gym/envs/base/legged_robot.py:_reward_feet_air_time`使用当前/上一帧接触OR；本次复用项目已有相同思想开关，没有移植四足动作或物理模型。

原始数据根位于`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_jump_heading_bounded150`与`chapter3_heading_online`，训练在`chapter3_jump_heading_contact150`。本包仅保留必要图、逐例摘要、配置和身份，不包含pt权重、body_contact、缓存或密钥。

当前六根能力表和零物理复算入口见[RESULTS能力表节](RESULTS.md#当前冻结技能的六根能力表零新增训练)；原严格1/36、20ms19/36，预测支持17/20有效决策状态，未进行MLP拟合或额外PPO。真实在线2/2仅指前述两个开发根，不把离线表当部署表现。

本轮所有已启动作业的[完成快照](delivery_config/completion_snapshot.json)；四个补采根的配置位于delivery_config/capability_root002/004/005/008.json。最初running_snapshot保留历史时点，不表示当前仍在运行。

新验证006/021已冻结原36条train证据运行：C6两条20ms接续通过、strict均未过；异质批量上游差异使固定/在线成绩差不能解释为选档收益。一次同原点诊断两条上游退出，未改全局默认、未加训。见[新验证与限制](RESULTS.md#冻结36条训练证据后的新验证状态与批量在线入口)。
