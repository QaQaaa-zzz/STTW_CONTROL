# V5.2：父模型离线审计与训练暂停

2026-10-09。当前状态：**LOWER_TRACKING_LIMITED / 用户明确暂停训练**。新物理步0，新PPO更新0，工程小批尚未执行。

## 原始 XY 证据

- [straight_hold，10s](evidence/r196_preference_v51_20261009/figures/update100_straight_hold_10s_xy.png)
- [gentle_positive，10s](evidence/r196_preference_v51_20261009/figures/update100_gentle_positive_10s_xy.png)
- [gentle_negative，10s](evidence/r196_preference_v51_20261009/figures/update100_gentle_negative_10s_xy.png)
- [fast_turn，10s](evidence/r196_preference_v51_20261009/figures/update100_fast_turn_10s_xy.png)
- [steer_reversal，10s](evidence/r196_preference_v51_20261009/figures/update100_steer_reversal_10s_xy.png)
- [speed_changes，10s](evidence/r196_preference_v51_20261009/figures/update100_speed_changes_10s_xy.png)
- [fast_turn，16s延伸](evidence/r196_preference_v51_20261009/figures/update100_fast_turn_16s_xy.png)

## best 与 last

V5.2尚无训练模型，故没有V5.2 best或last。父模型为对应alpha的V5.1 update100 last；以下是按V5.2物理标准离线评定的候选，不能冒称已训练V5.2 best，也不把last自动称best。正式best保存器已实现，工程集成未执行。

|端点|物理失败场景|工作界失败场景|主目标失败场景|10s联合保持失败场景|物理排序Q|qualified|
|---|---:|---:|---:|---:|---:|---|
|alpha0|0|1|0|3|9.031971|false|
|alpha1|0|1|0|5|11.175389|false|

两端主目标按有效静稳样本均通过，但都在fast_turn越过共同工作界。alpha0仅3/6场景满足10s联合保持，alpha1仅1/6；不能称整体合格。

|场景|alpha0峰值roll|alpha0保持10s/16s|alpha1峰值roll|alpha1保持10s/16s|
|---|---:|---|---:|---|
|straight_hold|0.007485|True/None|0.009913|False/None|
|gentle_positive|0.106893|False/None|0.096700|False/None|
|gentle_negative|0.108787|False/None|0.110878|True/None|
|fast_turn|0.315400|False/False|0.367610|False/False|
|steer_reversal|0.184054|True/None|0.188043|False/None|
|speed_changes|0.013229|True/None|0.016716|False/None|

## 上层提出与底层执行：fast_turn [2,4)

速度单位m/s、转角单位rad；下表为均值，不是RMSE。请求和后步状态来自同一5ms区间，未平移。

|端点/通道|raw|保护前proposal|governed|actual|upper=governed−raw|lower=actual−governed|task=actual−raw|
|---|---:|---:|---:|---:|---:|---:|---:|
|alpha0/speed|2.600000|2.363879|2.363722|2.285841|-0.236278|-0.077881|-0.314159|
|alpha0/steer|0.250000|0.238520|0.238520|0.232153|-0.011480|-0.006367|-0.017847|
|alpha1/speed|2.600000|2.643582|2.643582|2.550435|0.043582|-0.093147|-0.049565|
|alpha1/steer|0.250000|0.222281|0.222281|0.210331|-0.027719|-0.011950|-0.039669|

## 底层持续跟踪超限

按下发参考连续平稳≥0.5s后再检查：速度|e_lower|>.15m/s，或转角|e_lower|>.04rad持续>.5s。

|方法|区间(s，右端开)|持续(s)|通道|有符号lower误差范围|执行器/残差裁剪率|
|---|---|---:|---|---|---|
|B0|[5.445,6.205)|0.76|steer|[0.042181,0.144419]|0/0|
|B0|[6.315,8.745)|2.43|steer|[-0.091599,-0.040079]|0/0|
|alpha0|[9.020,10.240)|1.22|steer|[-0.071315,-0.043075]|0/0|

B0这些时段下发参考为[2.3m/s,0rad]；alpha0约[2.374m/s,0.026–0.028rad]。区间内参考温和且没有电机或残差裁剪，但存在姿态/航向恢复状态；不能把所有转角反打解释成驱动能力不足，也不能把两个不同闭环轨迹当独立训练种子。冻结日志没有完整ESO/历史状态快照，因此具体内部原因**未验证**。没有换R196、改摩擦/增益或补新仿真。

用户在看到B0与alpha0的持续误差后明确选择“按附件暂停训练”。本轮因此暂停扩大上层训练；允许的一次短复现尚未执行，不自动转为参数搜索。

## 全程、平稳、过渡、恢复误差

所有18条轨迹逐步恒等式 `e_task=e_upper+e_lower` 通过，零时间平移。完整报告逐场景/方法/通道包含bias、RMSE、P95、最大值及发生时刻、越限比例、全部连续越限区间；同时保留全程、[1,6)、[2,4)、governed静稳≥0.3s、governed过渡、转弯后恢复窗口。速度一般诊断阈值.10m/s，转角.04rad。

- [完整分窗口、全部越限区间 JSON](evidence/r196_preference_v52_audit_20261009/lower_tracking_report.json)
- [可比较统计 CSV](evidence/r196_preference_v52_audit_20261009/lower_tracking_summary.csv)
- [原始全时间轴5ms NPZ](evidence/r196_preference_v51_20261009/data/update100_timeseries.npz)；[离线误差重建脚本](../learning/src/sttw_control/lower_tracking_audit.py)可重建全量误差，不重复上传原始波形
- [持续错误门槛与姿态、裁剪来源](evidence/r196_preference_v52_audit_20261009/lower_tracking_gate.json)

`wheel_speed_proxy=轮轴速度×0.1m` 与实际车体速度分别保留；proxy−actual是速度差，`slip_proxy`不是经接触运动学验证的真实滑移率。

## 是否仍改善、是否继续

- alpha0父训练每25更新平均step reward：-0.05931381, -0.05317716, -0.04799679, -0.03734911；末窗口负成本幅值改善22.18%。
- alpha1父训练每25更新平均step reward：-0.06801755, -0.05641128, -0.05393193, -0.04142840；末窗口负成本幅值改善23.18%。

这些是父V5.1训练片段趋势，不是完整回合/后期固定验证收敛证明。V5.2没有新数据，不能判断其改善或收敛。目前不增加更新；先解决/解释上述具体恢复段的跟踪限制，再决定是否恢复已声明的200/max400计划。

## 实施与验证状态

基于26c3fed，并先合入此前4090D优化7ab2cd8（合并405cda1）。实现仅三处奖励差异、5080分项上界/失败目标、Actor+log_std热启动、Critic/Adam重置、完整回合统计、晚期数值验证、best SHA加载和有界扩展。普通物理/观测/动作/底层不改；默认批量评估、PPO同步及编译缓存保留。

7项必要CPU检查通过：新奖励/峰值/cap/失败、配置继承、误差区间、best安全排序/partial拒绝/身份加载、旧B0在新奖励下离线重评分。只读审查发现并修复B0新分项缺失会使奖励图漏基线的问题。工程8×16×2检查与正式训练**均未执行**；训练集成和自动扩展尚无运行验证，不能称V5.2已验证或可部署。

本轮用户已明确授权上传结果，交付分支为 `experiment/r196-v52-best-tracking`。训练仍暂停；模型、缓存和重复大轨迹不提交。

- [交付身份与文件哈希](evidence/r196_preference_v52_audit_20261009/manifest.json)
- [alpha0完整物理选择指标](evidence/r196_preference_v52_audit_20261009/parent_alpha0_physical_selection.json)
- [alpha1完整物理选择指标](evidence/r196_preference_v52_audit_20261009/parent_alpha1_physical_selection.json)
- [暂停状态](evidence/r196_preference_v52_audit_20261009/status.json)
