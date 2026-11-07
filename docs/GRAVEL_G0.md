# 固定石子 G0 与 P0 配置

当前仅完成代码/几何准备；真实碰撞和网格响应验收待G0运行。冻结H75与橙色Local300，物理/底层/中层分别0.2/5/20ms。复用原热初态和参数101，S2/S5各154颗固定石子，同中心/半径，仅高度10/20mm变化。局部1cm/5mm水平与1mm竖向量化；重叠取max，世界固定几何。

## 资产与预算

`learning/configs/isaac_h75_gravel_scenes.json`含可重现的全部四组参数。运行根目录`runs/h75_gravel_g0`链接数据盘，assets下每项含terrain.npz、identity、实际几何审计、中心表和预览。旧无效浮点拼接产物明确保留在invalid目录，不参与运行。

G0四条10秒/20万物理lane-step/零优化，全部alpha.5、2.3m/s、参数101。10mm与5mm对照门槛：首末实际触石事件差≤10ms、法向载荷积分相对差≤5%、轮中心高度RMSE≤2mm。超出时选择5mm并记录，不冒5mm已经独立证明收敛。原数据中0.8–5秒逐0.2ms保存高频接触，所有10秒低频轨迹/动作/奖励/终止完整保留。

## 命令

工作目录为本分支工作树，环境`PYTHONPATH=$PWD/learning/src JAX_PLATFORMS=cpu`。

```bash
# 生成某一资产；输出必须为新目录，避免覆盖原证据
PYTHONPATH=learning/src /home/qy/miniconda3/envs/mjp/bin/python learning/cli/build_route_pebbles.py --scene-config runs/h75_gravel_g0/S2_10mm.json --output runs/h75_gravel_g0/assets/S2_10mm
# 四lane检查，包含共享GPU队列、自动分析、Bark完成或失败通知
bash learning/scripts/run_gravel_g0.sh
```

生成命令对应资产现已存在，不应直接重复执行覆盖。运行后查看evaluation/RESULTS.md及gravel_contact_events.json，先看xy_g0.png，再看contact_g0/reward_g0。

P0的144条唯一条件在runs/h75_gravel_g0/p0_matrix.csv，配置为learning/configs/isaac_h75_gravel_sweep.json。G0后assets.json自动记录各地形采用分辨率。P0尚未启动；验收通过后使用既有evaluate_isaac_terrain_sweep入口、assets.json和该sweep配置。仅未来完整P0可使用--batch-cases24，不以四条G0替代P0能力证据。参考速度误差对原请求，A/B/C阈值按任务书，保持原到达/末段保持判据。后续S1–S7/P1–P4保留阶段门槛，不提前拟合选择器。

长任务或排队：通知开始等待和预计区间后结束助手回复；后台运行及一次完成通知继续。用户说继续时只收取一次状态；仍长等则再次结束回复。

## 已完成运行结果

四条物理采样完整结束（1486秒）；S2两个分辨率均10秒存活，S5分别1.93/2.27秒物理终止。分析脚本时间轴错误已修复，仿真没有重复执行。四条均有实际前/后轮触石，原生接触缓冲扩到每prim128后未再溢出。

S2粗细网格响应超过既定工程容差，按方案采用5mm；S5的末接触与响应比较因终止而截断，不能建立收敛结论。资产映射已选5mm，但G0不能称任务能力全部通过，也不由这四条推断alpha无效。完整P0未启动。

## P0批量筛查接续

用户继续推进后，采用G0实际接触已核实的5mm资产进入P0筛查，保留尚未证明数值收敛的限制。S5工程检查失败是待比较的任务难度，不通过降低石子高度删除。修复P0三组物理参数映射和720万lane-step上限；24个场景×速度×参数组合，每组合五alpha加Zero，共144lane。

新增run_gravel_p0.sh仅调用已有evaluate_isaac_terrain_sweep，完成后分析A/B/C任务合格及真实接触/低载/饱和，不自动启动P1。所有对照从同原热启动重新运行。原G0配置要求保存executed_commands/body_poses，但旧CLI未收录这两个result字段；G0实际有nominal/offsets/measurement与高频motor_torque，不能冒已保存完整执行命令。此次修复CLI保存和批量拆分轴，P0将完整保留请求字段；不为补字段重跑G0。

## P0 接触诊断加速

首批50个中层步耗时183.791秒；该次运行按用户加速要求主动停止，记录保留在runs/h75_gravel_p0/interrupted_duplicate_contact_aggregation，不能作为完整P0结果。复用既有按阶段采集原则：保留0.8–5秒内0.2ms原生接触点/法向/摩擦原始数据，跳过轮ContactSensor对相同缓冲的重复聚合及逐物理步聚合日志复制。主接触传感器的非轮碰撞终止判断照常逐步更新，控制/物理/模型/144lane批量不变。raw_contact_only为显式配置，默认关闭；原生接触索引另记传输同步与压缩耗时。语法和diff检查通过；实际吞吐改善待重跑进度核实，不声明预估倍数。本次重新执行同一144条/10秒/720万lane-step上限/零更新预算，不自动重试。

## GPU批量容量

按用户要求，地形sweep入口默认及P0脚本批量容量提高到341个case（每case固定五alpha加Zero，最多2046个实际环境，接近2048）。实际环境数为本批待评case数×6，不填充重复样本。当前P0仅24case/144环境，原本已全量并行，正在运行的批次不重启；此修改不扩大实验预算、种子或重复次数。后续阶段在其门槛通过后沿用更大批量；任务间仍使用共享GPU队列。
