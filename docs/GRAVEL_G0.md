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
