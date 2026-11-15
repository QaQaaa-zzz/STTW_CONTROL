# P5 分阶段α：驶离改善，尚非同交接状态因果对照

[map06全部实际/参考XY](map06_xy.png) · [map07全部XY](map07_xy.png)。每图覆盖36条件×5组，世界米制等比例，轨迹止于真实终点。虚线为分阶段α，实线为前后同α；标题G/T/D分别为目标完成/超时/跟踪域退出，按图例顺序。

## 结果

两个已知DEV地图×2根×2速度1.8/1.9m/s×9路线。新增144物理回合，复用此前216回合；4×36环境GPU批次共556.99秒，0训练。H75及shared225 RB冻结，实际交回事件切换α，不增加固定等待。沟前后2m平地、两侧转弯、STTW关节锁定与原成功判据不变。

|接近α/驶离α|map06目标|map07目标|合计|远岸接触/交回|
|---|---|---|---|---|
|0/0|36/36|36/36|72/72|72/72|
|.5/.5|24/36|29/36|53/72|72/72|
|1/1|6/36|5/36|11/72|72/72|
|.5/0|36/36|36/36|72/72|72/72|
|1/0|34/36|29/36|63/72|72/72|

.5/0相对.5/.5新增成功19、保留53、丢失0；1/0相对1/1新增52、保留11、丢失0。1/0剩余9次全部在驶离阶段跟踪域退出：map06两根route8/v1.9；map07两根route6、7、8/v1.9以及root7/route8/v1.8。不是跨沟失败，不能把全部360次交回当成整程成功。

## 配对边界与真实状态差异

所有来源的物理MJB、实际H75/RB权重、除α和说明外配置相同，配对初始qpos/qvel差0。**但原生GPU重放并非相同交接状态**：同接近α的比较中，.5和1在RB前的根XYZ最大分量差分别0.01920/0.05006m；严格早于两者较早交回时刻的最大分量差0.09643/0.10943m，交回时间最大差0.180/0.090秒。qpos/qvel全维最大差也保留在paired.csv，它们混合单位，不能解释成单一物理距离。

故本批证明这些分阶段组合在当前DEV协议下可执行，并显示描述性改善；尚不能把全部改善归因于驶离α，也未完成实速匹配或证明独立偏好收益。后续先复用完整快照接口做相同交回状态分叉；不以微小引擎误差对齐为目标，也不掩盖这里实际可见的状态差异。

[map06速度](map06_speed.png) · [map07速度](map07_speed.png) · [map06路径误差](map06_lateral.png) · [map07路径误差](map07_lateral.png)。速度实线是真实车体前向速度，点线为后轮轴速×.07m；执行目标检查使用水平航向投影速度。误差配对仅取双方真实记录共同时间且同为phase0/2，不补齐失败后的轨迹。详细360回合、432条配对记录见cases.csv/paired.csv；每组身份见identity_receipt.json。该覆盖协议无奖励定义，不制造奖励曲线。

## 阶段结论与复现

P5只完成五种阶段α组合的物理覆盖；完整九组合、条件世界模型、固定α/仅选速/速度+α的同预算与穷举比较、源跳跃/RB匹配接口均未完成。地图06/07是已知DEV，不是独立TEST。P4可靠性缺口仍在；不据本批采用可靠规划器，不进入P6。

原始路径：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p5_phase_alpha`。manifest/status记录全部命令与来源；既有只读JIT依赖不作修改。分析入口：

```bash
PYTHONPATH=learning/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /home/qy/mujoco_playground/.venv/bin/python learning/cli/report_playground_alpha_execution.py \
 --run /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p5_phase_alpha \
 --output docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_phase_alpha
```

分析脚本同时兼容旧同α清单；旧216回合重分析仍得到72/53/11，无物理重跑。冻结原始证据未覆盖。
