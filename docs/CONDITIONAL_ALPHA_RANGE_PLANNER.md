# 容扰—任务条件α区间查询接口

这是查询接口，不直接控制执行器。当前事实、采用与后续实验只在研究台账维护；本文件定义可复用接口。

实现：[preference_outcome_prediction.py](../learning/src/sttw_control/preference_outcome_prediction.py)。
入口为 `FrozenAlphaRangePlanner`，保持冻结中层与底层不变，不启动PPO。

```python
from sttw_control.preference_outcome_prediction import FrozenAlphaRangePlanner

planner = FrozenAlphaRangePlanner(frozen_outcome_model_directory)
plan = planner.query(
    current_native_observation,   # 现有U170管线的351维float32观测
    current_world_yaw,            # 当前世界航向rad，不是未来起扰航向
    current_task_time_s,          # 保留原任务时钟，不重置回合
    hypothetical_disturbances,   # 预先声明的假设扰动族
    user_alpha=original_user_alpha, # 与当前执行α分开保留
    candidate_alphas=[0., .25, .5, .75, 1.],
    no_fall_threshold=.8,
    task_threshold=.5,
    max_path_error_m=4.,          # 原路径RMSE限制；不是瞬时横向边界
)
print(plan['estimated_intervals'], plan['empty'])
```

4m只是已声明开放平面强扰动示例的宽松RMSE限制，不是通用道路容差。严格路径限制可能使集合为空；不能为了输出非空区间而静默放宽。

每个假设扰动为：

```python
dict(
    force_N=24.,                         # 沿下列方向的有符号等效力N
    force_direction_world=[-.7173560909, .6967067093, 0.],
    duration_s=.4,
    preparation_s=.6,                    # 现在切换α到假设起扰的准备时间
)
```

方向必须是固定世界水平单位向量，不随不同候选的车体分别旋转。假设列表不能取自实际未来施扰日程。当前实验为世界固定等效风载荷，不是经气动标定的风速或CFD。

输入观测必须来自相同冻结模型的真实观测组装、顺序和归一化，不能重新拼接。`observation[336]`是当前中层的条件α，临时调度时可能与用户偏好不同；协调器应显式传入原始`user_alpha`。仅两者相等时才可省略。查询会保留用户给定值，即使它不在声明网格上，并分别返回用户α和查询前的条件α。普通五档仍完整保留；默认研究九点网格用于验证五档之间的内部点，不自动改变实际执行候选集。

中层对各候选输出参考修正；两层16单元的小网络预测不触地、存活并进入原目标圆、原路径RMSE及upright速度峰值。原路径RMSE采用源状态到首次到达的窗口；未到达的存活样本用剩余任务窗口。失败样本不参与误差/速度回归，以免把早失败的短窗口学成低代价。

对于多个假设，取最小不触地/任务概率和最大路径误差筛选。查询返回候选掩码、每个候选的聚合预测和可多段区间；中间候选不合格就断开，空集合不隐含安全回退。速度预测当前是诊断值，不是经证明的速度约束。

区间连接的是相邻合格查询点，未查询的连续内部没有保证。概率是小样本模型分数，不能写成安全认证。已做的四状态α插值验证与新状态、外扰方向、准备时间的泛化是不同问题；严格时应把超出研究域的查询标为未验证。

接口只读取模型、配置和当前状态，不读取轨迹或未来结果标签。调用方负责选择执行α、变化率/驻留及返回用户偏好的时机；几何接回仍由已有10Hz模块完成，中层50Hz和底层200Hz不变。首次查询包含编译开销，应在任务前初始化/预热；局部测得的查询耗时不是系统硬实时保证。

区间之后可用无学习的 `tolerance_coordination.select_from_alpha_range` 生成执行候选：

```python
from sttw_control.tolerance_coordination import select_from_alpha_range
decision = select_from_alpha_range(
    plan, user_alpha=original_user_alpha, previous_alpha=last_executed_alpha,
    publication_permitted=current_reference_permission,
)
```

原用户α合格时优先保持或恢复；否则在已查询且合格的点中最小化与原用户α、上次执行α的平方距离（默认权重1、0.1）。不会从区间端点插出未经查询的动作。空集合或当前参考禁止发布时，显式保持上次执行值，并返回 `degraded=True`，不称该回退安全或可行。该函数只返回候选和原因；接入仿真执行、驻留和连续调度仍需对应运行证据，不能以单元测试冒充闭环实验。
