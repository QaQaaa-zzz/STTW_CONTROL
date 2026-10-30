# Disturbance-Tolerance-Guided Preference Coordination for a Single-Track Robot

Internal TCST manuscript. Evidence updated 2026-10-30. Model and data provenance are recorded in the companion evidence manifest; the current study uses frozen U170 control, a continuation-conditioned predictor, and 10 Hz geometric reconnection.

## Abstract

A tracking preference changes more than the nominal balance between path accuracy and speed regulation in a single-track robot: it also changes the closed-loop state from which the robot must withstand a disturbance. This paper develops disturbance-tolerance-guided preference coordination for low-speed turning. A frozen preference-conditioned middle policy supplies reference corrections to an existing balance controller. A compact supervised network evaluates candidate preferences together with preparation time, hypothetical disturbances, and a specified feedback policy for returning to the user's preference. A nonlearned coordinator retains the user setting when its predicted outcomes satisfy the task criteria and otherwise selects a temporary preference. Feedback restoration and geometric reconnection preserve the original task. Matched simulations expose distinct outcomes for balance recovery and goal completion. In two 24 N disturbance cases, both the fixed user preference and a fixed intermediate preference contact the ground before arrival, whereas the proposed execution reaches the original goal at 6.935 and 6.980 s and survives the full observation window. A separate three-case held-versus-return comparison achieves three arrivals with restoration and one with continued holding. These results establish the usefulness of coordinating preparation and preference restoration in the tested low-speed regime. [E250–E258,E276–E278]

## 1 Introduction

Single-track robots must coordinate steering, forward motion, and balance while following a task reference. In a tight turn, a request for smaller path error can induce a different speed and reference history from a request for stronger speed regulation. For a user, these settings represent legitimate tracking preferences. For the physical robot, they create different states, actuator demands, and responses to a subsequent disturbance. A controller that performs well on the nominal task can therefore encounter a disturbance from an unfavorable state. The control problem is to preserve the user's task while deciding when a temporary change of preference is useful. [E250,E259–E261]

The preference parameter in this study is denoted by alpha: zero emphasizes path tracking and one emphasizes speed regulation. It enters a learned middle policy that produces bounded reference corrections. The middle policy and the lower balance controller are frozen during disturbance-tolerance learning and evaluation. This architecture provides a small set of executable alternatives without retraining a complete control policy for each disturbance scenario. The coordinator changes the executed preference, while the original path, destination, and user preference continue to define the task. [E259–E261]

Two aspects make this coordination problem temporal. First, a preference command acts through reference filters, actuator dynamics, and vehicle motion. Its disturbance response depends on the state reached after a preparation interval. Second, maintaining a disturbance-resistant preference indefinitely may prevent completion of the original task. The temporary intervention must therefore be evaluated together with what happens afterward. Our predictor evaluates the candidate preference under a specified feedback-restoration policy, and our experiments distinguish ground contact, original-goal arrival, and original-user task cost. [E251,E253,E276]

### 1.1 Related work

Learning-enhanced bicycle control provides a relevant foundation. Hierarchical residual reinforcement learning combines attitude and position subtasks with conventional control for unmanned-bicycle path tracking. [R3] Adaptive data-enabled policy optimization has been combined with inner-loop feedback linearization for bicycle lean tracking. [R4] Disturbance-rejection approaches also coordinate trajectory tracking and balance through velocity planning and nonlinear feedback. [R5] These studies motivate the use of a capable frozen control stack and place the focus here on how its tracking preference is executed over time.

Reference and command governors modify requests to address constraints in an existing closed loop. [R6] Predictive safety filters assess a proposed control action and construct an alternative through model-based prediction. [R7] Recovery RL separates task behavior from recovery behavior using learned information about violations. [R8] Our coordinator similarly acts above an existing controller, but its decision variable is the tracking preference of a frozen middle policy. Its supervised target is the joint disturbance and task outcome under a candidate preparation and a prescribed subsequent restoration policy. This defines a concrete application of outcome prediction to preference execution in a steering-balanced single-track robot.

### 1.2 Contributions

The paper develops three connected contributions:

1. **Preference-conditioned closed-loop characterization.** A frozen middle policy exposes the path–speed tradeoff, and matched low-speed experiments characterize how preference-dependent state and reference histories change disturbance outcomes.
2. **Preparation- and continuation-conditioned tolerance prediction.** A compact model evaluates candidate preferences using the available preparation interval and a specified feedback-return policy, jointly predicting contact avoidance, original-goal completion, path deviation, and speed.
3. **Task-preserving preference coordination.** A user-priority selection rule combines these predictions with feedback restoration and geometric reconnection. Matched fixed-preference and held-versus-return experiments evaluate its effect on completing the original task.

The first contribution identifies the control behavior available to the coordinator, the second evaluates the consequences of using that behavior, and the third connects a temporary intervention to task completion. The experiments accordingly separate nominal preference behavior, preparation outcomes, and the effect of returning to the user's preference. [E250–E261,E276–E278]

## 2 问题定义与实际控制系统

当前固定中层为STTW_MIDDLE_H75_LOW_SPEED_MIXED_U170，由H75加入低速小弯混合场景续训后选出。后续容扰学习不再更新中层或ECBC、ESO、local300底层。中层50Hz，底层200Hz，每底层步含25个0.0002s物理子步；前后轮摩擦系数均0.8。重规划模块按10Hz检查/更新接回路径，原世界目标不变；这不是小网络以10Hz连续重新决策的证据。[E259,E261]

```text
原路径、原目标、速度请求、alpha_user
       ↓ 几何跟随 / 10Hz因果接回
当前观测、前方曲率、候选中层参考修正、假设扰动和准备时间
       ↓ 小型联合结果预测器
通过抗扰/到达/路径阈值的候选集合 → 区间表示
       ↓ 用户优先选择 + 当前传感器反馈恢复
alpha_exec → 冻结U170（50Hz）→ 原参考滤波与权限
       → ECBC + ESO + local300（200Hz）→ 车辆
```

物理完整快照包括车辆、ESO、执行器、参考滤波与历史观测，分支接续不重置这些量。预测器只使用当前可构造观测的子集和候选查询，不读取完整未来轨迹或真实起扰日程。当前观测含真实车体速度；实车应提供相应估计，后轮轴速乘半径不能无验证替代。所有新结果均为仿真。[E259–E261]

本文的风为固定世界水平向量上的质心等效外力：方向(-sin0.8, cos0.8, 0)，持续0.4s。20N或24N是力幅值，不是气动标定风速。低速检验使用完整直立观察窗真实前向速度绝对值不超过2.1m/s。该门槛是本研究工况定义，不是稳定性定理。[E250,E254–E256]

## 3 后续策略条件的容扰—任务预测

令x_k为含车体、执行器、ESO、滤波、历史观测及协调器状态的完整闭环状态，r为固定原任务，z_k=H(x_k,r)为预测输入可观测部分。对某个候选a及给定返回规则π_return，闭环分支记为

```text
x_(k+1) = F_cl(x_k, α_exec,k, r, d_k)
α_exec,k = Π(a, α_user, current sensors, plan age; π_return)
Y_nf = 1{从源时刻到原任务10s均未触地且无工程故障}
Y_goal = Y_nf × 1{故障前曾进入原世界目标1m圆}
```

这里F_cl表示实际仿真闭环，不是另一个已学习动力学模型。相同源状态的候选分支只改变声明的α和假设扰动；控制隐藏状态保持一致。工程故障不混入物理触地训练标签。网络由有限确定性分支的监督结果拟合，分类输出作为模型筛选分数使用；三个初始化成员平均并不自动给出真实扰动分布下的校准概率。[E251,E259–E261]

### 3.1 预测对象与监督标签

定义 f_ψ(z,a,d,τ,α_user; π_return) → (q_no-contact, q_goal, ê_path, v̂_peak)。其中z取源时刻最新状态与路径上下文，a为候选偏好及U170对应参考修正，d为假设扰动方向、幅值、持续时间，τ为准备时间；π_return为冻结的传感器返回规则。q_goal标签要求到达原目标且完整窗口存活。窗口截止原任务10s，触地轨迹在真实失败处截止；到达为接触/故障前进入原目标1m圆，不要求终端停车、航向对齐或驻留。[E251,E259–E261]

43个特征包括最新观测34维、候选/假设查询8维及原用户α一维。观测含速度、姿态、参考滤波以及当前/前方曲率等信息。架构为43→16ELU→16ELU→4，三个随机初始化成员分别预测后平均；每成员1044参数。两分类头使用BCE，路径与速度头学习log1p目标，损失权重1/1/0.25/0.25，失败样本不参与连续量回归，避免把早失败短轨迹误当低代价。AdamW学习率及权重衰减均0.001，三个成员各5000次CPU监督更新，固定末轮。训练拟合良好不等于校准或泛化。[E251,E252,E260,E268]

训练数据为232条查询、188条独特物理轨迹：185条“原用户α等于执行α”的恒定行为等价查询；44条在允许返回前已经触地的因果前缀失败查询；3条真实反馈返回轨迹。44条派生查询不计作新增物理实验或独立重复。新验证轨迹未用于该次拟合；来源状态和邻近恒定策略数据属于已知开发域，不称独立holdout。[E251,E252]

### 3.2 区间与实际选择

对预先声明的假设族D，在每个候选取最小不触地/到达分数和最大路径误差预测。候选集合定义为：

```text
A_ok = {a in A_query : min_d q_no-contact(a,d) >= eta_q,
                       min_d q_goal(a,d) >= eta_g,
                       max_d e_path(a,d) <= e_max}
```

本文实例采用eta_q=0.8、eta_g=0.5、e_max=4m。4m是开放平面强扰动实例的原路径RMSE门槛，不是道路边界；RMSE从源时刻算至首次到达，未到达存活轨迹用剩余观察窗。速度头仅作诊断，不作为已证明上界。不能将上述分类分数直接解释为具有覆盖率保证的安全概率。[E254–E258,E260]

原五档0/.25/.5/.75/1保留，当前研究另查询.875。区间连接相邻合格查询点，遇到不合格点断开；尚未查询的连续内部未验证。α_user在集合内时优先执行原值，否则从已查询的合格点最小化 (a−α_user)² + 0.1(a−α_previous)²。原方案中完整J_user目标目前落实为任务/路径预测门及偏好距离项，未实现长时域J_user优化，不将其写成现有算法。空集合或参考禁止发布时保持前值并标记退化，不宣称退化安全。[E260,E261]

### 3.3 执行与反馈返回

当前闭环实验在声明的源时刻查询一次区间。候选执行至少2s后，当当前侧倾绝对值≤15°、侧倾角速度≤0.35rad/s、前向速度≥1.5m/s且参考允许发布，连续满足0.3s后以每秒1α的上限返回α_user；条件变坏时暂停。定时从方案起点计算，不读真实外力的开始/结束标签。用户α用于原任务奖励，实际α用于中层查询，切换不清空闭环状态。[E253–E256,E261,E265]

预测器必须与这一后续策略一致：旧42特征恒定α模型不能直接评价反馈返回方案。新的manifest绑定43特征、返回参数和实现身份。这种一致性是方法复现要求，不是稳定性证明。全程连续重选区间、改变返回规则或改变用户偏好均需要对应验证。[E251,E260,E261]

### 3.4 可复现的执行顺序与有限性质

```text
源时刻（本阶段每回合仅一次）:
  读取当前z、用户偏好和参考发布权限；保留完整闭环状态
  对每个候选a生成冻结U170参考修正
  对每个预声明扰动假设查询结果预测器；形成A_ok
  若权限不允许或A_ok为空：保持此前α，记录退化
  否则若用户α合格：使用用户α
  否则：在合格查询点中最小化偏好距离与切换代价
随后每20ms:
  最小准备期未满时保持准备α
  满足当前姿态/速度/权限条件并累积驻留时间后进入返回阶段
  返回阶段条件满足时按限速向用户α移动，否则暂停
  已返回时保持用户α；几何接回按独立100ms时钟检查
```

该实现直接支持三个有限性质。其一，非退化选择属于实际查询的合格点，区间内部不会未经查询直接执行。其二，选择与返回接口不接收真实未来起扰/结束时刻；离线假设家族必须在执行前固定。其三，固定用户偏好时返回阶段使用

```text
α_(k+1) = α_k + clip(α_user − α_k, −ρΔ, ρΔ),  Δ=0.02s, ρ=1/s
```

故返回阶段向用户值单调靠近且不越过用户值，数学限速为ρΔ；实现含1e−6完成吸附容差。条件不满足时保持α，首次进入返回阶段需要驻留；进入后暂停再恢复不重新要求完整驻留。源时刻从用户α选择准备α可能发生跳变，上述限速只适用于返回阶段，不能冒称整个α信号都限速。原参考滤波/执行权限继续在底层生效。这些都是代码级性质，不证明状态稳定、任务必达或退化安全。[E261]

### 3.5 数据单位和评价口径

本文分别统计物理轨迹、监督查询和来源状态。共享完整来源的分支是配对干预，不是独立训练重复；由一条物理失败轨迹派生的标签不能增加独立样本数。各阶段固定物理模型和控制频率，保留成功与失败，只有明确同源同事件的缓存才用于机制对照。不同α加权奖励不直接排名，返回对照始终使用原用户偏好重评分。[E251,E253,E275]

| 证据组 | 单位与数据角色 | 能支持的结论 | 不能支持的结论 |
|---|---|---|---|
| 232查询/188物理轨迹 | 当前预测器TRAIN | 训练对象及数据构造可复现 | 232独立实验或概率校准 |
| 三条返回与同源恒定对照 | 相关开发轨迹，返回三条已用于TRAIN | 此组返回规则改善原目标到达 | 独立泛化、神经预测器不可替代 |
| 五条冻结后验证 | 四条NN选择、一条固定区间端点；DEV | 有限域的实际闭环实例 | 五次独立重复、全区间安全 |
| 四条新镜像阵风 | 新几何DEV，全部空集退化 | 三完成一失败、预测与行为范围的限制 | 有效神经调度、整个假设族真假分类 |
| 两条失败后候选 | 事后诊断DEV；复用原α0缓存 | 已测三个候选均不能完成左24N | 任意α和时机均不可恢复 |

五条冻结后成功、四条镜像验证与两条事后诊断用途不同，不合并为“8/11成功率”评价总体可靠性。报告逐工况实际结果，不给共享来源的小样本制造独立二项置信区间。[E254–E258,E269–E275]

## 4 仿真证据

### 4.1 同一扰动下的偏好差异

![五档固定偏好的实际与原始参考轨迹](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/u170_low_speed_five_alpha_load_boundary_v1/analysis/left90_R1p3_v2p0_t2.2_F+24_fixed_candidate_xy.png)

图1：U170，2m/s请求、R1.3m左90°弯，24N等效力在原任务2.2s开始、持续0.4s，五档α各一条相关确定性轨迹。实际轨迹在真实终点截止，原目标不变。[E250]

| 固定α | 10s物理结局 | 原目标到达 |
|---|---|---|
| 0 | 2.675s触地 | 否 |
| .25 | 触地 | 否 |
| .5 | 存活 | 8.680s |
| .75 | 存活 | 否 |
| 1 | 存活 | 否 |

α0与α1受扰前1–2.2s最低真实速度分别约1.448和1.660m/s；全直立窗最高速度均低于2.1m/s。α0名义侧倾角幅值并不更大，因此不能将“姿态角更大”或速度单一因素当作已证明原因。结果显示同一偏好产生的闭环状态/参考历史与抗扰结局相关；固定α=.5也能完成这个事件，不能据此宣称所有固定α均失败或必须使用学习调度。[E250]

### 4.2 返回用户偏好对原目标的作用

三条开发轨迹从同一完整早状态准备α=.875，在实际起扰1.9/2.0/2.1s时均存活并到达，时间6.165/8.645/6.225s。认证复用同源同扰动的恒定.875轨迹，后者三条都存活但仅一条到达。返回规则改善了这组原目标结果；不同起扰条件共享源状态，不能作为三个独立重复估计成功率。[E253]

| 起扰时刻(s) | 恒定.875到达(s) | 返回到达(s) | 恒定最近目标距离(m) | 返回最近目标距离(m) | 恢复用户0(s) |
|---:|---:|---:|---:|---:|---:|
| 1.9 | 未到 | 6.165 | 1.02545 | 0.05810 | 4.18 |
| 2.0 | 9.050 | 8.645 | 0.05117 | 0.00723 | 4.46 |
| 2.1 | 未到 | 6.225 | 2.07887 | 0.07313 | 4.42 |

第一条恒定轨迹只比1m目标阈值多约2.5cm，属于边界案例；第三条的目标接近改善更明显。所有返回/恒定对照均存活，因此此处证明的是该组原任务结果的变化，而非额外存活收益。按同一用户0重评分的三个原路径代价分项也并非全部改善；到达判据、连续距离、原用户代价分别报告，不用一个二值成功数代替全部控制品质。[E276]

![同源返回与恒定偏好的实际轨迹及同用户奖励](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/u170_causal_preference_return_v1/analysis/cached_hold_comparison_02.png)

全部三对的原始缓存身份、共同前缀核验和原用户代价保存在E276；图示起扰2.1s的一对，不代替表中全部结果。未重跑缓存，也未改变成功阈值。

### 4.3 冻结网络后的五条新物理验证

![网络选择与返回的实际轨迹](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/u170_return_policy_nn_confirmation_v1/analysis/left90_R1p3_v2p0_source_t1p0_a0_prep0.925_F+24_fixed_candidate_xy.png)

图2：网络在原任务1s查询固定假设族并选.875，真实起扰1.925s未传入选择器，反馈恢复原用户0。明显绕行仍保留，不将到达称为原路径精确跟踪。[E254]

| 验证类型 | 速度请求 / 半径 | 真实起扰 / 力 | 模型区间；执行α | 到达时间 | 最终α |
|---|---|---|---|---:|---:|
| 新时刻NN选择 | 2m/s / 1.3m | 1.925s / 24N | [.875,1]；.875 | 6.935s | 0 |
| 新时刻NN选择 | 2m/s / 1.3m | 2.075s / 24N | [.875,1]；.875 | 6.980s | 0 |
| 区间上端实测 | 2m/s / 1.3m | 2.000s / 24N | [.875,1]；固定测试1 | 6.735s | 0 |
| 另一来源状态NN选择 | 2m/s / 1.3m | 2.000s / 20N | [0,.5]；0 | 4.930s | 0 |
| 不同低速弯道NN选择 | 1.6m/s / .9m | 2.000s / 20N | {0}；0 | 6.960s | 0 |

五条均完整窗口存活、到达原目标并通过全直立窗速度≤2.1m/s检查；其中四条由网络实际选择，一条为预测区间端点测试。前三条执行返回，后两条始终保持用户0。共两个几何、三个来源状态，不能写成五个独立随机重复或统计可靠性。全部逐步奖励、实际α、原路径误差与真实失败截止均保留。神经选择前的实际未来事件输入检查为False。[E254–E258,E264]

相同早状态的模型分数阈值从.8升至.9999时，输出从[.875,1]收缩为{1}，说明接口响应阈值；该数字不表示99.99%真实不触地概率。不同低速来源的24N假设查询可返回空集，空集没有被删除。[E262,E266]

### 4.4 预测局限与已知失败

旧恒定策略V2在新准备时间上的四条实际推荐中三条触地，尽管之前32条α插值查询表现较好。增加监督更新改善训练拟合，也未自动修复时间间隙。新43模型将原用户及返回策略明确纳入预测对象；这些先前失败保留在研究台账和独立run，不能从新五条成功回写成旧模型可靠。[E263,E267]

新五条在默认筛选门槛下未出现预测可行而触地/未到达的案例，但路径预测存在误差；更严格2m路径门槛下，先前两条时刻验证存在一次误判。速度峰值头在上端案例低估约0.276m/s，因此不能用其作约束证书。五条正例不足以校准失败风险。[E257,E258]

### 4.5 前瞻镜像场景：空集退化与一个失败

在模型和阈值冻结后，新增1.8m/s、R1.1m的左右90°弯，bank0、α_user=0。两条无扰动源轨迹分别于4.815s和4.880s到原目标，全10s存活，最高真实速度均为1.95944m/s。随后从各自task1.0s完整状态续接，每方向20/24N世界固定方向等效阵风，实际起扰2.0s、持续0.4s。控制器查询预先给定的20/24N×准备0.9/1.0/1.1s家族，不读取实际事件时序。四条均得到空集，执行“保持此前α=0”的退化规则，不能计为网络找到可行偏好的成功案例。[E269–E271]

| 方向 | 力幅值(N) | 原目标到达(s) | 物理结局 | 续接窗直立最高真速(m/s) |
|---|---:|---:|---|---:|
| 左弯 | 20 | 5.590 | 全10s存活 | 1.83220 |
| 左弯 | 24 | 未到达 | 2.465s触地 | 1.82098 |
| 右弯 | 20 | 5.310 | 全10s存活 | 1.83653 |
| 右弯 | 24 | 5.430 | 全10s存活 | 1.83653 |

![新左弯24N：空集退化保持α0后触地，轨迹在真实失败处截止](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/u170_tcst_mirrored_gust_v1/analysis/left90_R1p1_v1p8_prep1_F+24_fixed_candidate_xy.png)

全部四条实际XY、每步与累计奖励及分项：[完整镜像阵风面板](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/u170_tcst_mirrored_gust_v1/analysis/README.md)。

四条均独立重建原用户奖励并核对实际α，全部保持0，无偏好返回动作。三条完成任务并不证明整个假设家族可行，也不能直接把聚合空集判作错误；但本轮未提供有效偏好调度，且左右同24N的结局不同，限制了现有模型的可用范围。场景为新几何开发验证，bank0仍共享，不是独立训练种子统计。[E270,E271]

一个具体的观测缺口是：task1s查询的当前、前0.5m和1m曲率均为0；原路径前方约1.5–3m才包含有符号弯道，前2m曲率约±0.9091/m。两源各有12个查询特征超出既有训练范围，但未触发归一化截断。该诊断提示更远几何预览与训练状态覆盖值得修正，尚未证明它们是预测外推失准的唯一原因。不得把当前空集分数解释成校准后的真实失效概率，也不因发现失败就修改阈值隐藏结果。[E272,E273]

### 4.6 失败后同源候选诊断：不把空集失败直接归咎于预测器

在4.5左24N失败后，追加同一完整状态的α=0.5与1两个准备候选，沿用反馈返回规则、24N/.4s@2s、原目标与全部物理参数。两条分别在2.480s和2.555s触地，均未到达；原缓存α=0在2.465s触地。三条源时刻之后的直立最高真速度为1.82098、1.82251、1.82220m/s，仍为低速。失败都发生在最小准备期结束前，不能解释为返回用户偏好过早。[E274,E275]

![同一完整状态下α0缓存与新α0.5/1实际轨迹、真实速度和原用户奖励](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/u170_tcst_left24_candidates_v1/analysis/cached_user_candidate_comparison.png)

该结果更正了把左24N失败直接归因于预测器选择不当的解释：尚无证据表明此次已测试候选能救回任务，空集与这三个候选的实际失败一致。未测α或准备时刻仍可能有效，因此也不能宣布该工况不可恢复。本阶段停止围绕此强扰动继续扫参，把它保留为能力边界。远端几何特征提取函数已通过接口测试，但尚未训练或接入新预测器，不宣称特征修正带来性能提升；现有冻结模型和已有成功实验保持原身份。[E274,E275]

### 4.7 Matched fixed-preference baselines

![Same-source actual and reference trajectories with fixed-preference baselines](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/u170_matched_fixed_baselines_v1/analysis/matched_comparison/group_00_comparison.png)

Four additional baseline continuations compare fixed preferences 0 and 0.5 with the two previously completed neural-selection cases. Each comparison starts from the identical complete closed-loop state at task time 1 s and retains the same U170 policy, lower controller, original path and goal, 10 Hz geometric reconnection, and 24 N disturbance lasting 0.4 s. The disturbance begins at either 1.925 or 2.075 s. Baselines execute their stated preference from the shared source; the earlier prefix belongs to the original user-zero trajectory. All task costs are rescored for user preference zero. [E254,E277,E278]

| Gust onset (s) | Fixed alpha=0 | Fixed alpha=0.5 | Tolerance-guided selection and return |
|---:|---|---|---|
| 1.925 | Contact at 2.390 s | Contact at 2.430 s | Goal at 6.935 s; survives to 10 s |
| 2.075 | Contact at 2.495 s | Contact at 2.530 s | Goal at 6.980 s; survives to 10 s |

The proposed controller selects alpha=0.875 and restores the original user preference at 4.26 and 4.38 s. Both fixed baselines contact the ground before reaching the goal in both cases. These matched comparisons demonstrate a task-completion benefit over the original preference and a fixed compromise under the two tested events. They complement the held-versus-return comparison in Section 4.2: preparation preserves balance through the disturbance, and restoration enables subsequent task completion. The full comparison includes both event timings and preserves each physical endpoint. Original-path running costs are reported separately from contact and arrival; shorter failed trajectories are not ranked favorably by their smaller accumulated cost. [E253,E254,E277,E278]

### 4.8 Preset preparation with identical feedback restoration

![Preset preparation and tolerance-guided execution under the same gust](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/u170_preset_return_baselines_v1/analysis/matched_comparison/group_01_comparison.png)

A second comparison replaces the predictor's choice with a preset preparation preference of 0.75 or 1. Both baselines use the identical source state, feedback-restoration rule, original goal, and geometric reconnection as the proposed method. Their preparation is initiated at the same declared task boundary; neither baseline receives the actual future gust timing. These four additional continuations separate the effect of executing a preparation-and-return sequence from the effect of choosing its preparation preference. [E279,E280]

| Gust onset (s) | Preset 0.75: arrival (s) / intervention (alpha·s) | Preset 1: arrival (s) / intervention (alpha·s) | Network-selected 0.875: arrival (s) / intervention (alpha·s) |
|---:|---|---|---|
| 1.925 | 7.275 / 2.093 | 6.410 / 2.789 | 6.935 / 2.432 |
| 2.075 | Not reached / 2.410 | 5.530 / 2.830 | 6.980 / 2.446 |

All six compared trajectories survive to task time 10 s. Intervention is the integral of the absolute difference between executed and original user preference over the shared source-to-end window. Preset 1 and network-selected 0.875 both complete the two tasks. The network-selected execution uses 12.8% and 13.6% less preference intervention, while preset 1 arrives earlier and has smaller primary running task cost in both cases. Preset 0.75 reduces intervention in the first case but does not reach the second goal. The comparison illustrates the tradeoff among preparation strength, task completion, and departure from the user preference; it motivates state-dependent selection across a broader set of tasks rather than a claim of uniformly superior performance in this pair. [E279,E280]

## 5 面向TCST的论点与最小后续工作

现有数据已经形成可写作的应用控制故事：路径优先可能降低未来容扰能力，过晚切换可能失效，提前准备与反馈恢复可以同时服务存活和原目标。论文价值应落在这一闭环问题、可执行的策略条件预测及实际效果上，不要求所有场景全面领先。但目前五条成功轨迹仍不足以支撑宽泛可靠性或神经预测器不可替代性。

后续按以下顺序推进；已完成的镜像面板见4.5，下面的修正与验证尚未执行：

1. **保留能力边界，集中论文论证。** 镜像面板和两个追加候选已完成；左24N三个已测候选均失败，当前不追加训练/扫参去追求全场景成功。几何预览补强暂为接口准备，待出现能区分预测不足与行为能力不足的证据后再安排受控对照。论文同时保留成功、退化和失败，而不以新的简单场景替换负结果。
2. **任务与方法归因。** 已有同源恒定α缓存的配对归因已在4.2整理，分开到达、连续目标距离、原用户代价和路径偏离；后续不重复运行这些缓存。现有固定.5成功反例必须保留。按用户最新要求，4.7已补齐同条件固定α0/.5基本基线；下一补充简单因果调度规则与少量跑道条件，保持统一物理、几何接回及原用户评分，不以全面胜过所有基线为目标。
3. **实车与论文定稿。** 在明确实车授权及传感器/执行接口后验证关键低速转弯—扰动—恢复过程；同步完成最近邻文献对照、作者核验和IEEE格式。实车尚未执行，不把仿真触地或等效力结果写成硬件验证。

当前方法不包含已证明的UCEDT牛顿阈值。UCEDT可作为平衡潜力的物理解释候选；若在正文使用其公式，需要另行核验原文、适用条件与当前闭环差异。本版容扰依据为实际闭环不触地标签，避免将解析势能量直接当真实外力安全界。

## 6 结论

本文实现了面向低速单轨机器人、以冻结偏好控制行为为基础的容扰—任务区间规划原型。共同扰动下存在固定偏好的触地差异；在明确的后续反馈策略条件下，小型预测器可以生成依赖状态与阈值的候选区间，并支持新时刻及相关不同低速弯道中的任务完成。现有证据支持有限域可行性及继续面向TCST开展验证，不证明连续区间保证、独立泛化、概率校准或实车性能。

## 证据与流程说明

E250–E276为本版当前证据，路径与SHA由本地builder解析；每项标记AI artifact_checked，人类核验未记录。旧E001–E249保留在历史主稿与证据明细，不能与当前物理/策略混算分母。本文仅在本地整理与导出；未上传未发表稿件、未投稿。作者、基金、利益冲突、数据公开许可与最终AI披露尚待人类确认，不编造声明。

[R1] IEEE Control Systems Society. Transactions on Control Systems Technology: Aim & Scope. https://www.ieeecss.org/publication/transactions-control-systems-technology （访问2026-10-29）。

[R2] IEEE Control Systems Society. Author Information: Transactions on Control Systems Technology. https://www.ieeecss.org/publication/transactions-control-systems-technology/author-info （访问2026-10-29）。

[R3] Huo, B., Yu, L., Liu, Y., and Chen, Z. Hierarchical residual reinforcement learning based path tracking control method for unmanned bicycle. Robotics and Autonomous Systems, 190, 104996, 2025. https://doi.org/10.1016/j.robot.2025.104996 （出版方检索摘要及摘录核对2026-10-29，全文直接访问受限）。

[R4] Persson, N., Zhao, F., Kaheni, M., Dörfler, F., and Papadopoulos, A. V. An Adaptive Data-Enabled Policy Optimization Approach for Autonomous Bicycle Control. arXiv:2502.13676, 2025. https://arxiv.org/abs/2502.13676 （作者预印本摘要核对2026-10-29；此处引用该版本，不混用未核验期刊卷页）。

[R5] Coordinated Trajectory Tracking and Self-Balancing Control for Unmanned Bicycle Robot Against Disturbances. Actuators, 15(1), 49, 2026. https://doi.org/10.3390/act15010049 （出版方摘要核对2026-10-29；作者列表及全文待核验，内部草稿暂存条目）。

[R6] Garone, E., Di Cairano, S., and Kolmanovsky, I. V. Reference and Command Governors for Systems with Constraints: A Survey on Theory and Applications. MERL TR2016-102, 2016; published in Automatica, 2017. https://merl.com/publications/docs/TR2016-102.pdf ; https://doi.org/10.1016/j.automatica.2016.08.013 （作者机构全文的摘要/引言及出版方元数据核对2026-10-29）。

[R7] Wabersich, K. P., and Zeilinger, M. N. A predictive safety filter for learning-based control of constrained nonlinear dynamical systems. arXiv:1812.05506, revised2021. https://arxiv.org/abs/1812.05506 （作者预印本v4摘要与元数据核对2026-10-29；此处不声称已复核其全部定理）。

[R8] Thananjeyan, B., Balakrishna, A., Nair, S., Luo, M., Srinivasan, K., Hwang, M., Gonzalez, J. E., Ibarz, J., Finn, C., and Goldberg, K. Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones. arXiv:2010.15920, revised2021. https://arxiv.org/abs/2010.15920 （作者预印本v2摘要与元数据核对2026-10-29，页面注明RA-L/ICRA2021）。

流程软件：Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. 2026. https://doi.org/10.48550/arXiv.2609.00065 （当前arXiv v2元数据核对2026-10-29）。scientific-writing用于本地证据组织与表述检查，不是控制方法或实验性能依据。
