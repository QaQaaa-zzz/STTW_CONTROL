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

## 2 System and Task Formulation

### 2.1 Frozen control hierarchy

The middle controller is `STTW_MIDDLE_H75_LOW_SPEED_MIXED_U170`, selected from a continuation of H75 training that includes low-speed tight turns. It supplies preference-conditioned reference corrections at 50 Hz. The lower controller combines ECBC, ESO, and the frozen local300 residual at 200 Hz. Each lower control interval comprises 25 physics steps of 0.0002 s. Front and rear wheel friction coefficients are both 0.8. The tolerance-learning stage changes neither policy weights nor the lower controller. [E259–E261]

The original path, speed request, destination, and user preference define the task. A geometric follower provides nominal references. When departure requires reconnection, a deterministic geometric module checks and updates the connection at 10 Hz while retaining the original world-frame destination. The executed preference enters the frozen middle network; the resulting corrections pass through the existing reference filters and actuator permissions. Thus preference coordination changes reference-generating behavior rather than directly commanding motor torques. [E259,E261]

Let x denote the complete closed-loop state, including the vehicle, actuators, observer, reference filters, and observation history. Let z=H(x,r) denote the observable predictor features for original task r. Experimental branches originate from the same complete x; selecting a different preference does not reset hidden controller state. The simulation uses true vehicle forward speed in this interface. A physical implementation requires an estimator for that quantity in addition to the available attitude, steering, and localization signals. [E259–E261]

### 2.2 Task and disturbance outcomes

The user preference alpha_u remains fixed within each evaluation. The executed preference alpha_e may temporarily differ from alpha_u. Original-task metrics are always evaluated against the original path and destination, even while a reconnecting path supplies executable references. Arrival occurs when the robot enters the original destination disk of radius 1 m before contact or a controller fault. The arrival condition does not impose a terminal stop. [E250,E253–E258]

The disturbance is a horizontal force applied at the body center of mass in a fixed world direction, (-sin(0.8), cos(0.8), 0). The main matched experiments use a 24 N pulse lasting 0.4 s; additional cases use 20 N. Force magnitude and duration define the intervention, and the actual future pulse schedule is not an input to the coordinator. Low-speed outcomes are checked using actual forward speed over the full upright observation window, with an absolute-speed threshold of 2.1 m/s. [E250,E254–E256,E277,E279]

For a source state at task time t_s, the observation ends at task time 10 s or a physical failure. The binary labels are

```text
Y_nf   = 1 if the remaining observation window contains no ground contact;
Y_goal = Y_nf times 1 if the original goal is reached before failure.
```

Engineering faults are retained in evaluation records and excluded from physical-contact supervision. In addition to these outcomes, we record original-path error, actual speed, preference restoration time, and intervention I_alpha = integral |alpha_e(t) - alpha_u| dt. Full-window costs and costs over a common paired observation prefix are reported separately. [E251,E278,E280]

## 3 Preparation-Conditioned Preference Coordination

### 3.1 Predicting a complete candidate execution

A candidate consists of a preparation preference a and a specified subsequent restoration policy pi_r. The complete branch evolves according to

```text
x_(k+1) = F_cl(x_k, alpha_e,k, r, d_k),
alpha_e,k = Pi(a, alpha_u, current feedback, plan age; pi_r).
```

Here F_cl is the implemented closed loop. The learned model predicts outcomes of this execution rather than identifying a separate dynamics model. For hypothetical disturbance d and preparation interval tau, the prediction is

```text
f_psi(z, a, d, tau, alpha_u; pi_r)
    = (q_nf, q_goal, e_path_hat, v_peak_hat).
```

The classification outputs score contact avoidance and completion of the original goal over the remaining task window. The regression outputs summarize original-path RMSE and peak upright speed. Path RMSE is evaluated from the source until first arrival; a surviving nonarriving execution uses the remaining window. Conditioning on pi_r aligns the training label with the actual policy executed after preparation. Holding a preference indefinitely and later returning it define different prediction targets. [E251,E259–E261]

The input contains 34 current observation features, eight candidate/disturbance-query features, and the original user preference, giving 43 features. Its state information includes speed, attitude, reference-filter state, and local path curvature. Candidate features include the preference and associated middle-policy reference corrections. The disturbance query specifies hypothetical force and timing information, independently of the actual future event. [E260]

Three independently initialized 43–16–16–4 networks with ELU hidden activations are trained and their outputs averaged. Each member contains 1,044 parameters. Binary cross entropy trains the two classification heads. The continuous targets use log(1+y), with loss weights 1, 1, 0.25, and 0.25 for the four heads. Failed trajectories do not contribute continuous-target regression, so a short failure cannot be learned as an inexpensive successful task. AdamW uses learning rate 0.001 and weight decay 0.001, for 5,000 supervised updates per member. The fixed final models are used for the reported experiments. [E251,E252,E260,E268]

The training set contains 232 query labels from 188 distinct physical trajectories: 185 held-equivalent queries whose user and executed preferences coincide, 44 derived failure queries that terminate before restoration can start, and three physical feedback-return trajectories. Derived queries reuse physical evidence rather than creating new independent rollouts. The current scores are used for candidate ranking and thresholding; probability calibration is not part of the training objective. [E251,E252]

### 3.2 Candidate selection

Before execution, specify a finite hypothesis set D of possible disturbances and preparation intervals. For each queried preference, take the minimum classification scores and maximum predicted path error across D. The admissible candidate set is

```text
A_ok = {a in A_query:
        min_(d,tau in D) q_nf(a,d,tau) >= eta_nf,
        min_(d,tau in D) q_goal(a,d,tau) >= eta_goal,
        max_(d,tau in D) e_path_hat(a,d,tau) <= e_max}.
```

The reported instance uses eta_nf=0.8, eta_goal=0.5, and e_max=4 m. This path-error threshold is an open-plane task criterion. The five original candidate preferences are 0, 0.25, 0.5, 0.75, and 1; the preparation study additionally queries 0.875. Adjacent qualifying queried points can be displayed as ranges, while execution chooses only an evaluated point. [E254–E258,E260]

If alpha_u is in A_ok and reference publication is permitted, the coordinator retains it. Otherwise, it chooses a queried admissible point minimizing

```text
(a - alpha_u)^2 + 0.1 (a - alpha_previous)^2.
```

This rule prioritizes the original preference while penalizing a change from the previous execution. When the set is empty or reference publication is prohibited, the current implementation holds the previous preference and records that outcome. The task-output heads determine eligibility; the preference-distance objective determines which eligible candidate is selected. [E260,E261]

### 3.3 Feedback restoration

After selection, preparation lasts at least 2 s from the source boundary. Restoration becomes eligible when all of the following hold: absolute roll is at most 15 degrees, absolute roll rate is at most 0.35 rad/s, forward speed is at least 1.5 m/s, and reference publication is permitted. These conditions must hold continuously for 0.3 s before entering the return phase. The return update is

```text
alpha_e,k+1 = alpha_e,k
              + clip(alpha_u - alpha_e,k, -rho*Delta, rho*Delta),
Delta = 0.02 s, rho = 1/s.
```

If the feedback condition becomes invalid during restoration, the preference is held until the condition becomes valid again. A resumed return does not restart the initial dwell requirement. An implementation tolerance of 1e-6 completes the final approach to the user value. Neither the actual gust onset nor its end enters this rule. [E253–E256,E261,E265]

For fixed alpha_u, each permitted return update moves alpha_e toward alpha_u without overshoot. Its increment is bounded by rho*Delta, apart from the stated completion tolerance. The initial preparation selection may jump from the user preference; the return-rate property applies to the restoration phase. Existing reference filters and actuator permissions remain active throughout.

### 3.4 Execution and verification

The reported coordinator performs one neural selection at the declared source boundary. The return rule subsequently runs at the middle-policy update rate, independently of the 10 Hz geometric reconnection clock:

```text
At the source boundary:
  Read current observations, user preference, and reference permission.
  Evaluate every candidate against the fixed hypothetical disturbance set.
  Retain the user preference when eligible; otherwise select a qualified point.
  Start the preparation clock without resetting the vehicle or lower controller.
At each subsequent middle update:
  Maintain preparation until the minimum age is reached.
  Accumulate the initial valid-feedback dwell and enter restoration.
  Move toward the user preference when feedback permits; otherwise hold.
  Continue geometric reconnection on its independent clock.
```

Each prediction package binds the feature schema, model identity, and restoration parameters to the evaluated continuation policy. Results are grouped by complete source state and actual force event. Shared-source branches provide paired interventions. They are not independent random repetitions. Fixed and dynamic baselines use the same vehicle model, frozen control stack, goal, and reconnection mechanism. Original-user costs are independently reconstructed from saved trajectories, and trajectories terminate at their actual physical endpoint. [E251,E261,E275,E278,E280]

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

### 4.9 Nominal low-speed two-turn track segments

![Three preferences on a left-leading S track segment](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/u170_track_segments_nominal_v1/analysis/S_left_60_R1p3_v2p0_zero_fixed_candidate_xy.png)

The nominal task panel is extended to mirrored S segments comprising two opposite 60-degree bends of radius 1.3 m, separated by a 3 m straight. Requested speed is 2 m/s and the original goal lies 4 m beyond the second bend. U170, the lower controller, friction, and geometric reconnection are unchanged. Each direction is evaluated at fixed preferences 0, 0.5, and 1 for 10 s, without an external force. [E281]

| Track segment | Alpha=0 arrival (s) | Alpha=0.5 arrival (s) | Alpha=1 arrival (s) |
|---|---:|---:|---:|
| Left-leading S | 6.685 | 6.635 | 6.510 |
| Right-leading S | 6.720 | 6.645 | 6.530 |

All six trajectories reach the original goal and survive the complete window. Their highest upright actual forward speed is 2.0413 m/s, within the 2.1 m/s low-speed criterion. None requires geometric reconnection. The speed-prioritized preference arrives 0.175 s earlier on the left-leading segment and 0.190 s earlier on its mirror than the path-prioritized preference. These nominal results establish a two-turn task setting for subsequent disturbed evaluation while retaining the same frozen control stack. [E281]

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
