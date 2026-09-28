# STTW_CONTROL：动态遥控偏好控制 V1 — Codex 执行规格

版本：2026-09-28 / V1.0
基准仓库：QaQaaa-zzz/STTW_CONTROL
已核对分支：experiment/fixed-endpoints-20260928
已核对提交：b446abae131f7a3698c884f1872f8049e9ba3c6e
配套配置：STTW_teleop_preference_governor_v1.json

本文是实施任务，不是已验证的控制结果。参数是明确的第一轮设计初值；验收失败时按本文停止和诊断，禁止声称必然成功、无限时域安全、实时可部署或全局最优。

## 0. 本轮唯一方法及交付范围

实现“**完整闭环模型预测 + 有限候选搜索 + 词典序偏好选择 + 显式航向恢复**”控制器。

本轮不训练 PPO、SAC 或其他强化学习；不训练神经网络；不加载旧策略参与控制；不另写一个需要辨识的神经动力学模型；不引入全局路径规划。

保留 ECBC＋ESO 与现有双通道残差执行接口。参考修正产生的底层控制差值仍必须限制在旧的残差权限内。不能绕过旧权限，直接用修改后的 ECBC 输出替换全部控制量。

第一阶段的控制对象和预测器都使用仓库现有完整 MuJoCo/MJX 模型。目标是先证明：在同一个受限闭环中，两种偏好确实能产生不同但有效的控制结果，并在冲突后恢复航向。这个阶段使用完整仿真状态，是模型已知、状态可得的仿真验证，不是实机状态估计方案。不要把它写成 sim-to-real 已完成。

本轮交付代码、配置、测试、固定场景结果、失败分类、计算成本和可复现报告。神经网络蒸馏、学习残差、简化预测器、实机部署留到后续独立任务，不能自动启动。

## 1. 任务定义与不得改变的语义

### 1.1 外部指令

用户操纵两个量：前向速度请求 v_c（m/s）和前轮转向角请求 delta_c（rad）。

不使用“速度＋偏航角速度”作为新任务的原始输入。omega_c 只能由 v_c 与 delta_c 派生，不能独立重新采样。本文使用的是目标速度型油门，不是电机转矩型油门。

alpha 在每条回合内固定，只允许 0 或 1。一个确定性控制器接受 alpha 参数；不需要两个神经网络。alpha 不得影响物理模型、残差权限、安全约束、候选采样密度或传感器质量。

### 1.2 两个端点

CONFLICT 状态下：

- alpha=0：首先最小化实际前轮转角对原始转向指令的误差，再最小化实际前向速度误差。可以减速，但不能奖励减速本身。
- alpha=1：首先最小化实际前向速度误差，再最小化实际转向误差。可减小转向，但不能奖励少转本身。

当前原始指令安全可行且没有待恢复航向误差时，零残差通过；不为了制造 alpha 差异而故意牺牲任何目标。

### 1.3 恢复任务

参考航向来自原始限幅限速后的指令积分。冲突解除并具备预测余量后，允许暂时修正当前转向，恢复此前积累的航向误差。

恢复阶段采用两端相同的规则，不再坚持“转向必须始终等于原始遥控角”。这是必要的：若少转之后一直严格执行 delta_c=0，则历史少转角不会自动消失。

V1 只对“航向恢复”作通过/不通过判定。记录参考 XY、实际 XY、横向偏差和纵向落后，但不奖励/惩罚位置误差，不强迫追赶时间位置，不宣称完整路径已经恢复。横向位置恢复不是本轮默认功能。

## 2. 仓库与权限保护

先读取根 AGENTS.md、PROJECT.md、learning 相关说明和当前局部 instructions。记录 git status、HEAD、当前分支和运行任务；不得停止现有训练，不得覆盖已有 runs、配置或 checkpoint。

从上述基准提交新建独立分支 experiment/teleop-pref-governor-v1，独立输出目录 runs/teleop_pref_governor_v1/<UTC时间戳>。当前工作区有未提交改动时，不执行 reset/clean/stash 覆盖；使用独立 worktree。若指定 SHA 不存在或必要依赖不可用，输出阻断原因，不从错误默认分支继续。

保留旧实验、旧 reward、旧 CLI 与 checkpoint 读取语义。新任务使用独立 config 类型/版本号，不把旧配置重新解释成新任务。

本轮允许本地提交独立分支；不自动推送，不修改默认分支，不 force push。

冻结以下内容：

| 项目 | 规则 |
|---|---|
| XML、质量、惯量、接触、摩擦、求解器、模型物理步长 | 从基准现有模型读取，不能为通过测试修改 |
| 底层控制周期 | 0.005 s，即 200 Hz |
| ECBC 增益、模型系数、ESO 参数与原更新公式 | 不调参，不改变符号和更新顺序 |
| ECBC 内部转向角速度限幅 | 保留 controller.py 中原值，不与执行器最终限幅混为一谈 |
| 执行器最终前轮角速度限幅 | 现有配置 3 rad/s |
| 执行器最终后轮轴角速度限幅 | 现有配置 60 rad/s |
| 转角机械边界 | 现有配置 ±0.8 rad；以 XML 兼容检查为准 |
| 双通道残差权限 | 前轮角速度 ±1.5 rad/s，后轮轴角速度 ±10 rad/s |
| 残差组合 | additive，strength=1，base_output_scale=1，project_base=false |
| 物理失败判据 | 保留侧倾 0.7 rad、非轮部件触地、非有限状态等原判据 |
| 旧策略 tanh 与动作映射 | 原封保留；本轮不调用神经策略，不对解析残差再套 tanh |

唯一明确取消的旧任务设置：新任务中 controller.fixed_roll_reference=null、learning_roll_reference=null。新任务不调用旧 tracking_reward，因此不再包含固定 6.84° 惩罚/最终门槛、路径主奖励、离带倒计时或自动容忍带收紧。历史配置文件不改。

如果运行时读取到的冻结参数与表格不同，先报告差异并停止功能实验，不默默改物理去匹配本文。

## 3. 两个模型，各自负责什么

### 3.1 虚拟参考：运动学自行车模型

仅用于定义用户意图：

kappa(delta) = cos(lambda) * tan(delta) / L
omega_c = v_c * kappa(delta_c)

L、lambda 来自 ControllerConfig；基准代码为 L=0.408 m、lambda=25°。轮半径代理 R=0.1 m 沿用当前接口。不得同时在几何转换和测量接口重复翻转符号。

用每个 5 ms 区间开始时的 (v_c, delta_c) 推进虚拟位姿。采用区间内恒定 v、omega 的精确积分：

- theta = omega_c*dt；
- p_next = p + v_c*dt*sinc(theta/2)*[cos(psi+theta/2), sin(psi+theta/2)]；
- psi_next = psi + theta；
- 此处 sinc(z)=sin(z)/z，z=0 时取 1；不能误用未换算的 numpy.sinc。

psi_c 保持连续、未模 2pi 的累计值。实际 psi 也由逐步 wrap 后的增量累计得到 psi_unwrapped。控制误差 e_psi=psi_c_unwrapped-psi_unwrapped。保留 sin/cos 诊断，但不能仅用模 2pi 的误差让转完整一圈被记作恢复。

初始化只在任务开始时将虚拟位姿设为共同实际初态。任务中不能重新锚定、平移、重置参考以消除误差。

### 3.2 预测器：完整受限闭环模型

使用同一份已加载 MuJoCo/MJX model、同一 solver、同一物理 substeps、同一 ECBC＋ESO、同一残差限幅和执行器伺服/力矩限制。默认 plant=mjx、predictor=mjx、impl='jax'，复用仓库现有兼容实现；不升级依赖，不改成 Warp。

预测模型不是单纯 kinematic bicycle，也不是只模拟 roll、steer 的简化 ODE。运动学只能生成参考，不能用 v²*kappa 小于某个阈值来代替摔倒/瞬态预测。

预测状态至少包括完整 mjx.Data 或 CPU integration state，ECBC/ESO state，执行器 pending/previous，控制 tick/ESO 启用状态，当前执行参考、原始指令、虚拟位姿、实际未展开航向、恢复状态。CPU 完整复制现有 mjData；MJX 使用不可变结构复制。不能只复制 qpos/qvel 而丢掉 warmstart、actuator state 或 ESO。

第一阶段 StateProvider 明确返回 observation_mode='oracle_full_state'。上层速度用世界平面速度在车身前向上的投影；同时记录 wheel_speed_proxy 与 slip_proxy。ECBC 内部继续使用现有 rear_rate*0.1，不擅自更换其测速输入。后续传感器/观测器问题不能通过给实机虚构完整仿真状态规避。

预测未来外部指令采用当前 (v_c,delta_c) 零阶保持；不读取 command_schedule 的未来行，不读取下一随机目标，不读取未来扰动事件。第一阶段外扰明确关闭。已知当前外部命令变化率也不外推，保证预见条件简单一致。

## 4. 最关键的执行实现：参考修正不能扩大残差权限

定义 c=[v_c,delta_c]，候选目标 z=[v_goal,delta_goal]，速率限制后的执行参考 g=[v_g,delta_g]。

从同一个“测量后、尚未更新本周期控制器”的状态 x_k 分别计算：

u_nom = ECBC_with_rear_reference(x_k, v_c, delta_c)
u_goal = ECBC_with_rear_reference(x_k, v_g, delta_g)

新类型优先用命名字段，避免数组顺序混淆。外部指令类型是(v,delta)，但旧_prepare的command_override顺序是[delta,v]，适配器必须显式交换。u 的通道顺序严格为 [front_steer_rate, rear_shaft_rate]，单位均为 rad/s。其第二项按当前实现为 v_reference/R；不是线速度本身。

解析残差：

a = clip((u_goal-u_nom)/[1.5,10.0], -1, 1)
u_prelimit = u_nom + [1.5,10.0]*a

随后原样调用 apply_residual，包括最终限幅、位置边界、延迟和现有命令变化率约束。MuJoCo 的实际轮轴 ctrl 符号按已有 _step 映射保留。

“候选目标被限幅后没有完全实现”是正常现象：预测和评分必须使用最终物理输出及实际运动，不能仍按 u_goal 认定已经实现目标。记录 requested_residual、applied_residual、residual_clipped 与 final_command_clipped。

绕开上述限幅直接执行 u_goal 是禁止的；这会把性能变化与扩大权限混为一谈。

### 4.1 ECBC / ESO 更新次数

controller_step 当前既返回控制输出又更新内部状态。必须为 nominal/goal 两个候选传入相同的旧 state，分别在局部变量计算，不能把 nominal 更新后的 state 再传给 goal。

当前实现中 ESO/gain 状态更新不依赖转向参考，需写单元测试确认相同测量下两个 next_controller_state 相同。真实系统只提交一次 next_controller_state；候选 rollout 只更新各自的副本。

如果重构后这项不再相同，立即阻断，不用平均或任选状态蒙混。

### 4.2 统一时序

alpha必须严格验证为有限值0或1，其他值直接配置/输入错误，不默认截断为端点。新任务采用如下唯一时序：

1. 得到 x_k、c_k 和上一周期存储的 g_k；若到 governor tick，先在 x_k 副本上搜索；
2. 根据选定 z，将 g_k 限速更新为本区间执行参考；
3. 从同一旧 controller state 计算 nominal/goal，提交一次 ESO 更新；
4. 组合与限幅动作，推进恰好 dt=0.005 s 的物理过程；
5. 用本区间开始时的 c_k 积分虚拟位姿，更新实际未展开航向和误差；
6. 记录本次 transition，再推进外部命令发生器至 c_{k+1}。

不能同时保留旧 _prepare() 的隐式预计算更新，又再运行新 kernel 更新。不要让候选搜索调用真实环境的 _advance() 或推进真实回合 tick。

## 5. 固定初始参数及理由

| 参数 | V1 值 | 选择理由和限制 |
|---|---:|---|
| 底层周期 | 0.005 s | 保持既有 ECBC/ESO 离散实现 |
| governor 周期 | 0.05 s，10个底层tick | 与200Hz底层解耦；目标20Hz，实时性必须实测 |
| 性能预测段 H | 0.8 s，160 tick | 观察指令调整、侧倾和驱动响应，不只看一步；是否足够由首轮回放/失败诊断决定 |
| 延长检查段 H_tail | 0.4 s，80 tick | 同一目标继续保持，检查临近窗口末的失稳；不是不变集证明 |
| 候选总预测时长 | 1.2 s，240 tick | 只能宣称在该检查窗口内通过预测 |
| 用户速度范围 | [2.0,2.6] m/s | 先保留近期速度范围，不扩展低速任务 |
| 用户转向范围 | [-0.30,0.30] rad | 左右对称，并包含明显冲突候选 |
| 用户速度变化率 | 0.5 m/s² | 延续原速度slew量级 |
| 用户转向角变化率 | 0.30 rad/s | 约1秒由直行变到最大转向；是新的转角slew，不冒充旧yaw-slew |
| governor目标转向范围 | [-0.35,0.35] rad | 给恢复保留小量额外转向空间，不改变机械边界 |
| governor目标最低速度 | 1.5 m/s | 避免把停车/极低速当作解；该下限必须由模型验证 |
| governor速度目标变化率 | 1.0 m/s² | 比用户slew快一倍，允许必要减速；并非真实加速度保证 |
| governor转角目标变化率 | 0.60 rad/s | 比用户slew快一倍，仍远低于执行器最终角速度上限 |
| 工作侧倾限值 | 0.30 rad | 继承已有工作范围，不等于物理摔倒阈值 |
| 工作侧倾角速度限值 | 1.5 rad/s | 首轮瞬态约束，用于拒绝高速倾倒趋势；需按结果审查 |
| 终端侧倾角速度限值 | 0.30 rad/s | 要求末端已有收敛趋势，不只窗口内未摔 |
| 终端平衡偏差 | 0.08 rad | 允许模型/瞬态误差，排除显著离平衡末态 |
| 恢复余量侧倾界 | 0.24 rad | 比工作上限留0.06 rad余量 |
| 恢复余量角速度界 | 0.8 rad/s | 恢复动作不在强烈倾倒瞬态加入 |
| 恢复可行保持 | 0.20 s | 连续4个governor tick确认，减少模式抖动 |
| 恢复启用渐变 | 0.30 s | 6个governor tick由0到1 |
| 航向恢复增益 | 0.8 s^-1 | 线性近似时间常数1.25秒，只是初始响应尺度 |
| 恢复附加yaw-rate限幅 | 0.4 rad/s | 避免大航向误差产生极端纠偏 |
| 恢复附加转向限幅 | 0.10 rad | 限制单次恢复意图的干预量 |
| 恢复允许欠速目标 | 0.10 m/s | 恢复阶段不重新引入大幅降速 |
| 航向恢复激活阈值 | 0.08 rad | 小偏差不频繁启用恢复 |
| 航向恢复退出阈值 | 0.03 rad，保持0.5 s | 与激活阈值形成滞回 |
| 词典序转向松弛 | 0.005 rad RMS | 约0.29°内视为主目标近似等价，再改善次目标 |
| 词典序速度松弛 | 0.02 m/s RMS | 避免为极小速度改进付出巨大次目标损失 |
| 初始准备 | 3.5 s，v=2.3、delta=0、零残差 | 完整闭环落稳并让原ESO启用时序完成 |

这些数值不得被描述成已经辨识的最优参数。不要为了验收将0.30工作界自动放大到0.7，或把不足的轨迹标为“参数还可调，所以通过”。

## 6. 候选搜索：确定性粗到细，不使用随机优化器

### 6.1 通用规则

每个候选是一个二维恒定目标 z，不是1.2秒的任意动作序列。底层参考按第5节slew接近 z。真实执行仅使用下一个0.05秒，再重算。

这是有限候选、单目标保持的滚动参考修正器，不是全局最优NMPC。找不到可行候选，只能写“本候选族未找到”，不能写“整个物理系统无可行解”。

### 6.2 CONFLICT 候选域

v_goal ∈ [max(1.5, v_c-1.0), v_c]
delta_goal ∈ [-0.35,0.35]

所有alpha使用同一候选域。保留负/反向转向候选，允许平衡所需瞬态；不要硬限制为与用户转向同号。上界v_c约束目标不主动超速，但真实瞬态超速仍需记录。

### 6.3 RECOVER 候选域

v_goal ∈ [max(1.5,v_c-0.10),v_c]
delta_goal ∈ [max(-0.35,delta_c-0.10*g), min(0.35,delta_c+0.10*g)]

g为恢复启用程度，初始0，每个governor tick增加0.05/0.30，最大1。原始指令重新不可行时，g立即归零；CONFLICT候选的执行参考仍按各自限速。转入已预测安全的bypass时按第9节例外处理，不瞬间改物理状态。

### 6.4 粗网格

9个等距速度点×15个等距转向点，共135个。外加9槽，固定总形状144：

1. 原始ECBC零残差直通候选（special bypass）；
2. 上次目标；
3. 当前执行参考；
4. (v_c,delta_c)；
5. (v_low,delta_c)；
6. (v_c,delta_rec)；
7. (v_low,delta_rec)；
8. (v_c,0)；
9. (v_low,0)。

除bypass外按当前域裁剪。允许重复但保持固定索引；无效槽masked，不能拿无效槽的零分参与选择。bypass的残差严格为0，g参考在副本中跟随原始c；不要误把它再过governor目标slew而失去基线身份。

### 6.5 细化

从粗网格共同结果中确定三个与alpha无关的中心：E_delta最小可行候选、E_v最小可行候选，以及RECOVER时C_psi最小可行候选/CONFLICT时平滑代价最小可行候选。重复中心不扩大预算。

每个中心使用5×5局部网格，偏移分别为[-0.5,-0.25,0,0.25,0.5]乘粗网格的速度/转角间距。至多75点，padding到80。裁剪候选域后预测，最终从144+80全部有效结果中选择。

粗网格没有可行候选时，三个中心改为共同安全违反代价最小的前三个有限候选，做同一轮细化；不能立刻宣称无解。每次solve最多一轮细化，不能无限增加候选。

候选局部状态按jax.vmap并行，时间按jax.lax.scan推进，jit固定形状。只保存每个候选的累计统计及终端状态；常规运行不把所有完整候选轨迹写盘。按需要仅保存获选、原始、最好转向、最好速度的诊断轨迹。

## 7. 可行性与安全边界

候选在完整1.2秒内所有物理substep/可观察控制tick满足：无非有限值、无原始物理失败、|phi|≤0.30、|phi_dot|≤1.5，执行器最终命令符合冻结界。若物理引擎有substep，触地/摔倒至少沿用原有逐substep检查。

预测终端 additionally 满足：|phi_dot|≤0.30；|phi-phi_eq|≤0.08。

phi_eq 用 controller._system 的当前速度ratio关系与预测末端实际转角计算 phi_eq=delta_actual/ratio(v_wheel_proxy)，不是固定6.84°，也不是用未实现的候选转向角代替实际转角。使用与ECBC相同的速度floor和符号。

安全筛选不加入实际速度必须始终等于当前速度命令的硬约束，也不加入固定±0.05 m/s硬超速带，以免把命令突变时不可避免的瞬态当作物理不可行。报告超速峰值、暴露时长、积分。

所有约束对alpha相同。0.30是操作范围，0.7才是继承的侧倾物理失败界。基线超0.30但没倒时，应称“违反操作范围”，不是“基线摔倒”。

### 7.1 无可行候选

保留所有候选的失败原因。若全部非有限，终止回合，标记numerical_failure。

若存在有限候选但无可行候选，进入EMERGENCY，按以下共同违反代价选择有限候选：先最小化预测真实物理失败指示，再最小化

V = max_t(max((|phi|-0.30)/0.06,0), max((|phi_dot|-1.5)/1.5,0))
    + mean_t[max((|phi|-0.30)/0.06,0)^2 + max((|phi_dot|-1.5)/1.5,0)^2]
    + max((|phi_dot_T|-0.30)/0.30,0)^2
    + max((|phi_T-phi_eq_T|-0.08)/0.08,0)^2.

再用平滑代价、候选索引打破平局。此处不使用alpha。

EMERGENCY选到动作不等于安全保证。记录fallback_used=true，该回合不得计入完整任务通过。可继续仿真收集失败证据，物理失败时终止。不把“急刹并回正”固定当成总是安全的备份。

## 8. 偏好选择的数学定义

性能指标只计算前H=0.8秒，均基于预测的实际物理量。未来原始命令按当前值保持。

E_delta = sqrt(mean((delta_actual-delta_c)^2))，单位rad。
E_v = sqrt(mean((v_forward-v_c)^2))，单位m/s。

CONFLICT中：

- alpha=0：取E_delta最小值m，保留E_delta≤m+0.005；在保留集合内求E_v最小值n，保留E_v≤n+0.02；最后最小化共同平滑代价。
- alpha=1：取E_v最小值m，保留E_v≤m+0.02；再求E_delta最小值n，保留E_delta≤n+0.005；最后最小化共同平滑代价。

不要换成10:1、100:1加权和。主目标在有限候选集合与给定RMS松弛内优先；不宣称闭环跨回合全局单调。

平滑代价定义：

C_smooth = 0.02*mean(||(u_final[i]-u_final[i-1])/[3,60]||²)
         + 0.001*mean(||(u_prelimit[i]-u_nom[i])/[1.5,10]||²).

第一步的u_final[i-1]来自真实上一执行器状态。所有min/排序只作用于有限且通过对应筛选的候选；浮点比较按配置显式松弛，不另加未声明容忍。末级仍平局时选候选固定索引最小者。不能让数组不稳定排序改变可复现性。

RECOVER中，两种alpha使用同一选择规则：先保留E_v≤min(E_v)+0.02的可行候选，再最小化

C_psi = mean((e_psi/0.10)^2) + 2*(e_psi_at_H/0.10)^2。

在C_psi≤min(C_psi)+0.001的集合中最小化C_smooth。此阶段允许的转向偏离已由RECOVER候选域约束。不再把原始delta跟踪列为第一优先级，否则无法补回航向。

原始bypass参与比较。如果没有可行恢复候选比bypass的C_psi降低至少1e-4，则本次执行bypass并标记recovery_blocked=true；不人为提高恢复增益。

## 9. 模式判定和航向恢复

每个governor tick先预测原始ECBC零残差候选。这一步与alpha无关。

raw_feasible：通过第7节工作和终端条件。
raw_reserve：raw_feasible，且全窗口|phi|≤0.24、|phi_dot|≤0.8。每个governor tick：raw_reserve为真则reserve_ticks加1，否则清零；连续4次才达到0.20秒。

模式优先顺序：

1. raw_feasible=false：CONFLICT，清空reserve连续计数、g=0，搜索偏好候选。
2. raw_feasible=true但raw_reserve连续不足0.20秒：TRACK，执行原始零残差；保存航向欠账，不启动恢复。
3. raw_reserve连续≥0.20秒且|e_psi|>0.08，或已经在RECOVER且尚未满足退出条件：RECOVER。
4. 其他情况：TRACK，原始零残差。

第2条中若本周期已在RECOVER但余量丢失，也退出至TRACK、g归零，执行已预测的bypass。bypass是明确的例外：残差立即为0，内部虚拟执行参考对齐当前原始指令；不修改真实物理状态。这种切换必须包含在raw预测内，最终执行器限制始终有效。非bypass候选才使用governor参考slew。不存在用“当前误差大”作为raw不可行的替代判断。

恢复退出：|e_psi|≤0.03、|v_forward-v_c|≤0.10、|delta_actual-delta_c|≤0.03连续保持0.5秒。然后g=0，转TRACK。存在剩余小误差不清零参考。

恢复候选注入点：

r_extra = clip(0.8*clip(e_psi,-pi/2,pi/2), -0.4,0.4)
r_des = omega_c+r_extra
delta_rec_raw = atan(L*r_des/(cos(lambda)*max(v_forward,1.5)))
delta_rec = delta_c + clip(delta_rec_raw-delta_c,-0.10*g,0.10*g)
最后裁至[-0.35,0.35]。

这里只生成搜索候选，不直接保证安全。积分欠账保持原值；控制用clip避免巨大角度产生极端命令，日志不能clip。未展开航向误差超过pi时另外标记large_heading_debt=true；绕整圈不能以wrapped误差小而通过。

不设置“离带3秒后强制恢复原速度”。8秒恢复时限仅用于第13节固定实验验收，不进入控制代价或模式触发。

## 10. 需要新增/调整的代码框架

优先新增独立模块，最小范围抽取共享物理函数。建议文件名固定如下；若仓库已有同功能模块则复用，并在manifest写明映射，不建立第二套相互矛盾的实现。

| 文件 | 必须实现的职责 |
|---|---|
| learning/src/sttw_control/teleop_commands.py | 原始(v,delta)指令、幅值/变化率限制、因果输入接口、固定/随机schedule |
| learning/src/sttw_control/teleop_reference.py | 独立虚拟位姿精确积分、实际yaw展开、航向欠账；不依赖策略 |
| learning/src/sttw_control/governor_state.py | typed dataclass/NamedTuple：Snapshot、GovernorState、Candidate、PredictionSummary、Decision |
| learning/src/sttw_control/closed_loop_kernel.py | 200Hz纯闭环步、同状态nominal/goal、残差权限、单次ESO提交、物理推进 |
| learning/src/sttw_control/closed_loop_predictor.py | 同状态副本rollout、vmap/scan、工作/终端可行性、无未来输入 |
| learning/src/sttw_control/preference_governor.py | 候选网格、细化、词典序、模式、恢复、fallback |
| learning/src/sttw_control/teleop_env.py | 独立新任务环境；不隐式路由到旧TimedRecoveryEnv的任务奖励 |
| learning/src/sttw_control/teleop_metrics.py | 指令跟踪、航向恢复、超速/欠速、权限、真实失败、工作范围违反、时延 |
| learning/cli/teleop_governor_check.py | 契约测试/短回放/模型与符号检查的统一入口 |
| learning/cli/teleop_governor_review.py | 固定面板、配对初态、三方法、阶段gate、budget、日志和报告 |
| learning/configs/teleop_pref_governor_v1.json | 配套json的仓库副本；不改旧配置 |
| learning/tests/test_teleop_*.py | 第12节测试 |

### 对现有文件的改动边界

controller.py：数学公式、增益与ESO不改；最多增加不改变结果的纯preview封装。

actuator.py：组合与限幅不改。解析残差直接使用[-1,1]接口；不让未来Actor重复tanh。

env.py：仅在确有必要时抽出physics-only transition供新旧环境共享；旧_step仍沿原时序调用旧_advance。禁止把新任务塞进旧timed_reference字段导致v/delta被当v/yaw。

observation.py、tracking_reward.py、rsl_training.py：旧模块不改语义。本轮不训练，不能为了新任务重写它们。新日志字段用独立schema；不以旧mean_step_reward选模型。

model.py：复用加载；记录原XML hash一次。保留仓库对MJX disabled actuator的兼容处理，不把不支持的bias伺服重新开启。

### 状态边界

Snapshot仅包括当前完整状态，不携带未来schedule；Controller API参数只提供当前c、alpha与snapshot。未来schedule留在外部测试器。这一隔离必须由单元测试覆盖，不仅靠注释。

Decision至少返回goal、normalized_residual、mode、raw_feasible、raw_reserve、selected_feasible、fallback_used、recovery_blocked、cost_summary、candidate_counts、predictor_ticks、latency_ms。

GovernorState至少含current_reference、last_goal、reserve_ticks、recovery_gain、recovery_hold_ticks、mode。候选副本中的这些状态不得污染真实对象。

## 11. 初始化、比较基线和预算

### 11.1 初始化

新kernel的初始controller必须处在本周期尚未更新的时刻；不要把RecoveryEnv.reset中已调用_prepare更新过的controller直接当作同一时刻的旧state再次更新。可复用物理初始化代码，再显式初始化一次controller/actuator。建立同一个完整准备状态：v=2.3、delta=0、零残差、动态ECBC，运行3.5秒。沿用原ESO启用规则，不能任务归零时顺手重置ESO。任务时钟可以重置，但物理/ESO/执行器时间信息必须独立保存。

准备末端要求：无物理失败；|phi|≤0.05、|phi_dot|≤0.10、|v_forward-2.3|≤0.10、|delta|≤0.03。失败就阻断本轮，不自动延长准备或把初态拼接成理想值。

每个场景B0/G0/G1从同一个完整准备快照开始，输入同一已经冻结的原始命令流。不通过重新独立reset伪装“同初态”。

### 11.2 三个主要方法

B0：动态ECBC＋ESO，零残差。
G0：同一个动态ECBC＋ESO＋本governor，alpha=0。
G1：同上，alpha=1。

固定6.84°的旧基线只作历史对照，不代替B0。无神经网络、无训练种子；随机命令seed不能称为训练seed。

### 11.3 本轮预算

训练更新=0，训练转移=0。

主要闭环面板最多42回合（14场景×3方法），每回合任务16秒，加共享准备状态3.5秒。允许另4回合航向恢复消融（±0.25两个场景、两个alpha、disable_recovery=true），总闭环回合上限46。

独立预测回放契约最多128条×1.2秒；网格单元测试使用合成模型，不消耗真实预测预算。

全任务预测累计硬预算80,000,000个control-tick-equivalents，包含初始化检查、真实模型rollout、padding计算和性能计时重复。一个candidate推进一次0.005秒闭环计1；若有物理substeps另记录physics_steps。闭环执行与候选预测分别计数。

每solve粗144+细80最多224候选，每候选240控制tick，即上限53,760候选控制tick；加raw检查最多再240。超预算前停止新增solve并生成budget_exhausted报告。此时不得把未完成面板标complete，不缩短H/减少候选隐藏预算问题，不自动开启新预算。先跑第13节的gate顺序，确保最有价值的证据优先得到。

这是有界诊断预算，不承诺所有场景都一定在该预算内完成。正常TRACK可跳过全网格，只做raw预测，节约计算。

## 12. 先测试，后闭环：必须通过的契约

1. 单位与符号：正delta对应的实际yaw方向与参考一致；后轮正前向速度与MuJoCo ctrl符号一致；前轮动作不是转角。
2. 输入限速：随机10000步检查|dv_c|≤0.5dt、|ddelta_c|≤0.30dt及范围；target变化不能绕开限速。
3. 虚拟参考：直线、恒定转向、正负对称、omega→0连续、跨pi未展开、sinc实现均通过解析测试。
4. 零修正等价：同一新kernel内goal=raw与bypass零残差完全等价；不得因计算两次ESO产生差异。
5. 单次ESO：一个真实200Hz tick更新一次；144候选预测后真实controller state完全不变。
6. 权限：所有alpha和mode，applied_residual逐通道不超过1.5/10；最终命令不超3/60；端点转角限制仍生效。goal无法实现时记录clip，不能扩大权限。
7. 因果：两份schedule在当前时刻以前一致、未来不同；同一snapshot的Decision必须一致。
8. 预测回放：同引擎、同完整初态、同恒定未来指令、同候选，一步和1.2秒rollout与执行kernel重放一致；各物理量max绝对差≤1e-5。失败先修时序/状态，不改容差掩盖。
9. CPU/MJX：另做短回放记录差异，不要求不同引擎逐位相同，不把差异自动归因于reward。CPU物理独立审计至少从B0左右温和转向记录中各取一个snapshot做1.2秒短回放，计入上述128条契约回放，不增加完整闭环回合；不能通过修改XML对齐。
10. 偏好排序：合成候选A(Edelta=.01,Ev=.3)、B(.1,.02)、C(.2,.4)，同安全集合alpha0选A、alpha1选B；不安全候选不能因成本低入选。
11. 主目标松弛：验证0.005rad/0.02m/s边界、NaN、重复候选、无有效候选和稳定索引tie-break。
12. 候选公平：同snapshot不同alpha产生完全相同的粗候选、细化中心及安全结果；只有选择次序不同。
13. 恢复：持续冲突不会随时间被强制恢复；余量未满足不能启用；反向债务符号正确；alpha两端恢复规则相同。
14. 无伪恢复：真实车绕完整圈时未展开误差不被清零；虚拟参考不随实际位置重定位。
15. 旧兼容：原仓库测试集不退化；旧配置加载、reward重建、checkpoint身份与旧CLI不被破坏。

## 13. 固定实验、阶段gate与通过标准

所有指令时刻单位秒，目标经过共同限幅/slew后才是c。每回合16秒。初始c=(2.3,0)。表中未写的后续目标保持至结束。

### Gate A：六个非冲突基础场景

N0：全程(2.3,0)。
N1：t=2目标(2.6,0)，t=6目标(2.0,0)，t=10目标(2.3,0)。
N2：t=2目标(2.3,+0.08)，t=6目标(2.3,0)。
N3：N2的负号镜像。
N4：t=2目标(2.3,+0.08)，t=5目标(2.3,-0.08)，t=8目标(2.3,0)。
N5：t=0目标(2.6,0)，t=2目标(2.6,+0.08)，t=6目标(2.0,+0.08)，t=10目标(2.3,0)。

先跑B0六场景。验收各稳定目标段最后0.5秒：速度RMSE≤0.12m/s、转角RMSE≤0.04rad；全程无物理失败、|phi|≤0.30。若不通过，停止后续大面板，只出baseline_not_ready报告；不得训练网络补救未理解的基线问题。

再跑G0/G1。raw_feasible且不恢复时残差应为0；同snapshot决策无alpha差异。温和工况如果存在参考运动学系统偏差，按日志报告，不能重置参考掩盖。

### Gate B：四个有限冲突与恢复场景

C25L：t=0目标(2.6,0)，t=2目标(2.6,+0.25)，t=5目标(2.6,0)。
C25R：C25L负号镜像。
C30L：同C25L，但+0.30。
C30R：C30L负号镜像。

顺序先C25L和C25R，再C30L/C30R；每个场景B0、G0、G1配对。统一冲突评价窗口[3,5)秒；禁止由策略自己选择评价时段。原始限速指令在此已达到目标。

每个被计作偏好成功的场景，同时要求：

- G0/G1无物理失败、无EMERGENCY，实际侧倾不超过0.30+0.002rad；0.002只作数值验收容差，不改预测上界；
- E_delta(G0)≤0.8*E_delta(G1)，且差值≥0.01rad；
- E_v(G1)≤0.8*E_v(G0)，且差值≥0.05m/s；
- 主目标绝对质量：E_delta(G0)≤0.05rad，E_v(G1)≤0.15m/s；
- G0展示真实减速：相对[1.5,2)秒平均速度至少降低0.15m/s，连续持续0.20秒；不是仅比上升后的速度请求低；
- G1在冲突窗口的平均转向幅值比原始目标小至少0.03rad；记录瞬态反向，不按其否决正常平衡动作；
- 恢复以后未展开航向误差、速度/转角误差满足下述标准。

恢复计时起点由外部命令定义：t≥5后，首次|delta_c|≤0.005。不能以governor自己宣布恢复可行为起点拖延计时。

在此起点后8秒内，|e_psi|≤0.05rad、|v-v_c|≤0.10m/s、|delta-delta_c|≤0.03rad连续保持0.5秒。8秒仅验收，不进入控制器代价。纵向落后允许；横向距离如实报告，不宣称其恢复。

若某场景原始B0从未违反工作范围、raw预测也始终可行，则标not_a_conflict，而不是强迫两个alpha分化。该情形不计“偏好通过”，也不宣称模型无解。

任何一对基础C25左右场景不满足时，停止新增大场景；保存现有结果和第15节失败报告。不要重复微调参数直到只剩成功曲线。

### Gate C：四条随机动态命令

使用np.random.Generator(np.random.PCG64(seed))生成固定输入目标序列；seeds=[65001,65002,65003,65004]。生成器只在评估侧持有未来列表。

0–2秒目标(2.3,0)。从t=2开始，每次按固定顺序独立抽v_target~U[2.0,2.6]、delta_target~U[-0.30,0.30]、保持时长U[0.8,1.6]；在t=10截断所有随机段，10–16秒目标(2.3,0)。再共同通过用户slew，保存实际c CSV。

三个方法使用完全相同序列。逐回合报告物理失败、操作界违反、EMERGENCY、随机阶段速度/转向误差和恢复段最终航向。

本轮随机样本少，全部成功也只称该固定面板通过，不称全域保证。随机阶段不要求处处都呈现全程RMSE严格反序；按共同原始指令片段和预测冲突标记提供局部分析，避免把不同占比的普通时段混起来。

### 恢复消融

仅在Gate B通过且预算足够时，C25L/R各跑G0/G1的disable_recovery=true版本，共4条。保持偏好搜索、物理、输入、权限不变，证明最终航向差异来自恢复机制，而不是更换了任务。没有神经网络，不能写“学习提高了恢复能力”。

## 14. 日志与可视化交付

每个真实5ms步保存一次CSV或等价可审计数组：

time、case_id、method、alpha、raw_target_v/delta、limited_v_c/delta_c、omega_c、goal_v/delta、governed_v/delta、actual_forward_speed、wheel_speed_proxy、actual_delta/delta_rate、phi/phi_dot、yaw_wrapped/yaw_unwrapped、reference_yaw_unwrapped、e_psi_unwrapped、实际/参考XY、along/lateral诊断、mode、g、raw_feasible/raw_reserve、fallback、recovery_blocked、u_nom/u_goal/requested_residual/applied_residual/final_ctrl、全部clip标志、physical_failure/end_code。

每个governor tick另保存：144/80候选数、有效/可行数、各否决原因、获选指标、最好速度与最好转向指标、是否细化、预测工作/终端余量、模型步数、同步求解时延。

每场景独立绘制：速度、实际转向、侧倾/角速度、未展开参考/实际航向、航向误差、双通道残差、模式、XY。注明冲突窗口和恢复起点，展示原始命令而非只展示修改后的目标。不要只交累计reward曲线。

生成INDEX.md、REPORT.md、metrics.csv、case_manifest.json、frozen_config.json、source_manifest.json、budget.json。source_manifest只记录代码commit、配置hash、XMLhash和运行依赖版本一次，不反复扫描/哈希大量图片。

REPORT必须区分：接口测试通过、预测模型一致、非冲突基线通过、两端偏好通过、航向恢复通过、随机面板通过、实时性通过。任何一项不通过都不能用“整体complete”掩盖。

## 15. 性能审计与失败分类

### 15.1 实时性

同步计时必须包含候选生成、预测、筛选和结果可用等待；JAX使用block_until_ready。记录模型加载/JIT编译单独耗时，不计为稳定运行单步时延，但不得隐藏。

以首次真实闭环中的已预热决策累计统计P50/P95/P99/max；需要重复benchmark时计入预测预算。记录硬件、设备、batch、版本。

20Hz governor的周期是50ms。p99≤40ms仅作为留余量的工程目标；p99>50ms则明确real_time_ready=false。第一阶段离线验证可以继续到预算上限，但不能用更慢的墙钟执行宣称实时20Hz，不自动改变控制周期或缩短预测。

### 15.2 不通过时输出什么

按以下原因分别标记，不混写为“PPO还没收敛”：

A. baseline_not_ready：取消固定侧倾后原始动态ECBC本身不能完成温和命令。
B. timing_or_state_mismatch：ESO、warmstart、动作延迟或区间时序不一致。
C. limited_authority：候选所需残差常被1.5/10截断，实际命令无法产生需要的动作。
D. no_candidate_found：有限候选族在当前窗口内未找到可行项，不等同全系统无解。
E. horizon_or_terminal_rejection：路径阶段可行但终端条件拒绝；提供原因分布，不自动放松。
F. reference_semantics_or_bias：转向符号、运动学参考与实际运动存在系统差异。
G. priority_not_separated：两端无足够差异；检查该场景是否真冲突与候选前沿，禁止奖励人为拉开动作。
H. recovery_blocked：原始或恢复候选余量不足，或C_psi无改善；保留历史欠账。
I. runtime_or_budget：计算未达实时或预算已耗尽，标记未完成实验。

报告至少保存一个失败snapshot及其原始、转向最好、速度最好、获选候选对照；不要从失败状态重新置零ESO后再给“修复结果”。

首轮不自动修改H、阈值、网格、ECBC或权限。可以在报告中给出一个有证据的下轮单因素建议，但不自行执行新预算。

## 16. CLI和最终执行顺序

新增CLI至少支持 --config、--output、--phase、--predictor-budget、--cases、--methods、--dry-run。phase只允许contracts/baseline/core/random/ablation/all。dry-run不推进真实或预测物理。

推荐入口（由Codex实现，而不是假定仓库已有）：

python learning/cli/teleop_governor_check.py --config learning/configs/teleop_pref_governor_v1.json --output <run>/contracts
python learning/cli/teleop_governor_review.py --config learning/configs/teleop_pref_governor_v1.json --output <run> --phase all --predictor-budget 80000000

--phase all严格按 Gate A→Gate B→Gate C→消融运行，并服从失败停止和预算；不是绕过gate的无条件全跑。若分阶段命令被重复调用，读取同一run的累计budget，不能重置计算账本。

依次完成：仓库审计→新类型/配置→时序与闭环kernel→契约测试→raw基线→预测器→网格/词典序→恢复→固定面板→随机面板→可用预算内消融→结果报告→本地commit。

结束时给出真实完成项、未完成项、测试结果、各gate指标、原始证据路径、已用预算和本地commit；不承诺后台继续，不启动训练。

## 17. 本轮明确不做

不更改摩擦/质量/执行器权限；不固定真实侧倾；不恢复固定6.84°目标；不使用旧几何路径奖励；不按时间追赶虚拟位置；不让alpha影响安全线；不按alpha改变候选域；不读取未来遥控；不为美观删失败；不称有限预测为严格安全证明；不以参考目标差异代替真实运动差异；不训练网络；不把神经网络蒸馏提前混入这一轮。

## 18. 资料与审计依据

源代码路径均相对于基准SHA：
- learning/src/sttw_control/controller.py：ECBC模型、更新顺序、fixed_roll_reference覆盖。
- learning/src/sttw_control/actuator.py：additive、最终限幅、延迟、端点约束。
- learning/src/sttw_control/env.py：_prepare更新ESO，_step的轮轴符号与物理推进，MJX disabled actuator兼容。
- learning/configs/fixed_endpoint_directional_alpha0.json：现行6.84°目标、残差权限和原训练配置。
- learning/src/sttw_control/timed_env.py：旧v/yaw及几何路径任务，本轮不复用其目标语义。

方法依据（仅支持方法类别，不提供本文参数的实验验证）：
- Garone, Di Cairano, Kolmanovsky, “Reference and Command Governors for Systems with Constraints: A Survey on Theory and Applications”, Automatica 75, 306–328. MERL页面：https://www.merl.com/publications/TR2016-102
- Wabersich & Zeilinger, “A predictive safety filter for learning-based control of constrained nonlinear dynamical systems”, Automatica 129, 109597 (2021). https://arxiv.org/abs/1812.05506
- MuJoCo simulation state / warmstarts：https://mujoco.readthedocs.io/en/3.6.0/programming/simulation.html
- MuJoCo MJX JAX batch接口：https://mujoco.readthedocs.io/en/3.3.5/mjx.html

以上完整模型搜索、RMS词典序、恢复状态机和参数是本任务的设计规格，不是对文献已有安全定理的直接复现。实现必须遵守本文的证据边界。

## 19. 配套配置与文档一致性

配套JSON是新schema，不是可以直接塞进旧TaskConfig的旧任务文件；实现严格的新类型读取并拒绝未知字段。单位以字段后缀为准，参数含义以本文公式为准。不要将turn-angle slew读取为yaw-rate slew。若发现JSON与本文不一致，停止并报告，不任意选一方。

Gate B状态初始化与[1.5,2)预转弯速度窗口必须包含t=0目标2.6经过用户限速的真实加速过程，不能直接把actual速度写成2.6。
