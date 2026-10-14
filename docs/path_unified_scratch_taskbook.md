# STTW 统一偏好上层从零训练：任务执行书 / Implementation Plan

> 给实施 agent：使用 subagent-driven-development 或 executing-plans 按任务实施；先完成代码与必要验证，再在执行会话的训练授权范围内运行明确预算。本文是规划交接，不是已经实施或训练成功的记录。
>
> 日期：2026-10-14，Asia/Shanghai。规划 agent 只写本文，不修改控制/训练源码，不启动、停止或续训任何任务。

**目标：** 从零初始化一个共享的 α 条件 Actor/Critic，用已有加权奖励直接学习速度—路径偏好。先实现每回合固定 α；达到本阶段行为目标后做小规模动态 α 检查。

**架构：** 复用 alpha_modulated_v1，Actor351 / Critic352，128→128→64 ELU，隐藏层2/3的 α gain/shift 调制，每网384个新增参数。部署只有一个 Actor；无教师、蒸馏、模仿、双专家、父 Actor 继承或模型权重平均。

**技术栈：** 现有 Python、PyTorch/RSL DirectCommandPPO、JAX/Flax、MuJoCo；解释器 /home/qy/mujoco_playground/.venv/bin/python。复用现有依赖，不升级库。

## 0. 已核验的代码基线和操作边界

- 基线提交：b0ae7106cbdb78bcd38637341524faa1fb2269b4。
- 交付前工作树HEAD新增7e5fa0927a3a9a8a68e53b45f73324d0285b8fe4，仅更新验证文档；本任务仍固定以上b0ae710源码基线。
- 基线工作树：/home/qy/STTW_CONTROL/runs/worktrees/path-strong-preference。
- 该提交已包含调制网络、Torch/Flax导出、v2奖励、α1热启动、两段课程、偏好选模、动态检查及相关测试。不要从6b541be重新实现这些功能。
- 读取期间，原热启动工程检查已完成：128策略转移、512底层ticks；随后原热启动正式运行已启动，PID记录3811918，读取时为第0轮validation、0训练转移。此为时间快照，执行前必须重新读其status；不要靠PID快照发信号。
- 活动任务：原工作树/runs/path_strong_preference_200_20261014/training/status.json。禁止在其工作树热改代码或冻结配置。本任务不授权停止该运行；已有热启动运行由其执行会话单独处理，不为本任务额外安排热启动对照。
- scratch使用新工作树 /home/qy/STTW_CONTROL/runs/worktrees/path-unified-scratch，建议分支 experiment/path-unified-scratch。已存在时先检查归属和状态，不覆盖或强制重建。
- 正式scratch身份：path_unified_scratch_200_20261014。不得把热启动模型、工程更新或旧run改名为scratch。
- 开工先读根AGENTS、research-hub/AGENTS.md、PROJECT_STATE.md及工作树override。本用户新任务针对scratch；旧override中α1热启动要求只适用于旧运行。执行agent应在新工作树增量记录本任务边界，不修改旧工作树治理文件。
- 仅改STTW_CONTROL。不改DVGC/JIT，不停其他项目进程。
- 保留local300及其SHA、物理模型、原路径、外环、ECBC/ESO、执行器、修正权限、滤波和slew。upper_alpha不得传给冻结的lower_alpha。
- 时序保持物理0.0002s、底层0.005s、上层0.02s；20秒回合、真实失败与终止语义保持原合同。
- 不因放宽本阶段验收而降低reward安全成本、物理失败条件或路径域边界。

## 1. 本阶段验收：比上一版宽松，历史结果另行保留

### 1.1 为什么调整

旧Coverage端点已达到原主任务门槛和最终保持；“旧模型从未达到这些要求”不符合原始数据。旧α1真正不满足的是后来加入的0.40m走廊边界。连续α相邻单调性也是后来加给统一网络的要求。

旧结果按完整原弯段0.5m等权分箱：

| 策略 | 高速弯 | 速度RMSE m/s | 路径RMSE m | 20秒roll峰值 rad | 20秒路径峰值 m |
|---|---|---:|---:|---:|---:|
| Coverage α0 | 左 | 0.268047 | 0.058366 | 0.259799 | 0.091404 |
| Coverage α0 | 右 | 0.203583 | 0.077741 | 0.283519 | 0.179387 |
| Coverage α1 | 左 | 0.026349 | 0.498789 | 0.243791 | 1.089159 |
| Coverage α1 | 右 | 0.021533 | 0.320333 | 0.298563 | 0.610649 |

旧主任务实际使用χ>0窗口：α0左右路径RMSE为0.063075/0.079699，α1速度RMSE为0.029078/0.023813，均通过旧0.10m/0.05m/s门槛。α1六场景末1秒最大绝对误差分别不超过速度0.026133m/s、路径0.014561m、航向0.000753rad。原始入口为 path-feedback-v1/runs/path_coverage_v2_resume_20261011/alpha0/final_best 与 alpha1/final_best，α1 checkpoint SHA前缀a1e9b7f1，α0为f4755e7a。

本次放宽的理由是先完成“一个共享网络具备可用的端点偏好”，不要求首轮同时达到成熟端点、严格连续性与新走廊标准。不得回写历史成功标签。

### 1.2 新研究阶段门槛

新schema：path_preference_research_v1。

| 项目 | 上一版统一网要求 | 本阶段要求 |
|---|---|---|
| 高速α0路径RMSE | ≤0.10m | ≤0.15m |
| 高速α1速度RMSE | ≤0.05m/s | ≤0.10m/s |
| Dv=Ev(α0)−Ev(α1) | ≥0.03m/s | ≥0.02m/s |
| Dy=Ey(α1)−Ey(α0) | ≥0.02m | ≥0.001m；只要求方向超过1mm数值识别带 |
| 最终保持窗口 | 最后1秒 | 最后0.5秒 |
| 保持速度/路径/航向 | ≤0.10m/s、0.10m、0.05rad | ≤0.15m/s、0.20m、0.10rad |
| 相邻α单调性 | 反向变化超过0.01即失败 | 只报告曲线和反向变化，不否决 |
| 0.302rad、0.40m | 历史严格任务门槛 | 并行诊断，不否决本阶段偏好验收 |
| 实际物理/路径域失败、目标完成、数据完整 | 必须满足 | 保持 |

- 主目标及Dv/Dy必须在高速左右弯分别满足，不把左右平均后抵消失败。
- 高速弯全部五个α必须覆盖完整原弯段；不能只比较共同走过的部分。
- 六场景×五α均检查真实物理/路径域失败、目标到达和新的末0.5秒保持。
- 不强迫低速普通场景产生行为分离；高速中间α只检查任务完成和实际响应曲线。
- 通过称为“固定α研究阶段行为合格”，不能改称严格安全认证或全区间单调偏好已证明。
- Dy≥0.001m只是偏好方向可识别的最低推进门槛，不是“偏好性强”的证据。另报告左右弯是否达到原强差值目标Dv≥0.03m/s且Dy≥0.02m，以及全部实际差值；该诊断不阻塞本阶段推进，也不增加仿真。
- 如中间α响应集中在某个小区间，明确报告“端点可调，中间响应不足”，而非隐瞒曲线；本次不因此追加训练。
- 继续输出旧严格1秒保持、.302/.40、旧主目标及上一版严格偏好指标，不删除负结果。

## 2. 新配置与身份（任务A）

**新增配置，禁止覆盖原热启动JSON：**

- learning/configs/STTW_Path_Unified_Scratch.json
- learning/configs/path_unified_scratch_run.json
- learning/configs/path_unified_scratch_engineering.json

以b0ae710的对应Strong配置复制为新实验配置，源码仍原位维护，不复制模块。新plan schema固定为 sttw_path_scratch_preference_training_plan_v1。

新run plan的相关字段：
~~~json
{
  "schema": "sttw_path_scratch_preference_training_plan_v1",
  "policy_mode": "shared_conditional_v1",
  "design_config": "STTW_Path_Unified_Scratch.json",
  "seed": 911401,
  "updates": 200,
  "automatic_extension": false,
  "conditional_extension": false,
  "initialization": {
    "mode": "scratch",
    "parent_alpha": null,
    "checkpoint": null,
    "checkpoint_sha256": null
  },
  "preference_selection_schema": "path_preference_research_v1"
}
~~~

这是在原完整plan中替换上述字段，其余lower/manifest/分布/时序等字段继承并冻结。沿用已有checkpoint字段名，不再新造parent_checkpoint配置别名。

设计配置中：
~~~json
{
  "alpha_conditioning": {
    "mode": "shared_conditional_v1",
    "sampling_mode": "fixed_mixture",
    "seed": 911402,
    "alpha_index": 336,
    "constant_per_episode": true,
    "evaluation_alphas": [0, 0.25, 0.5, 0.75, 1],
    "endpoint_probability_each": 0.4,
    "uniform_probability": 0.2
  },
  "preference_selection_schema": "path_preference_research_v1",
  "preference_acceptance": {
    "spatial_bin_width_m": 0.5,
    "endpoint_path_rmse_m": 0.15,
    "endpoint_speed_rmse_m_s": 0.10,
    "min_speed_gap_m_s": 0.02,
    "min_path_gap_m": 0.001,
    "final_hold_s": 0.5,
    "final_speed_error_m_s": 0.15,
    "final_path_error_m": 0.20,
    "final_heading_error_rad": 0.10,
    "adjacent_monotonicity_role": "diagnostic",
    "adjacent_speed_reverse_m_s": 0.01,
    "adjacent_path_reverse_m": 0.01,
    "roll_limit_role": "diagnostic",
    "corridor_limit_role": "diagnostic"
  }
}
~~~

- network.architecture保留alpha_modulated_v1；network.initialization描述更新为fresh scratch，不能仍写actor_only_zero_alpha_column。
- task_selection_schema保留path_conditional_task_v1，用于历史诊断。
- reward_version保留path_safety_preference_v2、primary_speed_multiplier=2；这是上一轮已选定的候选奖励，尚不称已训练验证的旧奖励。
- training_sampler仍demand_exposure_v2，route_seed=910201。
- dynamic_preference_checks显式enabled=true，trigger='qualified_research_preference'；保留已有rate_per_s=.5、first_change_s=.5、return_s=6、max_lower_ticks=24000。
- 工程plan只改为1 update、engineering_check=true、engineering_dimensions=[8,16]、discard_engineering_update=true；不做DEV或动态评测。
- 身份至少包含初始化mode/seed、null父上层SHA、完整alpha_conditioning SHA、preference_acceptance SHA、网络/奖励/配置/lower/source SHA及实际预算。scratch无父代不代表取消lower身份检查。
- 本阶段阈值固定在配置。未经新声明不得在50/100轮临时放宽，也不得用训练结果反向修改门槛。

验收：新scratch配置可静态加载；scratch携带非null父checkpoint/SHA/parent_alpha应拒绝；未知初始化或采样模式应报错；旧两种warmstart配置校验逐值保持。

## 3. Scratch初始化和固定采样（任务B）

### 3.0 一个网络具体如何形成偏好

所有α的数据更新同一个参数集合θ，不按α切换模型。每次上层决策将状态历史与α拼成351维输入，网络输出二维潜在动作，再经过现有变换、限幅、滤波和slew得到Δv、Δδ。Critic以352维输入估计同一α下的价值；部署时仅需要Actor。

已实现的隐藏层2/3按以下式子计算，不需要再造网络：
~~~text
q = 2*alpha - 1
h1 = ELU(W1*obs + b1)
z2 = W2*h1 + b2
h2 = ELU((1 + q*gain2)*z2 + q*shift2)
z3 = W3*h2 + b3
h3 = ELU((1 + q*gain3)*z3 + q*shift3)
mu = W4*h3 + b4
~~~

gain/shift是共享网络的可训练向量，不是两个端点专家；所有W、b、gain、shift都由混合α的PPO回报共同更新。奖励中α0偏路径、α1偏速度，普通需求区继续兼顾两者；安全和动作成本共用。具体仍调用现有path_command_reward.costs，不另写“加权回报”替换它，也不把不同α的轨迹交换奖励后当作on-policy样本。

“从零”特指上层Actor/Critic，无任何上层父模型；冻结local300仍是已训练底层。初始输出为零但隐藏参数随机，不是整网零初始化。α直接输入加上隐藏调制提供条件表达能力；是否学出强差异由最终同图轨迹和指标回答，结构本身不保证。

### 3.1 需要修改的代码位置（行号基于b0ae710）

learning/src/sttw_control/path_command_conditioning.py：
- is_strong()约14行：它仅表示架构，不能再由它同时推断初始化、奖励、采样或验收schema。
- curriculum_probabilities()约17行：新fixed_mixture返回[0.4,0.4,0.2]；与absolute_update无关。旧配置缺省继续使用原课程/原固定分布，不能改变历史输出。
- validate_conditional_plan()约44行：新增scratch schema分支，单独验证初始化、采样、reward与acceptance；旧schema保留原严格合同。
- initialize_conditional_actor()约74行：保留原warmstart实现；scratch不调用它，不折bias、不清零第336列。
- conditional_identity()约101行：允许scratch父SHA=null；不要访问不存在的curriculum。完整alpha_conditioning SHA已能覆盖固定采样；新scratch不伪造课程SHA。
- complete_resume_validation()：按显式preference_selection_schema重建当前运行的研究选模，不能因为仍是FiLM就固定调用旧严格profile。

learning/src/sttw_control/path_command_training.py：
- load_plan()约137/158行：仅非scratch解析父checkpoint；baseline_parent继续表示B0缓存来源，不是上层预训练父代。
- prepare()约186/207行：scratch跳过上层父SHA检查，保留lower、物理、normalizer、prepared-bank全部检查。预算仍为单learner，不恢复两端×2统计。
- 约386行保护清单：仅非null上层父路径加入清单；其他现有保护保留。
- 约455–458行：首次scratch启动保留make_algorithm()创建的fresh learner，不调用initialize_conditional_actor()。
- 约621行日志：fixed_mixture记录mode、固定概率和实际采样数，不能访问curriculum[0]或写成正在切换阶段。
- resume路径必须先恢复本次scratch checkpoint，不再执行fresh初始化。记录本次run自身Actor/Critic/Adam/log_std/lr/拒绝计数/RNG/counters恢复收据。
- 沿用已有resume语义：learner状态恢复，物理批次按prepared bank重启；不能把它称为完整闭环逐帧接续。

建议给初始化分派增加一个薄函数，避免在多个位置复制规则：
~~~python
def initialize_upper_policy(algo, plan, config, source):
    mode = plan["initialization"]["mode"]
    if mode == "scratch":
        assert not algo.optimizer.state_dict()["state"]
        return {
            "mode": "scratch",
            "parent_checkpoint": None,
            "parent_checkpoint_sha256": None,
            "restored": [],
            "critic": "fresh",
            "optimizer": "fresh",
            "log_std": "configured_fresh",
            "rng": "fresh",
            "update": 0,
        }
    return initialize_conditional_actor(
        algo,
        plan["initialization"]["checkpoint"],
        plan["initialization"]["checkpoint_sha256"],
        expected_lower_sha=source["candidate"]["actor_sha256"],
        config=config,
    )
~~~

该函数仅用于首次启动；resume走现有restore_checkpoint，不能用initialization.mode覆盖resume动作。

### 3.2 网络初始化直接复用，禁止重写网络

make_algorithm()约24–51行已实现：
- 隐藏层随机正交、bias零；
- Actor最后Linear权重和bias零；
- Critic随机初始化；
- std=[.25,.10]；
- FiLM gain/shift零；
- fresh Adam。

保留第336列随机权重。不要把所有Dense置零。初始确定性均值动作=0，在相同完整初态与filter/offset归零条件下回归B0；随机Gaussian探索不等于B0轨迹。

零输出层下第一次反向传播的隐藏层/FiLM梯度可能为零，这是正常的。用非零输出权重或已发生有效输出更新的状态测试FiLM梯度，不要因此把正确scratch初始化判为错误。

### 3.3 α固定分布

沿用sample_episode_alpha()的独立purpose RNG，以及(env_id, episode)计数，不把α绑定到路线难度或左右方向。新概率：
~~~python
p0 = p1 = 0.4
pu = 0.2
alpha = where(draw < p0, 0.0,
              where(draw < p0 + p1, 1.0,
                    (draw - p0 - p1) / pu))
~~~

仍每次reset采样、回合内保存在state.alpha。新方案没有100轮课程切换。保留absolute_update接口兼容旧运行，但新分支不使用它改变概率。

验收：update=1/100/101/200同seed/env_id/episode得到相同新分布结果；旧课程在100/101的原结果保持。新alpha_conditioning内容进入缓存/配置身份。

## 4. 独立研究验收、选模和报告（任务C）

主要位置：path_command_selection.py约308–416行。当前硬编码：
- summarize_preference()约347行：.10/.05主目标；
- 约355行：.03/.02差值；
- 约369–370行：相邻反向变化.01计失败；
- 约379–391行：使用旧goal_hold_failures决定资格和排序。

### 4.1 新接口

新增两个函数，接口合同如下：

| 函数签名 | 输入与返回约定 |
|---|---|
| `resolve_preference_acceptance(config: dict) -> dict` | 根据显式selection schema返回完整的阈值和role字典；旧schema返回旧常数，新schema读取第2节冻结字段。未知schema/字段、非有限或非正数值、非整数tick保持窗口均抛ValueError。 |
| `preference_hold_metrics(trace: dict, config: dict, acceptance: dict) -> dict` | 读取time、actual_forward_speed、v_user、path_cross_track、path_heading_error、physical_failure、domain_exit；返回final_hold、hold_lower_ticks、hold_window_s、final_hold_max_speed_error_m_s、final_hold_max_path_error_m、final_hold_max_heading_error_rad。没有末段数据时max字段为null、final_hold=false。 |

旧schema解析的常数为0.10/0.05主目标、0.03/0.02差值、末1秒0.10/0.10/0.05，以及相邻0.01硬验收；旧返回字段和排序保持兼容。新schema的role仅允许第2节列出的diagnostic，数值配置在本运行中固定；当前实现中的roll/corridor在strong偏好资格中已经是诊断项，保留这一行为。

保持helper必须锚定原20秒终点，而不是失败轨迹的最后0.5秒。建议用整数tick：
~~~python
dt = float(config["timing"]["lower_dt_s"])
horizon = float(config["training_future_phase_C"]["episode_s"])
tick_end = np.rint(np.asarray(trace["time"]) / dt).astype(np.int64) + 1
end_tick = round(horizon / dt)
hold_ticks = round(acceptance["final_hold_s"] / dt)
mask = (tick_end > end_tick - hold_ticks) & (tick_end <= end_tick)
complete_hold = np.array_equal(
    tick_end[mask],
    np.arange(end_tick - hold_ticks + 1, end_tick + 1)
)
~~~
新profile应当有100个真实lower ticks；不足100、早失败或非有限不能通过。

### 4.2 保留旧诊断，另算研究资格

summarize_preference(panels,config)仍调用现有完整性检查和summarize_conditional，但：
- 原qualified_task、goal_hold_failures、working_failures、corridor_failures含义保持旧标准。
- 新增research_goal_hold_failures、preference_direction_failures、incomplete_turn_conditions、qualified_research_preference。
- 每个case输出研究保持、旧保持、完整分箱Ev/Ey、Dv/Dy、相邻反向变化及当前acceptance。
- qualified_preference在新schema下可等于qualified_research_preference以兼容现有消费者，但必须携带selection_schema及qualification_role='fixed_alpha_research_behavior'；它不代表旧严格合格。
- 另输出legacy_strict_preference_qualified，用同一已保存panel按旧严格阈值纯CPU复算，不增加仿真。
- 相邻α反向变化只进入诊断，不进入新failures、score_tuple或normalized_preference_deficit。缺失分箱仍失败，不能借diagnostic放过漏走路段。
- 验收配置不允许改变任何reward参数或stage_B_baseline。

高速左右弯完整0.5m分箱沿用_full_turn_metrics()。直行和低速只作共同任务检查，普通场景不硬要求α分离。

### 4.3 新字典序选模

新preference_score_key：
~~~text
(
  physical_failures + tracking_domain_failures,
  incomplete_turn_conditions,
  research_goal_hold_failures,
  highspeed_endpoint_failures,
  preference_direction_failures,
  worst_endpoint_normalized_error,
  normalized_preference_deficit,
  source_update
)
~~~

- highspeed_endpoint_failures：两侧各检查Ey0/.15和Ev1/.10，共4项。
- preference_direction_failures：两侧各检查Dv≥.02和Dy≥.001，共4项。
- worst_endpoint_normalized_error=max(两侧Ey0/.15, 两侧Ev1/.10)。
- normalized_preference_deficit=sum(max(0,(limit-value)/limit))，仅对两侧Dv、Dy计算，limit分别.02、.001。
- 缺失指标保留null；沿用有限最大rank sentinel，不填零、不造成NaN排序。
- source_update仅最终平局时选较早模型。
- .302/.40、旧严格保持、相邻单调性均不得暗中进入这个排序。
- 指针继续best_preference_model.json，但仅在新的scratch运行目录；保存selection schema、acceptance及acceptance_sha256。
- 原best_conditional_model.json照旧保留。最终scratch主报告和动态入口读取新研究best，不默认读取旧共同best。

### 4.4 跨模块同步，避免“名义放宽、实际仍被旧标准阻挡”

- path_command_dynamic.py:_metrics()约32–70行：复用preference_hold_metrics；新研究profile用末0.5秒。旧严格保持另算保留。新trigger读取qualified_research_preference，不依赖旧qualified_task。
- dynamic_contract、run_dynamic_preference_checks及training尾部：新增enabled/trigger明确分派，不能仅因is_strong()为真就隐式发起评测。保持原六回合、.5/s slew、.5s变化、6s返回、24,000ticks全lineage上限和已完成/部分恢复防重复逻辑。
- path_command_training.py约690–695行：依据preference_selection_schema选择研究best和报告；工程模式不执行动态检查。
- path_command_reporting.py约536–559行：报告显示新门槛、研究资格、旧严格诊断以及单调性仅诊断的含义，删除把continuity仍写为硬验收的旧文案。
- path_command_evaluation.py、complete_resume_validation()：重建分数调用当前schema；panel/source/best身份包含acceptance SHA。
- path_command_cache.py:reuse_conditional_baseline()约297–325行：在现有B0物理身份白名单中排除纯preference_acceptance元数据。完整配置/选模身份仍保留它；Actor、物理、lower、路径、时序等身份不得放宽。
- 改acceptance后可重新计算已有匹配轨迹的研究分数，不可把旧score直接继承为相同资格。新正式训练配置冻结后，恢复时不允许偷偷换acceptance。
- 新scratch的动态检查启动条件是本研究阶段合格，不再要求旧0.40m、1秒严格保持、强单调性合格。

验收：同一合成panel可同时得到旧严格失败、新研究合格；新排序确实不被旧hold或单调失败左右；反向端点偏好、未完成弯段、真实失败仍不能合格。

## 5. 必要测试（任务D）

新增两个行为测试文件：
- learning/tests/test_path_scratch_initialization.py
- learning/tests/test_path_research_selection.py

扩展现有conditional cache/reporting/dynamic测试，不复制整套测试模块。旧strong tests保持不改预期。

测试清单：
- [ ] 无上层父文件时scratch静态加载和prepare成功；scratch夹带父身份报错。
- [ ] fresh同seed逐tensor相同；hidden和第336列非零；Actor确定性输出零；Critic不同于旧父模型；std正确、Adam为空、恢复键列表为空。
- [ ] scratch不调用initialize_conditional_actor；恢复scratch自身checkpoint确实恢复Actor/Critic/FiLM/Adam/RNG/counters，不重新归零。
- [ ] fixed_mixture在1/100/101/200轮结果一致；同一回合α不变；路线流与α流独立；旧warmstart和课程回归通过。
- [ ] 非零FiLM的Torch/Flax前向、导出、加载和梯度检查复用已有测试。
- [ ] 研究hold精确检查原20秒最后100tick；早终止不能拿其末100tick通过；固定/动态使用同一profile。
- [ ] 新schema放宽主目标与hold，旧schema保持原输出。
- [ ] 相邻非单调仅诊断；端点Dv<.02、Dy<.001或任一高速主目标超限则研究失败。
- [ ] 漏任一完整原弯段箱、非有限、失败后补零轨迹不合格。
- [ ] acceptance变化使选模身份变化，拒绝复用旧best分数；匹配B0物理仍可复用。
- [ ] 工程检查的checkpoint不能作为正式训练父代。
- [ ] 动态lineage预算与已完成缓存复用原测试继续通过。

研究门槛的代表性测试输入必须覆盖：
~~~text
高速两侧完整turn：
Ev0=.12, Ev1=.08 -> Dv=.04，速度通过新.10但不通过旧.05；
Ey0=.12, Ey1=.13 -> Dy=.01，路径通过新.15但不通过旧.10；
末0.5秒：|ev|=.12, |ey|=.15, |eh|=.08 -> 新hold通过，旧hold失败。
各中间α任务完整；故新qualified_research_preference=True，旧qualified_task=False。
另把α=.5的Ev抬到.16，保持终点hold不变：新资格不因相邻反向而改变，诊断必须记录反向。
再将Ev1改为.13：高速速度超限且Dv反向，新资格必须False。
~~~

从现有test_path_strong_selection.py的preferred()/panels()夹具构造上述数据；不要修改生产奖励来制造通过。

CPU命令（在新工作树）：
~~~bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python -m pytest learning/tests/test_path_scratch_initialization.py learning/tests/test_path_research_selection.py learning/tests/test_path_conditioning.py learning/tests/test_path_strong_conditioning.py learning/tests/test_path_modulated_policy.py learning/tests/test_path_strong_selection.py learning/tests/test_path_strong_reward.py learning/tests/test_path_conditional_training.py learning/tests/test_path_conditional_reporting.py learning/tests/test_path_conditional_cache.py learning/tests/test_path_dynamic_preference.py learning/tests/test_path_terminal_contract.py -q
~~~

必要测试通过后不为了接口反复重跑完整物理面板。新scratch仅一次8×16 GPU工程检查；原warmstart工程记录不能替代新初始化入口的检查。

## 6. 预算、运行入口和交付（任务E）

### 6.1 有界声明

- 正式：200×512×128=13,107,200策略转移，最多52,428,800底层ticks。
- 工程：8×16=128策略转移，最多512底层ticks；更新丢弃，无DEV/动态。
- 固定DEV：0/50/100/150/200 × 5α × 6场景 × 4000ticks，最多600,000ticks。50轮是检查点，不是提前失败停止线。
- 条件动态：仅研究阶段合格后，选中同一模型，左右高速弯×0→1/1→0/0→1→0，共6条20秒轨迹，最多24,000ticks，跨resume不重复扩预算。
- 以上合计上限53,053,312底层ticks。不含别的热启动run，不允许混账。
- B0优先复用匹配物理轨迹并按v2重评分。若缓存身份不符，先报告具体不匹配项，不隐式增加基线仿真预算。
- 不新增从零/热启动/奖励多组消融，不改成每端200，不自动延长至400。
- 科学指标未过仍执行至声明200；工程故障、非有限、身份或预算错误沿用现有保护。物理失败样本保留，不能因失败而补采超预算。
- 正式阶段从fresh seed重新创建learner，不从工程update1继续。

### 6.2 操作顺序及命令

以下是给实施agent的执行入口，不表示规划agent已执行。

先创建隔离工作树（目标不存在时）：
~~~bash
git -C /home/qy/STTW_CONTROL worktree add -b experiment/path-unified-scratch /home/qy/STTW_CONTROL/runs/worktrees/path-unified-scratch b0ae7106cbdb78bcd38637341524faa1fb2269b4
~~~

代码完成、相关测试通过后，在新工作树使用已有CLI，不另写慢速trainer。

无物理执行的正式声明：
~~~bash
/home/qy/mujoco_playground/.venv/bin/python learning/cli/train_path_command.py --config learning/configs/path_unified_scratch_run.json --output runs/path_unified_scratch_200_20261014/declaration --check-interface
~~~

唯一GPU工程检查：
~~~bash
/home/qy/mujoco_playground/.venv/bin/python -u learning/cli/train_path_command.py --config learning/configs/path_unified_scratch_engineering.json --output runs/path_unified_scratch_200_20261014/engineering --execute
~~~

正式启动（在执行会话已有明确运行授权、工程检查完成且GPU资源已协调后）：
~~~bash
/home/qy/mujoco_playground/.venv/bin/python -u learning/cli/train_path_command.py --config learning/configs/path_unified_scratch_run.json --output runs/path_unified_scratch_200_20261014/training --execute
~~~

输出目录已存在时不要覆盖或删除后重跑；按现有--resume机制声明独立恢复目录、携带历史消耗和源identity。scratch身份是首次初始化来源，不能用其掩盖resume状态。

### 6.3 性能、可见性和交付

- 复用现有DirectCommandPPO、同形状场景vmap、scan(50)、共享JAX缓存 /home/qy/STTW_CONTROL/runs/.jax_compilation_cache。
- 不在本任务引入额外性能算法。记录实际初始化、采样、优化、评测耗时。
- TensorBoard日志与旧run分离。服务端口先核对占用，优先项目入口6006；不得停止其他项目服务。验证HTTP和本run真实reward scalar后交付链接，仅event文件不算完成。
- 记录每组α实际样本数、χ>0及χ≥.5覆盖、成本分项、loss/KL、接受更新数、物理/路径域失败；训练统计不能替代固定DEV。
- 首屏交付同场景五α+B0+原始路径XY；同时输出速度/路径误差、滚转、每步与累计reward及分项、原请求→修正参考→最终命令→实测运动。保留PNG和原NPZ，不重复大包。
- 同一评测面板五个α必须属于同一个checkpoint。selected与last并列报告；新研究门槛和旧诊断并列，不只报qualified=true。
- 若中间α响应不足，保留真实曲线；一seed固定DEV不写成泛化或全区间Pareto最优证明。
- 本文作为唯一任务书交接即可；执行agent可将其复制到新工作树docs/path_unified_scratch_taskbook.md并随代码提交，不再另建多套过程文档。
- 执行agent更新新工作树PROJECT/VALIDATION/METHODS及override；持research-hub/.write.lock，锁内重读共享台账后增量记录scratch新任务、阈值版本、代码/运行状态，保留旧热启动及其他项目记录。
- 完成相关测试与git diff --check，显式stage任务文件并提交；push遵循执行会话当次授权，不冒称已推送。
- 最终收据列出commit、实际新run身份、初始化mode、actor/critic/optimizer来源、测试范围、实际消耗与预算、TB链接、研究资格/旧严格资格、当前限制。

## 7. 完成标准

- [ ] 原位代码支持scratch + fixed_mixture + research acceptance，旧warmstart/课程/严格验收兼容。
- [ ] 新配置父上层字段null；模型不继承任何上层权重，lower身份仍核验。
- [ ] CPU行为检查通过；一次新scratch GPU工程检查通过并丢弃更新。
- [ ] 正式任务仅一组200，必要评测和条件动态不超预算，日志/TB可核验。
- [ ] 新研究资格真正使用0.15/.10主目标、.02/.001差值、末.5秒宽松保持；旧严格门槛只并行报告。
- [ ] 指针、缓存、报告和动态触发使用同一acceptance身份，不存在表面放宽、实际旧门槛仍否决。
- [ ] 原始失败和所有声明α/场景保留，未达标如实交付，不追加轮数或事后改门槛。

本任务书只规定下一次实施；截至规划交付，没有执行scratch代码修改、测试、仿真或训练，也未改动/停止原热启动任务。

