# Path safety preference implementation and finite validation

2026-10-12：四组200轮已完成。下文原实施合同保留为历史；当前重新分析及
**尚未执行**的定向修补指令见[下一步实施计划](#next-repair-plan)。

Approved user contract, 2026-10-11. Base `2b0896a`; implementation uses an isolated
worktree. This document specifies code and engineering verification. Formal
training and full recovery evaluation are separate explicitly executed stages.

## Global constraints

Freeze local300 actor SHA `d408a94e886fa7f6309aa2c7635eb62e273c197405293ac9c2f1859014f25bcc`,
ECBC/ESO, plant, actuator authority, PP, filters/slew, immutable original path,
351/352 observations, 128/128/64 ELU networks and independent alpha0/alpha1.
Physics .0002 s x25, lower .005 s, upper .02 s; finite task 20 s.
Retain Coverage V2 sampler and terminal_contract_v2. No hot edits to active
Coverage, automatic formal training, checkpoint overwrite, lower retraining,
constraint relaxation, JIT changes or automatic push.

## Task 1: Versioned reward and independent reconstruction

Modify path_command_reward.py, path_command_env.py and path_command_reporting.py.
Missing path_reward.reward_version means the historical reward. New version is
`path_safety_preference_v1`. Unknown versions fail closed. Keep the original ten
cost keys, caps and 5080.2 failure bound; merge additions BEFORE individual caps.
Keep original task costs and chi. H(x)=x*x for abs(x)<=1 else 2abs(x)-1.

New configured constants under path_reward:
`risk_prediction_s=.10`, `predictive_roll_weight=20`,
`predictive_roll_start_rad=.28`, `predictive_roll_scale_rad=.02`,
`working_exposure_weight=30`, `working_exposure_width_rad=.002`,
`corridor_weight=40`, `corridor_band_m=.40`, `corridor_scale_m=.10`.
rho=max(abs(phi),abs(phi+.10*phi_dot)); roll adds
20H(max(rho-.28,0)/.02); working_roll adds
30clip((peak_roll-.30)/.002,0,1); path adds
40H(max(abs(ey)-.40,0)/.10). All additions are alpha independent.
costs gains roll_angle=None; missing signed roll in new version is an error.
Log actual interval offset_rate and diagnostic subcosts without double counting.
NumPy independently reconstructs all ten costs from raw trace measurements;
offset_rate is (end_offset-before_offset)/.02 for each four-tick block, including
partial terminal blocks. Missing slice context must fail, never assume zero.
Failure block ordinary rewards are replaced by the single original failure tail.
Tests: historical numerical fixtures, mirrors/cross-zero/inward roll, missing
roll, caps, new JAX/NumPy reconstruction, offset-rate and 1/2/3/4-tick failure.

## Task 2: Safety-first selection and preference acceptance

Add path_safety_task_v1 without changing path_task_best_v2 or old pointers.
Lexicographic key: physical_failures, tracking_domain_failures,
working_failures, max_roll_excess (relative .302), total_exceedance_seconds,
corridor_failures (abs ey>.40), max_corridor_excess, goal_hold_failures,
primary_band_failures, primary_score, worst_error. New qualification requires
all count gates zero and original task gates unchanged. Separate immutable
safety best pointer/generations. Accept explicit complete six DEV only.
Preference uses fixed .5m bins covering the ENTIRE original turn; equal-bin
mean MSE followed by sqrt. Require Ev0-Ev1>=.03 m/s and Ey1-Ey0>=.02 m separately
on both high-speed turns only after both endpoint task gates pass. Missing bins
or failures invalidate preference; low speed need not separate.

## Task 3: Weights-only reward forks, physical reuse and executable preparation

Add explicit weights_only_reward_fork alongside existing distribution fork.
Preserve actor.* and log_std only; fresh critic/Adam/RNG/LR/rejections/counters.
Old-reward control and new-reward candidate share each endpoint's exact initial
weights and fresh learner initialization; seed911101 plus alpha. Strict parent
identity/SHA/full learner checks and no mutation of existing fork allowlist.
Resolve parents only from COMPLETED Coverage best_task_model receipts and freeze
both references. Prepare may explicitly report parent_not_ready, execution must
fail closed until completion. Parent physical prefix is not learner continuation.
Reuse physical trajectories only after exhaustive dynamics/path/bank/actor/alpha/
timing/termination identity check; reward/selection differences allowed only in
explicit reweighting mode. Rebuild rewards into separate derived artifacts.
Compiled evaluator keys include configuration identity; retain vmap/scan50/JAX
cache/DirectCommandPPO. Four future arms control0,candidate0,control1,candidate1,
50x512x128 each; total13107200 policy transitions/52428800 lower ticks.
Save every10, DEV0/50; initial/final selected physical panels reused when equal.
No automatic extension/wall deadline. Formal status/TensorBoard per arm.

## Task 4: Full-state recovery and delivery

New path_command_recovery.py integrates existing environment/evaluator, no new
plant or stepping rules. Snapshot first B0 upper boundary with .302<abs(phi)<=.40,
one each left/right R2 at2.6. Store complete PathState, route, v_user, original
horizon/time, plant/lower/bank/config identities, source control and checksum.
Never synthesize qpos or reset ESO/history/filter. Fork only upper alpha.
Use B0 and four final selected arms; deduplicate identical actor identities.
No available qualifying snapshot means NOT_VERIFIED, not a replacement state.
Recovery: complete .5s consecutive peak_roll<=.28 within1s after takeover;
then no peak_roll>.302, no physical/domain faults, abs ey<=.40 and original
20s goal/last1s hold. Report full initial violation, later peak/duration and
speed/path costs. Prefix+10 tails <=48000 lower ticks. Explicit execution only.
Serialization/twin deterministic short continuation tests; no failure padding.

## Delivery

Run necessary CPU tests and exactly one discarded8x16 GPU engineering rollout
(<=512 lower ticks, <=1 optimizer update). It needs separate run/TensorBoard,
actual reward HTTP receipt and stage/error/completion notification. Full panels
and formal arms are NOT automatically executed. Read-only old NPZ reward audits
and XY plots are permitted. Persist results to docs/VALIDATION.md and the evidence
index, update shared PROJECT_STATE.md under its exclusive .write.lock, explicitly
stage/commit relevant files; push follows current authorization.

## 实施交接

工作树：`/home/qy/STTW_CONTROL/runs/worktrees/path-safety-preference`；分支
`experiment/path-safety-preference`。运行下列命令前切到该工作树。
解释器 `/home/qy/mujoco_playground/.venv/bin/python`；运行设置：

```bash
export PYTHONPATH=learning/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/home/qy/STTW_CONTROL/runs/.jax_compilation_cache
```

本轮正式训练与自然超限恢复均未启动。等 Coverage **整个流水线**完成，
再执行父身份冻结；运行中、仅训练阶段结束、最终面板/source不匹配都会拒绝。
冻结文件不可覆盖，并包含checkpoint/Actor及六个物理轨迹/原路径SHA。

```bash
/home/qy/mujoco_playground/.venv/bin/python learning/cli/freeze_path_safety_parents.py \
  --parent /home/qy/STTW_CONTROL/runs/worktrees/path-feedback-v1/runs/path_coverage_v2_resume_20261011 \
  --output runs/path_safety_parent_identities.json
```

两份计划各顺序执行α0/α1，合计四组50更新；DEV仅0/50，0轮和最终选中
轨迹按身份复用。控制与候选都使用新安全选模，但控制保留历史奖励。
默认CLI只准备、无物理采样；后续正式阶段执行命令为：

```bash
/home/qy/mujoco_playground/.venv/bin/python learning/cli/train_path_command.py \
  --config learning/configs/path_safety_control_run.json \
  --output runs/path_safety_control_50 --execute
/home/qy/mujoco_playground/.venv/bin/python learning/cli/train_path_command.py \
  --config learning/configs/path_safety_candidate_run.json \
  --output runs/path_safety_candidate_50 --execute
```

每个端点独立TensorBoard子目录与状态/优化器；每份计划独立服务器/日志。
新的reward必须通过HTTP加载才继续。严守50更新，不自动追加。阶段结束通知
与最终图/报告完成通知分开。需要exact resume时使用新输出目录和`--resume`；
它恢复当前新奖励阶段完整learner，和首轮weights-only分叉不同。

完整四组结束后，单独执行已声明的两B0前缀和十个配对接续；模板：
`learning/configs/path_safety_recovery_request.json`。修改运行路径只允许指向
实际完成、通过SHA审核的对应端点选中模型；不能指向last或临时best。

```bash
/home/qy/mujoco_playground/.venv/bin/python learning/cli/path_command_recovery.py \
  --mode capture --request learning/configs/path_safety_recovery_request.json \
  --output runs/path_safety_recovery_capture --execute
/home/qy/mujoco_playground/.venv/bin/python learning/cli/path_command_recovery.py \
  --mode resume --request learning/configs/path_safety_recovery_request.json \
  --output runs/path_safety_recovery_results --execute
```

找不到自然超限首状态，输出NOT_VERIFIED，不创建人工滚转状态。
恢复判据独立于正常六场景主任务/偏好验收；主目标RMSE保留描述性数据，
不能额外添加用户未声明的恢复门槛。输出PNG/NPZ/统一索引，保留初始超限段。
当前工程和CPU证据见[实现验证索引](evidence/path_safety_preference_v1/INDEX.md)。

CPU历史轨迹测试复用归档`runs/path_reward_peak_review_20261011`；本隔离工作树
以ignored符号链接指向原worktree该归档，不复制大产物。新checkout运行相关
测试前须接入同身份归档；缺少它是fixture不可访问，不能解释为物理失败。
最终完整16个path模块110项CPU检查通过；独立源码审阅所有已记录问题已修正。


### Authorized paired total budget200 (2026-10-11)

User selected all four groups total200, with50 as checkpoint. Preserve the active50-stage code/config/artifacts. In isolated budget worktree: test explicit bounded plan and completed50 resume guards, allow only budget200 with DEV50/100/150/200, keep frozen design and best identity unchanged, and use existing full learner restore (including Adam learning rates and RNG), never weights-only reinitialization at50. All four arms reset physical/controller state to prepared bank at the same stage boundary, consistent with existing learner resume semantics; no claim of frame-continuous training.

Implementation steps: (1) add failing budget/resume tests; (2) centralize budget schedule validation in path_command_fork and connect plan loader/evaluator; (3) add independent200 plans referencing existing50 runs and same frozen parent identity; (4) run targeted CPU tests and source-diff checks; (5) preregister incremental budget exactly four150x512x128=39,321,600 transitions (total52,428,800), queue continuation after all four50 arms complete, monitor and verify TB when continuation produces reward. No policy, reward, physics, sampler, optimizer algorithm or safety gate changes.

Verification completed:38 related CPU tests passed, including full learner state and Adam learning-rate restore. Both200 plan completed-parent audits passed. Continuation checkpoint identity retains the original50 best_resolution to load exact states; the explicit authorized_budget checkpoint_updates field records the expanded50/100/150/200 candidate comparison scope. Total four-group budget52,428,800 policy transitions; incremental remaining39,321,600; total max209,715,200 training lower ticks and max384,000 DEV lower ticks (24 cases per four checkpoints, update0 reused), recovery remains separately bounded48,000 ticks.

<a id="next-repair-plan"></a>

## 2026-10-12 下一步实施计划：限制弯中让步，保留已学回归

> 给实施agent：按 `executing-plans` 或 `subagent-driven-development` 分任务实施并复核。
> 本节是用户要求的分析与交接建议，**不是已经执行的实验或本轮训练启动记录**。

**目标：** 保护α0已合格的行为，从仍有良好最终保持的α1父代，修补弯中越出.40m与后续学习遗忘。
**架构：** 独立两端、冻结local300、原路径外环与PPO保持。先做固定Actor诊断，再有条件比较两个α1修补臂；α0不再重复训练。
**工具：** 现有MuJoCo/MJX、JAX vmap/scan(50)、PyTorch DirectCommandPPO、NumPy独立重评分。

### 事实与设计边界

[重新分析及原始证据](evidence/path_safety_preference_v1/paired200_analysis/INDEX.md)：Coverage父代已经产生预期取舍，α1原有六场景滚转/目标/保持/速度主任务均通过，但三个场景超过新corridor限制。新奖励200轮没有解决它，且丢失最终保持。不要把“新奖励失败”改写为“架构无效”，也不要把左弯1.010m解释成已接近.40m。

从本交接提交建立隔离工作树；代码起点至少包含`6ce0ba0`。先检查上游修复和工作区，不热改旧运行、不改旧JSON或best指针。新代码在原模块维护，新配置用独立身份。正式训练启动前登记解析配置、身份、预算及停止条件，向用户明确本节条件阶段的执行范围；本轮不启动它。

冻结原物理、351/352维、128→128→64、20秒、.0002/.005/.02秒时序、修正权限/filter/slew、Coverage V2、γ=.997、λ=.99和`terminal_contract_v2`。α只改变偏好，不能改变共同门槛、原速度/路径请求和执行能力。所有20ms块变化率仍为块末减块前除.02，包括不足四tick的失败块。新安全成本仍不能提供逐步安全保证。

### Task 1：固定模型身份与八回合因果诊断

**文件：** 修改`learning/src/sttw_control/path_command_evaluation.py`、`path_command_cache.py`；新增薄入口`learning/cli/path_command_diagnostic.py`与`learning/configs/path_safety_repair_diagnostic.json`；测试`learning/tests/test_path_diagnostic.py`。不要复制物理循环。

冻结α0比较基准（不是宣布整个两端系统采用）：

```text
/home/qy/STTW_CONTROL/runs/worktrees/path-safety-budget-200/runs/path_safety_control_200_20261011/alpha0/safety_task_best_generations/d674afdac56d97b520882592853ff8e2e5c87b011ee6c6e588ad2178caea595a/best_model.pt
checkpoint_sha256=d674afdac56d97b520882592853ff8e2e5c87b011ee6c6e588ad2178caea595a
同目录best_actor.pkl SHA=b66df1d960dd29d60031c306b46d78007f389f435fea271264d5bdcf496818e6
```

α1修补起点固定Coverage新增200父代，不用新奖励u200或动态best：

```text
/home/qy/STTW_CONTROL/runs/worktrees/path-feedback-v1/runs/path_coverage_v2_resume_20261011/alpha1/task_best_generations/a1e9b7f1e8d78c5335ab84b5a30068409cb6808378be113d6e56b6ede7079be3/best_model.pt
checkpoint_sha256=a1e9b7f1e8d78c5335ab84b5a30068409cb6808378be113d6e56b6ede7079be3
同目录best_actor.pkl SHA=269d5f43753f798d16e43f140e9b5b0c596a5fa687c45acedbc26649a070f152
```

- [ ] 核验两组checkpoint/Actor SHA、lower SHA、本地准备状态与六场景原轨迹身份。α1父代对应现有paired update_0000；记录所用完整prepared状态，不能仅恢复qpos/qvel。
- [ ] 在`BatchedPanelEvaluator._compiled_group`中Actor输出后、`env.policy_step`前增加可选`diagnostic_transform=None`和显式序列化`diagnostic_identity`。None走旧路径；identity进入编译缓存及物理轨迹缓存。只允许该诊断CLI启用，禁止正式训练/选best调用。复用vmap/scan(50)与共享JAX缓存。
- [ ] 父α1左右高速弯各运行三种预声明干预：`beta=.75`、`.5`、`.5且额外减速.05*chi m/s`。转向**物理目标修正**乘beta，不乘名义参考或实际执行器命令。仅第三种干预速度；chi来自当前原路径/原速度需求，非未来标签。转换后仍走原filter、slew、幅值clip与底层。核心为：

```python
# z is the original Actor latent; c is residual_interface; chi is current demand.
# Return z directly when beta == 1 and speed_backoff == 0 (exact no-op).
target = map_latent(z, config)
target_steer = beta * target[1]
steer_z = arctanh(clip(target_steer / c['steer_scale_rad'], -1+1e-6, 1-1e-6))
speed_target = target[0] - speed_backoff * chi   # speed_backoff is 0 or .05
speed_target = clip(speed_target, -c['speed_negative_scale_m_s'], c['speed_positive_scale_m_s'])
scale = where(speed_target >= 0, c['speed_positive_scale_m_s'], c['speed_negative_scale_m_s'])
speed_z = z[0] if speed_backoff == 0 else arctanh(clip(speed_target / scale, -1+1e-6, 1-1e-6))
z_applied = stack([speed_z, steer_z])
```

记录目标clip次数。额外−.05χ仅是减速量级探针，可能使原α1需求速度RMSE超过.05；即使滚转/路径改善，也不得据此放宽速度门槛或称共同合同可行。

- [ ] 新奖励α1 u200的两直道，各从原完整prepared状态运行一次：t<10秒完全原Actor，t∈[10,10.5]转向目标乘`max(0,1-(t-10)/.5)`，其后目标为0，速度通道不变，直到原20秒。用已有u200轨迹核对10秒前逐tick闭环输出；不另跑原分支。干预结束后的offset由原滤波/slew决定，不能直接清零控制状态。这是诊断干预，不是部署补丁。
- [ ] 测试None及beta=1无操作的数值/RNG一致，±转向对称，非对称速度尺度正确，修改前后目标均记录；不同诊断identity禁止缓存串用，真实终止不补零。随后才执行最多8×4000=**32,000 lower ticks**，不更新Actor。
- [ ] 八条全部导出原参考XY、roll、ev/ey/eh、每步/累计奖励与请求→名义/修正→最终执行器命令→实测链。基线已有轨迹复用；不同干预不是训练成功。对每组注明安全、.40m、主目标、目标截面及最后1秒保持。

**诊断分流：** 某一预声明设定在两高速弯均保持roll≤.302并比父代降低路径峰值，支持定向修补空间；两高速弯都满足全门槛才可称该干预实现可行示例。若三个设定均不能同时保持两弯滚转安全并降低路径峰值，不等于系统不可行，但暂停自动训练，报告哪条约束发生冲突。两直道撤销转向修正后回正且保持安全，才增强“该修正导致稳态漂移”的因果证据；若未改善，暂停Task 3的保持项并继续定位，不凭相关性启动。

### Task 2：路径成本截断及失败上界的独立修订

**文件：** 原位修改`path_command_reward.py`、`path_command_reporting.py`、`path_command_env.py`中的版本分支/诊断元数据、`path_command_fork.py`、`path_command_training.py`、`path_command_cache.py`；新增配置`learning/configs/STTW_Path_Safety_CapBound.json`；扩展`test_path_safety_reward.py`、`test_path_reward_cache.py`、`test_path_reward_fork.py`、`test_path_reward_budget.py`。

- [ ] 新奖励身份`path_safety_cap_bound_v2`复用V1公式和十个成本键，只把path cap从200改1000，同时`failure_cost_rate_bound`从5080.2改5880.2。历史缺省与V1逐值不变、未知版本拒绝。现有DEV raw path最大988.447支持这个候选量级，不能宣称训练样本都不会再cap。

```python
cfg['path_reward']['reward_version'] = 'path_safety_cap_bound_v2'
cfg['path_reward']['component_caps']['path'] = 1000.0
cfg['path_reward']['failure_cost_rate_bound'] = 5880.2
assert abs(sum(cfg['path_reward']['component_caps'].values()) - 5880.2) < 1e-6
```

- [ ] 统一定义“是否启用V1安全公式”的版本判断，接通JAX成本、环境日志与NumPy独立重建，拒绝新版本缺signed phi。失败尾罚自动用新上界；仍只记一次、保留原真实终止。明确B是**cap＋失败上界组合修订**，不是纯cap单因素。
- [ ] 新计划身份`path_safety_repair_v1`只允许`alphas=[1]`、本文父代、声明B/C和预算；原四组计划仍只允许原`[0,1]`及原预算。不能删掉旧严格校验来“支持新配置”。物理身份只排除新增且确定纯训练用途的保持配置；实际执行诊断、原路径/时序/底层变化仍使物理身份不同。
- [ ] 重评分现有全部120条DEV，不运行物理，保留V1原分数及V2新分数；逐项报告饱和时长变化。测试raw path=640时V1截200/V2保640，raw=1200时V2截1000；总cap=5880.2、失败剩余折扣尾罚、一次记账、20ms offset_rate/幅值clip/短终止块、JAX与NumPy一致。旧reward与旧四组fork回归必须过。

### Task 3：普通合格区父Actor转向保持（独立于奖励）

**文件：** 修改`path_command_training.py`及`direct_command_ppo.py`；新增`learning/configs/STTW_Path_Safety_CapBoundKeep.json`；测试`learning/tests/test_path_actor_retention.py`与现有`test_direct_command_ppo.py`。不改RSL库，不加观测维度，不增加策略运行时的老师网络。

- [ ] `DirectCommandPPO`接受缺省None的Actor辅助损失回调。在当前`policy.act(obs)`得到分布均值后，用`policy.action_mean`计算student steer；teacher为冻结Coverage α1 Actor副本，eval/no_grad、无采样，输入同一当前on-policy观测。teacher不进优化器，不读取旧轨迹当PPO样本。
- [ ] 采样**动作前**从当前完整状态计算mask（非本次奖励后步状态）；用当前原路径投影与可观测物理量，定义：

```python
healthy = (live_finite & (chi <= 1e-6)
           & (max_abs_original_curvature_at_s_s05_s1 <= 1e-3)
           & (maximum(abs(phi), abs(phi + .10*phi_dot)) <= .26)
           & (abs(ey) <= .10) & (abs(eh) <= .05) & (abs(ev) <= .10))
```

mask放入TensorDict的独立`retention_mask`键，随同一obs写入RolloutStorage；Actor/Critic的obs_groups仍只含policy/critic。沿用RSL原shuffle/minibatch索引，不另外randperm，不从已clip的归一化观测反推mask。没有新增网络输入或未来信息。

- [ ] 保持损失只在合格普通区阻止漂移，不固定弯中/大误差恢复动作；先用预声明候选系数.1、物理转向尺度.02rad，不在同一运行在线调参：

```python
with torch.no_grad():
    parent_mu = teacher(obs['policy'])
mask = obs['retention_mask'].reshape(-1).to(student_mu.dtype)
delta_difference = config['residual_interface']['steer_scale_rad'] * (student_mu[:, 1].tanh() - parent_mu[:, 1].tanh())
row = (delta_difference / 0.02).square()
keep_loss = 0.1 * (mask * row).sum() / mask.sum().clamp_min(1)
loss = original_ppo_loss + keep_loss
```

这是优化器辅助项，不写入reward/GAE，不冒充仅奖励变化。.01/.02rad目标差对应辅助损失.025/.1，避免未归一化小量使实验无效。空mask得到可微零；不对Critic/log_std直接反传，Actor共享隐藏层仍可能间接影响速度，须同时监测。原KL拒绝/回滚、梯度限幅、std限制继续覆盖完整PPO更新。系数和尺度为待验证设计，不宣称足够。

- [ ] teacher从冻结父Actor深复制并加载，初始化后不消费paired RNG；严禁从随机新ActorCritic构造造成下一步RNG改变。B/C同端点新Actor/log_std、Critic、Adam空状态、LR、拒绝计数和Torch/CUDA/NumPy/Python RNG逐张量/状态相同。A已有初始化作为核对基准。teacher身份存checkpoint，恢复时重建并核验，不能误装为student或Adam参数。
- [ ] 测试默认None及零权重的旧更新/RNG精确回归、不同minibatch mask对齐、空mask有限、teacher无梯度、辅助项对Critic/std无直接梯度、故障回滚和50轮完整恢复。用非零且不相同的父/学生输入验证损失能改变steer，不写只复述实现的空断言。
- [ ] 记录mask数量/分母、物理steer目标差、辅助loss、相同预clip时点PPO Actor梯度与辅助Actor梯度范数/夹角、按ordinary/strong/failure-near分组的TD残差与优势。梯度诊断只在每个update首minibatch执行`autograd.grad`、不写.grad、不改变RNG；其余记录标为未测，不能冒充全批次。mask为零/大幅减少必须报告，不自动扩大mask或系数。

failure-near仅作事后日志分组：同一环境同一回合、真实物理/路径域失败前最多50个策略步，窗口不跨reset；批次结尾尚不知后续结果的段标为截尾，不能当无失败样本。该分组不进观测、mask或奖励。

现有DEV后步lower tick代理mask覆盖父代约72.5%–98.5%，新u200仅2.3%–11.9%；这不是训练pre-step覆盖率。它支持从父代**防遗忘**，不支持从已退化u200直接修复。训练实际覆盖率须另记录。

### Task 4：有条件的两臂α1预算、验收与交付

**文件：** 新增`learning/configs/path_safety_cap_bound_run.json`、`path_safety_cap_bound_keep_run.json`及各自50→200阶段配置，复用`learning/cli/train_path_command.py`、现有报告/恢复入口，必要修改`path_command_selection.py`和报告器以接入单端点及冻结α0。不改旧排序与旧best。

| 组 | 训练起点/初始化 | 相对变化 | 新预算 |
|---|---|---|---:|
| A | 已有新奖励V1 α1，Coverage父代weights-only、新Critic/Adam/RNG | 参考臂，旧200数据复用 | 0 |
| B | 同父Actor/log_std、同fresh Critic/Adam/RNG | V2 cap＋失败上界 | 最多200轮 |
| C | 与B初始化完全相同 | B＋普通合格区父steer保持 | 最多200轮 |

旧奖励α1作为额外历史参照，冻结旧奖励α0 u200作偏好配对。A复用须核对原版本、父SHA、seed、sampler、PPO及50轮物理重置协议，不能只因同父Actor而称完全配对；若不匹配，拒绝该因果对照，报告差异，不自动补跑第三臂。

- [ ] 在Task 1给出支持证据后，冻结两个新配置与试验声明。每臂seed911102（plan seed911101＋α1），512环境×128策略步，50/100/150/200固定六DEV；0轮复用同身份物理、按各自reward重评分。匹配原A在50轮的**完整learner恢复＋物理重置**，包括原实现的route counter递增/reset语义；不是200轮连续仿真也不是50轮重新初始化。
- [ ] 实施后仅一次8环境×16策略步GPU工程检查，最多128策略转移/512 lower ticks、一次优化，丢弃更新；CPU测试覆盖B/C分支，不以接口测试重跑大面板。保持vmap批评测、scan(50)、共享编译缓存、现有DirectCommandPPO，计时初始化/采样/优化/评测。
- [ ] 新正式训练总上限`2*200*512*128=26,214,400`策略转移，最多104,857,600 lower ticks；DEV`2*4*6*4000=192,000` lower ticks。不能因旧A存在就把B/C预算算成零，也不能不经声明追加种子/更新。
- [ ] 每臂独立TensorBoard、status、checkpoint与通知；启动后核实HTTP与本臂真实reward标量加载。50轮是检查点，200是上限，不是收敛承诺。新增保留门槛：任一固定六场景DEV检查点出现物理/路径域失败、任一场景roll>.302、或父代已通过的任一最终保持退化，保留反例并停止该臂自动推进，等待结果审查；不改原字典序挑一个较小corridor掩盖退化。主目标/corridor暂不达标可在保留门槛通过时继续至预算。
- [ ] 保留原六场景完整20秒共同任务门槛：roll≤.302、|ey|≤.40、目标截面、末1秒每步|ev|≤.10、|ey|≤.10、|eh|≤.05，以及α0需求路径RMSE≤.10/α1需求速度RMSE≤.05和普通场景原门槛。两端合格后才检验左右高速完整弯段.5m等权分箱`Dv>=.03m/s, Dy>=.02m`，不剔除未覆盖箱。没有共同任务合格，不宣称有限偏好成功。
- [ ] 正常任务通过后才安排独立自然超限恢复：保留既有B0首个上层边界`.302<|phi|<=.40`完整快照合同，不能用Task 1的安全干预状态替代。B0、冻结α0、Coverage父α1、B、C五方法×左右两状态，前缀＋接续最多48,000 lower ticks；继续至原20秒。两端须1秒内完成连续.5秒roll≤.28，随后不再>.302，并通过原路径/.40m/最终保持；完整计入接管初始超限段。找不到快照报告未验证，不改角度造状态。
- [ ] 正式训练、DEV、诊断、一次工程检查和条件恢复合计最大105,130,112 lower ticks；其中工程128策略转移单列，不混入正式26,214,400。条件不触发时记录实际更少消耗，不补满预算。
- [ ] 完整交付六场景同图XY、roll/ev/ey/eh、每步与累计奖励/分项、修正命令链、父代能力保留/新增/丢失表，带精确父/学生SHA、seed、reward/辅助loss身份。PNG＋原NPZ，不复制权重或重复大包。固定DEV是已用于设计的开发集，单seed不能称泛化。任何一项失败保留反例、停在预算内，不回写成功阈值。
- [ ] 更新原位验证/方法文档，持共享`.write.lock`重读增量更新台账；提交相应代码/配置/必要证据，推送依当次授权。最终先交付实现及测试收据，明确哪些阶段实际执行、哪些只准备。

**后续决策：** 先完成固定Actor诊断；证据支持后比较B/C；只有共同任务、有限偏好、真实超限恢复三者均通过，再讨论多训练种子与连续α单网。当前没有依据重训底层、放宽.40m或提高已完成四组的200轮上限。

### 2026-10-12 执行回执：Task1完成物理交互，验证/支持门槛失败

用户要求按本节开始工作。隔离分支`experiment/path-safety-repair`从0356fc7实施，运行代码6daed97/387abc5。61相关CPU检查与独立静态审阅通过；source/config/bank/Actor/缓存来源、完整终点及原路径严格校验。

实际执行全部八20秒回合、32000lower ticks、PPO更新0。三个同设定左右配对都不能保持共同roll<=.302同时减少path峰值；两直道淡出后path接近原路径，却因末段航向振荡保持失败。因此Task2/3/4条件阶段**未触发**，未实现新cap/保持优化，也未启动两新200轮或自然超限恢复。

GPU前缀核对另有错误：低速直道所有原字段完全匹配；高速直道转角差2.3832e-5rad超过预声明2e-6。两条在10秒前应用的latent均与原Actor输出相同；源身份和批形状相同，数值差来源仍待验证。ACCOUNTING=error/exit1原记录保留，不修改容差、不重跑预算。高速淡出结果只能描述，不能称严格配对因果验证。分析交付完成与实验验证未通过分别记录。

[全部八条件XY/奖励分项/误差/命令链和数值审计](evidence/path_safety_repair/diagnostic/INDEX.md)。原计划与负结果保留；今后任何新诊断或训练须重新声明实际变化、预算与有效性条件，不借本执行回执自动扩展。
