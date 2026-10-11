# Path safety preference implementation and finite validation

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
