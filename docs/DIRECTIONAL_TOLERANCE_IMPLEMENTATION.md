# Directional tolerance diagnostic implementation

2026-10-22; isolated experiment/tolerance-calibration-20261022 based on 2b0896a.

The numerical module derives from the supplied mathematical verification script. Its linear constraints now retain every strictly positive slope, independent of constraint units; supplied slope accuracy and model validity remain caller responsibilities. Nominal violations are not silently forgiven by an absolute epsilon. A numerically rank-deficient Gramian returns UNKNOWN_NUMERICAL and NaN margins, never a physical unreachability certificate. Input state normalization is still required. No arbitrary regularization is added.

The optional `force_pulse=[absolute_start,absolute_end,fx,fy,fz]` uses controller-clock seconds and newtons in world coordinates. It applies a force at the vehicle subtree COM as an equivalent force/torque on the chassis, recomputing the moment arm each physical substep. Fractional substep overlap scales the applied force; each actual substep contributes to logged `applied_force_impulse`. It does not inject qpos/qvel or alter model parameters. `None` is the default. The force direction is captured at perturbation onset in the CLI and remains fixed throughout the pulse.

CPU tests verify zero-pulse equality and 0.0048 s partial-substep pulse impulse. The actual frozen-policy MJX preflight compares complete states for None vs zero pulse. The calibration CLI validates model identities and uses vmap batch7 / scan50 with persistent compilation cache.

The first panel attempt failed before force application because separately propagated identical GPU lanes differed numerically. It consumed 1400 lower ticks plus 8 preflight ticks. The next attempt branches all variants by copying ONE complete actual nominal state/history at pulse onset, and uses that same actual nominal prefix for all reported trajectories. Redundant SIMD prefix simulation is charged, not treated as independent evidence. This is genuine snapshot branching; it is not state-coordinate injection or a relaxed replay threshold.

Run artifacts are immutable per attempt. `status.json` reports declared/completed cases and active lower ticks. Scoped ACTIVE_RUN entries reuse the existing Bark-only watcher; automatic repair and email remain disabled. No training, new reward, observation schema, model adoption or online correction is part of this task.

Commands from this worktree:

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python -m pytest learning/tests/test_directional_tolerance.py learning/tests/test_tolerance_algebra.py learning/tests/test_tolerance_pulse.py learning/tests/test_path_feedback.py learning/tests/test_path_training.py learning/tests/test_path_terminal_contract.py -q
PYTHONPATH=learning/src JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qy/mujoco_playground/.venv/bin/python learning/cli/calibrate_directional_tolerance.py --output runs/NEW_IMMUTABLE_ATTEMPT
/home/qy/mujoco_playground/.venv/bin/python learning/cli/analyze_tolerance_pilot.py runs/COMPLETED_ATTEMPT
```

Do not rerun a completed panel merely to regenerate plots. One route, one speed, three onset times and a finite force grid are development evidence only. No calibrated force prediction from UCEDT is claimed by this implementation.


## Training integration audit (2026-10-22)

Source inspected: path_command_reward.costs/failure_reward, PathBatch.advance_one, and path_command_training.run. No reward or training changes made.

- Current Coverage V2 sum(component_caps) = failure_cost_rate_bound = 5080.2. If a bounded tolerance cost with cap C is added, the failure-tail bound must include that cap (and any coefficient) under the same scale/time units. Otherwise the previously derived terminal-failure penalty may cease to dominate the future cost. Recomputed reporting and COST_KEYS/statistics dimensions must include the same component.
- Existing reward is negative cost rate times scale 0.1 times lower dt 0.005; the PPO reward sums four lower ticks per upper interval 0.02. A per-upper tolerance cost must be integrated once with consistent duration, not accidentally multiplied four times.
- Keep original chi and speed/path preference terms fixed. A tolerance proxy must not replace chi, relax the original reference, alter success thresholds, or add observation dimensions to an existing Actor.
- Training uses restored Coverage model/optimizer identities only through explicit fork contracts. A changed reward is a new experiment and cannot masquerade as exact continuation under the old reward. Comparison arms must start from the same endpoint checkpoint and declare optimizer handling equally.
- Current 20-second complete closed-loop probe rollouts are offline labels, not a low-cost per-step feature. Current six left-turn states include only one directional failure boundary; censored labels cannot be converted to exact regression targets. A cheap state proxy and route/state-separated validation remain prerequisites for reward adoption.

Next integration evidence must include a zero-weight unchanged-transition/reward check, bounded/nonfinite/unknown-proxy behavior, component accounting plus failure-bound check, matched checkpoint/budget controls, and independent physical pulse outcomes. These checks supplement the scientific test of the proposed margin; they cannot replace it. No training has been launched by this audit.


## Concrete training edit map (2026-10-22, before model audit outcomes)

The current full-panel execution and its frozen analysis are unchanged. The following is an integration dependency map, not evidence that a margin has passed validation or that training has started.

1. `path_command_env.py:record_frame` records the post-policy-step state. Its raw20 frame is current at the saved pulse boundary, not a one-policy-step-old observation. Any cheap model computation must use the controller rear-wheel speed proxy and current ESO disturbance, consistently with `tolerance_state_features`.
2. `path_command_reward.py:costs` owns lower-tick cost integration. Add an independently bounded component only after selecting a validated metric; keep chi and the original ten terms. At zero coefficient, bypass the optional computation so an unknown model value cannot contaminate the baseline through zero times NaN.
3. `path_command_statistics.py:COST_KEYS` drives raw/effective/cap tensors and reports. `path_command_training.py` currently also hardcodes ten components in `PathBatch.reset`, the window accumulator, each-update accumulator, and window reset. Change these together to a config-resolved component schema; do not just append a reward dictionary key.
4. `failure_cost_rate_bound` must cover the sum of effective component caps including the new coefficient. Four lower-tick costs sum to one policy interval; an upper-tick proxy must be charged once, with 0.02 s duration, rather than repeated four times at that duration.
5. Training initialization currently enforces checkpoint identity in `restore_checkpoint`, while Coverage mode uses `path_command_fork`. A changed reward requires an explicit new fork contract with matching initialization in the comparison arm. Do not bypass an identity check or present a changed-reward run as exact resume.
6. Before production PPO, verify zero-weight behavior, nonfinite/unknown handling, bounded cost and failure-tail accounting, diagnostics dimensions, and a short actual rollout. Preserve frozen local300 and the 351/352 interfaces. Then compare equal-initialization/equal-step arms using original physical success criteria and independent pulse cases; reward improvement alone is not the endpoint.

Direct input audit: all eight onset-1 positive 0.5 N probes in `cross_state_v1_attempt2` have the identification sample at 1.0 s, finite roll rate, and actual first-tick impulse approximately 0.00249743 N s. This verifies availability/timing of the force-mapping input, not the quality of the resulting predictor. Large-force outcomes were not read for this edit map.
