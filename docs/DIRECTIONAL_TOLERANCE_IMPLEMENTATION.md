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
