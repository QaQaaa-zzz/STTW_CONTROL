# Direct command policy V3

The sole V3 method is one shared alpha-conditioned Actor (345 inputs, 69186 mean parameters, two Gaussian latents). It changes speed/steer references relative to the currently issued raw command. The original ECBC, ESO and bounded actuator residual interface remain. There is no parameter governor, q-based action projection, analytic heading controller or candidate rollout search in the V3 execution path.

Numerical authority: [STTW_Direct_Command_V3.json](../../learning/configs/STTW_Direct_Command_V3.json). Protocol: [attached specification](STTW_Codex_Direct_Command_V3.md). Implementation parent: local teleop `92afde6`; implementation snapshot: `8211d89`. Changes to shared teleop helpers preserve old defaults and add only an explicit prefiltered reference passthrough and zero-residual actuator-preview diagnostics.

Run in this worktree using the already installed environment:

```bash
PYTHONPATH=learning/src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/home/qy/mujoco_playground/.venv/bin/python learning/cli/train_direct_command.py \
  --config learning/configs/STTW_Direct_Command_V3.json \
  --output runs/NEW_UNIQUE_RUN --updates 20 --compute-wall-budget 1800
```

`--dry-run` checks the exact frozen configuration and creates status/budget without physics. Unknown or altered schema values and updates outside 1..20 are rejected. Existing run identities are immutable; this command cannot resume or overwrite a completed or interrupted run. No remote push occurs.

The command launches one owned worker with an external supervisor. Persistent stage ceilings are compile 300 s, prepared states plus interface checks plus engineering smoke 120 s, pilot 1200 s, review 180 s; total new compute 1800 s. Tests and interrupted attempts are charged to the same ledger. The two interface checks are capped at 4 simulation seconds, engineering smoke is 8x16x2, and pilot starts a fresh 512x128 shared policy. Normal finite task ends have no bootstrap; live rollout cuts preserve complete closed-loop state. The pipeline saves each completed checkpoint to preserve the endpoint at any budget stop, with 5/10/15/20 included.

Review uses the prescribed prepared state (2.3 m/s initial setpoint, zero initial roll), last completed pilot checkpoint, and fixed main/random88001 command streams. Three methods run as independent batched physical trajectories. Each 1 s simulated chunk saves all 5 ms samples, so an interrupted review leaves a real prefix. The per-episode 60 s and total-review 180 s ceilings are unchanged by batching. Failure stops that environment at its true endpoint.

Authoritative artifacts in the run:

- `manifest.json`, `frozen_config.json`, `observation_action_schema.json`: source/config/model/interface identities.
- `status.json`, `budget.json`, `launch.json`, `execution.log`: current stage, durable accounting and stop reason.
- `prepared_bank.pkl`, `prepared_metrics.json`, `interface_traces.npz`, `interface_checks.json`: full state and short interface evidence.
- `smoke/`, `pilot/`: separate fresh policies, metrics, finite checkpoints with optimizer and RNG, deterministic environment case manifests, two logged diagnostic environments per update.
- `tensorboard/`: separate phase scalars; TensorBoard HTTP service is recorded and verified by the execution operator.
- `review/`: frozen schedules, B0/alpha0/alpha1 physical traces, actual-observation alpha sensitivity, same-window metrics and PNG/PDF plots, plus explicit missing/partial evidence.

The review overview plots realized final-command change, computed from `actual_normalized_residual × [1.5, 10]` rad/s. `applied_residual` is the bounded additive request before the final actuator clip and is reported separately in `metrics.json` and each step CSV. For each of the two declared cases, `*_rewards.png/pdf` compares the three available methods' actual 5 ms scored reward and cumulative reward. Per-method `*_components.png/pdf` shows raw and effective cost rates and their cumulative component integrals. Curves stop at their observed physical failure or partial endpoint; failure reward replacement appears in scored reward, not in normal interval component costs.

Reward plots pair α0 policy with B0 scored under α0 and α1 policy with the same B0 physical trajectory rescored under α1. The derived `B0_rescored_alpha1.npz` and CSV preserve the physical trace, reweight speed/steer components from the frozen JSON, recalculate the cap and retain the whole 20 ms failure replacement. Rescoring first audits B0's stored α0 components and scored reward. If that audit fails or required fields are absent, the report labels the α1 baseline unavailable and does not draw a false comparison.

Training reward, gradient changes, latent differences and survival are not substitutes for the prescribed actual-speed/steer, work-range and unwrapped-heading results. One seed and six planned review episodes are a pilot, not convergence, generalization or safety proof. No real vehicle execution is authorized by this run.


## Explicitly authorized 500-update trial

After the original20-update pilot, the user requested500 updates. The separate
`learning/configs/direct_command_500.json` declares fresh512x128x500 sampling,
32,768,000 policy transitions,131,072,000 control ticks maximum,15000s training and
16200s total compute. It changes only run-length/sample budgets; the frozen V3 method,
reward, physics, initialization and restricted final-review contract are unchanged.
The CLI derives omitted update/wall arguments from the selected frozen config and
rejects values beyond that config. The original JSON still defaults to20/1800.
No checkpoint import or automatic further continuation is introduced.
