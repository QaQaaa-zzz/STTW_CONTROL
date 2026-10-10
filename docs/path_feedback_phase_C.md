# Geometric path Phase C — implementation and preparation record

Subsequent explicit user authorization launched the200-update endpoint campaign. See the [dated work report and launch snapshot](evidence/path_feedback_v1_20261010/INDEX.md) for actual progress; the preparation results below retain their original no-training scope.

Authoritative design: [spec](path_feedback_v1_spec.md), [supplied JSON](../learning/configs/STTW_Path_Feedback_V1.json). Executable preparation: [run configuration](../learning/configs/path_feedback_phase_C_run.json). Current research decisions remain in `/home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md`.

The minimum training path is now wired. This is engineering preparation, not trained alpha behavior. Phase B remains two zero-upper closed-path diagnostics; its final-heading failures are preserved and baseline perfection is not a prerequisite for the main-line experiment.

- New schema `sttw_geometric_path_actor351_v1`: Actor351/Critic352, fresh 128→128→64 ELU, zero Actor output layer, independent fixed-alpha0/1 policies. No old345 loading or slot substitution. Frozen local300 config, normalizer, checkpoint, physical identity and timing are checked.
- Reset-only immutable route sampler: 30% straight/gentle, 50% left/right rounded60–100°, 20% S, sampled speed1.8–2.6m/s/radius1.8–4m. Endpoint runs use identical environment keys. Physical prepared bank index3 is unchanged; 20% reset perturbations move only the route anchor once. Ordinary half straight/half30° R4 and S opposite bends with3m separation are explicit engineering resolutions in the run configuration. Fixed100m grid; no online reanchoring or curvature speed regulation.
- Training and evaluation call the same `PathCommandEnv.policy_step`, including actual-pose PP and first-order upper Δv/Δδ filter followed by existing slew. No ECBC/final-actuator filtering. The ten capped geometric reward components retain the supplied5080.2 failure bound.
- Shared existing RSL `DirectCommandPPO` update/storage, separate Actor/Critic rates, initial std[.25,.10], soft/hard KL and epoch rollback; three consecutive rejected batches stop. Finite20s tasks are terminal without bootstrap; live rollout boundaries bootstrap. Physical failure has its prescribed tail. Domain/nonfinite faults preserve state and abort without updating or ranking them as physical failures.
- Immutable full learner checkpoints every10 updates. Resume validates schema/config/lower identity and checkpoint hashes, restores optimizer/RNG and rejection streak, then restarts physical/controller/filter/history state from the prepared bank. Per-environment route counters restart at the next unused key; this is not frame-continuous simulation. Verified parent geometric best is inherited. Completed endpoints skip training; an endpoint with no saved boundary starts fresh. Old teleop best pointers are protected and never written.
- Six fixed DEV cases: straight/left90_R2/right90_R2 ×2.0/2.6m/s,20s each. Candidate evaluations100/150/200, no20/25 evaluation. Ranking order is physical failures, working-limit failures, endpoint-primary normalized RMSE, goal/final-hold failures, worst normalized RMSE. Primary score is continuous, not an invented pass threshold. Goal requires both signed-section crossing and path progress. Qualification covers physical/work/goal/final hold only, not alpha preference. Final report reloads selected best, pairs each alpha with the same-alpha rescored zero-upper trace, and retains XY, nominal/target/filter/correction/governed/actual/lower error, per-step/cumulative reward and signed components. No post-hoc smoothing.

Prepared budget: **200 updates per endpoint**,512 environments×128 policy intervals,13,107,200 transitions per endpoint /26,214,400 total; maximum104,857,600 training lower ticks. Six-case validation/zero-upper/final-best upper bound216,000 additional lower ticks. No automatic extension and no default wall cutoff. Elapsed time and training/validation counts are separate. This budget is a reviewable proposed launch configuration, not evidence that training occurred.

Default command performs preparation only:

```bash
JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python learning/cli/train_path_command.py \
  --config learning/configs/path_feedback_phase_C_run.json \
  --output runs/path_phase_C_new_preparation --check-interface
```

`--execute` is the separate launch switch and was **not used**. Training would create isolated per-endpoint TensorBoard logs, verify the actual reward scalar over HTTP, record its URL/PID, and write `status.json`, `metrics.jsonl`, checkpoint boundaries and notifications. No server was started for preparation. Future GPU compilation/runtime, performance, TensorBoard serving and complete physical PPO rollouts remain unverified; do not call CPU tests a training smoke.

Verification: [preparation receipt](evidence/path_feedback_v1_20261010/phase_C_preparation.json), [engineering receipt](evidence/path_feedback_v1_20261010/phase_C_validation.json). Abstract single/batched step tracing verifies351/352 observation and reset/terminal shapes without executing physics. Targeted tests cover deterministic route sampling, terminal/fault semantics, a real CPU PPO update on synthetic observations, checkpoint identity/counters, geometric selection, saved Stage-B reward reconstruction and terminal-tail accounting. A/B NPZ and all old results were reused unchanged. Review corrected route-counter replay, incumbent best loss, weak goal-plane passage and an unauthorized default wall cutoff; sequential-endpoint resume also skips an already-completed endpoint.

No new vehicle simulation, lower retraining, alpha policy training, adoption or push in this continuation.
