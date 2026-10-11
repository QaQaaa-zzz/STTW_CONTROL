# Path safety preference v1 — implementation evidence

Historical XY entry (prior Phase C, **not new safety-stage results**):
[Six-case actual/reference XY](/home/qy/STTW_CONTROL/runs/worktrees/path-feedback-v1/docs/evidence/path_feedback_v1_20261010/phase_C_results_20261011/ALL_XY.png).

New-stage fixed-six DEV and natural-overlimit recovery panels: **NOT_RUN**.
This delivery qualifies implementation wiring only; no policy/adoption claim.

- [Contract and runnable handoff](../../path_safety_preference.md)
- [CPU verification](verification.json): 110 passed, 13.92s.
- [One discarded GPU check](receipt.json): 8x16, 128 policy transitions,
  512 lower ticks, 1 optimizer update, 4 accepted epochs, learner discarded.
- [Completed engineering status](status.json): 53.043s total;
  setup/compile42.376s, sampling10.388s, optimization0.259s.
- [Actual reward HTTP receipt](engineering_reward_http_verified.json):
  [TensorBoard](http://localhost:6007), engineering reward step16 loaded.

The GPU check uses archived peak-review alpha0 best100 weights, fresh
Critic/Adam/RNG and the candidate reward. It is **not** a frozen current
Coverage parent. Complete real closed-loop batch state pickle roundtrip and
one identical cloned-state paired continuation pass within the same512-tick
budget. No additional replay, broad DEV, formal arm, or recovery physics.
The final diagnostic rho log field was added after this GPU check and covered
by final CPU verification; no numerical reward/physics changes followed it.

The integration review found and fixed component field naming, resumed DEV0
cache preservation, terminal/TensorBoard gates, original cached path attestation
and exact turn boundary coverage. Frozen parent receipts now include six
trace/path SHA pairs. A CPU offline end-to-end initial0 selection/resumed
final-panel reuse passes; modified geometry is rejected before registration.

Formal parents remain unfrozen until current Coverage **pipeline** completes.
Future control/candidate share frozen parent Actor/log_std and the same new
Critic/Adam/RNG seed; both use safety selection and unchanged CoverageV2
sampling. Two paired-endpoint plans = four50-update arms. Formal budgets,
qualification, plots and natural recovery remain an independent stage.

[Final independent source review and fix receipt](review.md): clean; repeated-resume0 candidate and recovery arm provenance gates corrected.
