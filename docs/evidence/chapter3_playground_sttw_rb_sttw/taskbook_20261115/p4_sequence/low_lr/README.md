# P4 lower-rate continuation: handoffs recovered, calibrated planning still incomplete

All actual/reference/predicted XY across six evaluated candidates: [36 map06](map06/README.md), [36 map07](map07/README.md), [18 mechanism candidates](map48/README.md). [Learning curves](learning_curves.png) preserve every evaluated checkpoint and failed last states.

**Meaningful model progress, no reliable planner adopted.** State-consistent handoffs recover from0–3/72 to72/72 for several candidates. Full-chain raw admission reaches68/72 for the preselected screen leader. A separate seed1 loss-best reaches4/4 true admissions with0/14 false admissions on the known mechanism map, but its coverage is only53/72 on the original DEV set. Own calibrated systems still reject all mechanism candidates.

The guarded-entry diagnostic independently showed original/displacement RB models fail even from real states/history. Three seeds then continued from their displacement loss-best checkpoints for1000 supervised updates, changing optimizer learning rate3e-4→3e-5 while preserving Adam moments, architecture, loss, data split, normalizers, estimated-entry exposure and all robot/physics contracts. This is extended training with a reduced learning rate, **not a matched-budget learning-rate ablation**. Runtime76.28/77.04/76.58s; all123 checkpoints saved every25 updates, TensorBoard6006 verified38 tags per seed. Zero new policy training or physical trajectories.

|Seed|Parent update /loss|Loss-best update /loss|Last loss|Last10 checkpoint loss range|
|---|---|---|---|---|
|0|2475 /27.73|3475 /2.890|2.890|2.890–17.495|
|1|2450 /43.82|3450 /2.485|2.485|2.485–9.878|
|2|2250 /18.09|2625 /3.475|5.204|3.749–17.316|

Loss improves substantially but still fluctuates; sustained convergence is not established. Screening all123 saved models by predicted state-event readiness on original DEV72, then mean joint position error, selected S0/3475,S0/3425,S1/2725. Mechanism outcomes did not enter that ranking. All loss-best and last checkpoints were also evaluated, deduplicated to six candidates:

|Candidate|Raw DEV admission /72|Mechanism true admit /4|False admit /14|Selected completion /2|
|---|---:|---:|---:|---:|
|S0/3475 (screen leader)|68|2|1|0|
|S0/3425|67|4|1|1|
|S1/2725|69|2|1|1|
|S1/3450 (seed1 loss-best)|53|4|0|2|
|S2/2625 (seed2 loss-best)|11|0|0|0|
|S2/3250 (seed2 last)|13|0|0|0|

The raw false admission for S0/3475 is root1002/route1: actual collision before RB at1.93s, recorded clearance−0.00345m. Root1001 has no plan. This is not a jump recovery failure. For S1/3450, C and I both select successful route3 for both roots; C rejects an extra infeasible candidate that I admits, but this does not improve their selected2/2 outcomes. G has no plan and Oracle2/2. Same cached predictor/constraints/candidate strong backtracking and exhaustive choices agree throughout. The recorded zero clearance in successful physical trajectories is not a positive clearance certificate.

Matched current comparisons: [S0/3475](comparison/seed0_update3475/README.md) and [S1/3450](comparison/seed1_update3450/README.md). I original-DEV admission is12/72 for S0 and72/72 for S1. These checkpoint-dependent differences do not prove C generally beats I. New gated results must not be mixed with old ungated baselines.

Calibration keeps the existing per-spatial-region/per-axis maximum AABB support miss+0.02m rule, with own/shared/cross diagnostics. The first historical-mode fit included unsupported horizon suffixes; its outputs are retained under `calibration/`. The added `--supported-only` fit excludes candidates with no predicted handoff and explicitly records every inclusion/exclusion; excluded candidates remain rejected, not relabeled as successes. It uses no actual-success filter and does not manually shrink margins.

|Model|Valid C/I calibration cases /72|Post-region lateral C/I margin(m)|Own/shared mechanism admission|
|---|---|---|---|
|S0/3475|72 /12|0.220 /0.152|0/18 for both methods|
|S1/3450|54 /72|0.338 /0.243|0/18 for both methods|

Only S0-C using the cross-I margin admits1/4 feasible mechanism candidates,0/14 failures. That cross-margin was fitted on just12 supported I cases and is a diagnostic, not an adopted uncertainty bound. All original cases and fit-domain exclusions remain visible. Margins are empirical spatial diagnostics, not safety guarantees, and unsupported suffixes do not represent an execution plan.

[Candidate results](candidate_summary.json), [123-checkpoint screen](screen_checkpoints.json), [training summary](training_summary.json), [checkpoint hashes](checkpoint_identity.json), [S0 conditional calibration](supported_calibration/seed0_update3475/summary.json), [S1 conditional calibration](supported_calibration/seed1_update3450/summary.json). Detailed geometry rejections and fit exclusions accompany those files. Raw commands/logs/checkpoints/NPZ: `STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p4_low_lr`.

All results are known DEV, reuse actual frozen trajectories and include shared prefixes; no independent TEST or physical improvement is claimed. P4 remains incomplete and P5/P6 have not started. No more training is scheduled from this budget. Next inspect which valid-domain geometry errors produce calibrated false rejections, separately from actual narrow-clearance candidate limits; preserve the successful raw mechanism case without claiming robust calibrated planning.
