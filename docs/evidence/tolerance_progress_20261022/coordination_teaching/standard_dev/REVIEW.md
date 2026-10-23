# Coordination teaching: completed development evaluation

[All-alpha nominal actual XY](nominal_review/actual_xy.png)

Fixed final400 actor retains the ordinary351D alpha-conditioned two-channel interface, frozen local300, V3 reward and control limits. No mask or oracle is present during evaluation.

| Panel | Parent tasks | Taught tasks | Parent working | Taught working | Domain exits parent→taught |
|---|---:|---:|---:|---:|---:|
| Moderate |200/200|200/200|143|95|0→0|
| Strong |189/200|195/200|67|57|9→4|
| Straight |10/10|10/10|10|10|0→0|
| Low-speed DEV |53/60|58/60|26|28|0→0|

The original low-speed target subset improves1/8→6/8. Across the declared panels11 paired task gains and0 losses; repeated zero-force/onset contexts mean470 cases are not independent statistical replicates. Do not call this a final-test success probability. Standard fresh evaluation used815232 lower ticks; low-speed120000; prior TRAIN check48000. Costs of probe144000 and record collection144000 are additional. Supervised optimization has400 updates,0physics/PPO interactions.

Unique standard nominal contexts:30/30 tasks in both parent and taught. All four turn conditions preserve strict speed-RMSE decrease over alpha0,.25,.5,.75,1. Taught path endpoint preference direction4/4; intermediate path errors are not claimed monotonic. No online-alpha-switching test has been performed.

There is a real tradeoff: nominal working30→24. Six2.8m/s cases reach peak roll0.3094–0.3250rad. Moderate48 working losses have peaks0.3023–0.3472rad. Strong task recovery improves while working67→57. Domain-exit reductions are measured; trajectories after remaining domain exits are physically UNKNOWN, not proven falls or successful survival.

## Decision

Retain taught final400 as the leading development recovery candidate, retaining parentu325 as immutable comparison/fallback. Do not discard the recovery gain solely because slightly stricter working thresholds turn red; do not hide those costs. This is not deployment certification or proof of broad superiority. No further reward/coverage/limit sweeps now.

Next verify reproducibility with two additional fixed supervised seeds on the same authenticated dataset and fixed final400 rule. Evaluate the same development protocol, keep training-domain and DEV evidence separate, then freeze an independent evaluation panel before further tuning. The tolerance model remains an auxiliary directional diagnostic; these gains came from state-conditioned teaching, not demonstrated tolerance shaping. If a candidate is retained, its measured tolerance must be assessed for that closed-loop policy rather than reusing parent calibration as a guarantee.

## Evidence

- numeric_results.json: three standard panels and matched outcomes.
- nominal_review/summary.json:90 unique policy/context rows and trace hashes.
- paired_continuous_review.json: strong improvements and alpha-specific working counts.
- moderate/working_regression_review.json: continuous roll costs of threshold crossings.
- Panel analysis directories: full saved reward/component/XY reports.
- ../preference_coordination_teaching_low_speed_dev_v1/analysis: complete low-speed results.

One training seed, one reset snapshot lineage and development-used panels limit inference. No new literature novelty claim is made by this experiment.
