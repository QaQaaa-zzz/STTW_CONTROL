# Predicted-entry exposure: bounded RB world-model continuation

All actual/reference/best/last XY: [map06,36 conditions](map06/README.md), [map07,36](map07/README.md), [mechanism map48,18 candidates](map48/README.md). S0/S1/S2 are the retained checkpoints; Before is the original residual model with execution-consistent history reset. L0/L2 show failed last checkpoints on map48.

|Candidate|DEV admitted /72 actual feasible|Map48 true admit /4|Map48 false admit /14|Selected map48 completion /2|
|---|---:|---:|---:|---:|
|Before|44|0|0|0|
|S0 best2425|56|2|5|0|
|S1 best2600|30|3|0|2|
|S2 retained2175|26|1|0|1|
|S0 last2575|not evaluated|0|0|0|
|S2 last2475|not evaluated|0|0|0|

No candidate is adopted as a reliable planner. S1 has a local2/2 benefit but loses14 feasible admissions on the original DEV set; S0 admits five actually infeasible mechanism candidates. S2's best is its starting checkpoint, with no accepted improvement.

Three seeds each received300 supervised updates with frozen upstream-estimated physical/history states replacing50percent of sampled TRAIN full-entry initializations. Labels remain actual RB-relative sequences; initial timing uncertainty is not eliminated. Additional transition loss scales hip/knee position by0.01rad and velocity by0.1rad/s, from the original lock contract. World-model weights changed; robot policies/physics/task gates did not. Original normalizers remain TRAIN-only, map split unchanged.

[Checkpoint curves](learning_curves.png) and [best/last comparison](checkpoint_comparison.json) show non-convergence: seed0 loss77.42→best39.52→last358.71; seed1 56.36→25.49; seed2 44.11→last845.98. Best selection uses the mean true-entry/predicted-entry DEV loss under this new loss, not the old score. The old auxiliary event readout is not reused with changed hidden states.

Even S1 predicts hip≈−1.24rad at phase2 versus home≈−1.20rad; current raw phase-classifier decisions do not prove lock-ready state consistency. Known DEV only; all actual labels reuse existing full-snapshot/collection traces. No new physical episodes or independent TEST. P4 remains incomplete; do not expand to P5/P6 or shrink calibrated margins.
