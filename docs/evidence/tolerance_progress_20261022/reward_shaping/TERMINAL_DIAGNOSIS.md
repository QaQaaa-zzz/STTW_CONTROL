# Tolerance comparison: terminal recovery diagnosis

All six training jobs completed 50 updates (19,660,800 policy transitions total). Evaluation saved 1152/1152 cases. This diagnostic recomputes final-second errors from saved NPZ trajectories; no new training or physics. Original result labels are unchanged. Incomplete trajectories fail the practical diagnostic.

| Alpha | Policy | Original task | Practical diagnostic | Final path error range (m) |
|---|---|---:|---:|---:|
| 0 | baseline | 133/144 | 142/144 | .0124–.0153 |
| 0 | static | 134/144 | 144/144 | .0011–.0309 |
| 0 | dynamic | 103/144 | 139/144 | .0100–.0164 |
| 1 | baseline | 141/144 | 144/144 | .0401–.0506 |
| 1 | static | 0/144 | 144/144 | .1088–.1319 |
| 1 | dynamic | 0/144 | 144/144 | .1019–.1342 |

Practical diagnostic: last second (time>=19s, 200 samples), maximum absolute path error<=.15m, heading error<=.05rad, speed error<=.1m/s, peak roll<=.33rad, reached goal, no physical/domain failure. This is post-hoc sensitivity informed by the user's acceptance of small errors, not an independent acceptance test or safety certificate. Ranges above use complete trajectories only; incomplete cases are retained as failures in counts. Nominal rows are repeated across onset groups.

Alpha1 static/dynamic original failures are terminal path-threshold failures, not falls: all 144 reached goal, no physical/domain failures; speed error <=.0111/.0202m/s, heading <=.0011rad. A path bias 1–3.4cm above the original 10cm limit explains 0/144. Do not reject these models solely on this binary label.

Nevertheless usefulness is mixed. Alpha0 dynamic loses 30 original successes versus baseline, with 38 working-roll failures vs9; its practical count remains lower (139 vs142). Alpha0 static gains only1 original success (2 improved,1 regressed), practical gains2. Alpha1 dynamic roll p95 decreases .2864 to .2798rad and maximum .3140 to .3072rad, but path RMSE increases .2656 to .2828m and speed RMSE .0302 to .0336m/s. Static alpha1 path RMSE decreases to .2457m but terminal bias increases. These full-trace case-mean RMSEs have early-termination caveats in alpha0.

Decision: accept alpha1 residual biases as tolerable for development; do not retrain merely to turn original labels green. Retain baseline as control, static as a modest candidate, and dynamic as mixed/negative evidence. No broad robustness-improvement claim is supported by this one-seed panel. Next analyze pulse-induced degradation against each policy's matched zero-force trace to separate nominal behavior changes from perturbation recovery; use saved trajectories first, with no extra training. Preserve the alpha325 single-network development selection separately.
