# Frozen single-alpha policy real-pulse evaluation

120-case development diagnostic; fixed local300, original V3 and10s/1s hold. This is a frozen-policy evaluation, not training or final TEST.
Task recovery = no physical/domain failure + original goal + final1s path≤0.1m, heading≤0.05rad, speed≤0.1m/s. Performance additionally requires working roll≤0.302rad.
Zero-force controls may repeat across pulse onsets; unique nominal contexts are20 per policy. Cases are paired deterministic branches, not independent training seeds.
Formula parameters are frozen from earlier dual-network calibration; no fitting to this screen. Predicted outside does not imply unavoidable failure.

| policy | α | task | performance | physical | domain | nominal | predicted-within nonzero | optimistic task errors | conservative task passes |
|---|---|---|---|---|---|---|---|---|---|
| u325 | 0 | 9/12 | 8/12 | 0 | 2 | 4/4 | 0 | 0 | 5 |
| u325 | 0.125 | 10/12 | 9/12 | 0 | 2 | 4/4 | 0 | 0 | 6 |
| u325 | 0.5 | 11/12 | 10/12 | 0 | 1 | 4/4 | 0 | 0 | 7 |
| u325 | 0.875 | 11/12 | 8/12 | 0 | 0 | 4/4 | 1 | 0 | 6 |
| u325 | 1 | 10/12 | 8/12 | 0 | 0 | 4/4 | 0 | 0 | 6 |
| shuffled | 0 | 9/12 | 8/12 | 0 | 2 | 4/4 | 0 | 0 | 5 |
| shuffled | 0.125 | 10/12 | 9/12 | 0 | 2 | 4/4 | 0 | 0 | 6 |
| shuffled | 0.5 | 11/12 | 10/12 | 0 | 1 | 4/4 | 0 | 0 | 7 |
| shuffled | 0.875 | 11/12 | 9/12 | 0 | 0 | 4/4 | 1 | 0 | 6 |
| shuffled | 1 | 11/12 | 8/12 | 0 | 0 | 4/4 | 1 | 0 | 6 |

Per-context response overlays (all declared alphas; pulse shaded; actual endpoints marked):

- [shuffled_left60_R2p2_v2.35_t2](curves/shuffled_left60_R2p2_v2.35_t2.png)
- [shuffled_left60_R2p2_v2.75_t2](curves/shuffled_left60_R2p2_v2.75_t2.png)
- [shuffled_right60_R2p2_v2.35_t2](curves/shuffled_right60_R2p2_v2.35_t2.png)
- [shuffled_right60_R2p2_v2.75_t2](curves/shuffled_right60_R2p2_v2.75_t2.png)
- [u325_left60_R2p2_v2.35_t2](curves/u325_left60_R2p2_v2.35_t2.png)
- [u325_left60_R2p2_v2.75_t2](curves/u325_left60_R2p2_v2.75_t2.png)
- [u325_right60_R2p2_v2.35_t2](curves/u325_right60_R2p2_v2.35_t2.png)
- [u325_right60_R2p2_v2.75_t2](curves/u325_right60_R2p2_v2.75_t2.png)

Per-alpha reward/component/wheel-proxy PNG+PDF are alongside each overview. Raw logged arrays remain in the immutable run. No matched pure-ECBC real-pulse control was run in this diagnostic; each pulse response uses the same policy own nominal context.

Source: /home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/preference_task_generalization_v1/angle60_R2p2/evaluation; code e9f4b8954111123edab6449de9d31528aefbf0ed; actual lower ticks 227804/240000; evaluation PPO=0.
