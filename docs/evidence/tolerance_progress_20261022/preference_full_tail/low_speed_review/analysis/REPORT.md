# Frozen single-alpha policy real-pulse evaluation

60-case development diagnostic; fixed local300, original V3 and10s/1s hold. This is a frozen-policy evaluation, not training or final TEST.
Task recovery = no physical/domain failure + original goal + final1s path≤0.1m, heading≤0.05rad, speed≤0.1m/s. Performance additionally requires working roll≤0.302rad.
Zero-force controls may repeat across pulse onsets; unique nominal contexts are20 per policy. Cases are paired deterministic branches, not independent training seeds.
Formula parameters are frozen from earlier dual-network calibration; no fitting to this screen. Predicted outside does not imply unavoidable failure.

| policy | α | task | performance | physical | domain | nominal | predicted-within nonzero | optimistic task errors | conservative task passes |
|---|---|---|---|---|---|---|---|---|---|
| full10 | 0 | 8/12 | 4/12 | 0 | 0 | 4/4 | 0 | 0 | 4 |
| full10 | 0.125 | 9/12 | 5/12 | 0 | 0 | 4/4 | 0 | 0 | 5 |
| full10 | 0.5 | 12/12 | 6/12 | 0 | 0 | 4/4 | 0 | 0 | 8 |
| full10 | 0.875 | 12/12 | 6/12 | 0 | 0 | 4/4 | 0 | 0 | 8 |
| full10 | 1 | 11/12 | 5/12 | 0 | 0 | 4/4 | 0 | 0 | 7 |

Per-context response overlays (all declared alphas; pulse shaded; actual endpoints marked):

- [left120_R2p5_v2.35_t2](curves/left120_R2p5_v2.35_t2.png)
- [left120_R2p5_v2.75_t2](curves/left120_R2p5_v2.75_t2.png)
- [right120_R2p5_v2.35_t2](curves/right120_R2p5_v2.35_t2.png)
- [right120_R2p5_v2.75_t2](curves/right120_R2p5_v2.75_t2.png)

Per-alpha reward/component/wheel-proxy PNG+PDF are alongside each overview. Raw logged arrays remain in the immutable run. No matched pure-ECBC real-pulse control was run in this diagnostic; each pulse response uses the same policy own nominal context.

Source: /home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/preference_full_tail_failure_check_v1/evaluation; code 6d900928c640199f3c75634942a3e1465b96bf75; actual lower ticks 120000/120000; evaluation PPO=0.
