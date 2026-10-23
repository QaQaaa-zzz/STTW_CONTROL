# Frozen single-alpha policy real-pulse evaluation

200-case development diagnostic; fixed local300, original V3 and10s/1s hold. This is a frozen-policy evaluation, not training or final TEST.
Task recovery = no physical/domain failure + original goal + final1s path≤0.1m, heading≤0.05rad, speed≤0.1m/s. Performance additionally requires working roll≤0.302rad.
Zero-force controls may repeat across pulse onsets; unique nominal contexts are20 per policy. Cases are paired deterministic branches, not independent training seeds.
Formula parameters are frozen from earlier dual-network calibration; no fitting to this screen. Predicted outside does not imply unavoidable failure.

| policy | α | task | performance | physical | domain | nominal | predicted-within nonzero | optimistic task errors | conservative task passes |
|---|---|---|---|---|---|---|---|---|---|
| full10 | 0 | 35/40 | 13/40 | 0 | 5 | 8/8 | 4 | 0 | 23 |
| full10 | 0.25 | 38/40 | 13/40 | 0 | 2 | 8/8 | 4 | 0 | 26 |
| full10 | 0.5 | 38/40 | 14/40 | 0 | 1 | 8/8 | 4 | 0 | 26 |
| full10 | 0.75 | 39/40 | 14/40 | 0 | 0 | 8/8 | 3 | 0 | 28 |
| full10 | 1 | 36/40 | 10/40 | 0 | 0 | 8/8 | 3 | 0 | 25 |

Per-context response overlays (all declared alphas; pulse shaded; actual endpoints marked):

- [left90_R1p8_v2.6_t1](curves/left90_R1p8_v2.6_t1.png)
- [left90_R1p8_v2.6_t2](curves/left90_R1p8_v2.6_t2.png)
- [left90_R1p8_v2.8_t1](curves/left90_R1p8_v2.8_t1.png)
- [left90_R1p8_v2.8_t2](curves/left90_R1p8_v2.8_t2.png)
- [right90_R1p8_v2.6_t1](curves/right90_R1p8_v2.6_t1.png)
- [right90_R1p8_v2.6_t2](curves/right90_R1p8_v2.6_t2.png)
- [right90_R1p8_v2.8_t1](curves/right90_R1p8_v2.8_t1.png)
- [right90_R1p8_v2.8_t2](curves/right90_R1p8_v2.8_t2.png)

Per-alpha reward/component/wheel-proxy PNG+PDF are alongside each overview. Raw logged arrays remain in the immutable run. No matched pure-ECBC real-pulse control was run in this diagnostic; each pulse response uses the same policy own nominal context.

Source: /home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/preference_full_tail_post_v1/strong/evaluation; code 6d900928c640199f3c75634942a3e1465b96bf75; actual lower ticks 390440/400000; evaluation PPO=0.
