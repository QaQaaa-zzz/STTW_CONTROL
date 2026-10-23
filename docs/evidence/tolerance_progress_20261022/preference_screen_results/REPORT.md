# Frozen single-alpha u325 real-pulse screen

200-case development diagnostic; fixed local300, original V3 and10s/1s hold. This is a new frozen-policy evaluation, not training or final TEST.
Task recovery = no physical/domain failure + original goal + final1s path≤0.1m, heading≤0.05rad, speed≤0.1m/s. Performance additionally requires working roll≤0.302rad.
Zero-force controls appear twice across pulse onsets; nominal unique contexts are20. Cases are paired deterministic branches, not independent training seeds.
Formula parameters are frozen from earlier dual-network calibration; no fitting to this screen. Predicted outside does not imply unavoidable failure.

| α | task | performance | physical | domain | nominal | predicted-within nonzero | optimistic task errors | conservative task passes |
|---|---|---|---|---|---|---|---|---|
| 0 | 40/40 | 33/40 | 0 | 0 | 8/8 | 14 | 0 | 18 |
| 0.25 | 40/40 | 31/40 | 0 | 0 | 8/8 | 16 | 0 | 16 |
| 0.5 | 40/40 | 28/40 | 0 | 0 | 8/8 | 18 | 0 | 14 |
| 0.75 | 40/40 | 25/40 | 0 | 0 | 8/8 | 19 | 0 | 13 |
| 1 | 40/40 | 26/40 | 0 | 0 | 8/8 | 19 | 0 | 13 |

Per-context response overlays (all declared alphas; pulse shaded; actual endpoints marked):

- [left90_R1p8_v2.6_t1](curves/left90_R1p8_v2.6_t1.png)
- [left90_R1p8_v2.6_t2](curves/left90_R1p8_v2.6_t2.png)
- [left90_R1p8_v2.8_t1](curves/left90_R1p8_v2.8_t1.png)
- [left90_R1p8_v2.8_t2](curves/left90_R1p8_v2.8_t2.png)
- [right90_R1p8_v2.6_t1](curves/right90_R1p8_v2.6_t1.png)
- [right90_R1p8_v2.6_t2](curves/right90_R1p8_v2.6_t2.png)
- [right90_R1p8_v2.8_t1](curves/right90_R1p8_v2.8_t1.png)
- [right90_R1p8_v2.8_t2](curves/right90_R1p8_v2.8_t2.png)

Per-alpha reward/component/wheel-proxy PNG+PDF are alongside each overview. Raw logged arrays remain in the immutable run. No matched pure-ECBC real-pulse control was run in this diagnostic; comparisons here use the same u325 own nominal context.

Source: runs/preference_tolerance_screen_v1/evaluation; code 9a724199abae8ccdc932af3cf3815854681a72e6; actual lower ticks 400000/400000; PPO=0.
