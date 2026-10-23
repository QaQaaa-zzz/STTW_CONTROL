# Frozen single-alpha policy real-pulse evaluation

48-case development diagnostic; fixed local300, original V3 and10s/1s hold. This is a frozen-policy evaluation, not training or final TEST.
Task recovery = no physical/domain failure + original goal + final1s path≤0.1m, heading≤0.05rad, speed≤0.1m/s. Performance additionally requires working roll≤0.302rad.
Zero-force controls may repeat across pulse onsets; unique nominal contexts are4 per policy. Cases are paired deterministic branches, not independent training seeds.
Formula parameters are frozen from earlier dual-network calibration; no fitting to this screen. Predicted outside does not imply unavoidable failure.

| policy | α | task | performance | physical | domain | nominal | predicted-within nonzero | optimistic task errors | conservative task passes |
|---|---|---|---|---|---|---|---|---|---|
| control | 0 | 2/6 | 2/6 | 0 | 0 | 2/2 | 0 | 0 | 0 |
| control | 0.125 | 3/6 | 3/6 | 0 | 0 | 2/2 | 0 | 0 | 1 |
| speed_off | 0 | 3/6 | 2/6 | 0 | 0 | 2/2 | 0 | 0 | 1 |
| speed_off | 0.125 | 5/6 | 3/6 | 0 | 0 | 2/2 | 0 | 0 | 3 |
| steer_off | 0 | 4/6 | 2/6 | 0 | 0 | 2/2 | 0 | 0 | 2 |
| steer_off | 0.125 | 4/6 | 2/6 | 0 | 0 | 2/2 | 0 | 0 | 2 |
| both_off | 0 | 2/6 | 2/6 | 0 | 2 | 2/2 | 0 | 0 | 0 |
| both_off | 0.125 | 3/6 | 2/6 | 0 | 2 | 2/2 | 0 | 0 | 1 |

Per-context response overlays (all declared alphas; pulse shaded; actual endpoints marked):

- [both_off_left120_R2p5_v2.35_t2](curves/both_off_left120_R2p5_v2.35_t2.png)
- [both_off_right120_R2p5_v2.35_t2](curves/both_off_right120_R2p5_v2.35_t2.png)
- [control_left120_R2p5_v2.35_t2](curves/control_left120_R2p5_v2.35_t2.png)
- [control_right120_R2p5_v2.35_t2](curves/control_right120_R2p5_v2.35_t2.png)
- [speed_off_left120_R2p5_v2.35_t2](curves/speed_off_left120_R2p5_v2.35_t2.png)
- [speed_off_right120_R2p5_v2.35_t2](curves/speed_off_right120_R2p5_v2.35_t2.png)
- [steer_off_left120_R2p5_v2.35_t2](curves/steer_off_left120_R2p5_v2.35_t2.png)
- [steer_off_right120_R2p5_v2.35_t2](curves/steer_off_right120_R2p5_v2.35_t2.png)

Per-alpha reward/component/wheel-proxy PNG+PDF are alongside each overview. Raw logged arrays remain in the immutable run. No matched pure-ECBC real-pulse control was run in this diagnostic; each pulse response uses the same policy own nominal context.

Source: runs/preference_control_intervention_v1/evaluation; code 7d4c2b4d14320ef4c30fcf2544693f7cd08bc882; actual lower ticks 92024/96000; evaluation PPO=0.
