# Exact onset-state reserve audit

Loaded 128 saved complete nominal snapshots across parent/baseline/static/dynamic, two alphas, four route-speed combinations and four onsets. Used original measurement helper, saved controller ESO, and verified controller parameters against source manifest. CPU only; zero physics steps. The same hypothetical dynamic cost is evaluated on every policy.

| Alpha | Policy | Mean dynamic cost | Median minimum reserve (N) | Saturated / n |
|---|---|---:|---:|---:|
| 0 | baseline_update50 | 2.1452 | 5.8871 | 0/16 |
| 0 | static_update50 | 1.9861 | 6.3362 | 0/16 |
| 0 | dynamic_update50 | 2.0432 | 6.1782 | 0/16 |
| 1 | baseline_update50 | 0.8056 | 7.7053 | 0/16 |
| 1 | static_update50 | 0.8194 | 7.7666 | 0/16 |
| 1 | dynamic_update50 | 0.9633 | 7.9773 | 0/16 |

No saturated costs in these sampled states. This does not establish absence of saturation elsewhere in training. Dynamic-policy reserve median exceeds baseline slightly, but matched pulse-response roll errors did not improve. Reserve is a proxy, not a validated ordering of actual learned-policy robustness. Policy-specific snapshots differ in state, so this does not identify causality.

Decision: no further calibration precision gate; no identical PPO extension. Preserve this negative reward-integration result. Next experiment should test a concrete change in how reserve guides training (e.g. context selection for actual perturbation recovery), with actual recovery as the objective, against an equal-interaction baseline. Reward-only proxy reduction is insufficient evidence. Formulation and training-distribution checks are required before launch; no new experiment started here.
