# Reward mechanism audit

## Algebraic correction

For the scalar model, lambda²=a4 and z=(a2*delta+a4*phi+d_hat)/lambda+phi_dot. Under modeled steady-turn equilibrium a2*delta+a4*phi+d_hat=0 and phi_dot=0, z=0 even when phi and delta are nonzero. Therefore the earlier suspicion that ordinary lean is automatically penalized is NOT established. Simply subtracting nominal lean could break the existing equilibrium cancellation.

For feasible z, min(F_plus,F_minus)=(sqrt(E_budget*W)-abs(z))/abs(K). The dynamic penalty is a clipped penalty on abs(z) relative to this speed-dependent radius. For locally fixed speed, modeled phi_ddot=a1*delta_dot+a2*delta+a4*phi+d_hat implies z=(phi_ddot-a1*delta_dot)/lambda+phi_dot. Thus transients, steering rate and disturbance estimation can affect the penalty. This is an algebraic observation, not identified causal explanation of learned behavior. Near the nominal radius, clipping may reduce useful distinctions; measured saturation is not yet established.

## Existing training measurements

Effective tolerance cost rate, update1 -> update50 (training distributions differ across updates; these are optimization telemetry, not paired capability measurements):

| Arm | Alpha | Initial | Final |
|---|---:|---:|---:|
| static | 0 | .0037 | .0047 |
| dynamic | 0 | 2.0928 | .4748 |
| static | 1 | .0020 | .0064 |
| dynamic | 1 | 1.0264 | .3011 |

Static shaping is almost inactive on average. Dynamic shaping reduced its proxy cost, while the held-out-from-optimization development pulse panel did not improve overall rejection. Successful optimization of this cost does not establish useful robustness shaping. These observations do not prove formula inaccuracy or reward exploitation.

## Concrete next step

1. Keep baseline controls, alpha325 single-network candidate and all negative results intact. Do not extend the same PPO run.
2. Recompute nominal reserve at saved complete onset snapshots with exact physical measurement and controller ESO state, using the original controller configuration. Distinguish straight motion, turn entry and established turn; compare penalty activity with actual pulse response on the already collected contexts.
3. Only after that check, choose between a lower-weight/transition-aware use of the reserve and a stronger task-restoration reward. The available V3 reward is a candidate for task-restoration comparison, not an already demonstrated fix. Changing nominal lean reference without checking equilibrium is rejected.
4. Small alpha1 terminal path bias (about10–13cm) is accepted for development; do not launch a large training run merely to satisfy the historical10cm gate. The unresolved issue is actual incremental robustness benefit.

No physics, training or production model changed in this audit.
