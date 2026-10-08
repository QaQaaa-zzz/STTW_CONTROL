# Direct command V3 review

B0 = ECBC+ESO plus the pinned frozen residual Actor (lower alpha=1); upper alpha0/1 share that same lower controller. The lower causal governed path is a transfer adapter, not a reproduction of the source bend task.

Main preference: **criteria_not_met**. 
Comparative criteria require complete, matched 5 ms windows for B0, α0 and α1.
The main conflict window is [2.5, 4.5) s; final quality uses [14, 16) s.
Recovery timing starts when the raw steer reaches |δc|≤0.005 after the final return target; the criterion is the first qualifying 0.5 s hold for each learned method.
Final-window hold is descriptive. Physical failure invalidates it. XY is displayed without a position criterion.
Reward plots compare each learned alpha with B0 rescored under that same alpha. B0 alpha1 uses the same saved physical trace, with audited component reweighting and the same whole-policy-interval physical-failure replacement.

## main

| Method | Status | Observed ticks | Physical failure | Final hold (descriptive) |
|---|---|---:|---|---|
| B0 | complete | 3200 | False | False |
| pi_alpha0 | complete | 3200 | False | False |
| pi_alpha1 | complete | 3200 | False | False |

B0 alpha1 rescore: **audited**.


| Method | Conflict speed RMSE (m/s) | Conflict steer RMSE (rad) | Last 2 s speed RMSE (m/s) | Last 2 s steer RMSE (rad) | First recovery (s) | Peak |φ| (rad) |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.00937 | 0.12914 | 0.23116 | 0.23196 | — | 0.37532 |
| pi_alpha0 | 0.13613 | 0.09146 | 0.24887 | 0.19807 | — | 0.43727 |
| pi_alpha1 | 0.1375 | 0.09149 | 0.25093 | 0.19846 | — | 0.43775 |

Residual peaks below use rad/s. Realized means final command minus zero-residual final command, after actuator clipping; bounded request is before that clip.

| Method | Realized front/rear peak | Bounded request front/rear peak |
|---|---:|---:|
| B0 | 0.96905 / 1.34921 | 0.96905 / 1.34921 |
| pi_alpha0 | 1.5 / 5.77148 | 1.5 / 5.77148 |
| pi_alpha1 | 1.5 / 5.76795 | 1.5 / 5.76795 |

Clip fractions are reconstructed from saved requested residuals and final commands. The original logged flags remain visible because exact float inequality can flag rounding alone.

| Method | Residual true / logged | Final command true / logged |
|---|---:|---:|
| B0 | 0.0 / 0.0 | 0.0 / 0.0 |
| pi_alpha0 | 0.025 / 0.025 | 0.0 / 0.00031 |
| pi_alpha1 | 0.025 / 0.025 | 0.0 / 0.00031 |

[Metrics](metrics.json) · [Plot PNG](main_comparison.png) · [Plot PDF](main_comparison.pdf) · [Per-tick and cumulative reward](main_rewards.png) ([PDF](main_rewards.pdf))

[B0 raw/effective cost components](main_B0_components.png) ([PDF](main_B0_components.pdf))
[pi_alpha0 raw/effective cost components](main_pi_alpha0_components.png) ([PDF](main_pi_alpha0_components.pdf))
[pi_alpha1 raw/effective cost components](main_pi_alpha1_components.png) ([PDF](main_pi_alpha1_components.pdf))

## random

| Method | Status | Observed ticks | Physical failure | Final hold (descriptive) |
|---|---|---:|---|---|
| B0 | complete | 3200 | False | False |
| pi_alpha0 | complete | 3200 | False | False |
| pi_alpha1 | complete | 3200 | False | False |

B0 alpha1 rescore: **audited**.


| Method | Conflict speed RMSE (m/s) | Conflict steer RMSE (rad) | Last 2 s speed RMSE (m/s) | Last 2 s steer RMSE (rad) | First recovery (s) | Peak |φ| (rad) |
|---|---:|---:|---:|---:|---:|---:|
| B0 | — | — | 0.20193 | 0.28125 | — | 0.34947 |
| pi_alpha0 | — | — | 0.23438 | 0.21032 | — | 0.38776 |
| pi_alpha1 | — | — | 0.23453 | 0.21064 | — | 0.3878 |

Residual peaks below use rad/s. Realized means final command minus zero-residual final command, after actuator clipping; bounded request is before that clip.

| Method | Realized front/rear peak | Bounded request front/rear peak |
|---|---:|---:|
| B0 | 1.10048 / 1.69068 | 1.10048 / 1.69068 |
| pi_alpha0 | 1.5 / 5.60472 | 1.5 / 5.60473 |
| pi_alpha1 | 1.5 / 5.60135 | 1.5 / 5.60135 |

Clip fractions are reconstructed from saved requested residuals and final commands. The original logged flags remain visible because exact float inequality can flag rounding alone.

| Method | Residual true / logged | Final command true / logged |
|---|---:|---:|
| B0 | 0.0 / 0.0 | 0.0 / 0.00031 |
| pi_alpha0 | 0.03469 / 0.03469 | 0.0 / 0.00063 |
| pi_alpha1 | 0.035 / 0.035 | 0.0 / 0.0 |

[Metrics](metrics.json) · [Plot PNG](random_comparison.png) · [Plot PDF](random_comparison.pdf) · [Per-tick and cumulative reward](random_rewards.png) ([PDF](random_rewards.pdf))

[B0 raw/effective cost components](random_B0_components.png) ([PDF](random_B0_components.pdf))
[pi_alpha0 raw/effective cost components](random_pi_alpha0_components.png) ([PDF](random_pi_alpha0_components.pdf))
[pi_alpha1 raw/effective cost components](random_pi_alpha1_components.png) ([PDF](random_pi_alpha1_components.pdf))

A complete six-episode review is one prepared state and one random sequence; it does not establish generalization or hard safety.
