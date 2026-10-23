# Single-alpha practical sensitivity

Source: /home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_distilled_v3_curriculum_400_20261021/training/validation/update_0350_stage3

Post-hoc analysis of saved 10s trajectories; original pass/fail labels preserved. Limits below are diagnostic options, not a proven safety envelope. Training is not modified.

| Alpha | Turn-window speed RMSE* | Turn-window path RMSE* | Largest final path error |
|---|---:|---:|---:|
| 0 | 0.3222 m/s | 0.1286 m | 0.1360 m |
| 0.25 | 0.2849 m/s | 0.1063 m | 0.1394 m |
| 0.5 | 0.2106 m/s | 0.1310 m | 0.1408 m |
| 0.75 | 0.1348 m/s | 0.1833 m | 0.1419 m |
| 1 | 0.0979 m/s | 0.2270 m | 0.1438 m |

*Source metrics use geometric-turn samples for turning cases and the full trace for straight cases. Table values average the six case-level RMSEs; they are not pooled full-trajectory RMSE.

## Endpoints

| Case | Alpha1 speed advantage (m/s) | Alpha0 path advantage (m) |
|---|---:|---:|
| left90_R2_2.6 | 0.2914 | -0.0112 |
| left90_R2_2.8 | 0.2457 | 0.0007 |
| right90_R2_2.6 | 0.3515 | 0.2556 |
| right90_R2_2.8 | 0.4641 | 0.3827 |
| straight_2.6 | -0.0035 | -0.0265 |
| straight_2.8 | -0.0037 | -0.0110 |

## Sensitivity

| Path limit (m) | Roll limit (rad) | Alpha | Pass / total |
|---:|---:|---:|---:|
| 0.1 | 0.302 | 0 | 0/6 |
| 0.1 | 0.302 | 0.25 | 0/6 |
| 0.1 | 0.302 | 0.5 | 0/6 |
| 0.1 | 0.302 | 0.75 | 0/6 |
| 0.1 | 0.302 | 1 | 2/6 |
| 0.1 | 0.33 | 0 | 0/6 |
| 0.1 | 0.33 | 0.25 | 0/6 |
| 0.1 | 0.33 | 0.5 | 0/6 |
| 0.1 | 0.33 | 0.75 | 0/6 |
| 0.1 | 0.33 | 1 | 2/6 |
| 0.15 | 0.302 | 0 | 5/6 |
| 0.15 | 0.302 | 0.25 | 5/6 |
| 0.15 | 0.302 | 0.5 | 5/6 |
| 0.15 | 0.302 | 0.75 | 6/6 |
| 0.15 | 0.302 | 1 | 6/6 |
| 0.15 | 0.33 | 0 | 6/6 |
| 0.15 | 0.33 | 0.25 | 6/6 |
| 0.15 | 0.33 | 0.5 | 6/6 |
| 0.15 | 0.33 | 0.75 | 6/6 |
| 0.15 | 0.33 | 1 | 6/6 |
| 0.2 | 0.302 | 0 | 5/6 |
| 0.2 | 0.302 | 0.25 | 5/6 |
| 0.2 | 0.302 | 0.5 | 5/6 |
| 0.2 | 0.302 | 0.75 | 6/6 |
| 0.2 | 0.302 | 1 | 6/6 |
| 0.2 | 0.33 | 0 | 6/6 |
| 0.2 | 0.33 | 0.25 | 6/6 |
| 0.2 | 0.33 | 0.5 | 6/6 |
| 0.2 | 0.33 | 0.75 | 6/6 |
| 0.2 | 0.33 | 1 | 6/6 |

[Existing matched-alpha XY figure](/home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_distilled_v3_curriculum_400_20261021/training/validation/update_0350_stage3/XY.png)

Interpretation: check turn trade-offs, straight-line bias, and final recovery separately. Do not force alpha1 to worsen its path merely to create separation. A relaxed threshold does not remove measured residual bias.
