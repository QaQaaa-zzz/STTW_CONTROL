# Single-alpha practical sensitivity

Source: /home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_distilled_v3_curriculum_400_20261021/training/validation/update_0325_stage3

Post-hoc analysis of saved 10s trajectories; original pass/fail labels preserved. Limits below are diagnostic options, not a proven safety envelope. Training is not modified.

| Alpha | Turn-window speed RMSE* | Turn-window path RMSE* | Largest final path error |
|---|---:|---:|---:|
| 0 | 0.3079 m/s | 0.0967 m | 0.0588 m |
| 0.25 | 0.2682 m/s | 0.0764 m | 0.0553 m |
| 0.5 | 0.1945 m/s | 0.1117 m | 0.0461 m |
| 0.75 | 0.1162 m/s | 0.1702 m | 0.0334 m |
| 1 | 0.0679 m/s | 0.2318 m | 0.0642 m |

*Source metrics use geometric-turn samples for turning cases and the full trace for straight cases. Table values average the six case-level RMSEs; they are not pooled full-trajectory RMSE.

## Endpoints

| Case | Alpha1 speed advantage (m/s) | Alpha0 path advantage (m) |
|---|---:|---:|
| left90_R2_2.6 | 0.2917 | -0.0008 |
| left90_R2_2.8 | 0.2687 | 0.0493 |
| right90_R2_2.6 | 0.3944 | 0.3382 |
| right90_R2_2.8 | 0.4949 | 0.4737 |
| straight_2.6 | -0.0052 | -0.0222 |
| straight_2.8 | -0.0047 | -0.0276 |

## Sensitivity

| Path limit (m) | Roll limit (rad) | Alpha | Pass / total |
|---:|---:|---:|---:|
| 0.1 | 0.302 | 0 | 6/6 |
| 0.1 | 0.302 | 0.25 | 6/6 |
| 0.1 | 0.302 | 0.5 | 6/6 |
| 0.1 | 0.302 | 0.75 | 6/6 |
| 0.1 | 0.302 | 1 | 6/6 |
| 0.1 | 0.33 | 0 | 6/6 |
| 0.1 | 0.33 | 0.25 | 6/6 |
| 0.1 | 0.33 | 0.5 | 6/6 |
| 0.1 | 0.33 | 0.75 | 6/6 |
| 0.1 | 0.33 | 1 | 6/6 |
| 0.15 | 0.302 | 0 | 6/6 |
| 0.15 | 0.302 | 0.25 | 6/6 |
| 0.15 | 0.302 | 0.5 | 6/6 |
| 0.15 | 0.302 | 0.75 | 6/6 |
| 0.15 | 0.302 | 1 | 6/6 |
| 0.15 | 0.33 | 0 | 6/6 |
| 0.15 | 0.33 | 0.25 | 6/6 |
| 0.15 | 0.33 | 0.5 | 6/6 |
| 0.15 | 0.33 | 0.75 | 6/6 |
| 0.15 | 0.33 | 1 | 6/6 |
| 0.2 | 0.302 | 0 | 6/6 |
| 0.2 | 0.302 | 0.25 | 6/6 |
| 0.2 | 0.302 | 0.5 | 6/6 |
| 0.2 | 0.302 | 0.75 | 6/6 |
| 0.2 | 0.302 | 1 | 6/6 |
| 0.2 | 0.33 | 0 | 6/6 |
| 0.2 | 0.33 | 0.25 | 6/6 |
| 0.2 | 0.33 | 0.5 | 6/6 |
| 0.2 | 0.33 | 0.75 | 6/6 |
| 0.2 | 0.33 | 1 | 6/6 |

[Existing matched-alpha XY figure](/home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_distilled_v3_curriculum_400_20261021/training/validation/update_0325_stage3/XY.png)

Interpretation: check turn trade-offs, straight-line bias, and final recovery separately. Do not force alpha1 to worsen its path merely to create separation. A relaxed threshold does not remove measured residual bias.
