# Single-alpha practical sensitivity

Source: /home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_distilled_v3_curriculum_400_20261021/training/validation/update_0400_stage3

Post-hoc analysis of saved 10s trajectories; original pass/fail labels preserved. Limits below are diagnostic options, not a proven safety envelope. Training is not modified.

| Alpha | Turn-window speed RMSE* | Turn-window path RMSE* | Largest final path error |
|---|---:|---:|---:|
| 0 | 0.2969 m/s | 0.1068 m | 0.0912 m |
| 0.25 | 0.2492 m/s | 0.1018 m | 0.1199 m |
| 0.5 | 0.1846 m/s | 0.1304 m | 0.1553 m |
| 0.75 | 0.0951 m/s | 0.2014 m | 0.1960 m |
| 1 | 0.0447 m/s | 0.2724 m | 0.2401 m |

*Source metrics use geometric-turn samples for turning cases and the full trace for straight cases. Table values average the six case-level RMSEs; they are not pooled full-trajectory RMSE.

## Endpoints

| Case | Alpha1 speed advantage (m/s) | Alpha0 path advantage (m) |
|---|---:|---:|
| left90_R2_2.6 | 0.3443 | 0.2216 |
| left90_R2_2.8 | 0.3562 | 0.3673 |
| right90_R2_2.6 | 0.3852 | 0.0506 |
| right90_R2_2.8 | 0.4396 | 0.1184 |
| straight_2.6 | -0.0060 | 0.1114 |
| straight_2.8 | -0.0058 | 0.1248 |

## Sensitivity

| Path limit (m) | Roll limit (rad) | Alpha | Pass / total |
|---:|---:|---:|---:|
| 0.1 | 0.302 | 0 | 5/6 |
| 0.1 | 0.302 | 0.25 | 0/6 |
| 0.1 | 0.302 | 0.5 | 0/6 |
| 0.1 | 0.302 | 0.75 | 0/6 |
| 0.1 | 0.302 | 1 | 0/6 |
| 0.1 | 0.33 | 0 | 6/6 |
| 0.1 | 0.33 | 0.25 | 0/6 |
| 0.1 | 0.33 | 0.5 | 0/6 |
| 0.1 | 0.33 | 0.75 | 0/6 |
| 0.1 | 0.33 | 1 | 0/6 |
| 0.15 | 0.302 | 0 | 5/6 |
| 0.15 | 0.302 | 0.25 | 6/6 |
| 0.15 | 0.302 | 0.5 | 3/6 |
| 0.15 | 0.302 | 0.75 | 0/6 |
| 0.15 | 0.302 | 1 | 0/6 |
| 0.15 | 0.33 | 0 | 6/6 |
| 0.15 | 0.33 | 0.25 | 6/6 |
| 0.15 | 0.33 | 0.5 | 3/6 |
| 0.15 | 0.33 | 0.75 | 0/6 |
| 0.15 | 0.33 | 1 | 0/6 |
| 0.2 | 0.302 | 0 | 5/6 |
| 0.2 | 0.302 | 0.25 | 6/6 |
| 0.2 | 0.302 | 0.5 | 4/6 |
| 0.2 | 0.302 | 0.75 | 4/6 |
| 0.2 | 0.302 | 1 | 0/6 |
| 0.2 | 0.33 | 0 | 6/6 |
| 0.2 | 0.33 | 0.25 | 6/6 |
| 0.2 | 0.33 | 0.5 | 6/6 |
| 0.2 | 0.33 | 0.75 | 6/6 |
| 0.2 | 0.33 | 1 | 0/6 |

[Existing matched-alpha XY figure](/home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_distilled_v3_curriculum_400_20261021/training/validation/update_0400_stage3/XY.png)

Interpretation: check turn trade-offs, straight-line bias, and final recovery separately. Do not force alpha1 to worsen its path merely to create separation. A relaxed threshold does not remove measured residual bias.
