# Single-alpha practical sensitivity

Source: /home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_distilled_v3_curriculum_400_20261021/training/validation/update_0375_stage3

Post-hoc analysis of saved 10s trajectories; original pass/fail labels preserved. Limits below are diagnostic options, not a proven safety envelope. Training is not modified.

| Alpha | Full-trace speed RMSE | Full-trace path RMSE | Largest final path error |
|---|---:|---:|---:|
| 0 | 0.3132 m/s | 0.1111 m | 0.0924 m |
| 0.25 | 0.2715 m/s | 0.0906 m | 0.1105 m |
| 0.5 | 0.1863 m/s | 0.1159 m | 0.1322 m |
| 0.75 | 0.1084 m/s | 0.1780 m | 0.1565 m |
| 1 | 0.0681 m/s | 0.2397 m | 0.1852 m |

## Endpoints

| Case | Alpha1 speed advantage (m/s) | Alpha0 path advantage (m) |
|---|---:|---:|
| left90_R2_2.6 | 0.3559 | 0.0427 |
| left90_R2_2.8 | 0.3346 | 0.1162 |
| right90_R2_2.6 | 0.3388 | 0.1827 |
| right90_R2_2.8 | 0.4531 | 0.3043 |
| straight_2.6 | -0.0062 | 0.0555 |
| straight_2.8 | -0.0054 | 0.0702 |

## Sensitivity

| Path limit (m) | Roll limit (rad) | Alpha | Pass / total |
|---:|---:|---:|---:|
| 0.1 | 0.302 | 0 | 6/6 |
| 0.1 | 0.302 | 0.25 | 0/6 |
| 0.1 | 0.302 | 0.5 | 0/6 |
| 0.1 | 0.302 | 0.75 | 0/6 |
| 0.1 | 0.302 | 1 | 0/6 |
| 0.1 | 0.33 | 0 | 6/6 |
| 0.1 | 0.33 | 0.25 | 0/6 |
| 0.1 | 0.33 | 0.5 | 0/6 |
| 0.1 | 0.33 | 0.75 | 0/6 |
| 0.1 | 0.33 | 1 | 0/6 |
| 0.15 | 0.302 | 0 | 6/6 |
| 0.15 | 0.302 | 0.25 | 5/6 |
| 0.15 | 0.302 | 0.5 | 5/6 |
| 0.15 | 0.302 | 0.75 | 3/6 |
| 0.15 | 0.302 | 1 | 0/6 |
| 0.15 | 0.33 | 0 | 6/6 |
| 0.15 | 0.33 | 0.25 | 6/6 |
| 0.15 | 0.33 | 0.5 | 6/6 |
| 0.15 | 0.33 | 0.75 | 3/6 |
| 0.15 | 0.33 | 1 | 0/6 |
| 0.2 | 0.302 | 0 | 6/6 |
| 0.2 | 0.302 | 0.25 | 5/6 |
| 0.2 | 0.302 | 0.5 | 5/6 |
| 0.2 | 0.302 | 0.75 | 5/6 |
| 0.2 | 0.302 | 1 | 5/6 |
| 0.2 | 0.33 | 0 | 6/6 |
| 0.2 | 0.33 | 0.25 | 6/6 |
| 0.2 | 0.33 | 0.5 | 6/6 |
| 0.2 | 0.33 | 0.75 | 6/6 |
| 0.2 | 0.33 | 1 | 6/6 |

[Existing matched-alpha XY figure](/home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_distilled_v3_curriculum_400_20261021/training/validation/update_0375_stage3/XY.png)

Interpretation: check turn trade-offs, straight-line bias, and final recovery separately. Do not force alpha1 to worsen its path merely to create separation. A relaxed threshold does not remove measured residual bias.
