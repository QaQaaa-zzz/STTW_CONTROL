# Single-alpha practical sensitivity

Source: /home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_distilled_v3_curriculum_400_20261021/training/validation/update_0300_stage3

Post-hoc analysis of saved 10s trajectories; original pass/fail labels preserved. Limits below are diagnostic options, not a proven safety envelope. Training is not modified.

| Alpha | Turn-window speed RMSE* | Turn-window path RMSE* | Largest final path error |
|---|---:|---:|---:|
| 0 | 0.3394 m/s | 0.0946 m | 0.0161 m |
| 0.25 | 0.3010 m/s | 0.1030 m | 0.0057 m |
| 0.5 | 0.2412 m/s | 0.1601 m | 0.0206 m |
| 0.75 | 0.1772 m/s | 0.2249 m | 0.0601 m |
| 1 | 0.1329 m/s | 0.2871 m | 0.1424 m |

*Source metrics use geometric-turn samples for turning cases and the full trace for straight cases. Table values average the six case-level RMSEs; they are not pooled full-trajectory RMSE.

## Endpoints

| Case | Alpha1 speed advantage (m/s) | Alpha0 path advantage (m) |
|---|---:|---:|
| left90_R2_2.6 | 0.1947 | -0.0411 |
| left90_R2_2.8 | 0.1293 | -0.0693 |
| right90_R2_2.6 | 0.4250 | 0.4551 |
| right90_R2_2.8 | 0.4998 | 0.6225 |
| straight_2.6 | -0.0050 | 0.0856 |
| straight_2.8 | -0.0044 | 0.1020 |

## Sensitivity

| Path limit (m) | Roll limit (rad) | Alpha | Pass / total |
|---:|---:|---:|---:|
| 0.1 | 0.302 | 0 | 5/6 |
| 0.1 | 0.302 | 0.25 | 6/6 |
| 0.1 | 0.302 | 0.5 | 6/6 |
| 0.1 | 0.302 | 0.75 | 6/6 |
| 0.1 | 0.302 | 1 | 2/6 |
| 0.1 | 0.33 | 0 | 6/6 |
| 0.1 | 0.33 | 0.25 | 6/6 |
| 0.1 | 0.33 | 0.5 | 6/6 |
| 0.1 | 0.33 | 0.75 | 6/6 |
| 0.1 | 0.33 | 1 | 2/6 |
| 0.15 | 0.302 | 0 | 5/6 |
| 0.15 | 0.302 | 0.25 | 6/6 |
| 0.15 | 0.302 | 0.5 | 6/6 |
| 0.15 | 0.302 | 0.75 | 6/6 |
| 0.15 | 0.302 | 1 | 6/6 |
| 0.15 | 0.33 | 0 | 6/6 |
| 0.15 | 0.33 | 0.25 | 6/6 |
| 0.15 | 0.33 | 0.5 | 6/6 |
| 0.15 | 0.33 | 0.75 | 6/6 |
| 0.15 | 0.33 | 1 | 6/6 |
| 0.2 | 0.302 | 0 | 5/6 |
| 0.2 | 0.302 | 0.25 | 6/6 |
| 0.2 | 0.302 | 0.5 | 6/6 |
| 0.2 | 0.302 | 0.75 | 6/6 |
| 0.2 | 0.302 | 1 | 6/6 |
| 0.2 | 0.33 | 0 | 6/6 |
| 0.2 | 0.33 | 0.25 | 6/6 |
| 0.2 | 0.33 | 0.5 | 6/6 |
| 0.2 | 0.33 | 0.75 | 6/6 |
| 0.2 | 0.33 | 1 | 6/6 |

[Existing matched-alpha XY figure](/home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_distilled_v3_curriculum_400_20261021/training/validation/update_0300_stage3/XY.png)

Interpretation: check turn trade-offs, straight-line bias, and final recovery separately. Do not force alpha1 to worsen its path merely to create separation. A relaxed threshold does not remove measured residual bias.
