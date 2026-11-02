# U170 physical-domain task rescore

## Actual/reference XY before scores

Each panel overlays the original reference and all five executed alpha trajectories. The 16 PNG files are copied from the archived source analysis.

| Domain | Low single, bank0 | High S, bank0 | Low single, bank6 | High S, bank6 |
|---|---|---|---|---|
| test_00 | [XY](xy/domain0_bank0_low_single_left_zero_fixed_candidate_xy.png) | [XY](xy/domain0_bank0_high_S_left_zero_fixed_candidate_xy.png) | [XY](xy/domain0_bank6_low_single_left_zero_fixed_candidate_xy.png) | [XY](xy/domain0_bank6_high_S_left_zero_fixed_candidate_xy.png) |
| test_01 | [XY](xy/domain1_bank0_low_single_left_zero_fixed_candidate_xy.png) | [XY](xy/domain1_bank0_high_S_left_zero_fixed_candidate_xy.png) | [XY](xy/domain1_bank6_low_single_left_zero_fixed_candidate_xy.png) | [XY](xy/domain1_bank6_high_S_left_zero_fixed_candidate_xy.png) |
| test_02 | [XY](xy/domain2_bank0_low_single_left_zero_fixed_candidate_xy.png) | [XY](xy/domain2_bank0_high_S_left_zero_fixed_candidate_xy.png) | [XY](xy/domain2_bank6_low_single_left_zero_fixed_candidate_xy.png) | [XY](xy/domain2_bank6_high_S_left_zero_fixed_candidate_xy.png) |
| test_03 | [XY](xy/domain3_bank0_low_single_left_zero_fixed_candidate_xy.png) | [XY](xy/domain3_bank0_high_S_left_zero_fixed_candidate_xy.png) | [XY](xy/domain3_bank6_low_single_left_zero_fixed_candidate_xy.png) | [XY](xy/domain3_bank6_high_S_left_zero_fixed_candidate_xy.png) |

This uses the same task score as the later task-combination panel on already completed trajectories. The four physical domains were independent TEST draws for the original arrival/survival study, but this task score was applied after inspecting that study and is exploratory, not a fresh confirmatory domain test.

| Domain | Wheel friction | Chassis mass scale |
|---|---:|---:|
| test_00 | 0.932 | 0.975 |
| test_01 | 0.759 | 1.046 |
| test_02 | 0.887 | 0.927 |
| test_03 | 0.612 | 1.074 |

| Alpha | Goal | Survived | Narrow | Wide | Mean turn speed (m/s) | Mean maximum original cross (m) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16/16 | 16/16 | 3/16 | 16/16 | 2.135 | 0.426 |
| 0.25 | 16/16 | 16/16 | 2/16 | 16/16 | 2.162 | 0.474 |
| 0.5 | 16/16 | 16/16 | 1/16 | 16/16 | 2.192 | 0.528 |
| 0.75 | 16/16 | 16/16 | 1/16 | 16/16 | 2.227 | 0.582 |
| 1 | 16/16 | 16/16 | 1/16 | 16/16 | 2.263 | 0.636 |

The [JSON](domain_task_rescore.json) contains all 80 paired cases, physical-domain parameters, raw data paths and continuous outcomes.
