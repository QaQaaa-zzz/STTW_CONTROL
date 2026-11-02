# Matched 90-degree left/right turn experiment

## Actual/reference XY first

Four smooth finite-radius routes change heading by 90 degrees: left/right at low request 2.0 m/s, radius 1.6 m and high request 3.05 m/s, radius 3.0 m. This is not a zero-radius mathematical corner. Prepared banks 2/5, friction 0.8, 10 Hz causal path reconnection, original-goal task score, H10 and frozen controllers are shared. All 48 episodes are retained.

### u170

- [bank2_right_angle_high_left_90_R3p0_v3p05_zero_fixed_candidate_xy](xy/u170/bank2_right_angle_high_left_90_R3p0_v3p05_zero_fixed_candidate_xy.png)
- [bank2_right_angle_high_right_90_R3p0_v3p05_zero_fixed_candidate_xy](xy/u170/bank2_right_angle_high_right_90_R3p0_v3p05_zero_fixed_candidate_xy.png)
- [bank2_right_angle_low_left_90_R1p6_v2p0_zero_fixed_candidate_xy](xy/u170/bank2_right_angle_low_left_90_R1p6_v2p0_zero_fixed_candidate_xy.png)
- [bank2_right_angle_low_right_90_R1p6_v2p0_zero_fixed_candidate_xy](xy/u170/bank2_right_angle_low_right_90_R1p6_v2p0_zero_fixed_candidate_xy.png)
- [bank5_right_angle_high_left_90_R3p0_v3p05_zero_fixed_candidate_xy](xy/u170/bank5_right_angle_high_left_90_R3p0_v3p05_zero_fixed_candidate_xy.png)
- [bank5_right_angle_high_right_90_R3p0_v3p05_zero_fixed_candidate_xy](xy/u170/bank5_right_angle_high_right_90_R3p0_v3p05_zero_fixed_candidate_xy.png)
- [bank5_right_angle_low_left_90_R1p6_v2p0_zero_fixed_candidate_xy](xy/u170/bank5_right_angle_low_left_90_R1p6_v2p0_zero_fixed_candidate_xy.png)
- [bank5_right_angle_low_right_90_R1p6_v2p0_zero_fixed_candidate_xy](xy/u170/bank5_right_angle_low_right_90_R1p6_v2p0_zero_fixed_candidate_xy.png)

### rule

- [bank2_right_angle_high_left_90_R3p0_v3p05_zero_fixed_rule_xy](xy/rule/bank2_right_angle_high_left_90_R3p0_v3p05_zero_fixed_rule_xy.png)
- [bank2_right_angle_high_right_90_R3p0_v3p05_zero_fixed_rule_xy](xy/rule/bank2_right_angle_high_right_90_R3p0_v3p05_zero_fixed_rule_xy.png)
- [bank2_right_angle_low_left_90_R1p6_v2p0_zero_fixed_rule_xy](xy/rule/bank2_right_angle_low_left_90_R1p6_v2p0_zero_fixed_rule_xy.png)
- [bank2_right_angle_low_right_90_R1p6_v2p0_zero_fixed_rule_xy](xy/rule/bank2_right_angle_low_right_90_R1p6_v2p0_zero_fixed_rule_xy.png)
- [bank5_right_angle_high_left_90_R3p0_v3p05_zero_fixed_rule_xy](xy/rule/bank5_right_angle_high_left_90_R3p0_v3p05_zero_fixed_rule_xy.png)
- [bank5_right_angle_high_right_90_R3p0_v3p05_zero_fixed_rule_xy](xy/rule/bank5_right_angle_high_right_90_R3p0_v3p05_zero_fixed_rule_xy.png)
- [bank5_right_angle_low_left_90_R1p6_v2p0_zero_fixed_rule_xy](xy/rule/bank5_right_angle_low_left_90_R1p6_v2p0_zero_fixed_rule_xy.png)
- [bank5_right_angle_low_right_90_R1p6_v2p0_zero_fixed_rule_xy](xy/rule/bank5_right_angle_low_right_90_R1p6_v2p0_zero_fixed_rule_xy.png)

| Method | Goal | H10 | Narrow | Wide | Turn speed (m/s) | Max original cross (m) |
|---|---:|---:|---:|---:|---:|---:|
| Rule | 8/8 | 8/8 | 8/8 | 8/8 | 2.482 | 0.124 |
| U170 alpha=0 | 8/8 | 8/8 | 6/8 | 2/8 | 2.122 | 0.325 |
| U170 alpha=0.25 | 8/8 | 8/8 | 6/8 | 2/8 | 2.164 | 0.382 |
| U170 alpha=0.5 | 8/8 | 8/8 | 4/8 | 5/8 | 2.215 | 0.452 |
| U170 alpha=0.75 | 8/8 | 8/8 | 4/8 | 6/8 | 2.277 | 0.527 |
| U170 alpha=1 | 8/8 | 8/8 | 2/8 | 8/8 | 2.346 | 0.601 |

Exact case scores and raw run locations are in [comparison.json](comparison.json). This panel was requested after inspecting nominal and signed-pulse outcomes and is exploratory.

Run status: both physical evaluations and all 48 case analyses report `completed` with their declared lower-tick counts and zero optimizer updates. The rule wrapper recorded exit 0. The U170 wrapper session ended with signal exit 143 after its 40-case evaluation had already written `completed`; its separate `execute.exit` receipt is absent. The complete case index, raw trajectories and task scores were used here without rerunning physics.
