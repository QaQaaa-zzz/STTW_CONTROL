# DEV-tuned geometric rule on the new task combinations

## Actual/reference XY first

Eight frozen-rule panels cover all four new routes and prepared banks2/5; each plots actual XY against the original route. The five-alpha U170/H75 matched panels are in the linked middle result package.

- [bank2_holdout_high_S_left_70_R3p10_v3p10_zero_fixed_rule_xy](xy/bank2_holdout_high_S_left_70_R3p10_v3p10_zero_fixed_rule_xy.png)
- [bank2_holdout_high_single_right_96_R3p15_v3p05_zero_fixed_rule_xy](xy/bank2_holdout_high_single_right_96_R3p15_v3p05_zero_fixed_rule_xy.png)
- [bank2_holdout_low_S_right_72_R1p55_v2p00_zero_fixed_rule_xy](xy/bank2_holdout_low_S_right_72_R1p55_v2p00_zero_fixed_rule_xy.png)
- [bank2_holdout_low_single_left_76_R1p45_v1p90_zero_fixed_rule_xy](xy/bank2_holdout_low_single_left_76_R1p45_v1p90_zero_fixed_rule_xy.png)
- [bank5_holdout_high_S_left_70_R3p10_v3p10_zero_fixed_rule_xy](xy/bank5_holdout_high_S_left_70_R3p10_v3p10_zero_fixed_rule_xy.png)
- [bank5_holdout_high_single_right_96_R3p15_v3p05_zero_fixed_rule_xy](xy/bank5_holdout_high_single_right_96_R3p15_v3p05_zero_fixed_rule_xy.png)
- [bank5_holdout_low_S_right_72_R1p55_v2p00_zero_fixed_rule_xy](xy/bank5_holdout_low_S_right_72_R1p55_v2p00_zero_fixed_rule_xy.png)
- [bank5_holdout_low_single_left_76_R1p45_v1p90_zero_fixed_rule_xy](xy/bank5_holdout_low_single_left_76_R1p45_v1p90_zero_fixed_rule_xy.png)

The pure-pursuit curvature-cap rule uses no learned middle output. Its 3.5 m/s² lateral-acceleration cap and 6 m preview were selected on earlier DEV conditions, before executing these eight rule cases. The middle new-combination outcomes were already opened when this comparator was constructed.

| Method | Goal | H10 | Narrow | Wide | Mean turn speed (m/s) | Mean max original cross (m) |
|---|---:|---:|---:|---:|---:|---:|
| Rule | 8/8 | 8/8 | 7/8 | 7/8 | 2.469 | 0.233 |
| U170 alpha=0 | 8/8 | 8/8 | 6/8 | 2/8 | 2.094 | 0.273 |
| U170 alpha=0.5 | 8/8 | 8/8 | 4/8 | 6/8 | 2.178 | 0.409 |
| U170 alpha=1 | 8/8 | 8/8 | 4/8 | 8/8 | 2.300 | 0.532 |

Paired U170 endpoint versus rule: narrow: U170-only 1, rule-only 2, both 5, neither 0; wide: U170-only 1, rule-only 0, both 7, neither 0.

All eight individual rule scores, including failures and continuous margins, are in [comparison.json](comparison.json). The matched five-alpha middle cases are in [the separate full panel](../task_preference_holdout/HOLDOUT.md).
