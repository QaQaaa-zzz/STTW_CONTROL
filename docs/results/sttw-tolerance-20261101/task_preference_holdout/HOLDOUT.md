# H75 and U170 on new task combinations

Same local200 lower, 0.8 wheel friction, 10 Hz geometric reconnection, four newly declared low/high single/S routes, prepared banks 2/5 and all five fixed alpha settings. The relaxed score was selected before these exact route/bank outcomes. This checks new combinations, not unseen training distribution or physical-domain randomization.

| Model | Alpha | Original goal | H10 survival | Narrow | Wide | Mean turn speed m/s | Mean pre-goal max original cross-track m |
|---|---:|---:|---:|---:|---:|---:|---:|
| H75 | 0 | 8/8 | 8/8 | 8/8 | 7/8 | 2.200 | 0.223 |
| H75 | 0.25 | 8/8 | 8/8 | 5/8 | 8/8 | 2.253 | 0.317 |
| H75 | 0.5 | 8/8 | 8/8 | 4/8 | 8/8 | 2.325 | 0.391 |
| H75 | 0.75 | 8/8 | 8/8 | 4/8 | 8/8 | 2.401 | 0.482 |
| H75 | 1 | 8/8 | 8/8 | 3/8 | 5/8 | 2.458 | 0.615 |
| U170 | 0 | 8/8 | 8/8 | 6/8 | 2/8 | 2.094 | 0.273 |
| U170 | 0.25 | 8/8 | 8/8 | 6/8 | 2/8 | 2.131 | 0.334 |
| U170 | 0.5 | 8/8 | 8/8 | 4/8 | 6/8 | 2.178 | 0.409 |
| U170 | 0.75 | 8/8 | 8/8 | 4/8 | 8/8 | 2.234 | 0.473 |
| U170 | 1 | 8/8 | 8/8 | 4/8 | 8/8 | 2.300 | 0.532 |

## By route

| Model | Route | Alpha | Narrow | Wide | Turn speed m/s | Max cross-track m |
|---|---|---:|---:|---:|---:|---:|
| H75 | holdout_high_S_left_70_R3p10_v3p10 | 0 | 2/2 | 2/2 | 2.781 | 0.247 |
| H75 | holdout_high_S_left_70_R3p10_v3p10 | 0.25 | 1/2 | 2/2 | 2.836 | 0.403 |
| H75 | holdout_high_S_left_70_R3p10_v3p10 | 0.5 | 0/2 | 2/2 | 2.903 | 0.530 |
| H75 | holdout_high_S_left_70_R3p10_v3p10 | 0.75 | 0/2 | 2/2 | 2.981 | 0.658 |
| H75 | holdout_high_S_left_70_R3p10_v3p10 | 1 | 0/2 | 1/2 | 3.050 | 0.906 |
| H75 | holdout_high_single_right_96_R3p15_v3p05 | 0 | 2/2 | 2/2 | 2.667 | 0.300 |
| H75 | holdout_high_single_right_96_R3p15_v3p05 | 0.25 | 0/2 | 2/2 | 2.732 | 0.521 |
| H75 | holdout_high_single_right_96_R3p15_v3p05 | 0.5 | 0/2 | 2/2 | 2.829 | 0.697 |
| H75 | holdout_high_single_right_96_R3p15_v3p05 | 0.75 | 0/2 | 2/2 | 2.941 | 0.852 |
| H75 | holdout_high_single_right_96_R3p15_v3p05 | 1 | 0/2 | 0/2 | 3.019 | 0.990 |
| H75 | holdout_low_S_right_72_R1p55_v2p00 | 0 | 2/2 | 2/2 | 1.734 | 0.180 |
| H75 | holdout_low_S_right_72_R1p55_v2p00 | 0.25 | 2/2 | 2/2 | 1.774 | 0.185 |
| H75 | holdout_low_S_right_72_R1p55_v2p00 | 0.5 | 2/2 | 2/2 | 1.840 | 0.187 |
| H75 | holdout_low_S_right_72_R1p55_v2p00 | 0.75 | 2/2 | 2/2 | 1.896 | 0.278 |
| H75 | holdout_low_S_right_72_R1p55_v2p00 | 1 | 1/2 | 2/2 | 1.933 | 0.404 |
| H75 | holdout_low_single_left_76_R1p45_v1p90 | 0 | 2/2 | 1/2 | 1.619 | 0.164 |
| H75 | holdout_low_single_left_76_R1p45_v1p90 | 0.25 | 2/2 | 2/2 | 1.671 | 0.158 |
| H75 | holdout_low_single_left_76_R1p45_v1p90 | 0.5 | 2/2 | 2/2 | 1.729 | 0.150 |
| H75 | holdout_low_single_left_76_R1p45_v1p90 | 0.75 | 2/2 | 2/2 | 1.785 | 0.139 |
| H75 | holdout_low_single_left_76_R1p45_v1p90 | 1 | 2/2 | 2/2 | 1.829 | 0.162 |
| U170 | holdout_high_S_left_70_R3p10_v3p10 | 0 | 0/2 | 2/2 | 2.694 | 0.449 |
| U170 | holdout_high_S_left_70_R3p10_v3p10 | 0.25 | 0/2 | 2/2 | 2.741 | 0.516 |
| U170 | holdout_high_S_left_70_R3p10_v3p10 | 0.5 | 0/2 | 2/2 | 2.790 | 0.594 |
| U170 | holdout_high_S_left_70_R3p10_v3p10 | 0.75 | 0/2 | 2/2 | 2.830 | 0.660 |
| U170 | holdout_high_S_left_70_R3p10_v3p10 | 1 | 0/2 | 2/2 | 2.887 | 0.720 |
| U170 | holdout_high_single_right_96_R3p15_v3p05 | 0 | 2/2 | 0/2 | 2.507 | 0.211 |
| U170 | holdout_high_single_right_96_R3p15_v3p05 | 0.25 | 2/2 | 0/2 | 2.544 | 0.364 |
| U170 | holdout_high_single_right_96_R3p15_v3p05 | 0.5 | 0/2 | 2/2 | 2.602 | 0.557 |
| U170 | holdout_high_single_right_96_R3p15_v3p05 | 0.75 | 0/2 | 2/2 | 2.697 | 0.715 |
| U170 | holdout_high_single_right_96_R3p15_v3p05 | 1 | 0/2 | 2/2 | 2.807 | 0.850 |
| U170 | holdout_low_S_right_72_R1p55_v2p00 | 0 | 2/2 | 0/2 | 1.616 | 0.216 |
| U170 | holdout_low_S_right_72_R1p55_v2p00 | 0.25 | 2/2 | 0/2 | 1.647 | 0.229 |
| U170 | holdout_low_S_right_72_R1p55_v2p00 | 0.5 | 2/2 | 0/2 | 1.682 | 0.242 |
| U170 | holdout_low_S_right_72_R1p55_v2p00 | 0.75 | 2/2 | 2/2 | 1.719 | 0.256 |
| U170 | holdout_low_S_right_72_R1p55_v2p00 | 1 | 2/2 | 2/2 | 1.762 | 0.278 |
| U170 | holdout_low_single_left_76_R1p45_v1p90 | 0 | 2/2 | 0/2 | 1.558 | 0.215 |
| U170 | holdout_low_single_left_76_R1p45_v1p90 | 0.25 | 2/2 | 0/2 | 1.593 | 0.227 |
| U170 | holdout_low_single_left_76_R1p45_v1p90 | 0.5 | 2/2 | 2/2 | 1.637 | 0.241 |
| U170 | holdout_low_single_left_76_R1p45_v1p90 | 0.75 | 2/2 | 2/2 | 1.691 | 0.259 |
| U170 | holdout_low_single_left_76_R1p45_v1p90 | 1 | 2/2 | 2/2 | 1.744 | 0.280 |

Every underlying case is in [holdout_comparison.json](holdout_comparison.json). Actual/reference XY panels for every route, bank and model are in the source runs and portable package.
