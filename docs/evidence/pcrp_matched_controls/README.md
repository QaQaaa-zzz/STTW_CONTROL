# Matched controls on previously examined routes

Counts include every physical group. Each entry gives full 10 s / final hold / Task A / B / C counts. Conditions share the original route, speed request and lower controller. The curvature cap uses an independent simulation batch and alpha .5 for reward scoring. Zero disables middle corrections. Each row of episodes.csv retains the actual endpoint and tracking errors on both its observed lifetime and the prefix shared across all compared methods in its group.

| Condition | n per method | 0 | .25 | .5 | .75 | 1 | Zero | Cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s_bend_pulse | 5 | 5/0/0/0/0 | 5/3/0/0/0 | 5/5/0/0/0 | 5/5/0/2/0 | 5/0/0/0/0 | 5/0/0/0/0 | — |
| right_bend | 5 | 5/5/0/0/0 | 5/5/0/0/0 | 5/5/0/0/0 | 5/5/0/0/0 | 5/5/0/0/0 | 1/0/0/0/0 | — |
| flat_original | 8 | 8/8/8/0/8 | 8/8/8/0/8 | 8/8/8/8/8 | 8/8/0/8/8 | 8/8/0/8/0 | 8/0/0/0/0 | 8/0/0/0/0 |
| gravel010_layout0 | 4 | 3/3/0/0/0 | 1/1/0/0/0 | 4/4/1/0/1 | 3/3/0/2/2 | 4/2/0/2/1 | 0/0/0/0/0 | 1/0/0/0/0 |
| gravel010_layout1 | 4 | 2/2/0/0/0 | 3/3/1/0/0 | 2/1/0/0/0 | 4/3/1/1/0 | 3/3/1/2/1 | 1/0/0/0/0 | 1/0/0/0/0 |
| position_s3.9 | 8 | 8/8/5/0/2 | 5/5/0/0/0 | 2/2/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 |
| position_s4.2 | 8 | 1/0/0/0/0 | 8/8/1/0/5 | 8/8/0/0/8 | 1/1/0/0/0 | 1/1/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 |
| position_s4.5 | 8 | 0/0/0/0/0 | 0/0/0/0/0 | 7/5/3/0/4 | 0/0/0/0/0 | 8/4/0/0/0 | 1/0/0/0/0 | 0/0/0/0/0 |

## Per-group actual and reference paths

| Condition | Physical group | Figure |
|---|---:|---|
| s_bend_pulse | 101 | [XY](figures/s_bend_pulse_101.png) |
| s_bend_pulse | 102 | [XY](figures/s_bend_pulse_102.png) |
| s_bend_pulse | 103 | [XY](figures/s_bend_pulse_103.png) |
| s_bend_pulse | 104 | [XY](figures/s_bend_pulse_104.png) |
| s_bend_pulse | 105 | [XY](figures/s_bend_pulse_105.png) |
| right_bend | 101 | [XY](figures/right_bend_101.png) |
| right_bend | 102 | [XY](figures/right_bend_102.png) |
| right_bend | 103 | [XY](figures/right_bend_103.png) |
| right_bend | 104 | [XY](figures/right_bend_104.png) |
| right_bend | 105 | [XY](figures/right_bend_105.png) |
| flat_original | 401 | [XY](figures/flat_original_v2.6_seed401.png) |
| flat_original | 402 | [XY](figures/flat_original_v2.6_seed402.png) |
| flat_original | 403 | [XY](figures/flat_original_v2.6_seed403.png) |
| flat_original | 404 | [XY](figures/flat_original_v2.6_seed404.png) |
| flat_original | 405 | [XY](figures/flat_original_v2.6_seed405.png) |
| flat_original | 406 | [XY](figures/flat_original_v2.6_seed406.png) |
| flat_original | 407 | [XY](figures/flat_original_v2.6_seed407.png) |
| flat_original | 408 | [XY](figures/flat_original_v2.6_seed408.png) |
| gravel010_layout0 | 401 | [XY](figures/gravel010_v2.6_layout0_seed401.png) |
| gravel010_layout0 | 402 | [XY](figures/gravel010_v2.6_layout0_seed402.png) |
| gravel010_layout0 | 403 | [XY](figures/gravel010_v2.6_layout0_seed403.png) |
| gravel010_layout0 | 404 | [XY](figures/gravel010_v2.6_layout0_seed404.png) |
| gravel010_layout1 | 401 | [XY](figures/gravel010_v2.6_layout1_seed401.png) |
| gravel010_layout1 | 402 | [XY](figures/gravel010_v2.6_layout1_seed402.png) |
| gravel010_layout1 | 403 | [XY](figures/gravel010_v2.6_layout1_seed403.png) |
| gravel010_layout1 | 404 | [XY](figures/gravel010_v2.6_layout1_seed404.png) |
| position_s3.9 | 401 | [XY](figures/position_s3.9_v2.6_seed401.png) |
| position_s3.9 | 402 | [XY](figures/position_s3.9_v2.6_seed402.png) |
| position_s3.9 | 403 | [XY](figures/position_s3.9_v2.6_seed403.png) |
| position_s3.9 | 404 | [XY](figures/position_s3.9_v2.6_seed404.png) |
| position_s3.9 | 405 | [XY](figures/position_s3.9_v2.6_seed405.png) |
| position_s3.9 | 406 | [XY](figures/position_s3.9_v2.6_seed406.png) |
| position_s3.9 | 407 | [XY](figures/position_s3.9_v2.6_seed407.png) |
| position_s3.9 | 408 | [XY](figures/position_s3.9_v2.6_seed408.png) |
| position_s4.2 | 401 | [XY](figures/position_s4.2_v2.6_seed401.png) |
| position_s4.2 | 402 | [XY](figures/position_s4.2_v2.6_seed402.png) |
| position_s4.2 | 403 | [XY](figures/position_s4.2_v2.6_seed403.png) |
| position_s4.2 | 404 | [XY](figures/position_s4.2_v2.6_seed404.png) |
| position_s4.2 | 405 | [XY](figures/position_s4.2_v2.6_seed405.png) |
| position_s4.2 | 406 | [XY](figures/position_s4.2_v2.6_seed406.png) |
| position_s4.2 | 407 | [XY](figures/position_s4.2_v2.6_seed407.png) |
| position_s4.2 | 408 | [XY](figures/position_s4.2_v2.6_seed408.png) |
| position_s4.5 | 401 | [XY](figures/position_s4.5_v2.6_seed401.png) |
| position_s4.5 | 402 | [XY](figures/position_s4.5_v2.6_seed402.png) |
| position_s4.5 | 403 | [XY](figures/position_s4.5_v2.6_seed403.png) |
| position_s4.5 | 404 | [XY](figures/position_s4.5_v2.6_seed404.png) |
| position_s4.5 | 405 | [XY](figures/position_s4.5_v2.6_seed405.png) |
| position_s4.5 | 406 | [XY](figures/position_s4.5_v2.6_seed406.png) |
| position_s4.5 | 407 | [XY](figures/position_s4.5_v2.6_seed407.png) |
| position_s4.5 | 408 | [XY](figures/position_s4.5_v2.6_seed408.png) |
