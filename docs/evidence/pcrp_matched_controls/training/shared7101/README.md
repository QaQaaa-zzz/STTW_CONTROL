# Shared PCRP seed 7101, update 150

Each cell gives full 10 s / final hold / tasks A, B and C. The five columns use one fixed actor. Physical groups are paired across all policies. The CSV includes all current and archived runs and computes speed and path RMSE over the common observation window of all twelve lanes per group.

| Terrain | Model | n per preference | 0 | .25 | .5 | .75 | 1 |
|---|---|---:|---:|---:|---:|---:|---:|
| flat_original | shared7101 | 8 | 0/0/0/0/0 | 0/0/0/0/0 | 5/0/0/0/0 | 2/0/0/0/0 | 7/0/0/0/0 |
| flat_original | H75 | 8 | 8/8/8/0/8 | 8/8/8/0/8 | 8/8/8/8/8 | 8/8/0/8/8 | 8/8/0/8/0 |
| gravel010_layout0 | shared7101 | 4 | 2/0/0/0/0 | 2/0/0/0/0 | 2/0/0/0/0 | 1/0/0/0/0 | 1/0/0/0/0 |
| gravel010_layout0 | H75 | 4 | 3/3/0/0/0 | 1/1/0/0/0 | 4/4/1/0/1 | 3/3/0/2/2 | 4/2/0/2/1 |
| gravel010_layout1 | shared7101 | 4 | 2/0/0/0/0 | 1/0/0/0/0 | 2/0/0/0/0 | 2/0/0/0/0 | 2/0/0/0/0 |
| gravel010_layout1 | H75 | 4 | 2/2/0/0/0 | 3/3/1/0/0 | 2/1/0/0/0 | 4/3/1/1/0 | 3/3/1/2/1 |
| position_s3.9 | shared7101 | 8 | 0/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 |
| position_s3.9 | H75 | 8 | 8/8/5/0/2 | 5/5/0/0/0 | 2/2/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 |
| position_s4.2 | shared7101 | 8 | 0/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 |
| position_s4.2 | H75 | 8 | 1/0/0/0/0 | 8/8/1/0/5 | 8/8/0/0/8 | 1/1/0/0/0 | 1/1/0/0/0 |
| position_s4.5 | shared7101 | 8 | 0/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 |
| position_s4.5 | H75 | 8 | 0/0/0/0/0 | 0/0/0/0/0 | 7/5/3/0/4 | 0/0/0/0/0 | 8/4/0/0/0 |

Among 31 complete runs, the last-second limits were exceeded in 31 for lateral error, 0 for heading and 0 for speed.

Endpoint classifications for the new seed: contact_failure (5), domain_exit (94), full_window (31), roll_limit_failure (70).

[Per-lane physical metrics and raw trajectory paths](episodes.csv). Detailed local diagnostics include the paired Zero trajectory, five preference-specific reward reconstructions, the original reference and actual endpoints.

| Terrain | Physical group | Both actors and controls | Detailed diagnostics |
|---|---:|---|---|
| position_s3.9 | 401 | [XY](figures/position_s3.9_v2.6_seed401.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s3.9_v2.6_seed401/overview.png) |
| position_s3.9 | 402 | [XY](figures/position_s3.9_v2.6_seed402.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s3.9_v2.6_seed402/overview.png) |
| position_s3.9 | 403 | [XY](figures/position_s3.9_v2.6_seed403.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s3.9_v2.6_seed403/overview.png) |
| position_s3.9 | 404 | [XY](figures/position_s3.9_v2.6_seed404.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s3.9_v2.6_seed404/overview.png) |
| position_s3.9 | 405 | [XY](figures/position_s3.9_v2.6_seed405.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s3.9_v2.6_seed405/overview.png) |
| position_s3.9 | 406 | [XY](figures/position_s3.9_v2.6_seed406.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s3.9_v2.6_seed406/overview.png) |
| position_s3.9 | 407 | [XY](figures/position_s3.9_v2.6_seed407.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s3.9_v2.6_seed407/overview.png) |
| position_s3.9 | 408 | [XY](figures/position_s3.9_v2.6_seed408.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s3.9_v2.6_seed408/overview.png) |
| position_s4.2 | 401 | [XY](figures/position_s4.2_v2.6_seed401.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.2_v2.6_seed401/overview.png) |
| position_s4.2 | 402 | [XY](figures/position_s4.2_v2.6_seed402.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.2_v2.6_seed402/overview.png) |
| position_s4.2 | 403 | [XY](figures/position_s4.2_v2.6_seed403.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.2_v2.6_seed403/overview.png) |
| position_s4.2 | 404 | [XY](figures/position_s4.2_v2.6_seed404.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.2_v2.6_seed404/overview.png) |
| position_s4.2 | 405 | [XY](figures/position_s4.2_v2.6_seed405.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.2_v2.6_seed405/overview.png) |
| position_s4.2 | 406 | [XY](figures/position_s4.2_v2.6_seed406.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.2_v2.6_seed406/overview.png) |
| position_s4.2 | 407 | [XY](figures/position_s4.2_v2.6_seed407.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.2_v2.6_seed407/overview.png) |
| position_s4.2 | 408 | [XY](figures/position_s4.2_v2.6_seed408.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.2_v2.6_seed408/overview.png) |
| position_s4.5 | 401 | [XY](figures/position_s4.5_v2.6_seed401.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.5_v2.6_seed401/overview.png) |
| position_s4.5 | 402 | [XY](figures/position_s4.5_v2.6_seed402.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.5_v2.6_seed402/overview.png) |
| position_s4.5 | 403 | [XY](figures/position_s4.5_v2.6_seed403.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.5_v2.6_seed403/overview.png) |
| position_s4.5 | 404 | [XY](figures/position_s4.5_v2.6_seed404.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.5_v2.6_seed404/overview.png) |
| position_s4.5 | 405 | [XY](figures/position_s4.5_v2.6_seed405.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.5_v2.6_seed405/overview.png) |
| position_s4.5 | 406 | [XY](figures/position_s4.5_v2.6_seed406.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.5_v2.6_seed406/overview.png) |
| position_s4.5 | 407 | [XY](figures/position_s4.5_v2.6_seed407.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.5_v2.6_seed407/overview.png) |
| position_s4.5 | 408 | [XY](figures/position_s4.5_v2.6_seed408.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/position_s4.5_v2.6_seed408/overview.png) |
| flat_original | 401 | [XY](figures/flat_original_v2.6_seed401.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/flat_original_v2.6_seed401/overview.png) |
| flat_original | 402 | [XY](figures/flat_original_v2.6_seed402.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/flat_original_v2.6_seed402/overview.png) |
| flat_original | 403 | [XY](figures/flat_original_v2.6_seed403.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/flat_original_v2.6_seed403/overview.png) |
| flat_original | 404 | [XY](figures/flat_original_v2.6_seed404.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/flat_original_v2.6_seed404/overview.png) |
| flat_original | 405 | [XY](figures/flat_original_v2.6_seed405.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/flat_original_v2.6_seed405/overview.png) |
| flat_original | 406 | [XY](figures/flat_original_v2.6_seed406.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/flat_original_v2.6_seed406/overview.png) |
| flat_original | 407 | [XY](figures/flat_original_v2.6_seed407.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/flat_original_v2.6_seed407/overview.png) |
| flat_original | 408 | [XY](figures/flat_original_v2.6_seed408.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/flat_original_v2.6_seed408/overview.png) |
| gravel010_layout0 | 401 | [XY](figures/gravel010_v2.6_layout0_seed401.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/gravel010_v2.6_layout0_seed401/overview.png) |
| gravel010_layout0 | 402 | [XY](figures/gravel010_v2.6_layout0_seed402.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/gravel010_v2.6_layout0_seed402/overview.png) |
| gravel010_layout0 | 403 | [XY](figures/gravel010_v2.6_layout0_seed403.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/gravel010_v2.6_layout0_seed403/overview.png) |
| gravel010_layout0 | 404 | [XY](figures/gravel010_v2.6_layout0_seed404.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/gravel010_v2.6_layout0_seed404/overview.png) |
| gravel010_layout1 | 401 | [XY](figures/gravel010_v2.6_layout1_seed401.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/gravel010_v2.6_layout1_seed401/overview.png) |
| gravel010_layout1 | 402 | [XY](figures/gravel010_v2.6_layout1_seed402.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/gravel010_v2.6_layout1_seed402/overview.png) |
| gravel010_layout1 | 403 | [XY](figures/gravel010_v2.6_layout1_seed403.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/gravel010_v2.6_layout1_seed403/overview.png) |
| gravel010_layout1 | 404 | [XY](figures/gravel010_v2.6_layout1_seed404.png) | [Reward and errors](/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_train_20261121/shared7101_eval/diagnostics/gravel010_v2.6_layout1_seed404/overview.png) |
