# Fixed u200 high-speed preference test

[中文分析与结论](ANALYSIS.md) · [弯道XY局部放大](TURN_XY_ZOOM.png)

![All cases XY](ALL_XY.png)

![All cases per-step reward](ALL_STEP_REWARDS.png)

u200 sharedActor / frozenlocal300 / seed911401. No training. All speed levels are out of the original2.4–2.6strong training range. R2 left/right90deg, common10m prefix,10s horizon.3.0 retains cap3.0;3.2/3.5 explicitly use evaluation-only reference cap3.5; physical/steer/residual magnitude/filter/slew limits unchanged. No external disturbance.

## All36 physical episodes

Turn mean speed below uses complete original-turn equal spatial bins. Missing means incomplete bins, not zero error. Outward maximum uses the observed turn; roll uses the whole real prefix. Duration/failure flags must be read before comparing. Policy total curves use the actual logged scored_tick_reward; baseline curves use same-alpha reconstruction. Float32 failure-tail rounding is retained in the logged policy totals and failure component.

| Case | Method | Seconds | Approach v | Turn mean v | Turn speed RMSE | Turn path RMSE | Mean Δv | Observed turn outward max | Peak roll | Physical/domain fail | Goal/hold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| left90_R2_3.0 | B0 | 10.000 | 2.9796 | 2.8923 | 0.1207 | 0.1434 | 0.0000 | 0.0000 | 0.5476 | False/False | True/False |
| left90_R2_3.0 | alpha_0 | 10.000 | 2.9786 | 2.8804 | 0.1407 | 0.0654 | -0.0390 | 0.1237 | 0.3979 | False/False | True/True |
| left90_R2_3.0 | alpha_0p25 | 10.000 | 2.9786 | 2.8859 | 0.1334 | 0.0643 | -0.0320 | 0.1200 | 0.4035 | False/False | True/True |
| left90_R2_3.0 | alpha_0p5 | 10.000 | 2.9786 | 2.8912 | 0.1263 | 0.0638 | -0.0257 | 0.1180 | 0.4090 | False/False | True/True |
| left90_R2_3.0 | alpha_0p75 | 10.000 | 2.9785 | 2.8958 | 0.1199 | 0.0629 | -0.0201 | 0.1139 | 0.4142 | False/False | True/True |
| left90_R2_3.0 | alpha_1 | 10.000 | 2.9785 | 2.9000 | 0.1141 | 0.0624 | -0.0151 | 0.1116 | 0.4190 | False/False | True/True |
| right90_R2_3.0 | B0 | 9.400 | 2.9796 | 2.8980 | 0.1128 | 0.0687 | 0.0000 | 0.0000 | 0.5069 | False/True | True/False |
| right90_R2_3.0 | alpha_0 | 10.000 | 2.9786 | 2.8810 | 0.1455 | 0.0795 | -0.0497 | 0.1071 | 0.4042 | False/False | True/True |
| right90_R2_3.0 | alpha_0p25 | 10.000 | 2.9786 | 2.8897 | 0.1329 | 0.0775 | -0.0381 | 0.1055 | 0.4117 | False/False | True/True |
| right90_R2_3.0 | alpha_0p5 | 10.000 | 2.9786 | 2.8983 | 0.1198 | 0.0756 | -0.0269 | 0.1031 | 0.4192 | False/False | True/True |
| right90_R2_3.0 | alpha_0p75 | 10.000 | 2.9785 | 2.9051 | 0.1091 | 0.0735 | -0.0173 | 0.0997 | 0.4266 | False/False | True/True |
| right90_R2_3.0 | alpha_1 | 10.000 | 2.9785 | 2.9101 | 0.1004 | 0.0710 | -0.0092 | 0.0929 | 0.4336 | False/False | True/True |
| left90_R2_3.2 | B0 | 8.340 | 3.1799 | 3.0528 | 0.1609 | 0.1419 | 0.0000 | 0.0000 | 0.6160 | False/True | True/False |
| left90_R2_3.2 | alpha_0 | 10.000 | 3.2143 | 3.0518 | 0.1822 | 0.0724 | -0.0574 | 0.1560 | 0.4573 | False/False | True/True |
| left90_R2_3.2 | alpha_0p25 | 10.000 | 3.2150 | 3.0587 | 0.1735 | 0.0732 | -0.0471 | 0.1543 | 0.4637 | False/False | True/True |
| left90_R2_3.2 | alpha_0p5 | 10.000 | 3.2157 | 3.0656 | 0.1641 | 0.0734 | -0.0374 | 0.1509 | 0.4700 | False/False | True/True |
| left90_R2_3.2 | alpha_0p75 | 10.000 | 3.2164 | 3.0717 | 0.1558 | 0.0749 | -0.0278 | 0.1503 | 0.4759 | False/False | True/True |
| left90_R2_3.2 | alpha_1 | 10.000 | 3.2171 | 3.0758 | 0.1495 | 0.0746 | -0.0208 | 0.1457 | 0.4823 | False/False | True/True |
| left90_R2_3.5 | B0 | 3.970 | 3.4287 | — | — | — | 0.0000 | 0.0000 | 0.7001 | True/False | False/False |
| left90_R2_3.5 | alpha_0 | 10.000 | 3.4468 | 3.2706 | 0.2589 | 0.1620 | -0.1094 | 0.2666 | 0.5371 | False/False | True/False |
| left90_R2_3.5 | alpha_0p25 | 10.000 | 3.4471 | 3.2817 | 0.2468 | 0.1656 | -0.0950 | 0.2712 | 0.5436 | False/False | True/False |
| left90_R2_3.5 | alpha_0p5 | 10.000 | 3.4475 | 3.2891 | 0.2371 | 0.1684 | -0.0825 | 0.2739 | 0.5511 | False/False | True/False |
| left90_R2_3.5 | alpha_0p75 | 10.000 | 3.4479 | 3.2988 | 0.2258 | 0.1717 | -0.0702 | 0.2783 | 0.5556 | False/False | True/False |
| left90_R2_3.5 | alpha_1 | 10.000 | 3.4466 | 3.3065 | 0.2164 | 0.1747 | -0.0587 | 0.2824 | 0.5591 | False/False | True/False |
| right90_R2_3.2 | B0 | 6.740 | 3.1799 | 3.0627 | 0.1532 | 0.0513 | 0.0000 | 0.0461 | 0.5894 | False/True | True/False |
| right90_R2_3.2 | alpha_0 | 10.000 | 3.2143 | 3.0526 | 0.1825 | 0.0773 | -0.0684 | 0.1388 | 0.4557 | False/False | True/True |
| right90_R2_3.2 | alpha_0p25 | 10.000 | 3.2150 | 3.0656 | 0.1663 | 0.0779 | -0.0508 | 0.1391 | 0.4637 | False/False | True/True |
| right90_R2_3.2 | alpha_0p5 | 10.000 | 3.2157 | 3.0773 | 0.1508 | 0.0786 | -0.0343 | 0.1392 | 0.4717 | False/False | True/True |
| right90_R2_3.2 | alpha_0p75 | 10.000 | 3.2164 | 3.0884 | 0.1355 | 0.0791 | -0.0192 | 0.1381 | 0.4797 | False/False | True/True |
| right90_R2_3.2 | alpha_1 | 10.000 | 3.2171 | 3.0980 | 0.1230 | 0.0793 | -0.0060 | 0.1354 | 0.4879 | False/False | True/False |
| right90_R2_3.5 | B0 | 4.035 | 3.4287 | — | — | — | 0.0000 | 0.1981 | 0.7002 | True/False | False/False |
| right90_R2_3.5 | alpha_0 | 10.000 | 3.4468 | 3.2903 | 0.2413 | 0.1050 | -0.1059 | 0.1935 | 0.5221 | False/False | True/False |
| right90_R2_3.5 | alpha_0p25 | 10.000 | 3.4471 | 3.3073 | 0.2221 | 0.1090 | -0.0847 | 0.1980 | 0.5315 | False/False | True/False |
| right90_R2_3.5 | alpha_0p5 | 8.560 | 3.4475 | 3.3211 | 0.2046 | 0.1132 | -0.0642 | 0.2035 | 0.5411 | False/True | True/False |
| right90_R2_3.5 | alpha_0p75 | 8.620 | 3.4479 | 3.3327 | 0.1892 | 0.1171 | -0.0474 | 0.2060 | 0.5510 | False/True | True/False |
| right90_R2_3.5 | alpha_1 | 8.780 | 3.4466 | 3.3420 | 0.1765 | 0.1202 | -0.0326 | 0.2070 | 0.5613 | False/True | True/False |

## Complete paired figures and raw evidence

- left90_R2_3.0: [XY/physical/errors](left90_R2_3.0_physical.png) · [step/cumulative rewards](left90_R2_3.0_rewards.png) · [signed components](left90_R2_3.0_components.png) · [cumulative components](left90_R2_3.0_cumulative_components.png) · [request/nominal/governed/actual/wheel speed](left90_R2_3.0_speed_chain.png)
- right90_R2_3.0: [XY/physical/errors](right90_R2_3.0_physical.png) · [step/cumulative rewards](right90_R2_3.0_rewards.png) · [signed components](right90_R2_3.0_components.png) · [cumulative components](right90_R2_3.0_cumulative_components.png) · [request/nominal/governed/actual/wheel speed](right90_R2_3.0_speed_chain.png)
- left90_R2_3.2: [XY/physical/errors](left90_R2_3.2_physical.png) · [step/cumulative rewards](left90_R2_3.2_rewards.png) · [signed components](left90_R2_3.2_components.png) · [cumulative components](left90_R2_3.2_cumulative_components.png) · [request/nominal/governed/actual/wheel speed](left90_R2_3.2_speed_chain.png)
- left90_R2_3.5: [XY/physical/errors](left90_R2_3.5_physical.png) · [step/cumulative rewards](left90_R2_3.5_rewards.png) · [signed components](left90_R2_3.5_components.png) · [cumulative components](left90_R2_3.5_cumulative_components.png) · [request/nominal/governed/actual/wheel speed](left90_R2_3.5_speed_chain.png)
- right90_R2_3.2: [XY/physical/errors](right90_R2_3.2_physical.png) · [step/cumulative rewards](right90_R2_3.2_rewards.png) · [signed components](right90_R2_3.2_components.png) · [cumulative components](right90_R2_3.2_cumulative_components.png) · [request/nominal/governed/actual/wheel speed](right90_R2_3.2_speed_chain.png)
- right90_R2_3.5: [XY/physical/errors](right90_R2_3.5_physical.png) · [step/cumulative rewards](right90_R2_3.5_rewards.png) · [signed components](right90_R2_3.5_components.png) · [cumulative components](right90_R2_3.5_cumulative_components.png) · [request/nominal/governed/actual/wheel speed](right90_R2_3.5_speed_chain.png)

Independent reward reconstruction max abs error=0.0041867448; normal ticks=3.0820248e-07, terminal tail abs=0.0041867448/relative=6.4124211e-06 (float32 simulation versus NumPy reconstruction). [All metrics and signed reward component sums](metrics.json). Original step data (local, not copied into Git): /home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/path_coverage_highspeed_probe_20261020/cap3p0|cap3p5/baseline and policy/alpha_*/case.npz, plus immutable original paths. No duplicatedCSV/PDF/ZIP or SHA scans. A physical/domain exit is recorded as failure, not certified speed-priority behavior.
