# Single-alpha final-stage review

Existing stage3 panels only. Practical sensitivity: final1s path<=0.20m, heading<=0.05rad, speed<=0.10m/s, peak roll<=0.33rad, no physical/domain failure and goal reached. Original labels remain in each source.

| Checkpoint panel | Practical pass | Expected turn trade-offs | Max final path error | Peak roll |
|---|---:|---:|---:|---:|
| [update_0300_stage3](update_0300_stage3/REPORT.md) | 30/30 | 2/4 | 0.142m | 0.306rad |
| [update_0325_stage3](update_0325_stage3/REPORT.md) | 30/30 | 3/4 | 0.064m | 0.298rad |
| [update_0350_stage3](update_0350_stage3/REPORT.md) | 30/30 | 3/4 | 0.144m | 0.312rad |
| [update_0375_stage3](update_0375_stage3/REPORT.md) | 30/30 | 4/4 | 0.185m | 0.323rad |
| [update_0400_stage3](update_0400_stage3/REPORT.md) | 24/30 | 4/4 | 0.240m | 0.324rad |

Next decision: inspect final and prior candidate XY/velocity curves, retain useful speed-path trade-offs, and distinguish tolerable steady bias from failures requiring training changes. This report alone does not mark user satisfaction or adoption.
