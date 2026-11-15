# Full-chain original DEV coverage

[Map06: all36 actual/reference/I/C XY](map06/README.md) · [Map07: all36](map07/README.md).

|DEV map|Actual successful|C feasible admitted|I feasible admitted|C false rejections|
|---|---:|---:|---:|---:|
|06|36/36|28/36|36/36|8|
|07|36/36|24/36|36/36|12|
|Total|72/72|52/72|72/72|20|

All20 C rejections lack predicted final goal hold within10s. None is rejected by world bounds, departure domain or missing events in this set. All original actual trajectories succeed; there are no true-negative candidates here, so this set cannot measure false-admission control. The known obstacle map48 remains a separate diagnostic.

Same old TRAIN/DEV map split, frozen models, route/speed-conditioned TRAIN typical interfaces and original8.55m trigger. No new training or physical episodes. No independent TEST or new terrain-family claim. Empty forbidden-box maps keep world bounds and task gates; first attempt assumed a box list and failed before writing results, recorded in raw failed_attempt.json and repaired without changing physics.

These outcomes reinforce that chained prediction currently underperforms coherent TRAIN-typical interfaces for goal-hold prediction. Calibration cannot by itself repair a wrong center trajectory or missing goal hold. Next: preserve these negatives while checking full-chain state/occupancy errors and same-rule model/shared margins; retain direct/residual comparison.
