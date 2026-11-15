# Matched direct/residual full-chain diagnostic

All actual/reference/I/C XY: [map06,36 cases](map06/README.md), [map07,36](map07/README.md), [mechanism map48,18 candidates](map48/README.md). Plot limits include the diverging predictions; failed predictions are not cropped to the road.

|Model structure / interface|DEV feasible admitted /72|Map48 feasible admitted /4|Map48 false admitted /14|Map48 selected completion /2|
|---|---:|---:|---:|---:|
|Residual / C|52|1|1|0|
|Residual / I|72|4|1|1|
|Direct / C|0|0|0|0|
|Direct / I|72|4|0|2|

Existing direct ground checkpoints were chosen by their original1s DEV criterion: approach seed0/update900, departure seed0/update600. RB and all physical/control/task contracts are shared. Direct latent state is not interchangeable with residual latent state, so a matched history bridge was trained with identical72TRAIN/24DEV, three seeds×600 updates, batch64/lr3e-4/clip1. Best updates325/500/600; seed1/update500 selected by normalized entry RMSE0.2138. No new control/world-model training or physics.

[All bridge checkpoint curves](bridge_curves.png) and [best/last diagnostics](bridge_convergence.json) are retained. Different seeds peak at different updates, including the boundary600; this is a bounded pilot, not formal convergence. TensorBoard6006 loaded all three runs.

Direct C loses all72 true-positive candidates: every one has invalid predicted posture, departure domain and missing final goal hold. Coordinates also diverge vertically (map06 minimum predicted z≈−5.60m), a model failure, not a physical execution. I uses coherent observed TRAIN interfaces and remains accurate in this limited set. These results do not identify a unique causal state component or prove universal residual superiority. They do show that lower single-stage error or an interface bridge alone cannot certify full-chain prediction.

Own, common, cross-interface and cross-structure calibrated margins were also applied. Every expanded configuration rejects all18 mechanism-map candidates, including all four feasible ones. See [direct margins/results](direct_calibration/summary.json) and [residual cross-structure results](residual_structure_cross/summary.json). The same failure remains under common margins; no margin shrinking is adopted.

Keep residual C as the less-failing development candidate, not an accepted planner. Do not replace it with direct C. Next work targets coherent physical/history propagation and free-running full-chain losses; retain I as the strong comparator. P4 is not accepted, and P5/P6 have not started.
