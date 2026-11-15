# P4 G/I/C/Oracle: first matched DEV diagnostic

[All18 actual/reference/I/C XY](xy.png). No new physical episodes; real outcomes reuse P2 full-snapshot branches.

|Method|True admit|False admit|False reject|True reject|Selected completed /2|No plan /2|Admitted failure /2|
|---|---:|---:|---:|---:|---:|---:|---:|
|G|0|0|4|14|0|2|0|
|I|4|1|0|13|1|0|1|
|C|2|1|2|13|1|1|0|
|Oracle|4|0|0|14|2|0|0|

I correctly admits all four feasible candidates, C only two: no evidence that continuous transfer improves this DEV matrix. Both select one successful task out of two, through different choices. I selects the false-admitted straight approach on root1002 and collides before jumping; C has no plan on root1001 and succeeds on root1002. No-plan is reported as a decision outcome, not a safe stopping policy or a simulated collision. Oracle completes both roots using approach1/departure0.

I uses one coherent observed TRAIN interface per upstream/speed/phase, selected by proximity to the TRAIN median xyz/full velocity; its physical state, quaternion and histories stay paired. It does not read actual DEV future states. C propagates predicted states/history throughout. G follows the reference at1.8m/s and fixed height with level posture/locked joints; it has no dynamic or aerial prediction. G rejects all candidates under the same raw body-envelope obstacle/world constraints.

Full backtracking traverses all nine candidates per root using exactly the same cached predictor, gates and cost as exhaustive joint selection; all eight decisions coincide. This establishes equivalence for this finite implementation, not an advantage in planning quality or a new physical experiment. `summary.json` separates model-query wall time (including loading) and cached selection time; GPU timing is not a deployment latency claim.

Real obstacle clearance and failure stage are preserved in decisions.json. Successful sources report0m for some samples in the existing native geometry-distance audit; do not interpret those values as a positive safety margin. Only two roots on one known DEV map;18 candidates share prefixes and are not18 independent task trials. No bootstrap/generalization claim.

P4 remains incomplete: per-model and shared/cross-margin calibration, direct/residual structural comparison, larger frozen validation and stage-error/occupancy diagnosis remain. These raw-envelope diagnostics do not replace calibrated final results.

[Stage-event state errors](stage_errors.csv) compare predicted versus actual RB-to-ground events (including event-time difference); approach failures are compared at the actual failure time. Event-state errors are not same-time trajectory errors. Downstream variants share the same preceding event and must not be counted as independent samples.
