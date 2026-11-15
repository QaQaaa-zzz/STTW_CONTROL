# Execution-consistent handoff history reset

All actual/reference/before/after XY: residual [map06](residual/map06/README.md), [map07](residual/map07/README.md), [map48](residual/map48/README.md); direct [map06](direct/map06/README.md), [map07](direct/map07/README.md), [map48](direct/map48/README.md).

The frozen real runner already invokes `middle.reset_mask` at RB return under `sttw_only_after_lock`. Actual handoff history has lane_ticks0 and cleared old frames; learned prediction instead carried a count around1090 and stale frames. The predictor now calls that same reset implementation using predicted qpos/qvel and fixed route/controller settings. This changes prediction only, not physical execution, actuator ownership or task criteria.

On12 saved actual event states, reconstructing the reset differs by at most0.00095849 in gyro-derived frame/ESO/filtered-roll-rate terms; remaining fields are within1e-5. Predicted body angular velocity is the gyro proxy. This is measured approximate parity, not bitwise sensor equivalence. Actual future is used only in this offline audit.

The semantic correction is retained despite no performance gain: residual C admissions decrease52→44 out of72 actually successful DEV trajectories, mechanism-map true/false admissions1/1→0/0. Direct C remains0/72. The missing reset was real but not the sole cause.

Predicted phase2 also contains hip errors around0.2rad, while actual native handoff uses position0.01rad and velocity0.1rad/s tolerances. Hip/knee values are not projected to ideal home and measured future states are not injected. Current classifier stop does not establish physical lock readiness. Subsequent bounded predictor training addresses upstream exposure and transition-state errors.
