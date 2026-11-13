# STTW turning capability diagnostics

[Original map: all actual/reference XY and 72 cases](evaluation_ccd512/RESULTS.md)

[Reduced maximum ridge: all actual/reference XY and 72 cases](evaluation_ridge40mm/RESULTS.md)

Original map: 41/72 goal holds, straight 18/18, turns 23/54. Reduced highest ridge 8cm to 4cm: 45/72; fixed alpha0 12/12 versus original 10/12. Zero-middle 6/12 in both. Selected alpha0 is a DEV candidate, not established robustness. Unchanged terrain cases can differ near thresholds between GPU batches; aggregate differences are descriptive, not an isolated deterministic causal estimate.

Initial CCD-overflow run is INVALID and preserved externally. Both reported runs have no overflow in captured runtime logs. Same locked pendulum, implicitfast, friction0.8, original wheels, position steering, 2m/s requests; no training. Frozen H75 and zero-middle ECBC/ESO, not Local300. No task reward implemented; no reward-performance claim. Raw configurations, trajectories, declarations and logs: /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_isaac_recovery/turning_suite

[Robustness: all 216 trajectories](robustness/evaluation/RESULTS.md): alpha0 98/108, zero-middle51/108. [Rejected 0.5s lookahead](robustness/evaluation_lookahead05/RESULTS.md):69/108 and15/108. Retain0.7s. [Isaac shared-protocol comparison](evaluation_isaac/RESULTS.md):13/72 versus Playground41/72. This cold-start comparison is not historical warmed/landing-entry performance and does not by itself prove restoration. Same terrain arrays, routes, initial body speed and zero wheel speed, locked pendulum and evaluation/controller loop; native Isaac velocity steering/2.5ms and Playground position steering/1ms differ explicitly.
