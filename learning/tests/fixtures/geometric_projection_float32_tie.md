# Float32 nearest-segment regression fixture

Source: `runs/geometric_random_recovery_20260918_continue200/standard_best252/evaluation/alpha_0/seed_49001/tight_turn__nominal/baseline/trace.npz`.

The NPZ retains only the first 1601 samples of the committed reference pose/command, actual pose, geometric features and projection progress. No model, optimizer, image or unrelated training state is included. Control dt is 0.005 s. This is an engineering regression fixture, not a performance dataset.

At step 1522, NumPy replay picks adjacent segment 1469 rather than runtime segment 1468. The maximum prefix differences are approximately 0.001041 rad heading and 0.001429 1/m curvature despite sub-micrometre cross-track differences. Replaying from zero progress with runtime JAX arithmetic matches the recorded features and progress exactly. The test also corrupts lateral error by 0.01 m and requires rejection; the diagnostic tolerance is unchanged.
