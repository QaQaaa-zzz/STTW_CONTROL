# P4 first C-only full-chain diagnostic

[All18 actual/reference/predicted XY](xy.png).

True admit2, false admit1, false reject2, true reject13;18conditions total, four actually feasible.

First C-only raw-envelope DEV diagnostic; no I/G/strong-backtracking comparison or new margin calibration; full approach/departure tracking-domain and all-stage posture/obstacle gates included

Only common first-frame physical/controller/RB histories enter the predictor. Ground states and estimated history feed RB; predicted RB event state/history feed departure. Real future trajectories/status are used only for offline comparison. Initial RB support assumes the declared flat approach zone; STTW locked hip/knee rates are zero in the ground model.

This is an engineering/DEV result, not deployment or independent TEST. No new physical episodes: actual outcomes reuse the full-snapshot Oracle. G/I/Oracle, same-candidate strong backtracking, calibration and actual selected-plan verification remain required.

## Error diagnosis and next gate

The false admission is root1002 / approach0 / departure1: predicted minimum raw clearance +0.00739m, but the actual approach hits a forbidden obstacle before jumping. This is not a successful jump followed by a failed departure. Both false rejections are root1001 / approach1 / departure0–1: the predicted goal hold is missing; departure0 also has negative predicted clearance. Original 10s/0.5s hold criteria remain unchanged.

History bridge: 72 TRAIN and 24 DEV approach sequences, three seeds with 600 updates each; best updates575/375/600 and normalized entry RMSE0.1730/0.1874/0.1745. Seed0 is used here. These are estimated controller/RB histories, not measured future states. Training completion does not establish convergence or safe planning. Raw logs/checkpoints reside in `taskbook_20261115_p4_history_bridge`; [all seed metrics](history_bridge_summary.json).

P4 remains open. Next: compare G/I/C/Oracle with the same candidates, predictor and constraints, including strong backtracking; diagnose stage occupancy errors and calibrate using declared DEV rules; verify selected-plan outcomes. P5/P6 do not start before this stage is assessed.

**Trigger-contract correction:** these historical predictions used hardcoded8.60m; the executed frozen configuration uses8.55m. Corrected prediction results are in [trigger_fix](trigger_fix/README.md). Original artifacts remain unchanged as evidence of the error.
