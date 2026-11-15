# P3 supervised full-joint model: true-entry DEV assessment

[Checkpoint curves](learning_curves.png). Full data and model source: `/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p3_world_fullentry`. No policy updates.

- [All36 map06, seed0 XY](map06_seed0_xy.png)
- [All36 map07, seed0 XY](map07_seed0_xy.png)
- [All36 map06, seed1 XY](map06_seed1_xy.png)
- [All36 map07, seed1 XY](map07_seed1_xy.png)
- [All36 map06, seed2 XY](map06_seed2_xy.png)
- [All36 map07, seed2 XY](map07_seed2_xy.png)

|Seed|Checkpoint|Mean full-segment max XYZ (m)|Worst XYZ (m)|Missing handoff /72|
|---|---|---|---|---|
|0|1800|0.0615|0.0838|0/72|
|1|1650|0.0909|0.1264|0/72|
|2|1800|0.0703|0.1150|0/72|

All72 DEV sequences per seed retained. Same-domain generated terrain layouts; not novel terrain types or independent TEST. Full-record duration is used only to score prediction, not as model input. Controller/RB histories are learned estimates after initialization. This report does not establish planning utility or P3 completion.


## Predicted event stop (no actual stop time as input)

The model queries at most125 steps and selects its first predicted STTW phase. All72 cases per seed report a handoff. Each predicted handoff state is compared with the recorded actual handoff, so this metric includes event timing error.

|Seed|Event MAE (s)|Worst event error (s)|Mean handoff position error (m)|Mean geometry AABB corner error (m)|
|---|---|---|---|---|
|0|0.0739|0.175|0.1468|0.1770|
|1|0.0903|0.705|0.2158|0.2474|
|2|0.0381|0.190|0.0984|0.1428|

These are different from the same-time trajectory metrics above. Seed1's severe premature handoff remains in the evidence. Predicted controller/RB history errors and all-joint errors are preserved in `predicted_stop_seed*.json`. AABB discrepancies are not collision safety guarantees. Phase/event quality and downstream decisions still require validation; no P4 planning-benefit claim.

Two implementation tests passed: serialized100=40+60 prediction and GPU predicted-event stop/missing-event behavior. Actual training/checkpoint logs remain on disk. No reward or PPO convergence claim: this is supervised prediction training. Best checkpoints are selected among saved25-update candidates under the declared DEV objective; they are not guaranteed global optima.
