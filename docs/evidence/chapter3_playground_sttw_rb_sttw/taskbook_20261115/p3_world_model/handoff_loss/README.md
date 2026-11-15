# P3 supervised full-joint model: true-entry DEV assessment

[Checkpoint curves](learning_curves.png). Full data and model source: `/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p3_world_handoffloss`. No policy updates.

- [All36 map06, seed0 XY](map06_seed0_xy.png)
- [All36 map07, seed0 XY](map07_seed0_xy.png)
- [All36 map06, seed1 XY](map06_seed1_xy.png)
- [All36 map07, seed1 XY](map07_seed1_xy.png)
- [All36 map06, seed2 XY](map06_seed2_xy.png)
- [All36 map07, seed2 XY](map07_seed2_xy.png)

|Seed|Checkpoint|Mean full-segment max XYZ (m)|Worst XYZ (m)|Missing handoff /72|
|---|---|---|---|---|
|0|2275|0.0424|0.0787|0/72|
|1|2300|0.0563|0.0906|0/72|
|2|2175|0.0427|0.0795|0/72|

All72 DEV sequences per seed retained. Same-domain generated terrain layouts; not novel terrain types or independent TEST. Full-record duration is used only to score prediction, not as model input. Controller/RB histories are learned estimates after initialization. This report does not establish planning utility or P3 completion.


## Decision and event audit

[All12 downstream XY, confusion counts and unchanged margins](continuation/README.md).

Named-joint loss mapping fixes the native rear/steer/front order. Rotation now uses a0.2rad scale; an explicit handoff-transition state loss is included. Old results stay in `../interchange` and `../full_entry`. Changes concern supervised loss only; frozen controllers, physics and success thresholds are unchanged.

Seed0/checkpoint2275 is a provisional DEV candidate: raw geometry admits all4feasible and rejects all8failed cases, whereas seeds1/2 admit1/0feasible. All three remain no-plan under the old frozen M1 margin. This is not a calibrated full-chain system or independent TEST result. Candidate identity is in candidate_selection.json.

Seed0 predicted handoff time MAE36.4ms, worst95ms; handoff position error mean6.74cm, worst19.24cm; collision-geometry AABB corner discrepancy mean8.71cm. Seed1 worsened substantially (handoff position mean1.20m), retained in event_summary.json.

**P3 remains incomplete: takeoff event sampling.** Raw5ms wheel-contact flags contain1–2no-wheel-contact samples in each of72DEV records. Subsampling to20ms retains such a sample in only13records. Therefore the current20ms takeoff readout is not a valid all72event evaluation. Original records are preserved; short-event labels must be corrected before claiming complete contact-transition prediction. This is a brief contact-flag gap, not evidence of a sustained aerial jump. Far-contact timing for seed0 is14.7ms MAE/40ms worst across72cases.
