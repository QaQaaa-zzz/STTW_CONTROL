## Current speed input and fit scope

Both ordered phase-speed pairs (1.8→1.9 and1.9→1.8) complete72/72; the runner changes request on actual RB handback. Physical initial speed magnitude is1.8m/s, rotated by initial yaw. A preparation manifest's2m/s description was incorrect and has a separate correction record; the physics was unchanged.

The supervised predictor now reads each trajectory's actual phase2 request in both training features and free rollout. Additional datasets keep source-prefixed case IDs and their existing TRAIN/DEV roots. The new two-speed fit uses18,997TRAIN transitions from108trajectories and36DEV cases, matchedM0/M1/M2;600updates per seed and25min total cap. All six fits completed in1026.087s; full geometry and withheld phase-switch evaluation completed in288.200s. Long-horizon tail errors remain, so the predictors are not adopted for full-horizon planning. The earlier six600-update models and all failed speed transfers are retained.

# Current shared-plant planning implementation

Current entry: `learning/cli/evaluate_playground_handoff.py`, with GPU shared execution in `playground_handoff_evaluation.py`; route publication in `playground_adapters.py`/`torch_middle_control.py`; finite bank screening in `playground_joint_planner.py` and `playground_rolling.py`. Configurations and checkpoints are frozen per run; the detailed current record is `research-hub/CHAPTER3_PROJECT_STATE.md`.

STTW uses frozen H75/ECBC/ESO with hip/knee equality brakes; RB225 owns all four targets through native execution and finite completion, then relocks at measured posture. Physics remains implicitfast1ms x5; lower5ms and skill20ms. Periodic planning uses measured state/controller history and the same-grid committed preview prefix. Query and publication delay is charged to continued old-plan physics. Empirical state/support envelopes are not safety proofs.

The adjacent-speed validation changes only requested ground speed, including after handback. Initial forward speed remains2m/s, native RB does not receive a new speed command, and the two-meter flat buffers remain fixed. Registry: `learning/configs/playground_validated_command_domain.json`; it does not authorize arbitrary within-episode switching.

Predictor entry: `train_playground_history_model.py`, shared model `playground_history_model.py`, terrain descriptor `playground_terrain_features.py`. M0 uses same-data linear ARX; M1 adds a GRU history/terrain correction; M2 uses the same inputs/head and integration but directly predicts dynamic increments. Three seeds share source split and update budget. Continuation restores model and Adam but restarts sampling RNG from the declared paired seed; it is not exact sampling-stream replay. No controller weights are changed.

The following sections describe historical simulator/interface work and are not the current shared-plant model schema.

## Historical two-interface implementation

Current revision: see [implemented review corrections](../evidence/chapter3_review_repairs/RESULTS.md). New models use55D compact upstream and19D Frenet context; original582D/world-x implementations and evidence remain supported. Joint retest26/30/27; shared horizon43/35/36. Earlier assessments below are historical where superseded.

# Two-interface model reproduction entry

Current model: `learning/src/sttw_control/interface_world_model.py`; training: `learning/cli/fit_interface_world_model.py` and `fit_handoff_continuation.py`; online panel: `prepare_interface_online.py` and `analyze_interface_online.py`. Frozen configuration and weights are under `/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_interface_all_routes/upstream_distance` and `downstream_goal_distance`; online raw data are under `chapter3_interface_online/repeat0`, `repeat1`, `repeat2`.

The current world model predicts valid handoff, conditional39D state/time descriptor, and ground-route goal probability. It does not propagate all six historical event targets. The operational physical definitions below remain applicable, while historical event-model targets must not be mistaken for the current output schema.

# Reproduction details: controller and event semantics

These predicates describe the frozen reported implementation, not proposed replacements. Sources: `learning/src/sttw_control/ditch_jump_adapter.py`, `offroad_jump_reward.py`, `ditch_ground_handoff.py`, `torch_middle_control.py`, `torch_path_reward.py`; online configuration `chapter3_offroad_distance_online/config.json` and its pinned `chapter3_long_bend_goal/task.json`.

## Ground controller interface

STTW names the existing speed/path-preference ground-control stack. H75 is its frozen learned middle-layer checkpoint, which receives causal motion/control history, path projection/preview and α, and adjusts speed/steering references above the ECBC balance controller and extended-state observer (ESO). The lower learned residual is zero. Front actuation uses steering-rate and rear actuation wheel-speed interfaces; the jump policy also controls the pendulum mechanism through the frozen interface.

α=0 favors path following; α=1 favors speed tracking, with finite secondary costs and unchanged actuator authority. This is the trained preference interface, not a guarantee that every episode realizes that ordering. It is not a motor-gain multiplier or a physical-limit relaxation. The existing `torch_path_reward.costs` defines training preference costs; no ground-controller training occurs in this paper's world-model fitting. After contact, continuation is fixed to1.5m/s,α0.5, while approach uses the selected speed/α.

## Event labels

All physical predicates are sampled every2.5ms; control-boundary events occur at20ms. Local wheel longitudinal positions `s` are measured from the near lip; gap width is0.30m. Scorer lane support requires |wheel y|≤1m. This support strip differs from the original-route corridor half-width0.6m.

|Model target|Actual operational rule|
|---|---|
|Jump entry|Ground-to-jump controller switch at root x≥9m; this is not measured takeoff.|
|Confirmed flight|Recorder latches scorer stage≥1. The scorer first arms on qualified near-bank support, then requires both wheels unloaded (contact norm≤1N), both lower-bound clearances>15mm, positive world-x speed, in-strip and valid physics for17 successive samples (40ms between first/last). Flight onset has front-wheel s∈[−1.2,0.30)m. Stage1 additionally requires front-wheel s≥0 and mean wheel advance≥0.05m from flight onset. Thus it is confirmed/progressed flight, not the first no-contact sample.|
|Far-bank contact|During a valid confirmed attempt, at least one wheel obtains a qualified far-bank target contact. All loaded filtered contact patches for that wheel must lie on the far bank x∈[10.6,25]m and |y|≤1m, with wheel center≥0.02m above contact points, measurable vertical support>1N, valid nonsaturated contact buffer, and wheel center s≥0.30m. The scorer touched flag is latched by the recorder.|
|STTW handoff|Alive jump-phase lane with observed far contact resumes ground control at the next control boundary. No1s stable-support gate is applied.|
|Goal continuation|Ground-continuation/bypass phase, root x∈[16.5,18.5]m, |y|≤0.5m, world vx≥0.5m/s, no physical/ground-control failure and no historical road violation, before original10s horizon.|
|Auxiliary clearance|Recorder latches scorer stage≥2. Both wheels must have been observed in the interior20–80% of the gap while unloaded, clearance≥5mm and in-strip; both centers then pass the far lip during a valid attempt. Recontact with near bank, unsupported obstacle contact, strip departure or reversing after clearance can abort a current attempt. Clearance is logged but excluded from Q's probability product.|

Near-bank arming in rough-support mode permits intermittent wheel contact: both wheels must have been observed on near support within80ms, remain before the lip, within strip, and have conservative upper clearance≤5cm; qualified support accumulates for17 samples. The scorer rejects physically discontinuous wheel position changes exceeding12dt+0.002m per sample. These details matter when interpreting the label 'flight'; a raw support-mask change alone is not the fitted target.

## Failure and horizon rules

The verified adapter declares failure for nonfinite state/action/contact values, body contact, root z<−0.15m, |root y|>3.9m, root x<0.5m, |upright-frame roll|>0.610865rad (35°), or |pitch|>1.308997rad (75°). Ground controller domain exit or numerical failure also terminates execution. No additional yaw-angle termination is used in the pinned jump physical-limit specification. The task horizon is10s from original episode start; it is not renewed at query/publication.

Distance from root XY to the current reference polyline greater than0.6m latches a road violation at physics frequency. This flag can invalidate eventual goal arrival without immediately terminating the episode. Physical validity, scorer attempt validity, corridor validity and final task success are distinct predicates.

The model's world-quaternion-derived Euler roll is not the upright-frame roll used for failure limits. Its nominal asset offset must not be interpreted as robot lean. Predicted motion error figures use normalized quaternions and circular differences.

## Task-conditioned joint selection extension

`InterfaceWorldModel.score(..., task_conditioned=True)` evaluates the frozen upstream handoff particles against three downstream geometries with the 58-input continuation ensemble. The same particle draws are shared across approach candidates. The selected route maximizes the probability averaged over particles and upstream ensemble members; there is no per-particle route oracle. `MovingDitchPlanner` publishes the approach candidate and retains its selected downstream route for handoff. The `planned` handoff comparator retains that route, while the task-conditioned feedback method queries the measured state again. Neither route publication resets the robot or controller history. Cold initialization still occurs once at the original jump-to-H75 switch.

The frozen joint panel restricts upstream candidates to the 15 straight-crossing speed/preference combinations represented in the task-conditioned collection. Three downstream routes share the same goal, physical map and corridor width. This extension is separate from the earlier 60-candidate panel; it does not establish joint generalization to S/90-degree approaches or detours. Per-lane inference wall time is rounded to physical control ticks. Timing is not a certification of total serial-batch deployment latency.

Multi-map preparation: descriptors now batch known map rasters and select each trajectory map on GPU, without adding map-ID features to the network. TRAIN-only normalization remains. Matched individual-map lookup and pose-gradient CPU tests pass; multi-map fitting has not run yet. The eight-map physical collection is separately queued under the shared GPU lock.

PG9 adds eight predeclared terrain maps (286/288 physical goals), six TRAIN/two DEV maps for matched M0/M1/M2 fitting, and full observed departure prediction on all 72 DEV conditions. Selected checkpoints include intermediate updates575/550; formal convergence and executed model-planner benefit remain unproven. Evidence: [multi-map report](../evidence/chapter3_playground_sttw_rb_sttw/p3_multimap_models/RESULTS.md) and its paired complete-departure XY figures. No independent P4 claim.

PG10 adds a static same-candidate model-planning DEV pilot: M0 10/12, M1/M2 12/12 with identical selections,24 unique physics runs. This avoids one no-plan fallback but does not show fewer false-feasible published plans or a structural advantage over M2. Measured-state integration and independent E1–E3 remain required.
