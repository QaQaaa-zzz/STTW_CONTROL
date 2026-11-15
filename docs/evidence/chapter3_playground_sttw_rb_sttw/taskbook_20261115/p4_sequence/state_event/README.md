# P4 state-conditioned predicted handoff

[All36 map06 actual/reference/predicted XY](map06/README.md), [all36 map07](map07/README.md), [all18 mechanism map48](map48/README.md). When no handoff is found, the saved terminal is the maximum-horizon state; plotted departure extensions are **unsupported diagnostics**, not accepted plans or simulated actions.

The predictor now chooses the first predicted phase2 sample that also satisfies the frozen joint tolerances, predicted far-wheel contact, native body conditions and geometric possibility of far-wheel support. The complete trace is checked once on GPU; no future actual inputs, weight updates, joint projection, enlarged horizon or changed control/physics. Missing events remain unsupported. `--handoff-event classifier` reproduces the prior event choice, with all saved prediction arrays identical in the18-case compatibility query.

The guard accepts84/84 saved real handoff conditions under the qvel roll-rate proxy and recorded contact flags. It is still not a contact safety proof or complete reconstruction of phase3/4 controller history. Contact probabilities use0.5; wheel AABB reach is a necessary geometric condition only.

|Frozen model|Events /72 original DEV|Admitted /72|Mechanism events /18|Mechanism admissions /18|
|---|---:|---:|---:|---:|
|Original RB|0|0|0|0|
|Displacement S1/2600|0|0|0|0|
|Displacement S1/2550|3|3|0|0|
|Displacement S1/2325|0|0|0|0|

All three admissions are map07/root7/speed1.9/upstream2, with three departure routes. They share the predicted approach prefix and are not three independent improvements. Their reused actual trajectories all succeed. Predicted handoff5.14s is earlier than actual5.360–5.375s, and position error is0.476–0.503m. Thus the event gate repairs impossible stop choices but does not make the state predictor accurate. No model is adopted as a reliable planner, and no current evidence supports C over I or a new generalization claim.

Four models ×90 known DEV conditions were queried plus one18-case legacy parity query; zero new physical rollouts and zero updates. Each query took about2.1–2.6s. Ten related behavior tests pass, including delayed valid event selection and a lane with no valid event; horizon/mask/serialized state semantics are preserved.

[Summary and query times](summary.json), [real-state gate audit](measured_gate_audit.json), [three predicted/actual handoff errors](found_event_errors.json), [classifier compatibility](classifier_parity.json), [frozen budget](contract.json). Full commands, cases and trajectories: `STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p4_state_event`.

P4 remains open. Stop extending this training budget. Next compare true-entry/estimated-entry and TRAIN-typical I under the **same corrected event rule**, using existing checkpoints, to separate upstream errors from RB model failures before any further model training. Original ungated baselines remain historical diagnostics; do not compare their admission totals as if the gates were unchanged. P5/P6 have not started.
