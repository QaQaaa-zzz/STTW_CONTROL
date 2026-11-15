# P4 predicted handoff lock audit

[All actual/reference/predicted XY, including failures](../entry_exposure/README.md). Predictions are unchanged by this gate: the fresh 18-case GPU query matches every saved prediction array exactly.

The original model plus three retained exposure checkpoints each fail the frozen hip/knee lock check in all 90 declared DEV cases: **0/360 model-condition states ready**. These are four predictions of the same 90 conditions, not 360 physical runs. Measured phase2 exists in 84/90 conditions (72 original DEV and 12 mechanism candidates), and all 84 meet the same joint tolerances; six mechanism candidates fail before RB. Shared approach prefixes are not independent episodes.

Home is taken from the native model initial_state keyframe: hip −1.2rad, knee2.5rad. Frozen tolerances remain strict position error<0.01rad and joint speed<0.1rad/s. No physical state projection, controller change, new training or new simulation was performed.

The evaluator now rejects a predicted terminal state that violates these necessary conditions and records both errors. This is **not** a complete event guard: contact/body readiness is not certified, and the original phase-classifier stopping time remains unchanged. The model still requires repair; rejecting every candidate is not task success.

Integration: retained S1 map48,18 candidates,3.49s GPU query; zero true/false admissions, four feasible rejections and14 infeasible rejections. Previous local selected2/2 is preserved as an ungated diagnostic, not a physically consistent predicted handoff. Eight skill behavior tests pass, including schema-order and false-phase lock checks.

[Per-case joint errors and measured comparisons](cases.json), [summary](summary.json), [budget](contract.json), [prediction parity](prediction_parity.json). Raw run: `STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p4_joint_gate`. Reproduce the saved-state audit with `learning/cli/audit_skill_handoff_lock.py --physical <map48/base> --prediction <saved prediction directory> --output <new directory>`; repeat --prediction for each model/map.

Next P4 diagnosis: inspect the full RB predicted joint/phase traces to separate early phase classification from persistent physical drift; then change one identified mechanism. Do not loosen lock tolerances or infer that more of the same training will repair it. P5/P6 remain unstarted.
