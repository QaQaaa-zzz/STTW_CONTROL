# P4 full-trace and joint integration diagnosis

[All actual/reference/predicted XY for the same conditions](../entry_exposure/README.md). [All288 actual-velocity integration errors](integration_error.png).

Four retained models ×90 known DEV conditions were replayed from their saved causal predicted entries, with the frozen125-step/2.5s RB horizon. The first phase2 terminal pose reproduces the prior saved pose exactly in all360 method conditions. Zero new physics episodes or training.

|Model|Phase2 and joint-ready somewhere in horizon|Mechanism map48|
|---|---:|---:|
|Original RB|6/90|0/18|
|Exposure S0|0/90|0/18|
|Exposure S1|0/90|0/18|
|Retained S2|0/90|0/18|

The same counts hold for any joint-ready sample at/after the original classifier event. Thus merely waiting later cannot generally repair these saved predictions. Post-classifier steps are diagnostic extrapolation, not validated execution of RB beyond its true handoff. Full contact/body readiness remains unproven even for the six joint-ready samples.

A separate representation check integrates **actual** sampled joint velocities using the model's trapezoidal rule and measures hip/knee position error at the same sampled phase2 endpoint:

|Data role|20ms median/max error(rad)|20ms error>=0.01rad|5ms median/max error(rad)|5ms error>=0.01rad|
|---|---|---:|---|---:|
|TRAIN216|0.01305/0.04328|155/216|0.00314/0.00692|0/216|
|DEV72|0.01184/0.03349|47/72|0.00314/0.00633|0/72|

This identifies a discretization mismatch: the fixed20ms joint integrator cannot exactly reproduce both observed positions and observed velocities. A learned velocity can compensate, but then incurs velocity error; the check is not a lower bound on every learned rollout's position error. It does not prove this is the only failure cause. Model interval, controller interval, physical parameters, locks and success thresholds were not changed. Actual velocities are offline diagnostic labels, never planner inputs.

Next bounded change: give the world model an explicit learned joint-displacement correction while retaining its rotation integrator and velocity prediction, so it can represent within20ms motion omitted by endpoint quadrature. Initialize the correction at zero to preserve old predictions; estimate correction scales only on TRAIN, retain the original model control, then compare checkpoint/phase/lock/whole-chain decisions with the same DEV conditions. This is a proposed model change, not an executed training result. Do not project terminal joints to home or loosen tolerances.

Reproduce with `learning/cli/audit_skill_handoff_trace.py --prediction <saved evaluation> --output <new directory>` (repeat --prediction) and `learning/cli/audit_skill_integration.py --data <p3_world_data> --output <new directory>`. [Per-condition full-trace decisions](trace_cases.json), [query summary](trace_summary.json), [paired integration errors](integration_cases.json), [integration summary](integration_summary.json). Raw full qpos/qvel/contact/phase arrays reside in `STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p4_trace_audit/results/trace_*.npz`. These are model-method conditions sharing physical roots, not independent experimental samples. P4 remains open.
