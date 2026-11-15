# P4 geometry and calibration diagnosis

First inspect the existing all-case actual/reference/predicted XY: [S0/3475](../comparison/seed0_update3475/xy.png), [S1/3450](../comparison/seed1_update3450/xy.png). These arrays are unchanged; no physical rollout or model training was repeated.

**The lateral uncertainty bound, supported by real center-prediction errors, blocks the current successful routes. Command-conditioned calibration does not fix the continuous model.** No new model or bound is adopted; P4 remains open.

## Geometry attribution

The audit covers both fixed checkpoints, both I/C methods, all18 mechanism candidates, actual and predicted paths, and raw/full/x-only/y-only/z-only expansion. Actual futures and axis-only bounds are offline diagnostic controls, never deployment inputs or proposed safe bounds. All failure cases remain in [geometry_cases.csv](geometry_cases.csv).

For S1/3450, each method's4 physically successful candidates passes raw geometry. Applying only the fitted y bound rejects all4, as does the full bound; x-only or z-only preserves all4. Even the actual successful trajectories expanded by those bounds are rejected4/4. The bound therefore exceeds available corridor room, independently of this candidate's center error. This does not justify shrinking the bound.

The largest C post-region lateral support miss is0.318076m at map07/root7/route2/v1.8, x13.579m/t6.700s, frontwheel geometry. Root lateral error is−0.308409m; translating the predicted root to the actual center in this offline diagnostic leaves0.009667m support miss at that same sample. The peak occurs before predicted goal x17.393m, on a monotone prediction: it is not a post-goal tail or x-sort reversal artifact. It comes from a different command than the mechanism's feasible routes3/4. All region/axis maximizers are in [margin_sources.csv](margin_sources.csv).

For S1 C, raw predicted obstacle AABB separation on the4 successes is0.041–0.141m versus0.135–0.207m from recorded actual poses. These are sampled axis-separation diagnostics computed from the collision shapes, **not MuJoCo signed contact distance or a continuous-time safety certificate**. Actual recorded zero clearance remains unchanged. S0 has additional non-geometric failures: geometry-only pass counts cannot replace complete event/domain/goal admission.

## Bounded command-conditioned experiment

Freeze three groups by commanded upstream route, available before prediction. Within each group retain exactly the same spatial regions, maximum AABB support miss plus0.02m rule, all declared support samples, and fixed event/domain/goal gates. Missing groups use pooled fitted bounds. Evaluate both already selected models; do not select a new checkpoint. Eight CPU analyses: two-map pooled parity, two-map conditional fit, map06-only fit, map07-only fit for each model. No physics, reward, actor, task, threshold, or candidate changes.

Both pooled reproductions preserve all previous numerical margins, counts,144 candidate/margin outcomes and rejection records exactly. Two-map conditional C/I systems still reject all18 mechanism candidates for both models. Map-disjoint coverage is:

|Model|Fit → validation map|C whole-trajectory support coverage|I whole-trajectory support coverage|Largest C/I excess(m)|
|---|---|---:|---:|---:|
|seed0_update3475|06 → 07|35/36|6/6|0.0105/0.0000|
|seed0_update3475|07 → 06|34/36|0/6|0.0039/0.0368|
|seed1_update3450|06 → 07|21/27|36/36|0.0398/0.0000|
|seed1_update3450|07 → 06|24/27|17/36|0.0059/0.1087|

Coverage denominators include only predicted-event-supported candidates (C72/I12 for S0 and C54/I72 for S1 across the two maps). Unsupported candidates remain rejected and all exclusions are in [fit_cases.csv](fit_cases.csv). Cases share prefixes; these are not independent successes. Both maps were previously used in model development; this is held-out-from-bound-fitting DEV, **not independent TEST**.

S1 I fitted only on map07 admits2/4 mechanism successes with0/14 false admissions, but covers only17/36 supported map06 trajectories and misses by up to0.10866m. Reject that apparent improvement as an adopted bound. All C variants still admit0/4 mechanism successes. Thus upstream grouping alone neither fixes C nor supplies a validated replacement margin.

## Decision and reproduction

Stop calibration tweaking for this iteration. The next informative change is to diagnose and reduce lateral center error under estimated RB exit states, comparing actual versus predicted exit inputs to the same ground predictor; retain this fixed calibration and negative comparisons. Do not add more RB updates solely because the planner has no plan. Do not enter P5/P6 yet.

Implementation: `learning/cli/audit_skill_calibration.py`; opt-in `--condition upstream --fit-maps map06` (or map07) in `calibrate_skill_sequence.py`. Default pooled behavior is preserved. The existing2 nominal/geometry behavior tests pass. [Frozen experiment contract](contract.json), [conditional decisions](conditional_cases.csv), [all method summaries](conditional_summary.csv), [held-out support coverage](heldout_coverage.csv), [all bounds/counts](calibrations.json).

Raw contracts, exact run inputs, all rejections, timing and outputs: `STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p4_low_lr/{geometry_audit,conditional_calibration}`. CLI contracts resolve raw inputs; previous checkpoint identities remain in [checkpoint_identity.json](../checkpoint_identity.json). Query bodies, models and physical logs were not changed.
