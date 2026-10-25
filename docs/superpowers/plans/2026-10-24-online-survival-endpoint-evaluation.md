# Frozen endpoint evaluation: preference utility and physical capability

Status: pre-endpoint design, not an evaluation launch. The64-update training continues unchanged. No endpoint has been selected and no independent test claim is made.

## Questions

1. Do terminal policies improve contact avoidance at the same user alpha, and what speed/path/hold costs do they pay?
2. Do different alphas still provide useful nominal choices rather than collapsing to one behavior?
3. Are changes competitive with zero middle correction, the original fixedalpha1 behavior, and a simple speed floor?

The initial frozen-H75 training batch is motivation only: right−32N gives0/4 survivors foralpha0/.5 and4/4 foralpha1, butalpha1 has6.805s mean domain hold. Pre-pulse speed is1.090/1.307/1.432m/s, steer−.393/−.316/−.272rad; yaw-projected lateral force is−31.755/−31.678/−31.475N. Speed, steering, roll and history co-vary; this does not isolate a speed mechanism or demonstrate recovery by switchingalpha at a shared onset state. Different alphas used different action-noise streams. Two arms have the same original Actor, so their96 executions are48 baseline conditions, not96 independent policy replications.

## Endpoint identity

Use exactly update64 from the common-source task-only and physical-credit runs; require actual completed paired checkpoints, native next_collection64, Actor248 steps and each Critic256 steps, model/config/normalizer/std/source hashes and source recovery receipts. Preserve inherited first-batch costs. Do not pick a checkpoint by this evaluation, substitute an earlier winner, or treat completed Critic warmup as Actor training.

Comparison roles: immutable H75 parent, task-only terminal, physical-credit terminal, deterministic zero-middle follower, and H75 with a fixed1.5m/s reference floor. The floor is a declared legal subset of the common broad reference permissions, not a new plant or a changed requested speed. Its filter/back-calculation behavior and actually applied references must be logged. Originalfixedalpha1 is already in the parent'salpha grid; do not duplicate it as an independent baseline.

## Two inference modes, never conflated

Deployment evidence uses deterministic Actor mean actions. The zero baseline is exactly zero correction; do not add exploration noise, fake a trained checkpoint or rely on numeric underflow to obtain deterministic behavior.

A separate stochastic panel uses each endpoint's actual saved Gaussian std and4 common-noise draws. This diagnoses the training distribution; it does not replace deterministic deployment results or supply4 independent training seeds. Keys should match across models, force pairs andalphas for each context/replicate, unlike the initial training batch's distinctalpha streams. Distinctalphas still generate distinct pre-pulse states; common noise does not make them shared-state interventions.

## Fixed primary panels

All controllers retain frozenlocal300, the same preparedbank3 task0,10s original window and current broad permission bounds: speed reference[1.0,3.5]m/s, steering bound.65rad, follower steering maximum.6rad. The floor role self-restricts only speed references to at least1.5. Its lower controller, actuator/physical limits and original task remain identical. No runtime source of the active training job is edited to prepare this evaluation.

- Low-speed physical panel: the two existing90degree/R1.1 left/right tasks, requested1.8/1.5m/s; zero and original signed32N world pulses,.1s at task ticks464/488. Fivealphas0/.25/.5/.75/1 test both trained preferences and interpolation. Four nonzero-Actor roles ×4 path/force contexts ×5alphas =80 deterministic episodes, plus4 unique deterministic zero-middle episodes =84. Secondary stochastic panel has the same four roles ×4 contexts ×5alphas ×4 noise draws =320 episodes. Zero's deterministic outcomes may be referenced as aliases, not recomputed or counted as320 noisy trials.
- Nominal preference panel: reuse left/right110degree/R2.6 geometries and requested2.35/2.75m/s from the old nominal-transfer study, but freshly evaluate all roles under the current common permissions. Four roles ×4 contexts ×5alphas +4 unique zero-middle trajectories =84 deterministic episodes. Geometry-specific path arrays may have different shapes; group kernels by real shapes rather than pad/change geometry.

Total488 actual episodes, at most976000 actual5ms lower-control transitions, with active and allocated cost and wall time reported separately. These known development geometries/forces with new noise are not an unopened finalTEST. No physics/evaluation has been launched by this document.

Why fresh nominal baselines are necessary: old nominal permissions were steering.35rad, speed floor1.5 and follower steering maximum.3. Reuse old geometry/code/analysis, but do not mix its success numbers with current broad-permission policies. Actual route names/radii/angles/speeds and effective configs must be saved explicitly, avoiding oldleft90_R2/2.4 names for actual110degree/R2.6/2.35 conditions.

## Metrics and limits

Record true nonwheel contact and complete physical window separately from task domain exit, task termination and held-reference tail. Report paired new/retained/lost contact-free conditions, zero-force changes, actual nonzero-pulse exposure and first contact interval. Do not require100percent success or zero task loss. Keep original XY references, actual world trajectories and true endpoints. Provide per-case/alpha/noise paired20ms rewards, signed components, errors and given/actual references; never invent task rewards or path errors during hold.

Cross-score nominal factual trajectories over user-preference weights0/.25/.5/.75/1 using authenticated current V3 cost reconstruction. Report continuous speed/path costs and useful preference ranges; preserve terminal penalties and incomplete cases. Secondary rescoring is not extra physical samples. Nativeworking-roll labels remain diagnostics, not physical fall labels.

A value baseline with inputsActor351+remaining-time estimates survival under its training disturbance distribution; it has no force-query input. Do not interpret its scalar as a Newton-valued tolerance radius. A pass at32N is a tested point, not a certified0–32N interval. If a useful endpoint effect exists, follow with separately declared force/direction curves and shared-complete-onset-state response checks, reusing authenticated old onset assets where their clock/force/history contracts match. Fulltask0 improvement can include preconditioning; it is not automatically a reactive recovery improvement. That distinction is a claim boundary, not a requirement to win every case.

If simpler fixed policies or ordinary task training cover the observed benefit, retain that result and stop the added physical-credit claim. Consider independent training repeats and wider confirmation only after establishing a useful development effect; do not stack new predictors or modify the oldenergy formula to preserve a story.
