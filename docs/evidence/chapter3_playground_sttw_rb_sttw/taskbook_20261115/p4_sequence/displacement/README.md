# P4 joint-displacement model: bounded negative result

Actual/reference/predicted XY cover all90 declared conditions for every evaluated model: [best checkpoints](best/map48/README.md) ([map06](best/map06/README.md), [map07](best/map07/README.md)); [last checkpoints](last/map48/README.md) ([map06](last/map06/README.md), [map07](last/map07/README.md)); [shortlist](shortlist/map48/README.md) ([map06](shortlist/map06/README.md), [map07](shortlist/map07/README.md)). Plot admission labels are the initial **joint-only** evaluation; corrected geometric-support verdicts are below. All prediction arrays are unchanged when adding the support checks.

**No candidate adopted. P4 remains incomplete.**

A zero-initialized linear head(hidden128→5 joints,645 parameters) adds a learned interval displacement to trapezoidal joint integration. Its five scales are RMS one-step quadrature defects from within-rollout TRAIN pairs only. Rotation/root integration and velocity prediction are unchanged; no terminal projection to home is applied. The robot model, policy,20ms/5ms/1ms timing and success thresholds stay frozen.

Three seeds received300 supervised updates each from the same original parents as the previous exposure experiment, with unchanged optimizer state, loss,50percent estimated-entry exposure and horizon schedule. Original parameters retain Adam state; the new head starts fresh moments. Zero-head CPU100-step parity and100=40+serialized60 pass; all three real initial DEV selection scores equal the old experiment exactly. Checkpoints every25 updates were retained. Runtime23.92/23.69/23.61s; TensorBoard6006 loaded38 tags per seed.

|Candidate|Selected update|DEV loss best /last|Joint-ready /90 at classifier event|Initially admitted DEV72 /mechanism18|
|---|---:|---|---:|---|
|S0 loss-best|2475|27.73 /255.62|0|0 /0|
|S1 loss-best|2450|43.82 /112.43|0|0 /0|
|S2 loss-best|2250|18.09 /268.65|0|0 /0|
|S0 last|2575|—|0|0 /0|
|S1 last|2600|—|15|12 /1|
|S2 last|2475|—|0|0 /0|

[Full curves](learning_curves.png) show large fluctuations and deteriorated last losses; convergence is not established. All39 saved checkpoints were screened with resident GPU entries, ranking by original DEV72 joint-ready count then mean joint error; mechanism outcomes did not affect ranking. Top3 were S1 updates2600/2550/2325, with DEV joint-ready12/12/6. Their full-chain queries initially admit DEV12/12/6 and mechanism1/0/0. These are intermediate, insufficient gates, not valid handoff results.

The apparent mechanism benefit is invalid: S1/2600 root1002/up1 predicts phase2 at4.66s and x9.413m, before the gap[10.3,10.6], while actual handoff is5.605s/x11.130m. Learned far-contact probabilities also falsely claim support there. Native body and probability checks alone did not remove it.

The evaluator now also requires **geometric possibility of the predicted far contact**: both wheel collision AABBs must reach the far bank when both far wheels are required. This is a necessary condition for the existing contact contract, not a stronger new task or contact guarantee. All84 measured handoff conditions pass this check. It rejects the early predicted contacts; all three shortlisted models admit0/72 DEV and0/18 mechanism after the correction. Other already-rejected candidates cannot become admitted by this additional condition. Old results are preserved. Native body checks use qvel roll rate as a gyro proxy; full phase3→4 transition-history consistency remains unverified.

No new physics rollouts or independent TEST; all labels reuse existing trajectories, and shared prefixes are not independent samples. No improvement in reliable whole-chain planning demonstrated. Selected-model manifest remains unchanged; old event-readout weights are not reused on new hidden states.

Next: change predicted event selection from first phase2 logit to the first state satisfying the frozen handoff conditions and inspect whether such a state exists within the declared horizon. This is an event-selection diagnosis, not another automatic training extension. If no valid state exists, retain no-plan and diagnose the model rather than projecting states or loosening thresholds.

[Per-case corrected gates](geometry_cases.json), [all evaluation summaries](all_evaluation_summary.json), [39-checkpoint screen](screen_checkpoints.json), [weights identities](checkpoint_identity.json), [training comparisons](checkpoint_comparison.json), [exact prediction parity](prediction_parity.json). Full raw commands, training logs, checkpoints and NPZ are in `STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p4_displacement`. Reproduce training with `train_skill_world_model.py --joint-displacement` plus frozen command_seed*.json arguments. Ten related skill behavior tests pass; nine final GPU queries complete without error. P5/P6 have not started.
