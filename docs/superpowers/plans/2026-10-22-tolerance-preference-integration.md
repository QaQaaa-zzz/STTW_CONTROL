# Tolerance + conditional preference integration implementation plan

> For agentic workers: execute task by task using the executing-plans workflow in the existing isolated tolerance-calibration worktree. No extra agents required.

**Goal:** verify directional margin training utility on the actual single-alpha preference network, preserving deployable preference behavior and frozen local300.

**Architecture:** bring the existing alpha_modulated_v1 architecture into the tolerance branch with a legacy default. Separate force/alpha selection from conditional state selection; compare uniform, physical-risk and stratified-shuffled risk. First measure frozen u325 under real pulses; only then declare the recovery training campaign.

**Tech Stack:** existing JAX/MJX, Flax, Torch/RSL3.2, existing physical configuration and bank provenance.

## Global constraints

- Physical0.0002s×25, lower0.005s, upper0.02s, local300 unchanged.
- u325 Actor351/Critic352 with alpha modulation; original V3 reward and10s horizon, final hold1s. No action interpolation, no added privileged Actor inputs.
- Retain legacy20s dual-endpoint experiment identities and all earlier artifacts. No edits in the dirty path-conflict worktree.
- Freeze baseline identities: actor fa4a05274190b606d2b814a78b75c598ebe91e48cf9795fc31678e5e916528de, checkpoint03b5067844e400e68315be7169279fb70c62432c720829e17c59512d4ff295f5.
- Immediate engineering faults remain failures/UNKNOWN as appropriate; no replay precision gate. One-seed DEV evidence is not final TEST.
- New executions registered for Chinese STTW_CONTROL Bark-only; no email/autofix. New training requires actual TensorBoard reward HTTP verification.

## Task1 — Architecture bridge

Files: path_command_policy.py, path_command_training.py, test_path_modulated_policy.py.

- [x] First failing tests: explicit factories, 384 modulation parameters, malformed identity rejection.
- [x] Port existing7e822c7 alpha modulation; preserve default legacy and all export parameters.
- [x] Test Torch/Flax gradients, optimizer and restore. Actual u325 strict restore/full export parity reproduced.
- [x] Independent pinned-source V3 parity: 1008 random/extreme inputs over seven alphas, failure horizons, zero diagnostic difference;14 tests.

## Task2 — Matched marginal sampler

Files: margin_state_sampling.py, test_margin_state_sampling.py.

- [x] Failing tests first.
- [x] P(state|force,alpha-bin) retains30% uniform support; focus near normalized-energy radius1 with bandwidth.2 and70% focused weight.
- [x] Five alpha strata: exact endpoints and three interior bins. Force draw and alpha-bin draw independent of arm; nominal episodes uniform.
- [x] Stratified risk permutation preserves per-force per-bin weight distribution; no physical randomization or reward change.
- [x] Review found and fixed within-bin alpha drift: exact_alpha_groups preserves actual alpha, including distinct values .1/.3 in one old bin.1024 paired draws verify exact matching; continuous singleton groups cannot be reweighted.
- [x] Real320-context bank audit: 4096 paired draws/arm, force-enable/actualalpha/nominalstate exactly equal at equal counters. Guided changes1682states; true-risk distance-to-boundary1.022→.551, shuffled1.034. Actual training exposures still required after reset divergence.

## Task3 — Frozen preference pulse screen

Files: preference_tolerance_screen.py and thin CLI; actual resolved declaration and status under a new run.

- [x] Resolved stage3 geometry R1.8, v2.6/2.8; original10s and V3.
- [x] Original200case panel completed260.65s/alltaskpass; additional0/±20/±32N200case panel completed216.92s/task189/200,9domain2holdfail,0recordedfalls. Both original10s/V3/local300, own-prefix/noPPO.
- [x] Saved actual XY, signed errors, logged rewards/components, impulses/features;96PNG/PDFpairs across two panels, task/working/physical labels separated.
- [x] Nominal4/4 turns retain strictly decreasing speedRMSE withalpha; moderate/stress200/200 complete. Training band declared±8/16/24/32N, independent realprefixes.

## Task4 — Recovery learning utility experiment

- [x] Collected320authentic prefixes/25573anchors in181.93s/511456lower ticks, bank6ce48aa4c8004b6caf6ffc9759f6b7552e7c7054050e1f6a9ea788a955d80bc0. Originalclock/historyvalidated; allprefixesvalid, nominal full8scompletion not proven. Actualαcounts128/21/22/21/128; noDEVstates.
- [ ] Resolve one shared reward/learner/force/alpha schedule for uniform/risk/shuffled arms; fresh immutable runs and critic warmup/parent retention.
- [ ] Test sampler actual exposures, complete-state resets and snippet bootstrap; no silent20s→10s switch.
- [ ] Run the declared matched campaign and paired full-task/preference evaluation; independent repeats and continuous metrics required before claiming method benefit.

This plan does not mark the objective achieved. Tasks3–4 must provide real runtime/model evidence; architecture compatibility alone is insufficient. Shared research ledger remains the sole current-state authority.
