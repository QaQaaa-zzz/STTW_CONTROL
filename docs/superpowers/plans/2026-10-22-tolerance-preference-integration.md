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
- [ ] Export real-bank probability/exposure audit before formal training; expected matching is not claimed exact after early reset divergence.

## Task3 — Frozen preference pulse screen

Files: preference_tolerance_screen.py and thin CLI; actual resolved declaration and status under a new run.

- [x] Resolved stage3 geometry R1.8, v2.6/2.8; original10s and V3.
- [ ] Parentu325 only, alpha0/.25/.5/.75/1, two directions, two speeds, two onsets1/2s, forces0/±8/±12N:200cases,max400000lower ticks. Own-policy complete prefix shared across force branches. No PPO.
- [ ] Preserve actual XY, signed errors, logged per-step rewards/components, pulse impulse and model features; report task recovery separately from working performance.
- [ ] Verify200/200 and nominal preference behavior before choosing the new TRAIN bank distribution.

## Task4 — Recovery learning utility experiment

- [ ] Collect a new authentic single-network TRAIN prefix bank; alpha provenance and nominal parent anchors retained.
- [ ] Resolve one shared reward/learner/force/alpha schedule for uniform/risk/shuffled arms; fresh immutable runs and critic warmup/parent retention.
- [ ] Test sampler actual exposures, complete-state resets and snippet bootstrap; no silent20s→10s switch.
- [ ] Run the declared matched campaign and paired full-task/preference evaluation; independent repeats and continuous metrics required before claiming method benefit.

This plan does not mark the objective achieved. Tasks3–4 must provide real runtime/model evidence; architecture compatibility alone is insufficient. Shared research ledger remains the sole current-state authority.
