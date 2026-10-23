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
- [x] Bound shared declaration: same u325/V3/10s,27slots=2critic+25PPO,3arms,5308416transitions/max21233664lower ticks,seed918301/pulseseed918203,actor3e-5/critic3e-4,parentKL.05. First two workers running; exact warmupActor invariance and actualTB rewardHTTP verified.
- [x] Actual512lane8step engineering completed16384ticks,432completepulses,41.03s,0physical/domain/engineering;351/352/original10s/4s snippet preserved. Actualalpha×force androute×force exposures in each training update. Existing timeoutbootstrap reused; full-task effects still pending.
- [x] All3trainingarmscompleted27slots/25PPO,5308416transitions; actual HTTP reward step27verified,400accepted actor minibatchesperarm. Scientific review andactualexposure/modeldifficulty audits published in preference_recovery_pilot/SCIENTIFIC_REVIEW.md.
- [x] Post pipeline completed1240freshcases/2454996ticks,400authenticatedparentcasesreuse; allfourpoliciesnominal30/30task/workingpass andspeedpreferencepreserved. Strongtaskparent/uniform/guided/shuffled189/186/190/193per200, no physical-guidance superiority. Original fixedfinal evaluation declaration:1240fresh cases/max2480000ticks plus400authenticatedcached parent cases; includes all30nominal cases/policy. Fixedfinalu27 export/learner/modulation identity verified before evaluation. Independent repeats and continuous metrics remain necessary before a method-benefit claim.

This plan does not mark the objective achieved. Tasks3–4 must provide real runtime/model evidence; architecture compatibility alone is insufficient. Shared research ledger remains the sole current-state authority.

## Task5 — TRAIN difficulty diagnosis before a new sampler

The user authorized continued scientific iteration after reviewing the conservative-boundary mismatch. Reuse recovery_margin_scan.collect rather than a second simulator. Preserve legacy20s and select the actor factory/horizon from the authenticated bank config. Explicit state roles override the old every-fourth default. Keep original full clocks,9signed forces, V3 logged components and wheel proxy; no PPO or changed physics.

- [x] Add failing tests for10s/20s scan contracts, declaration budget/clock mismatch and calibration-state separation.
- [x] Implement config-derived deadline/actor, per-state alpha/features and optional declared roles in the existing collector; preserve old defaults.
- [x] Before execution declare60uniformly sampled TRAINstates,12/exactalpha,8calibration+4audit each, forces0/±8/±16/±24/±32N, parentu325 suffix,540cases/max1080000ticks,0PPO. Selected without DEV labels/modelrisk. Existing TRAIN bank was used by pilot learners; this is parent-difficulty calibration, not independent trained-policy generalization.
- [x] Register ChineseSTTW Bark before launch; one worker. Completed540cases/786672ticks/152.80s,0PPO. Nominal54/60viable, conditionalpulsefailcalibration63/280,validation31/152; riskAUC.792/.854vsforce.707/.748,descriptiveonly. Nearradius1calibrationfails1/61, risk2–3fails27/82. ActualcompletionBarkaccepted,queueempty.
- [x] Publish all540scalars+6compactXY/rewardpanels,full60statepanelsserver,updatedsharedledger. Newdeclaredtarget/bandwidthplumbingtestedwithuniformprobabilityinvariance/exactalpha-force-statepreservation. Retain oldnegativepilot; no newPPO yet.

Next candidate: independently calibrated sampling center around2.5/bandwidth.5, fixedoriginalgamma/E_b; declare and test matcheduniform/physical/shuffled learning and full-task preference retention. These are exploratory development choices after examining the diagnostic; no safety guarantee, final-test claim or automatic adoption.

## Task6 — Test calibrated sampling scale

Use the existing training and post-evaluation pipeline. New guided/shuffled jobs keep the original parent,bank,V3,seed918301,pulse918203,2warmup+25PPO and all learner settings; target2.5/bandwidth.5 replaces1/.2. Original uniform learning is authenticated/reused because uniform probabilities do not depend on these parameters; it is not a new replicate. One GPU worker follows measured aggregate throughput. Fresh3538944policytransitions/max14155776lower ticks, no automatic extension. Post evaluation retains1240freshcases/max2480000ticks plus400exact parent cases, including re-evaluation of the reused uniform checkpoint.

- [x] Failing tests for reuse identity drift and actual final uniform export; reject changedseed/forces/budget/reward/parent/bank.
- [x] Add explicit uniform-only reuse provenance to existing candidate authentication; preserve normal candidate checks and use actual campaign job count in progress.
- [x] Freeze declaration/config/plan/post protocols, audit actual320-state uniform probability equality, register ChineseBark and separate actual TensorBoard before launching.
- [x] Engineering512lanes×8steps completed16384ticks/41.92s,432wholepulses,0engineering/physical/domain. Engineeringentrynowhonorsdeclaredsampler; initialenabledmeanrisk2.072. Code2a1ced8,34CPUchecks.
- [x] Launch existing scheduler1599712withoneworker1599744 andautomaticfinalizer1599776; actualrewardHTTP6009loadedthroughstep4. First2criticwarmupActorbitwiseunchanged,thenactualPPOaccepted16minibatches/update. Sampling~41s/update.
- [ ] Analyze fixed-final task gains/losses and preference retention, then decide next iteration. Goal remains unfinished until useful recovery improvement plus retained preference is verified.
