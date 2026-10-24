# Fresh full-prefix survival collector

**Goal:** Collect fresh stochastic H75 episodes from actual task0 and bind their legal decisions to observed finite-window physical outcomes. This is a sampling engineering check, not a training or paper-effect experiment.

**Scope:** Existing isolated worktree, local300/H75 frozen, contact-only diagnostic environment; actual prepared bank3, left90/R1.1 requested1.8m/s and right90/R1.1 requested1.5m/s, alpha0. Reuse prior fixed world-direction and task-time schedules. Four episodes: leftF0/+32N, rightF0/−32N, pulse.1s, original10s horizon, at most8000 actual lower ticks. Common random-number seed within each force pair. Do not expand forces/seeds/alpha or tune std on outcome. No PPO update or TensorBoard training claim.

## Implementation

1. New `survival_collection.py`: raw diagonal-Gaussian sampling and joint log density, actual frozen actor/physics step with legal pre-action observation/action/logprob/value baseline. Carry state+RNG+latched held reference; no reset/restart at pulse or domain exit. Actor decisions only inside path domain. Outside, reuse `make_hold_last_tick`; guard every lower tick at contact/fault/window. Use actual JAX vmap and scan50, shared compilation cache.
2. `test_survival_collection.py`: first fail missing sampler; compare its log density with independent Torch Normal and verify saved raw action precedes tanh. Real full-kernel semantics are verified by the four actual trajectories, not a fake physics unit fixture.
3. New thin CLI `collect_survival_full_prefix.py`: prepare immutable plan/model/config/source identities and notification registration; execute only with explicit flag. Save per50-control-step chunks before outcome analysis, freeze sampling std from authenticated checkpoint. Archive code at launch. Initial physical value baseline is explicit constant.5, not a trained safety Critic. Verify Torch/JAX means and saved raw-action joint logprob after collection.
4. After all episodes terminate, resolve actual continuous-clock targets. Contact/complete/engineering unknown remain separate. Check full observations/logprob are finite on legal actions, constant checkpoint identity, no held-row Actor data, no reset of clock, source force schedule and actual exposure; preserve failure-before-pulse as valid collector outcome. Provide all XY and signed original cost components on task-valid steps, not invented path rewards during hold.
5. Read-only adversarial review, 0-physics code checks before launch, shared-ledger registration before GPU. Verify Bark-only hook and genuine PID/status. Terminal process triggers analysis and Bark; low-frequency observation.

## Claims and next step

This establishes fresh sampled-data provenance and terminal-target association only. Contact-only collection uses original V3 cost formulas but a diagnostic contact/domain terminal contract; it is not a byte-identical reproduction of the old roll-terminal training environment. Hold is part of the deployed composite controller. Historical96 branches are not used for PPO. Masked recovery credit is known from SafeExplorer; no new gradient theorem is claimed. Dual-objective learner/storage integration and training effects remain separate work.
