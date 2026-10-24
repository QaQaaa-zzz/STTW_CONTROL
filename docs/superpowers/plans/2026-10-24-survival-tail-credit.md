# Survival Tail Credit Implementation Plan

> For agentic workers: execute inline using executing-plans; review each completed task before proceeding. Existing isolated tolerance-calibration worktree is retained.

**Goal:** Carry an observed finite-window no-contact outcome to legal preceding Actor actions while excluding held-reference ticks, engineering unknowns and stale-policy data.

**Architecture:** A host-side complete-episode resolver consumes explicit physical-tick event/action masks and collection-policy identity. A Torch PPO-loss adapter uses only resolved legal actions, without mixing task GAE or changing the production learner. A read-only replay audit applies this contract to the authenticated existing96 physical branches; it never treats them as fresh PPO data.

**Tech stack:** Python, NumPy, Torch, pytest; reuse original NPZ files and no simulation.

## Constraints

- Contact is actual non-wheel contact, not roll>.7 or path-domain exit.
- Physical horizon gamma is1. Original task reward/done/gamma are unchanged.
- No Actor observations or Actor loss after path handover; no off-policy historical training.
- Incomplete/fault cases yield unknown targets and zero learning masks; no-contact requires an observed full window.
- One policy identity per collected episode; optimizer consumption must match the collection identity.
- No new training/physics, no remote push, no frozen-weight edits. This is infrastructure for a fair baseline, not a novelty result.

### Task1: Complete-episode target and masked loss

Files: create `learning/src/sttw_control/survival_credit.py` and `learning/tests/test_survival_credit.py`.

Interface: `resolve_survival_credit(active_tick, actor_action_valid, contact_failure, engineering_fault, horizon_reached, physical_tick, window_start_tick, window_ticks, collection_policy_id, learner_policy_id)` takes equal1D Boolean masks, an actual integer transition clock and equal nonempty policy IDs. Check full continuous clock coverage, not logical array contiguity; return outcome (`survived/contact/unknown/pending`), scalar target, valid-action indices, per-tick targets/mask and counted physical/Actor ticks. The first observed physical terminal event closes the episode; fault at the same event is unknown. Domain exit is deliberately not a terminal input.

Interface: `survival_ppo_loss(new_log_prob, old_log_prob, values, targets, mask, old_values, clip)` indexes the mask before arithmetic, freezes sampled old log-prob/advantage, and returns actor/value losses plus sample count. `survival_ppo_batch(evaluate_selected, observations, actions, old_log_prob, old_values, targets, mask, clip)` additionally filters and detaches recorded observations/actions BEFORE network forward. Reject unresolved selected targets, stale data at target resolution, invalid probability values and empty selection. Old held-reference rows can contain NaN and must produce zero gradient rather than poison losses.

- [x] Write literal tests for a survivor with held-reference tail, contact during tail, fault/pending, stale identity and false action on inactive ticks. Example: active `[1,1,1,1]`, legal actions `[1,1,0,0]`, contact only finaltick -> both legal actions receive0, heldticks excluded.
- [x] Run `PYTHONPATH=learning/src JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qy/mujoco_playground/.venv/bin/python -m pytest -q learning/tests/test_survival_credit.py`; observe missing-module failure, then implement resolver.
- [x] Add a failing end-to-end Torch gradient test: two legal actions with outcome1 and0, same old/new logprob and value.5 -> opposite update directions. NaN held/padded rows have zero gradients. Implement the adapter and rerun targeted tests. Added actual NN pre-forward/detach and clock-coverage regressions; final18new+9related=27passed.

### Task2: Authenticated real-trace credit audit

File: create `docs/evidence/tolerance_progress_20261022/baseline_provenance_audit/audit_survival_credit.py`.

- [x] Authenticate combined physical analysis and all171 referenced raw-source hashes.
- [x] For each of96 branch rows locate raw NPZ in its snapshot directory, preserve logical4-tick groups, mark one Actor action per active normal policy block, never count lower ticks as independent Actor samples. Validate actual continuous physical clock and exact4s horizon; retain contact/fault events and compare outcome with original classifier.
- [x] Save per-case target arrays, Actor/physical/hold counts, contact-versus-task status and source hashes into a new analysis-only directory. All files are diagnostic/off-policy and not supplied to an optimizer.
- [x] Check known13 disturbed survivors with domain-hold retain outcome1; all96 original physical labels remain unchanged. These are suffix targets only, not proof of full-prefix PPO integration.

### Task3: Review, record and next interface

- [x] Run related failure-contract/continuation tests alongside new tests; obtain read-only adversarial review of targets, actual action masking and loss gradients. Final27passed; clock-coverage and pre-forward/detach regressions implemented. Final96branch receipt is `runs/survival_tail_credit_audit_v3/receipt.json`,45survived/51contact,13disturbed hold survivors;0newphysics/updates.
- [x] Record actual outcomes/limits in shared ledger under `.write.lock`, preserve both projects, commit only new module/test/audit/plan and intended documentation. Implementation commit959ca14, ledger revision1990. Chinese Bark first attempt failed; one retry accepted, original failure retained in `runs/survival_tail_credit_audit_v3/bark_receipt.json`.
- [ ] Next integration must collect fresh stochastic complete episodes under a frozen collection policy, preserve old logprob and true pre-action observation, and resolve physical tails before updating. No claim that production PPO already uses this adapter.
