# Geometric path Phase C — prepared, not_run

Authoritative design: [spec](path_feedback_v1_spec.md); exact supplied parameters: [JSON](../learning/configs/STTW_Path_Feedback_V1.json). Phase B is a path outer-loop diagnostic with zero upper action, not an alpha preference experiment.

Prepared interfaces:
- New `sttw_geometric_path_actor351_v1` / `task_mode=geometric_path`, 16x20 local history +16 masks +15 physical-scale path contexts; critic352 adds finite remaining fraction. Legacy345 Actor is rejected. New Flax Actor351→128→128→64→2 ELU has zero output layer; critic separate.
- Action is Δv/Δδ with one tanh, physical first-order state τ=.10/.08s at200Hz before existing asymmetric slew. Reference amplitude clipping back-calculates affected filter/offset channels; ordinary slew does not reset filter. Train and evaluation must invoke this same environment/function. Nothing filters ECBC or final motor commands.
- `path_command_reward.py` implements supplied ten independently capped costs, cap sum5080.2, no global clip, geometric chi from fixed path curvature. Offset rate is policy-endpoint20ms difference, accounted once. Normal reward is summed over5ms ticks; physical failure replaces current interval with finite remaining discounted tail. Phase B reward is a diagnostic at alpha0 with zero upper, not a learned policy comparison.
- Full scenario distribution, left/right/S fractions, reset-only route offsets, fixed evaluation route IDs, speeds, PPO/entropy/std/KL/checkpoint/best settings remain explicit in `training_future_phase_C` and `stage_C_evaluation` in JSON. No lower training, no old Actor initialization, no last-as-best adoption.

Not executed / remaining before a separate training authorization:
- No Actor/Critic/Adam optimization, no training rollout, no alpha0/1 path policy, no new best pointer.
- Phase-B runner consumes only two specified deterministic routes. Stochastic60–100degree/S-route sampler and vectorized PPO lifecycle integration (task truncation/domain-exit classification, bootstrap/final observation, epoch rollback/stop plumbing and fixed-path best protocol) are not yet wired into the existing trainer. The JSON prepares these settings; it is not an executable training launch file.
- Current `policy_step` returns state/logs for the bounded evaluator. Do not substitute it for the legacy training step without completing that lifecycle adapter. Validate same filter, schema, termination, and reward reconstruction before any training.

No automatic Stage C launch or claim that alpha task has been solved.

User steering, 2026-10-10: prioritize rapid validation of the frozen-lower plus preference-upper main line. Baseline perfection is not a prerequisite; preserve declared failures and use the zero-upper traces as comparators. This supersedes the attachment default acceptance gate in intent, but does not authorize automatic training or rewrite physical/metric thresholds. Complete the minimum PPO lifecycle before a bounded main-line experiment, rather than retuning the baseline indefinitely.
