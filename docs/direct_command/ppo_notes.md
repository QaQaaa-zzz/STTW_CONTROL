# V3 PPO adapter notes

`direct_command_ppo.make_algorithm(obs, steps, spec, device)` accepts an RSL
`TensorDict` with `policy: [envs,345]` and `critic: [envs,346]` and the complete
V3 JSON spec. The two stored actions are Gaussian latent samples `z`, before
`tanh`, reference mapping, filtering, ECBC, or actuator mapping. The actor's
alpha is `policy[:,336]`; the critic observation already includes alpha in
its first 345 entries and adds only the remaining task fraction.

RSL supplies rollout storage, full-rollout GAE/advantage normalization, and
sampling. The local optimizer uses separate actor (including `log_std`) and
critic parameter groups. Each PPO epoch measures exact old/new Gaussian KL
over the complete rollout, both pooled and by alpha. A mean above 0.01 ends
the current update after retaining that epoch. A mean above 0.03, a nonfinite
quantity, or an exception restores the model, optimizer, and Torch/CUDA,
NumPy, and Python RNG to the beginning of that epoch and stops the pilot.
Previously accepted epochs remain. No batch retry occurs.

The update result distinguishes attempted and accepted epochs/minibatches,
and reports preclip actor/critic gradient norms for both accepted and attempted
steps, pooled and per-alpha KL, retained projected std, optimizer learning
rates, and stop reason. If an epoch is rejected, its reported KL and attempted
gradient norms describe the rejected candidate while `projected_std` describes
the restored policy.
`hard_kl_stop` and `nonfinite_stop` prevent a further update. The caller must
persist the last completed checkpoint and surface these results; the adapter
does not claim that a KL guard ensures physical safety.

The finite 800-step task boundary and physical failure must both arrive as
`terminated=True` with no timeout bootstrap. A live 128-step rollout boundary
uses normal RSL bootstrap. The environment must preserve final observations
before resetting and feed only raw `z` to `process_env_step` storage.
