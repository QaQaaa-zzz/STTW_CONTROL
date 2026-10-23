# Reserve-guided actual-pulse training design

> 最新阶段结果及运行快照见 [容扰度进展报告](TOLERANCE_PROGRESS_20261022.md)。下文保留编写时的设计/审计状态，不代表当前未启动。

## Motivation and scope

The completed reserve-shaping experiment used PathBatch.advance_one -> env.policy_step without force_pulse. It trained under the existing route/state distribution with a proxy penalty and evaluated external pulses afterward. This is evidence about reward-only transfer, not about learning recovery from external-force exposure. The six-job result remains unchanged and is not a successful robustness claim.

Alternative1: reduce shaping weight; weak causal test because no actual pulse recovery exposure remains. Alternative2: replace the task reward with V3; useful future restoration option but confounds the role of reserve. Selected alternative3: actual pulse training in both arms, use reserve only to guide sampling, and keep task reward fixed. This changes the mechanism being tested; it does not retroactively validate the original reward idea.

## Paired experiment

- Frozen local300 and physical/observation/action contracts unchanged. One independent upper endpoint policy per alpha, no blending; alpha325 single-network candidate stays separate.
- Parents: completed baseline_alpha0/1 update0050 from tolerance_shaping_train_v1. Both arms for each alpha load identical actor/critic/std weights with fresh optimizer and the same declared RNG seed. Hash-pin all sources before launch.
- Arms: uniform pulse training versus reserve-guided pulse curriculum. No tolerance reward penalty in either arm. Retain historical baseline task reward to isolate sampling. Do not simultaneously introduce V3.
- 512 environments,128 upper steps/update,25 updates/arm,2 alphas:6,553,600 policy transitions total. Reuse vectorized physics and compilation cache; at most2 workers. This is a finite first comparison, not a user-imposed hard success gate.
- Keep the original20s episode contract. Train through genuine physical prefixes from existing reset states; no hand-crafted qpos/qvel injection. Half episodes remain nominal to preserve tracking. The other half receive one0.1s lateral force pulse, direction fixed in world coordinates at pulse onset, identical force application mechanism to prior evaluation.
- Shared candidate pulse distribution: onset sampled uniformly on the0.02s control-aligned grid over0.8–3.5s, signed amplitudes from±2,±4,±6,±8N. Avoid using final evaluation labels for curriculum selection. Both arms share candidate routes/onsets and expected pulse frequency.
- Uniform arm selects amplitude uniformly. Guided arm, at onset, computes directional reserve from current physical measurement and saved ESO using the existing formula; samples amplitudes with a mixture of50%uniform exploration and50%soft weighting near predicted boundary. Use dimensionless distance |abs(F)-R_direction|/max(1N,R_direction), weights exp(-distance), normalized across the same8 signed amplitudes. Finite reserves outside the candidate range are clipped for sampling weights, and invalid formula values fall back to uniform with a counted diagnostic. No magnitude is treated as certified safe.
- This is a curriculum comparison with different force exposure by design, not an equal-force comparison. Log force histograms and actual impulse, number of pulse/nominal episodes and all interactions; require comparable total interactions, not identical reward returns. If guided exposure concentrates on easy cases, report it rather than attributing the effect to better recovery learning.

## Evaluation and interpretation

Use identical fixed final checkpoints and the same1152-case development panel framework for both arms plus parent controls. It is already-used development data; do not call it final test. Preserve original labels and report predeclared practical diagnostic path<=.15m,roll<=.33rad with original heading/speed/goal/physical/domain conditions. Report nominal tracking, full-task success, pulse-minus-own-nominal response, valid-prefix coverage and early failures separately. Compare paired gains/losses and raw metrics; no requirement of exact replay or zero approximation error. No model adoption based only on training reward or reserve score.

## Implementation and launch checks

Add optional pulse state to the training episode: reset schedule/key, onset/end ticks, world force vector and fired flag. Freeze onset decision once per episode; preserve state/history/time through pulse. Pass force_pulse to existing policy_step; apply nothing before onset or after duration. Keep default PathBatch behavior unchanged. Use meaningful tests for schedule, once-only direction freezing, sign-specific reserve selection, uniform fallback and impulse duration. Short vectorized GPU smoke must demonstrate actual nonzero impulse and finite rollout before production launch.

Before launch save source/code/config/hash manifest, run identities and finite budgets; register existing Bark-only hooks, preserve email/autofix disabled; verify TensorBoard actual reward scalar. Do not add another watcher. Final reports and ledger updates remain required. No production training has been launched by this design document.
