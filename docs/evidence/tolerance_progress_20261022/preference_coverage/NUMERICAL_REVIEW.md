# Coverage-only recovery experiment: numerical review

470 fresh cases completed; report rendering still in progress at this audit. No new training or adoption.

| Panel | Policy | Task | Working | Domain exit |
|---|---|---:|---:|---:|
| moderate | parent | 200/200 | 143 | 0 |
| moderate | oldfull10 | 200/200 | 136 | 0 |
| moderate | coverage | 190/200 | 144 | 0 |
| strong | parent | 189/200 | 67 | 9 |
| strong | oldfull10 | 186/200 | 64 | 8 |
| strong | coverage | 177/200 | 63 | 7 |
| straight | parent | 10/10 | 10 | 0 |
| straight | oldfull10 | 10/10 | 10 | 0 |
| straight | coverage | 10/10 | 10 | 0 |
| low_speed | parent | 53/60 | 26 | 0 |
| low_speed | oldfull10 | 52/60 | 26 | 0 |
| low_speed | coverage | 50/60 | 26 | 0 |

## Interpretation

Coverage expanded 130/320 TRAIN contexts into low-speed, wide-turn routes, with corresponding nominal anchors. Same parent, reward, full-tail duration and 27-update schedule as the previous full-tail experiment. This tests the joint sampling/anchor change; it does not isolate speed from turn angle.

Prespecified low-speed target subset: parent, oldfull10, coverage each recover 1/8. Coverage tail path errors among the seven nonrecoveries range 0.1785–1.0171 m; heading errors 0.2314–0.5756 rad. These are persistent recovery problems, not just the 10 cm cutoff. Some paired errors improve, others worsen. Single-seed, development-context evidence does not establish that coverage can never help.

Standard-panel losses concentrate at alpha=1 with small tail-path offsets; see standard_regression_review.json. Unique nominal tasks: parent 30/30, oldfull10 30/30, coverage 29/30. All retain strictly decreasing turn speed RMSE across five alpha values in four turn contexts; intermediate path monotonicity is not demonstrated.

## Decision and next diagnostic

Keep parent u325 as reference; do not adopt coverage or automatically extend PPO. Full-tail duration and coverage alone have not produced a practical recovery gain in the targeted cases. Next inspect saved failed trajectories: original versus modified references, speed/steering correction saturation and oscillation, turn/exit phase, reward components and task errors. Determine whether actions reduce rewarded costs while delaying recovery, or whether an available recovery behavior fails to be learned. Compare matched successful/intervention traces before changing reward or training again. Tolerance remains an auxiliary diagnostic, not a required successful training signal.

## Artifacts

- [All-alpha nominal XY](nominal_review/actual_xy.png)
- [Target subset actual XY](matched_curves/actual_xy.png)
- [Target subset heading](matched_curves/path_heading_error.png)
- [Target continuous metrics](low_speed/target_continuous_review.json)
- [Nominal trace identities](nominal_review/summary.json)

## Saved-command and reward diagnostic

The seven low-speed target nonrecoveries already carry substantial tail path penalties: mean per-case path share is 84.1% (parent), 88.8% (oldfull10), 80.0% (coverage); speed contributes most of the rest. This excludes the simple explanation that these low-alpha tail failures receive no path signal. It does not measure PPO advantage quality or prove the reward is optimal. Primary is zero on the straight tail, but the persistent path channel is active.

[Executed command offsets and reward](matched_curves/command_reward_diagnostic.png) show similar repeated speed-reduction/steering-reversal cycles across all three policies. The successful matched case settles; the difficult cases keep oscillating. This is descriptive evidence for investigating closed-loop coordination/damping, not proof that one channel causes oscillation. Some candidates improve cumulative reward without recovering by the original deadline. Saved standard traces lack latent actions and clip flags, so actuator or policy saturation is not established. Source implementation logs `nominal` as the existing path-follower reference; plotted offsets are `governed - nominal`, not direct motor actions.

Next reuse the already completed same-state intervention traces with latent actions/clipping logs to determine whether removing one correction channel damps oscillation. Do not infer a deployment solution from channel removal or add a reward term solely from this descriptive plot.

## Existing same-state intervention follow-up

Reused the completed intervention run, no new simulation. After interventions end at 4.1 s, all four newly recovered speed_off cases reduce heading RMS by 0.0793–0.3163 rad and path RMS by 0.1117–0.4960 m versus matched control. Its one regression increases both. All four newly recovered steer_off cases reduce heading RMS, but three have larger whole-late-window path RMS despite reaching the terminal recovery criteria: terminal success and integrated path quality differ.

Control correction rate limits are active for 16.3–38.6% of post-4.1s ticks across these eight disturbed cases. Reference clipping is zero. These flags show active rate limiting, not proof that increasing limits would help; preserve physical/control limits. The low-speed recovery issue includes time evolution of coupled corrections. A blanket off switch is contradicted by regressions. Next design a state-conditioned coordination learning intervention using training-only recoverable branch examples, preserving the ordinary two-channel interface and nominal alpha behavior. Existing evaluation interventions may motivate the design but must not become training examples and then be reported as held-out evidence. Require a training-only pilot to demonstrate usable recovery targets before further PPO.

Evidence: existing_intervention_diagnostic.json and review_existing_intervention.py. No claim of unique root cause or proven new training method.
