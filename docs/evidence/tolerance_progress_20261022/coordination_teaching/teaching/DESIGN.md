# State-conditioned coordination teaching pilot

Pending valid dataset; no optimizer has run.

Purpose: test whether the existing parent alpha-conditioned Actor can reproduce the useful TRAIN recovery behavior through its ordinary two-channel interface, while retaining original successes and alpha preference. This is an imitation initialization experiment, not a new reward, new controller, tolerance proof, or automatic deployment.

Initialization: exact frozen u325 Actor weights, fresh Adam; lower controller and reward unchanged. Critic and exploration log_std are not trained by imitation and must not be presented as resumed PPO state. Fixed final Actor only; no best-checkpoint selection on DEV.

Proposed single trial: 400 supervised optimizer steps, seed 918501, Adam 3e-5, global gradient norm clip 1. Each minibatch draws 128 samples from each of four distinct strata: intervention, restored-parent recovery tail, successful control, nominal anchors. Equal group losses on the two pre-filter latent outputs; retain per-channel errors. Uniform sampling within each trajectory group is case-balanced because retained trajectories have equal durations; record actual group sizes. No timestamp, applied force, context ID, future reward or privileged Critic values appended to the 351D input.

Record initial/final whole-dataset error by group and channel, actual optimizer steps, gradients, all source hashes, actor/optimizer checkpoints and fresh TensorBoard scalars. Offline teaching has no newly measured rollout reward: label fit losses as fit losses, do not fabricate a training reward curve.

Evaluation after training: first ordinary unmasked execution on the 24 unique TRAIN pilot contexts versus control. Fit improvement is insufficient. If recovery behavior transfers, evaluate the existing low-speed DEV and standard/nominal preference panels with fixed final actor and retained parent controls. Report training-domain success separately from DEV and expose all regressions. No time-window masks during evaluation/deployment. Outcome-based training sample selection does not license training on DEV.

This pilot is small and derived from a single initial snapshot. Positive results establish a route worth expanding, not broad generalization or final paper evidence. If unmasked TRAIN recovery does not improve, inspect covariate shift/conflicting targets before more PPO, more sampling ranges, or more teaching steps. No automatic parameter sweep.
