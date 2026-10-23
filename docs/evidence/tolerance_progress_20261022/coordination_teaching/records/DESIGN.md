# TRAIN observation collection

Same 72 declared contexts and frozen parent as preference_train_coordination_probe_v1; only optional recording enabled. 144000 maximum lower ticks, zero actor/PPO updates. Existing pilot outcomes remain immutable. Added arrays: 351D pre-action actor observations, masked 2D latent target, unmasked parent means, policy timestamps. No oracle force, future return or case identity enters observations.

First compare outcomes and continuous traces with pilot; minor numerical differences acceptable, changes must be reported. Check 50Hz sample alignment, finite arrays, actor(obs) reproducing parent means, masked targets inside the declared interval and restoration afterward. All pairs retain source hashes and arm/context metadata outside the actor input.

Dataset selection uses TRAIN outcomes only: speed_off task recoveries with improved return versus failed control are recovery candidates; retain successful control contexts (including adverse intervention counterexamples), plus independently collected nominal anchors across all alpha values. Do not label failed control actions as good merely because a case exists. Sampling and optimization contract will be declared after valid record counts. This collection does not automatically launch training or select a final policy.
