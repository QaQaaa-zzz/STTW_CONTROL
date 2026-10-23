# State-conditioned coordination teaching: candidate evidence

[Low-speed DEV actual XY](low_speed_dev/analysis/all_cases_xy.png) · [TRAIN actual XY](train_check/analysis/all_cases_xy.png)

The fixed final400 supervised Actor improves the original low-speed DEV target subset from 1/8 to 6/8. All60 tasks improve 53→58, with five paired gains and zero regressions; nominal20/20 retained, working26→28, domain0. These contexts were not teaching samples, but were used in earlier development diagnosis: this is DEV evidence, not final TEST or multi-seed proof.

Ordinary unmasked TRAIN rollout: parent17/24→taught23/24; disturbed9/16→15/16, nominal8/8, six gains/zero regressions, working15→14. Standard410 DEV and five-alpha nominal review are still running. Candidate is not adopted.

## What changed

Same alpha-conditioned351D Actor, frozen local300, V3 reward, control filters/limits and original task. A TRAIN-only same-state diagnostic found temporary speed-target removal can help or hurt depending on state. Six improved-return recovery examples were recorded with actual pre-action observations; failed forced interventions were not copied indiscriminately. The Actor receives its ordinary observations, no force/timing/case oracle. Deployment has neither forced masks nor a selector.

Teaching uses four equal-sampled groups:600 intervention samples,1860 restored-parent tail samples,8500 successful-control samples,25573 nominal anchors. Fresh Adam3e-5,400 supervised steps,seed918501,gradient clip1; fixed final checkpoint. Critic/log_std unchanged; optimizer file is supervised-only and is not a PPO resume checkpoint. CPU training had0new physics; the trajectory collections/evaluations have separately charged costs.

## What supports it and what remains

72-case TRAIN diagnostic: control/speed_off/steer_off17/18/17 tasks; speed_off six gains/five losses. Records collection72cases:36000 decision samples,2160 parent-action recomputations max error4.17e-7; no task-label change versus pilot, max tail-path numerical difference2.54cm retained. All raw source hashes are preserved.

Lower full-tail duration and broader coverage alone had failed to improve the original1/8 target recovery. Teaching now transfers to unmasked execution and a nearby DEV panel. This does not yet prove overall robustness, alpha behavior preservation, online switching, physical limits, or novelty. Tolerance formula remains an auxiliary diagnostic; do not attribute these gains to formula-based training.

[Probe](probe/analysis/REPORT.md) · [Teaching contract](records/TEACHING_PLAN.md) · [TRAIN closed loop](train_check/analysis/REPORT.md) · [Low-speed DEV](low_speed_dev/analysis/REPORT.md)

[TensorBoard fit losses](http://localhost:6015): actual supervised fit scalars through400 verified; they are not rollout rewards. Raw checkpoints and full datasets stay server-side; publication_manifest.json maps compact evidence to sources.
