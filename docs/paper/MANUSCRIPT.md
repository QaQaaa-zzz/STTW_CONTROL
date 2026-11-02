# Task-Specified Path–Speed Preference Control for a Single-Track Robot

Internal working manuscript, 2026-11-01. **Not submission ready.** The nominal development results below have been checked against saved artifacts by an AI assistant; human verification, the two declared task experiments, an independent holdout, and final declarations remain outstanding. The earlier predictor-led draft is preserved in [the dated history](history/20261101_preference_coordinator_snapshot/MANUSCRIPT.md). The current PDF and HTML are generated from this source for local review.

## Abstract

Path tracking and speed regulation can conflict during a turn on a balancing single-track robot. We study a shared preference-conditioned middle controller that adjusts both speed and steering references while a common frozen lower controller handles local tracking and balance. A task specifies one of five evaluated preference values before execution; this study does not require online preference selection. In an already inspected nominal development panel of four route types and two prepared initial-state banks, the selected controller reached the original destination and remained upright in all 40 fixed-preference episodes. Across the eight paired route-bank contexts, mean speed during the turn increased at every step from path-priority to speed-priority preference, while original-path cross-track error also increased at every step. Mean turn speed changed from 2.115 to 2.295 m/s and mean common-prefix cross-track root-mean-square error from 0.173 to 0.328 m between the endpoint preferences. These observations establish a repeatable nominal behavior tradeoff within this development panel. They do not yet establish task-level superiority: spatially constrained passage, speed-preserving passage through an original exit checkpoint, strong simple-rule comparisons, and held-out combinations remain to be measured.

## 1 Introduction

A single-track two-wheel robot must coordinate steering, forward motion, and balance. On a demanding turn, a strict path objective may require reducing speed, whereas holding speed may increase the lateral excursion or prevent the original destination from being reached. An autonomous controller should expose these consequences to the task, rather than treat every request as the same tracking problem. The application question in this paper is whether **one shared controller** can produce useful behavior for two different, predeclared priorities under the same vehicle, lower controller and actuator limits.

Huo et al. [R1] studied a different bicycle-control problem: path tracking at a controlled 4 m/s in PyBullet. Their hierarchy separates attitude and position control and adds residual reinforcement learning to traditional control at both levels. Their published abstract reports improvements in path tracking and training behavior. This prior work means that hierarchy and residual learning alone cannot be our novelty. Our proposed distinction is **task-specified path–speed execution**: the same middle policy changes both reference channels in response to a user preference, while the original path and destination remain the evaluation targets. The complete Huo article's numerical tables have not been independently verified in this draft; no quantitative cross-paper comparison is made.

The controller is already trained. We therefore do not propose a new reinforcement-learning algorithm or a learned selector of the best preference. The evidence needed for a control-applications claim is more specific: first, that the preference controls a reproducible path–speed tradeoff; second, that preassigning different preferences helps complete distinct spatial and speed tasks; and third, that this effect survives fair comparison with a tuned simple rule and held-out operating conditions. The first proposition has preliminary development evidence. The latter two remain open.

The intended contributions, conditional on those pending tests, are (i) a task formulation that scores both original-path geometry and requested speed, (ii) a shared conditional middle controller that jointly adjusts speed and steering references while reusing one lower controller, and (iii) task-level evidence separating narrow-corridor passage from a wide-corridor, speed-preserving turn. These are control-application contributions, not claims of a new learning algorithm, formal stability guarantee or universal benefit of any particular preference value.

## 2 Control system and preference objective

### 2.1 Frozen controller and original task

The evaluated middle policy is STTW_MIDDLE_H75_LOW_SPEED_MIXED_U170. It was selected from a continuation of H75 training; the selected checkpoint is update 170, whereas the continuation ran through update 202. The lower policy is STTW_LOCAL_DOMAIN_U200. The middle controller runs at 50 Hz and produces bounded speed and steering-reference corrections. The lower ECBC/ESO plus local residual stack runs at 200 Hz. A deterministic causal geometric reconnection process runs at 10 Hz. Both wheel friction coefficients are 0.8 in the evaluated MuJoCo model. The evaluated task window is 10 s. The fixed original destination is a 1 m disk for the legacy nominal panel. These details and identities are recorded in the [experiment package](../results/sttw-tolerance-20261101/README.md) and [lower-selection record](../results/sttw-tolerance-20261101/lower_base/selection.json).

The original world-frame path, speed request and destination define what the user asked for. If the vehicle departs and receives a reconnecting execution path, task scoring still uses the original geometry and original destination. First entry into the destination disk, sustained post-turn tracking, full-window upright survival and speed tracking are separate quantities. The legacy first-entry measure has no stop, terminal heading or hold requirement; it must not be called complete path tracking.

The user chooses alpha from the evaluated set {0, 0.25, 0.5, 0.75, 1}. Zero emphasizes original-path tracking and one emphasizes requested speed. The same network weights and lower controller are used for all five values. Alpha is fixed for a task episode in the present study. Training sampled the endpoints with probability 0.4 each and a uniform alpha with probability 0.2; the five values are evaluation settings, not five independently trained policies [E1].

### 2.2 Effective training objective

The middle PPO continuation used the implemented path_persistent_preference_v3 reward [E1]. Let H(x) = x² for |x| ≤ 1 and H(x) = 2|x|−1 otherwise. Let e_v be error against the original speed request, e_y the signed original-path cross-track error, e_h the original-path heading error, and chi the configured curvature-demand gate. Before adding balance and reference regularizers, the effective endpoint-mixed running costs include:

- speed: (1−alpha)(10−9chi)H(e_v/0.1) + alpha·10H(e_v/0.1);
- path: 10(1−0.9alpha)[H(e_y/0.1) + 0.5H(e_h/0.1)];
- primary band: 20chi[(1−alpha)H((|e_y|−0.1)_+/0.1) + alpha H((|e_v|−0.05)_+/0.1)].

Each endpoint channel is capped before mixing. Roll, working-roll, roll-rate, overspeed, low-speed and reference-offset terms also apply; the per-lower-step reward is −0.1×0.005 times the sum of effective costs, with a separate early-failure term. The frozen configuration contains an older textual path_weight field, but the V3 implementation overrides it through path_secondary_ratio=0.1. The [reward implementation](../../learning/src/sttw_control/path_command_reward.py) and [frozen configuration](../../learning/configs/h75_low_speed_mixed_finetune_200.json) define the actual formula. The lower continuation was optimized for local speed and delivered-steering-reference tracking, with no alpha, original-path or goal term [E2]. Changing alpha therefore changes the middle task objective and reference output, not the lower policy's own reward.

Training reward is not itself the application outcome. The tests below compare actual speed, original-path error and task events. Different alpha values have different reward weights, so their weighted returns are not ranked as one common physical-performance score.

## 3 Experimental design and evidence roles

### 3.1 Completed nominal preference panel

The completed panel paired four route types—low-speed single turn, low-speed S turn, high-speed single turn and high-speed S turn—with prepared banks 0 and 6. Each of the eight contexts ran all five fixed alpha values with U170/local200, giving 40 physical episodes. Saved XY panels overlay the actual motion and original reference [E3]. The comparison record [E4] contains first arrival, survival, turn-interval mean speed and original-path cross-track RMSE over the common pre-arrival prefix. This is development evidence: the routes and banks were already inspected during controller selection. The two banks are not two independently trained policies or a final holdout.

### 3.2 Two task tests to complete

**Spatially constrained passage.** A declared centerline, corridor width, requested speed and original exit checkpoint define the task. A path-priority alpha is fixed *before* each evaluation. Every actual XY sample is scored against the original corridor, even if the causal execution path reconnects. A trajectory succeeds only if it remains upright, stays inside the corridor, enters the original exit checkpoint and satisfies the declared post-turn continuation condition. Speed loss is permitted but measured.

**Speed-preserving turn with allowed deviation.** The same controller and comparable geometry are used with a wider declared corridor. A speed-priority alpha is fixed *before* evaluation. The task records actual turn speed, requested-speed deficit, lateral excursion, contact, original exit-checkpoint entry and post-turn continuation. Keeping speed while missing the checkpoint is not success.

The corridor widths, exit geometry, speed tolerance and held-out route–speed–entry-state combinations have **not yet been frozen or tested**. They must be declared from task requirements and development feasibility before seeing held-out outcomes. The present manuscript does not insert invented success rates into either task.

### 3.3 Comparators, channel diagnosis and holdout

The planned matched comparison retains U170/local200 and the same initial complete state, geometry, filters and actuator bounds within a condition. It compares the two preassigned task preferences with a fixed middle preference alpha=0.5 and a tuned conventional path follower with speed/steering limiting. The conventional rule must be tuned on development data before the holdout is opened; a rule that merely follows an unsafe aggressive command is not a meaningful baseline. Report paired task-success counts with denominators, speed deficit, original-path error, roll/contact, arrival time and actual failure endpoint. Avoid evaluating the path-priority arm only by path error or the speed-priority arm only by speed.

Suppressing one of the middle policy's two correction outputs can reveal which channel drives an observed behavior. This output-masking diagnostic does not substitute for an equally trained or tuned single-channel policy. Any claim that joint correction outperforms the best single-channel method requires that fairer comparator.

The final evaluation should lock policy and comparator parameters before testing route, speed and entry-state combinations not used for selection or tuning. Changes in wheel friction or other physical-domain parameters are separate declared domain tests. A physical robot experiment would require a separate authorized protocol; none is claimed here.

## 4 Results available at this draft stage

![Representative nominal actual/reference XY panel: one route and prepared bank, all five alpha values. The other seven panels are linked in E3.](../results/sttw-tolerance-20261101/lower_base/bank0_low_single_left_xy.png)

Table 1 is recomputed from the selected local200 rows of [E4], eight route-bank contexts per alpha. All 40 episodes entered the original 1 m goal disk and survived the full 10 s observation window. The values are descriptive means over matched development contexts, not confidence intervals or independent training-seed statistics.

| Fixed alpha | First arrivals / episodes | Mean turn speed (m/s) | Mean original-path cross-track RMSE (m) | Mean first arrival (s) |
|---|---:|---:|---:|---:|
| 0 | 8/8 | 2.115 | 0.173 | 5.487 |
| 0.25 | 8/8 | 2.145 | 0.208 | 5.464 |
| 0.5 | 8/8 | 2.180 | 0.248 | 5.443 |
| 0.75 | 8/8 | 2.228 | 0.288 | 5.416 |
| 1 | 8/8 | 2.295 | 0.328 | 5.389 |

In each of the eight paired contexts, turn speed and cross-track RMSE increased at every alpha step [E4]. The first-arrival ordering is weaker: alpha=1 arrived later than alpha=0 in the two high-speed single-right contexts despite being faster during the turn. These results are consistent with a useful preference mechanism but do not establish either declared task-level advantage. A faster turn can produce a larger excursion and need not reach the exit earlier.

The earlier H75/local300 nominal screen reached 33 of 35 original goals without contact; alpha=1 missed both tested S-turn goals [E5]. A separate low-speed tight-turn screen reached 20 of 20 first goals without contact, yet one alpha=0 trajectory had substantial late path oscillation [E6]. Those screens used an earlier lower-controller identity and cannot be pooled numerically with Table 1. They explain why the final task must score sustained behavior and the original corridor instead of counting only first goal entry.

A disturbance-focused development panel with the later U170/local200 stack gave joint original-goal arrival and full-window survival in 11/12 cases for fixed alpha=0, 10/12 for fixed alpha=1, and 11/12 for both a learned-risk coordinator and its same-timing no-risk variant [E7]. This is a negative boundary for a separate predictor claim. It is not evidence that a tolerance selector improves the two task-specified applications studied here.

## 5 Discussion

The currently supported result is narrow but consequential: under one frozen control identity, five task preferences produce an ordered nominal path–speed tradeoff over eight development contexts while retaining first goal arrival and upright survival. This is more than a cosmetic five-color trajectory plot, because actual speed and original-path error move in opposing directions. It is less than proof of application value: no scored narrow corridor or wide speed-preserving exit task has been executed with preassigned preferences and a tuned simple comparator.

The lower-level controller was adopted because it improved delivered-steering-reference tracking in its development screen, while some original-path errors increased [E2]. This reinforces the need to measure the full task rather than infer path quality from lower-level tracking or training reward. Replanning supplies an execution reference after deviation but cannot redefine the original scoring path. First arrival without terminal continuation can hide a later tracking failure, as the earlier low-speed diagnostic shows.

The design deliberately excludes automatic alpha selection and disturbance-tolerance prediction from its central claim. The matched coordinator/no-risk result gives no consistent outcome advantage attributable to the predictor [E7]. A task-provided preference may still be valuable without an online selector. The remaining test is whether two different priorities lead to meaningfully different **successful tasks**, not whether alpha=1 is universally faster, alpha=0 is universally safer, or all five levels are monotone under every condition.

The current evidence has limited geometry, prepared initial banks and development reuse. We have not estimated uncertainty across independently seeded trainings, verified real-robot speed and pose sensing, or evaluated domain-randomized task success under the new two-task definitions. These limits belong in the final claims even if the pending experiment produces a positive result.

## 6 Conclusion

A shared preference-conditioned controller can alter both speed and original-path error during single-track turning while preserving nominal goal entry and upright survival in a bounded development panel. The present evidence supports the existence of an executable path–speed tradeoff. Whether this tradeoff delivers a practical advantage for predeclared narrow-corridor and wide-corridor tasks remains unresolved until the matched task comparisons and holdout are complete. The paper's final conclusion will be revised to the measured task outcomes.

## Evidence and references

[E1] [Middle reward, training and configuration map](../results/sttw-tolerance-20261101/README.md); [effective reward source](../../learning/src/sttw_control/path_command_reward.py); [training configuration](../../learning/configs/h75_low_speed_mixed_finetune_200.json). AI artifact checked; human verification pending.

[E2] [Lower-controller selection and objective](../results/sttw-tolerance-20261101/README.md); [40-case lower summary](../results/sttw-tolerance-20261101/lower_base/summary.json). AI artifact checked; human verification pending.

[E3] [Eight actual/reference XY panels](../results/sttw-tolerance-20261101/lower_base/). AI artifact checked; human verification pending.

[E4] [Portable copy of the 120-row lower-selection comparison](evidence/nominal_five_alpha_lower_selection.json), byte-identical to the [archived source](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/lower_domain_transfer_dev_v1/comparison/results.json). Select model=last200. Table 1 was recomputed from its 40 selected rows with the [analysis script](analysis/nominal_five_alpha.py) and [machine-readable summary](evidence/nominal_five_alpha_summary.json). AI artifact checked; human verification pending.

[E5] [Earlier five-alpha nominal H75 screen](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/h75_friction08_nominal_five_alpha_v1/analysis/README.md). Different lower controller; AI artifact checked; human verification pending.

[E6] [Earlier low-speed oscillation review](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/h75_friction08_low_speed_tight_turn_v1/analysis/README.md). Different lower controller; AI artifact checked; human verification pending.

[E7] [Cross-speed paired disturbance results](../results/sttw-tolerance-20261101/cross_speed/comparison.json). AI artifact checked; human verification pending.

[R1] Huo, B., Yu, L., Liu, Y., and Chen, Z. “Hierarchical residual reinforcement learning based path tracking control method for unmanned bicycle.” *Robotics and Autonomous Systems* 190 (2025), 104996. [Publisher article](https://doi.org/10.1016/j.robot.2025.104996). Abstract and accessible section snippets checked; full numerical tables remain to be checked.

[R2] Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. “Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.” arXiv:2609.00065 (2026). [Current arXiv record](https://doi.org/10.48550/arXiv.2609.00065). Its scientific-writing procedure supported evidence organization and draft preparation; the accountable human authors must review all scientific claims.
