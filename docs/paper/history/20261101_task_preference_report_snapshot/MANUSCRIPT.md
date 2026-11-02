# Task-Specified Path–Speed Preference Control for a Single-Track Robot

Internal working manuscript, 2026-11-01. The nominal, first frozen task-score and frozen-middle comparison panels have been checked against saved artifacts; a tuned conventional comparator, independent domain/initial-state validation, human verification and final declarations remain outstanding. The earlier predictor-led draft is preserved in [the dated history](history/20261101_preference_coordinator_snapshot/MANUSCRIPT.md). The current PDF and HTML are generated from this source for local review.

## Abstract

Path tracking and speed regulation can conflict during a turn on a balancing single-track robot. We study shared preference-conditioned middle policies that adjust speed and steering references while a frozen lower controller handles local tracking and balance. A task specifies one of five preference values before execution. In a 40-episode U170 nominal development panel, all preferences reached the original destination and survived 10 s; mean turn speed rose from 2.115 to 2.295 m/s as original-path cross-track RMSE rose from 0.173 to 0.328 m between endpoint preferences. A second 40-episode nominal screen preserved 40/40 arrivals and survival. In a frozen task-score panel, all 40 episodes again reached the goal and survived; preassigned path-priority alpha=0 met the original buffered narrow score in 2/8 contexts versus 4/8 for fixed alpha=0.5, while speed-priority alpha=1 met the wide score in 3/8 versus 2/8. We then compared four existing frozen middle checkpoints on matched low/high conflict routes with the same lower controller. Under a separately declared relaxed development score, original H75 achieved 8/8 narrow passages at alpha=0 versus 4/8 at alpha=0.5, while U170 achieved 8/8 wide speed-preserving turns at alpha=1 versus 4/8 at alpha=0.5. In a prospectively declared check of four new route–speed combinations and two paired initial banks, the same U170 network obtained 6/8 narrow-task successes at preassigned alpha=0 versus 4/8 at fixed alpha=0.5, and 8/8 wide-task successes at preassigned alpha=1 versus 6/8 at fixed alpha=0.5. The preference response is repeatable, and its task value depends on checkpoint and geometry. A tuned conventional comparator and independent physical-domain validation remain for an application-level claim.

## 1 Introduction

A single-track two-wheel robot must coordinate steering, forward motion, and balance. On a demanding turn, a strict path objective may require reducing speed, whereas holding speed may increase the lateral excursion or prevent the original destination from being reached. An autonomous controller should expose these consequences to the task, rather than treat every request as the same tracking problem. The application question in this paper is whether **one shared controller** can produce useful behavior for two different, predeclared priorities under the same vehicle, lower controller and actuator limits.

Huo et al. [R1] studied a different bicycle-control problem: path tracking at a controlled 4 m/s in PyBullet. Their hierarchy separates attitude and position control and adds residual reinforcement learning to traditional control at both levels. Their published abstract reports improvements in path tracking and training behavior. This prior work means that hierarchy and residual learning alone cannot be our novelty. Our proposed distinction is **task-specified path–speed execution**: the same middle policy changes both reference channels in response to a user preference, while the original path and destination remain the evaluation targets. The complete Huo article's numerical tables have not been independently verified in this draft; no quantitative cross-paper comparison is made.

The controller is already trained. We therefore do not propose a new reinforcement-learning algorithm or a learned selector of the best preference. The evidence needed for a control-applications claim is more specific: first, that the preference controls a reproducible path–speed tradeoff; second, that preassigning different preferences helps complete distinct spatial and speed tasks; and third, that this effect survives fair comparison with a tuned simple rule and held-out operating conditions. The first proposition has repeated nominal evidence. A first frozen task-score panel gives mixed outcomes; the tuned-rule and independent-domain comparisons remain open.

The intended contributions, conditional on those pending tests, are (i) a task formulation that scores both original-path geometry and requested speed, (ii) a shared conditional middle controller that jointly adjusts speed and steering references while reusing one lower controller, and (iii) task-level evidence separating narrow-corridor passage from a wide-corridor, speed-preserving turn. These are control-application contributions, not claims of a new learning algorithm, formal stability guarantee or universal benefit of any particular preference value.

## 2 Control system and preference objective

### 2.1 Frozen controller and original task

The original evaluated middle policy is STTW_MIDDLE_H75_LOW_SPEED_MIXED_U170. It was selected from a continuation of H75 training; the selected checkpoint is update 170, whereas the continuation ran through update 202. The frozen comparison additionally evaluates original H75, W400 and u325. The common lower policy is STTW_LOCAL_DOMAIN_U200. The middle controller runs at 50 Hz and produces bounded speed and steering-reference corrections. The lower ECBC/ESO plus local residual stack runs at 200 Hz. A deterministic causal geometric reconnection process runs at 10 Hz. Both wheel friction coefficients are 0.8 in the evaluated MuJoCo model. The evaluated task window is 10 s. The fixed original destination is a 1 m disk for the legacy nominal panel. These details and identities are recorded in the [experiment package](../results/sttw-tolerance-20261101/README.md) and [lower-selection record](../results/sttw-tolerance-20261101/lower_base/selection.json).

The original world-frame path, speed request and destination define what the user asked for. If the vehicle departs and receives a reconnecting execution path, task scoring still uses the original geometry and original destination. First entry into the destination disk, sustained post-turn tracking, full-window upright survival and speed tracking are separate quantities. The legacy first-entry measure has no stop, terminal heading or hold requirement; it must not be called complete path tracking.

The user chooses alpha from the evaluated set {0, 0.25, 0.5, 0.75, 1}. Zero emphasizes original-path tracking and one emphasizes requested speed. The same network weights and lower controller are used for all five values. Alpha is fixed for a task episode in the present study. Training sampled the endpoints with probability 0.4 each and a uniform alpha with probability 0.2; the five values are evaluation settings, not five independently trained policies [E1].

### 2.2 Effective training objective

The middle PPO continuation used the implemented path_persistent_preference_v3 reward [E1]. Let H(x) = x² for |x| ≤ 1 and H(x) = 2|x|−1 otherwise. Let e_v be error against the original speed request, e_y the signed original-path cross-track error, e_h the original-path heading error, and chi the configured curvature-demand gate. Before adding balance and reference regularizers, the effective endpoint-mixed running costs include:

- speed: (1−alpha)(10−9chi)H(e_v/0.1) + alpha·10H(e_v/0.1);
- path: 10(1−0.9alpha)[H(e_y/0.1) + 0.5H(e_h/0.1)];
- primary band: 20chi[(1−alpha)H((|e_y|−0.1)_+/0.1) + alpha H((|e_v|−0.05)_+/0.1)].

Each endpoint channel is capped before mixing. Roll, working-roll, roll-rate, overspeed, low-speed and reference-offset terms also apply; the per-lower-step reward is −0.1×0.005 times the sum of effective costs, with a separate early-failure term. The frozen configuration contains an older textual path_weight field, but the V3 implementation overrides it through path_secondary_ratio=0.1. The [reward implementation](../../learning/src/sttw_control/path_command_reward.py) and [frozen configuration](../../learning/configs/h75_low_speed_mixed_finetune_200.json) define the actual formula. The lower continuation was optimized for local speed and delivered-steering-reference tracking, with no alpha, original-path or goal term [E2]. Changing alpha therefore changes the middle task objective and reference output, not the lower policy's own reward.

Training reward is not itself the application outcome. The tests below compare actual speed, original-path error and task events. Different alpha values have different reward weights, so their weighted returns are not ranked as one common physical-performance score.

The U170 continuation log contains 150 sampled-training records, updates 53–202 [E11]. Mean reward per lower step was −0.108, −0.146 and −0.076 in the first, middle and last 40-record windows; the changing rollout mix and visible fluctuations do not establish a fixed-task reward plateau. The log records 119 domain exits, zero physical failures, three target-KL-shortened updates and no nonfinite optimizer stop. We select and assess the controller from executed task trajectories rather than infer task completion from this reward trace.

## 3 Experimental design and evidence roles

### 3.1 Completed nominal preference panel

The completed panel paired four route types—low-speed single turn, low-speed S turn, high-speed single turn and high-speed S turn—with prepared banks 0 and 6. Each of the eight contexts ran all five fixed alpha values with U170/local200, giving 40 physical episodes. Saved XY panels overlay the actual motion and original reference [E3]. The comparison record [E4] contains first arrival, survival, turn-interval mean speed and original-path cross-track RMSE over the common pre-arrival prefix. This is development evidence: the routes and banks were already inspected during controller selection. The two banks are not two independently trained policies or a final holdout.

### 3.2 Two task tests to complete

**Spatially constrained passage.** A declared centerline, corridor width, requested speed and original exit checkpoint define the task. A path-priority alpha is fixed *before* each evaluation. Every actual XY sample is scored against the original corridor, even if the causal execution path reconnects. A trajectory succeeds only if it remains upright, stays inside the corridor, enters the original exit checkpoint and satisfies the declared post-turn continuation condition. Speed loss is permitted but measured.

**Speed-preserving turn with allowed deviation.** The same controller and comparable geometry are used with a wider declared corridor. A speed-priority alpha is fixed *before* evaluation. The task records actual turn speed, requested-speed deficit, lateral excursion, contact, original exit-checkpoint entry and post-turn continuation. Keeping speed while missing the checkpoint is not success.

The completed bounded validation panel froze a **buffered-anchor corridor proxy** before physical execution [E9]: a 0.5 m radius disk around the recorded vehicle XY anchor must stay within an original-path half-width of 0.8 m for the narrow task or 1.4 m for the wide task until first entry into the original 1 m goal disk. Both tasks also require full H10 survival and original-path cross-track RMSE at most 0.2 m during 8–10 s. The wide task additionally requires actual mean turn speed at least 0.88 of the requested speed; the narrow task reports speed loss without a minimum. Alpha=0 and alpha=1 are preassigned to narrow and wide, with alpha=0.5 as a fixed compromise. The 0.5 m disk is an XY scoring proxy rather than an exact swept vehicle mesh. Thresholds were selected from the earlier DEV2 panel. The new route–speed combinations are prospective, but prepared banks may have prior project use; this is not an independent random-domain generalization test.

### 3.3 Frozen middle checkpoint comparison

We reused the task-score panel's four low/high single/S-turn routes, prepared banks 1 and 3, original path and destination, local200 lower controller, friction 0.8 and 10 Hz reconnection. The candidate set was U170, original H75, W400 and u325. All five fixed alpha values were executed for every declared candidate condition. H75 and W400 completed all four routes (40 episodes each); u325 completed the two selected conflict routes (20 episodes). A first H75 process was terminated externally after 400 lower steps; its record was kept and the same 20-case panel was manually rerun from a new run directory. No model was trained during this comparison.

For model selection, we declared a relaxed development score after inspecting U170's original task panel: the narrow buffered-anchor half-width is 0.9 m, the wide half-width 1.4 m, the post-goal 8–10 s original-path RMSE limit 0.25 m, and the wide turn-speed threshold 0.85 of the request. The 0.5 m anchor buffer, original-goal entry and H10 survival remain. This score is distinct from the earlier frozen 0.8 m/0.2 m/0.88 score in Table 2. It was applied to every model without rewriting Table 2 [E10].

### 3.4 New task combinations

After the checkpoint-selection panel, we froze four new exact route–speed–direction combinations: a low-speed single left turn (76°, R1.45 m, 1.90 m/s), low-speed S right turn (72°, R1.55 m, 2.00 m/s), high-speed single right turn (96°, R3.15 m, 3.05 m/s), and high-speed S left turn (70°, R3.10 m, 3.10 m/s). Prepared complete initial banks 2 and 5 had not been used in the preceding four-route task-score or checkpoint-selection panels. H75 and U170 each executed all four routes × two banks × five fixed alpha values, 40 physical 10 s episodes per model, under the same local200 lower controller, friction 0.8 and 10 Hz geometric reconnection [E12]. The same relaxed buffered-anchor score from Section 3.3 was fixed before these executions. This is a prospective exact-combination check; it is not a claim that the continuous geometry distribution, initial-state generator or physics domain was unseen during training.

### 3.5 Comparators, channel diagnosis and independent domain check

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

A prospectively declared second development screen used four new low/high single/S-turn route–speed combinations and two additional prepared banks with the same U170/local200 stack [E8]. All 40 fixed-preference episodes entered the original 1 m goal disk and survived H10. At the predeclared 0.50 m original-path centerline half-width, the five α levels passed 8/8, 8/8, 6/8, 4/8 and 4/8, respectively; mean actual turn speed increased from 2.079 to 2.278 m/s between α0 and α1. This is stronger evidence that the behavior tradeoff is repeatable across more nominal contexts, but the half-width test uses a centerline proxy, not the vehicle footprint, and does not score the requested-speed threshold or establish independent generalization.

![New task-score panel: low-speed S turn, original path and all five executed preferences.](../results/sttw-tolerance-20261101/task_preference_task40/bank1_tasktest_low_S_left_68_R1p35_v2p05_zero_fixed_candidate_xy.png)

The frozen task-score validation [E9] used four new low/high single/S-turn route–speed combinations, two prepared banks and all five fixed α values, again 40 physical episodes. Every episode entered its original goal and survived H10. Actual mean turn speed rose from 2.116 to 2.297 m/s between α0 and α1. Table 2 reports the two preassigned task arms against the fixed α0.5 compromise, using the declared buffered-anchor corridor, actual speed and 8–10 s original-path tracking.

| Route family (two prepared states each) | Narrow α0 | Narrow α0.5 | Wide α1 | Wide α0.5 |
|---|---:|---:|---:|---:|
| Low-speed single right | 2/2 | 2/2 | 0/2 | 0/2 |
| Low-speed S left | 0/2 | 2/2 | 2/2 | 0/2 |
| High-speed single left | 0/2 | 0/2 | 1/2 | 2/2 |
| High-speed S right | 0/2 | 0/2 | 0/2 | 0/2 |
| **All four routes** | **2/8** | **4/8** | **3/8** | **2/8** |

The narrow α0 failures on the low-speed S turn exceed the buffered boundary by 0.23 and 0.55 mm; these threshold crossings are too small to imply a meaningful geometry disadvantage. On the high-speed single-left route, one wide α1 failure has last-2-s original-path RMSE 0.2016 m against the frozen 0.2000 m limit. The unaltered success counts show a mixed task result: α1 gains two wide-task completions on the low-speed S turn but loses one on the high-speed single-left turn, and α0 has no aggregate narrow-task advantage. The per-case continuous metrics and eight actual/reference XY panels are retained in [E9].

![Original H75 actual/reference XY on a low-speed S turn: all five fixed preferences, prepared bank 1.](../results/sttw-tolerance-20261101/frozen_middle_comparison/H75/bank1_tasktest_low_S_left_68_R1p35_v2p05_zero_fixed_h75_xy.png)

Table 3 compares the three frozen middle policies with full four-route coverage under the common *relaxed development* score [E10]. Every one of their 120 evaluated cases entered the original goal and survived H10. All five alpha levels and the fourth policy u325's two-route screen are preserved in the result package. The table uses eight paired route/bank conditions per model and alpha.

| Middle | Alpha | Narrow task | Wide task | Mean turn speed (m/s) | Mean pre-goal maximum original cross-track (m) |
|---|---:|---:|---:|---:|---:|
| U170 | 0 | 6/8 | 3/8 | 2.116 | 0.369 |
| U170 | 0.5 | 6/8 | 4/8 | 2.195 | 0.396 |
| U170 | 1 | 4/8 | 8/8 | 2.297 | 0.539 |
| H75 | 0 | 8/8 | 4/8 | 2.202 | 0.149 |
| H75 | 0.5 | 4/8 | 8/8 | 2.339 | 0.371 |
| H75 | 1 | 0/8 | 6/8 | 2.452 | 0.705 |
| W400 | 0 | 6/8 | 6/8 | 2.279 | 0.187 |
| W400 | 0.5 | 4/8 | 8/8 | 2.400 | 0.429 |
| W400 | 1 | 0/8 | 6/8 | 2.463 | 0.715 |

H75 shows the clearest path-priority response: alpha=0 clears all eight narrow conditions, whereas alpha=0.5 clears four, while mean turn speed rises across its five alpha settings. U170 remains the strongest alpha=1 wide-task endpoint in this panel. The first two-route model screen favored W400; its remaining low-speed single-right route exposed post-goal tracking loss, so a two-route selection would have been misleading. Original H75 is the working base for the next path-priority experiment, with U170 retained as the speed-priority reference. The relaxed score is checkpoint-selection development evidence, not a replacement for the earlier frozen task test.

![U170 actual/reference XY on a new high-speed single right turn, prepared bank 2; all five fixed preferences.](../results/sttw-tolerance-20261101/task_preference_holdout/U170/bank2_holdout_high_single_right_96_R3p15_v3p05_zero_fixed_candidate_xy.png)

<div style="break-before: page"></div>

Table 4 reports the new-combination panel [E12], eight paired route/bank conditions per model and alpha. All 80 cases entered the original goal and survived H10. The score is the previously fixed relaxed development score; the route–speed–bank combinations were fixed before this panel.

| Middle | Alpha | Narrow task | Wide task | Mean turn speed (m/s) | Mean pre-goal maximum original cross-track (m) |
|---|---:|---:|---:|---:|---:|
| H75 | 0 | 8/8 | 7/8 | 2.200 | 0.223 |
| H75 | 0.25 | 5/8 | 8/8 | 2.253 | 0.317 |
| H75 | 0.5 | 4/8 | 8/8 | 2.325 | 0.391 |
| H75 | 0.75 | 4/8 | 8/8 | 2.401 | 0.482 |
| H75 | 1 | 3/8 | 5/8 | 2.458 | 0.615 |
| U170 | 0 | 6/8 | 2/8 | 2.094 | 0.273 |
| U170 | 0.25 | 6/8 | 2/8 | 2.131 | 0.334 |
| U170 | 0.5 | 4/8 | 6/8 | 2.178 | 0.409 |
| U170 | 0.75 | 4/8 | 8/8 | 2.234 | 0.473 |
| U170 | 1 | 4/8 | 8/8 | 2.300 | 0.532 |

Within U170, the preassigned path-priority endpoint improves narrow-task passage from 4/8 at fixed alpha=0.5 to 6/8, and the speed-priority endpoint improves wide-task completion from 6/8 to 8/8. These are gains of two paired contexts per task under the selected score; the [per-route table](../results/sttw-tolerance-20261101/task_preference_holdout/HOLDOUT.md) retains all route-level outcomes. H75 preserves its 8/8 alpha=0 narrow-task result on the new combinations, but its alpha=1 wide score is 5/8 and some high-speed alpha=0 traces show late speed oscillation. Thus neither checkpoint supports a claim that one endpoint is best for all tasks.

The earlier H75/local300 nominal screen reached 33 of 35 original goals without contact; alpha=1 missed both tested S-turn goals [E5]. A separate low-speed tight-turn screen reached 20 of 20 first goals without contact, yet one alpha=0 trajectory had substantial late path oscillation [E6]. Those screens used an earlier lower-controller identity and cannot be pooled numerically with Table 1. They explain why the final task must score sustained behavior and the original corridor instead of counting only first goal entry.

A disturbance-focused development panel with the later U170/local200 stack gave joint original-goal arrival and full-window survival in 11/12 cases for fixed alpha=0, 10/12 for fixed alpha=1, and 11/12 for both a learned-risk coordinator and its same-timing no-risk variant [E7]. This is a negative boundary for a separate predictor claim. It is not evidence that a tolerance selector improves the two task-specified applications studied here.

## 5 Discussion

U170 produces an ordered nominal path–speed tradeoff across two 40-episode panels. Its first frozen task score gave mixed task outcomes, but the subsequent new-combination panel gives a paired gain of two contexts for each preassigned endpoint against fixed alpha=0.5: alpha=0 on the narrow task and alpha=1 on the wide task. This establishes a bounded task-level use of one shared network under the fixed scoring rule. The checkpoint screen also matters: original H75 is stronger on the narrow task, yet its alpha=1 wide score falls below its fixed compromise. On the earlier selected routes, H75 alpha=0.25 passed both relaxed task scores in 8/8 contexts; on the new combinations its narrow score drops to 5/8. A fixed intermediate preference can be sufficient in some geometries, and task conditions determine when choosing an endpoint changes the outcome. The tuned simple-rule comparison must use the same task conditions.

The lower-level controller was adopted because it improved delivered-steering-reference tracking in its development screen, while some original-path errors increased [E2]. This reinforces the need to measure the full task rather than infer path quality from lower-level tracking or training reward. Replanning supplies an execution reference after deviation but cannot redefine the original scoring path. First arrival without terminal continuation can hide a later tracking failure, as the earlier low-speed diagnostic shows.

The design deliberately excludes automatic alpha selection and disturbance-tolerance prediction from its central claim. The matched coordinator/no-risk result gives no consistent outcome advantage attributable to the predictor [E7]. A task-provided preference may still be valuable without an online selector. The next test is whether a tuned simple rule can reproduce the observed speed–path choices on the same task conditions.

The current evidence has limited geometry, prepared initial banks and development reuse. We have not estimated uncertainty across independently seeded trainings, verified real-robot speed and pose sensing, or evaluated domain-randomized task success under the new two-task definitions. These limits belong in the final claims even if the pending experiment produces a positive result.

## 6 Conclusion

Frozen preference-conditioned middle policies produce repeatable path–speed behavior while preserving original-goal arrival and upright survival across the bounded panels reported here. Under the previously fixed relaxed score, U170's new route/bank combinations show a paired two-context task gain over fixed alpha=0.5 for each preassigned endpoint: 6/8 versus 4/8 narrow and 8/8 versus 6/8 wide. H75 keeps 8/8 alpha=0 narrow success but has 5/8 alpha=1 wide success in those combinations. The policy checkpoint and geometry therefore affect which preferences are useful. A tuned traditional rule, dual-channel diagnosis and independent physical-domain validation remain before a general task-level superiority claim.

## Evidence and references

[E1] [Middle reward, training and configuration map](../results/sttw-tolerance-20261101/README.md); [effective reward source](../../learning/src/sttw_control/path_command_reward.py); [training configuration](../../learning/configs/h75_low_speed_mixed_finetune_200.json). AI artifact checked; human verification pending.

[E2] [Lower-controller selection and objective](../results/sttw-tolerance-20261101/README.md); [40-case lower summary](../results/sttw-tolerance-20261101/lower_base/summary.json). AI artifact checked; human verification pending.

[E3] [Eight actual/reference XY panels](../results/sttw-tolerance-20261101/lower_base/). AI artifact checked; human verification pending.

[E4] [Portable copy of the 120-row lower-selection comparison](evidence/nominal_five_alpha_lower_selection.json), byte-identical to the [archived source](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/lower_domain_transfer_dev_v1/comparison/results.json). Select model=last200. Table 1 was recomputed from its 40 selected rows with the [analysis script](analysis/nominal_five_alpha.py) and [machine-readable summary](evidence/nominal_five_alpha_summary.json). AI artifact checked; human verification pending.

[E5] [Earlier five-alpha nominal H75 screen](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/h75_friction08_nominal_five_alpha_v1/analysis/README.md). Different lower controller; AI artifact checked; human verification pending.

[E6] [Earlier low-speed oscillation review](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/h75_friction08_low_speed_tight_turn_v1/analysis/README.md). Different lower controller; AI artifact checked; human verification pending.

[E7] [Cross-speed paired disturbance results](../results/sttw-tolerance-20261101/cross_speed/comparison.json). AI artifact checked; human verification pending.

[E8] [Prospective nominal screen, eight XY panels and all 40 case outcomes](../results/sttw-tolerance-20261101/task_preference_nominal40/TASK_SCREEN.md); [frozen configuration](../../learning/configs/task_preference_nominal40.json); [diagnostic code](analysis/task_preference_screen.py). DEV2, not independent TEST; AI artifact checked, human verification pending.

[E9] [Frozen task-score results, eight actual/reference XY panels and per-route outcomes](../results/sttw-tolerance-20261101/task_preference_task40/TASK_SCORES.md); [predeclared configuration](../../learning/configs/task_preference_task40.json); [scoring implementation](analysis/task_preference_task_score.py). AI artifact checked; human verification pending.

[E10] [Frozen-middle comparison, 28 actual/reference XY panels and all per-case outcomes](../results/sttw-tolerance-20261101/frozen_middle_comparison/README.md); [four-route machine-readable comparison](../results/sttw-tolerance-20261101/frozen_middle_comparison/full_route_comparison.json); [two-route four-model screen](../results/sttw-tolerance-20261101/frozen_middle_comparison/comparison.json); [comparison code](analysis/compare_full_route_models.py). Model-selection DEV with relaxed thresholds, not independent TEST; AI artifact checked, human verification pending.

[E11] [U170 continuation reward, domain exit, KL and critic-loss curves](../results/sttw-tolerance-20261101/training_review/README.md); [machine-readable training review](../results/sttw-tolerance-20261101/training_review/training_review.json); [review code](analysis/review_middle_training.py). Source is the frozen continuation `metrics.jsonl`; AI artifact checked, human verification pending.

[E12] [New task-combination panel, 16 actual/reference XY figures and all 80 outcomes](../results/sttw-tolerance-20261101/task_preference_holdout/README.md); [paired per-route table](../results/sttw-tolerance-20261101/task_preference_holdout/HOLDOUT.md); [analysis code](analysis/compare_holdout_middles.py). Exact route/bank combinations were fixed before execution; task thresholds came from earlier development. AI artifact checked, human verification pending.

[R1] Huo, B., Yu, L., Liu, Y., and Chen, Z. “Hierarchical residual reinforcement learning based path tracking control method for unmanned bicycle.” *Robotics and Autonomous Systems* 190 (2025), 104996. [Publisher article](https://doi.org/10.1016/j.robot.2025.104996). Abstract and accessible section snippets checked; full numerical tables remain to be checked.
