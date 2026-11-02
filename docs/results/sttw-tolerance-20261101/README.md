# STTW preference and disturbance results — 2026-11-01

Frozen comparison: middle `STTW_MIDDLE_H75_LOW_SPEED_MIXED_U170`, lower `STTW_LOCAL_DOMAIN_U200`, 50 Hz middle/200 Hz lower, 10 Hz geometric reconnection, wheel friction 0.8, 10 s task window. The original goal is a 1 m disk; arrival and full-window survival are separate. All routes here are development cases from one shared prepared-state bank, not an independent generalization test.

## Actual/reference XY first

The [12 matched low/high, single/S-bend panels](cross_speed/) show all five strategies under nominal and ±36 N side loads. The [three same-state intervention panels](directional/) show both disturbance directions and the later measured trigger. The [six lower-continuation panels](lower_u140/) show both speeds, nominal and ±36 N, and all five α values. The [nominal u1](lower_u1_nominal/), [nominal u200](lower_u200_nominal/) and [disturbed u1](lower_u1_disturbed/) folders cover every condition in their respective bounded checks. Each comparison folder also contains `comparison.json` with per-condition outcomes and original raw-file hashes. Raw trajectories, training logs, and model weights remain in the immutable local run archive and are not duplicated here.

![Low-speed right turn, −36 N: five matched strategies](cross_speed/case_09_xy.png)

![Low-speed right turn, +36 N: five matched strategies](cross_speed/case_11_xy.png)

## Results and decision

| Paired study | Original-goal arrival and H10 survival | Interpretation |
|---|---:|---|
| Four routes × nominal/±36 N, 12 conditions per method | Measured-return risk coordinator 11/12; same-timing without risk 11/12; fixed α0 11/12; fixed α1 10/12; short dwell 10/12 | Risk coordinator gains one and loses one versus the matched no-risk rule. Its added arrival versus short dwell is real but does not establish consistent risk-selection benefit. |
| Pulse-trained lower u140, two routes × five α × nominal/±36 N | Parent local200 23/30; u140 10/30; paired gains 0, losses 13 | Reject u140. High-speed nominal turning fails to reach the original exit in all five α cases. |
| Pulse-trained lower u1/u200, low/high nominal × α0/1 | Parent 4/4; u1 4/4; u200 2/4 | u1 retains nominal tasks; u200 regresses. |
| Pulse-trained lower u1, low-speed single right × α0/1 × ±36 N | Parent 1/4; u1 1/4; paired gains 0, losses 0 | No disturbance-recovery gain; retain local200. |

The lower continuation used the existing on-policy PPO framework: local200 Actor/Critic weights, fresh optimizer, 200 updates and 25,804,800 simulation transitions. Half of episodes had no added pulse; the others received a random signed 24–36 N world-Y pulse for 0.15 s at 1–6 s. Its local speed/steering commands and reward did not include the complete H75 preference-switch-and-return task. Training reward alone was not a reliable adoption criterion; the tested candidates are not adopted.

Same-complete-state directional intervention provides the mechanism question: at task time 1.40 s, held α values that survive [−36 N](directional/held_minus36.png) and [+36 N](directional/held_plus36.png) are disjoint; by the measured 1.72 s response, [all five tested held α continuations fail under −36 N](directional/feedback_minus36.png). This supports studying an observable, recoverable preparation window, while the present held-candidate predictor does not label the full online switch-and-return policy. The working [research design](../../paper/RESEARCH_DESIGN.md) separates these hypotheses from established results. The existing manuscript PDF describes an earlier controller generation and has not been updated to these results.

## Code entry points

- [Native closed-loop evaluation and original-goal outcome](../../../learning/src/sttw_control/native_goal_run.py), [risk coordination](../../../learning/src/sttw_control/tolerance_coordination.py), [matched lower comparison](../../../learning/cli/compare_lower_transfer.py).
- [Pulse sampling in the existing lower environment](../../../learning/src/sttw_control/lower_command.py), [PPO training](../../../learning/src/sttw_control/lower_command_training.py), and [frozen training configuration](../../../learning/configs/lower_pulse_finetune.json).

This is a compact review package: PNG XY panels plus numerical comparison files, with no checkpoints, duplicate PDFs, or raw trajectory archives.
