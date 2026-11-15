# P4 matched handoff-focus continuation: not adopted

Actual/reference/predicted XY for all12 checkpoints: best [map06](best/map06/xy.png), [map07](best/map07/xy.png), [map48](best/map48/xy.png); last [map06](last/map06/xy.png), [map07](last/map07/xy.png), [map48](last/map48/xy.png). Labels C/F denote control/focus loss arms, not independent I versus continuous C planning. [Learning curves](learning_curves.png) show all246 saved checkpoints, including regressions.

**The added handoff objective improves some exit-state errors but does not improve the calibrated planner. Reject it as the adopted loss for now.** The original-loss representative selects1/2 successful mechanism plans; the focus representative selects0/2. Both systems still reject every mechanism candidate under their own empirical margins. Last checkpoints regress markedly. Sustained convergence is not established, and P4 remains open.

## Matched experiment

Resume each previous low-rate seed's loss-best (updates3475/3450/2625) with Adam state. Two arms per seed use the same initial weights, sampling seed,1000-update budget, batch64, lr3e-5, saved25 cadence, frozen upstream predictions and estimated-entry exposure. Control retains the original objective. Focus adds weight1 times (balanced phase2-vs-rest BCE + actual-onset lateral-exit loss): world y scaled0.05m, yaw0.05rad, world vy0.1m/s, three squared terms averaged. This is a combined event/state-loss ablation, not isolation of individual terms. No network size, normalizer, frozen control skill, dynamics, success tolerance, event guard or calibration rule changed.

Total6000 supervised world-model updates,246 saved checkpoints, roughly7.8min; no PPO or new physical episodes. The six TensorBoard runs loaded40 scalar tags each at localhost6006. Training-stage Bark returnedHTTP200. Default-loss initial DEV scores exactly reproduce all three parent scores. The added-loss test checks early-event penalty, lateral bias and quaternion-sign invariance; all11 skill tests pass.

|Arm|Seed|Parent score|Best update / score|Last score|Last10 score range|
|---|---:|---:|---:|---:|---:|
|control|0|2.890|4300 / 2.351|2.678|2.351–20.154|
|control|1|2.485|4225 / 2.217|3.047|2.217–6.976|
|control|2|3.475|3525 / 2.686|5.819|2.686–5.869|
|focus|0|4.069|4425 / 2.952|6.120|2.952–6.120|
|focus|1|3.429|4375 / 2.632|7.185|2.632–7.285|
|focus|2|4.362|3550 / 3.543|6.801|3.543–24.795|

Scores across arms have different definitions and are not performance rankings. All best/last pairs are distinct, so12 checkpoints were evaluated on all90 knownDEV conditions (72 originalDEV plus18 mechanism), for1080 method conditions using the same saved physical trajectories. This is not1080 new or independent runs.

|Candidate|Best/last|DEV event found /72|DEV raw true admit /72|Mechanism true admit /4|False admit /14|
|---|---|---:|---:|---:|---:|
|control_s0_u4300|best|72|66|4|1|
|control_s0_u4475|last|60|57|2|1|
|control_s1_u4225|best|39|38|4|1|
|control_s1_u4450|last|24|23|4|0|
|control_s2_u3525|best|18|17|0|1|
|control_s2_u3625|last|0|0|4|0|
|focus_s0_u4425|best|72|65|2|1|
|focus_s0_u4475|last|3|3|0|0|
|focus_s1_u4375|best|54|49|4|1|
|focus_s1_u4450|last|0|0|0|0|
|focus_s2_u3550|best|24|24|0|1|
|focus_s2_u3625|last|0|0|1|0|

## Common-metric representative comparison

Choose one candidate per arm by originalDEV72 raw true admissions, then event-selected position MAE. Mechanism outcomes never enter that ranking. Selected control S0/4300 and focus S0/4425 both have72 supported events, so their originalDEV state errors cover the same cases. Position MAE is0.15145 versus0.12969m; lateral MAE0.04945 versus0.04613m; event MAE0.07333 versus0.06083s. This modest error reduction does not translate into better task admission:66/72 versus65/72.

Raw known-mechanism G/I/C/Oracle outcomes:

|Arm|G selected success /2|I selected success /2|C selected success /2|Oracle /2|
|---|---:|---:|---:|---:|
|control|0|1|1|2|
|focus|0|0|0|2|

Both representatives falsely admit root1002/route1, whose recorded execution collides before jumping at1.93s, clearance−0.0034518m. Focus additionally has no plan for root1001; control selects its successful route3. Full same-candidate/predictor/constraint backtracking and exhaustive selection agree in all comparisons. The recorded successful clearances of0 remain unchanged and are not positive-clearance certificates. [Control comparison](comparison/control_s0_u4300/README.md), [focus comparison](comparison/focus_s0_u4425/README.md).

Same-rule supported-domain spatial calibration (maximum AABB support miss +0.02m) rejects all18 mechanism candidates for both C/I systems under own or shared margins. Focus C under cross-I margins admits1/4 true candidates and0/14 failures; this remains a diagnostic, not a substitute for its own bound. Control C/I fits72/72 supported cases; focus C/I fits72/36, so the focus cross-I diagnostic uses only36 supported I cases. Post-region lateral bounds are control C/I0.2273/0.2432m and focus C/I0.2022/0.1519m. Fitting exclusions and per-case rejections are preserved under calibration/. No uncertainty bound was manually reduced.

## Decision and next step

Do not adopt the focus objective or either new representative as a reliable planner; retain the previous candidate identities and all negative records. This bounded run also shows that simply adding another1000 constant-rate updates is not a convergence solution: the last checkpoints often lose valid events despite a lower within-run best loss. The next bounded change should address optimization stability and evaluate event-sensitive checkpoints with the same physical thresholds; do not launch another identical constant-rate continuation or weaken the lock gate. P5/P6 remain deferred.

[All candidate metrics](candidate_cases.csv), [selection identities and hashes](evaluation_candidates.json), [training contract](contract.json), [evaluation contract](evaluation_contract.json), [exact commands](evaluation_status.json), [comparison commands](comparison_status.json). Four small orchestration scripts and the pre-run source patch accompany this report. The scripts are copies of run-local orchestration: execute them from the raw run directory below, where their relative inputs exist, not directly from this report folder. All data are TRAIN/knownDEV; no independent TEST, improved physical controller or validated safety guarantee is claimed.

Raw checkpoints, resident-GPU query arrays, logs and full calibration JSON: `STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p4_handoff_focus`. Source tools remain under learning/cli and learning/src/sttw_control; diagnostics never inject actual future data into causal queries. Separate physical/historical inputs used in the preceding departure-interchange study are not used here.
