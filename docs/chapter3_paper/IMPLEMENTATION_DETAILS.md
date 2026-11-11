# Reproduction details: controller and event semantics

These predicates describe the frozen reported implementation, not proposed replacements. Sources: `learning/src/sttw_control/ditch_jump_adapter.py`, `offroad_jump_reward.py`, `ditch_ground_handoff.py`, `torch_middle_control.py`, `torch_path_reward.py`; online configuration `chapter3_offroad_distance_online/config.json` and its pinned `chapter3_long_bend_goal/task.json`.

## Ground controller interface

STTW names the existing speed/path-preference ground-control stack. H75 is its frozen learned middle-layer checkpoint, which receives causal motion/control history, path projection/preview and α, and adjusts speed/steering references above the ECBC balance controller and extended-state observer (ESO). The lower learned residual is zero. Front actuation uses steering-rate and rear actuation wheel-speed interfaces; the jump policy also controls the pendulum mechanism through the frozen interface.

α=0 favors path following; α=1 favors speed tracking, with finite secondary costs and unchanged actuator authority. This is the trained preference interface, not a guarantee that every episode realizes that ordering. It is not a motor-gain multiplier or a physical-limit relaxation. The existing `torch_path_reward.costs` defines training preference costs; no ground-controller training occurs in this paper's world-model fitting. After contact, continuation is fixed to1.5m/s,α0.5, while approach uses the selected speed/α.

## Event labels

All physical predicates are sampled every2.5ms; control-boundary events occur at20ms. Local wheel longitudinal positions `s` are measured from the near lip; gap width is0.30m. Scorer lane support requires |wheel y|≤1m. This support strip differs from the original-route corridor half-width0.6m.

|Model target|Actual operational rule|
|---|---|
|Jump entry|Ground-to-jump controller switch at root x≥9m; this is not measured takeoff.|
|Confirmed flight|Recorder latches scorer stage≥1. The scorer first arms on qualified near-bank support, then requires both wheels unloaded (contact norm≤1N), both lower-bound clearances>15mm, positive world-x speed, in-strip and valid physics for17 successive samples (40ms between first/last). Flight onset has front-wheel s∈[−1.2,0.30)m. Stage1 additionally requires front-wheel s≥0 and mean wheel advance≥0.05m from flight onset. Thus it is confirmed/progressed flight, not the first no-contact sample.|
|Far-bank contact|During a valid confirmed attempt, at least one wheel obtains a qualified far-bank target contact. All loaded filtered contact patches for that wheel must lie on the far bank x∈[10.6,25]m and |y|≤1m, with wheel center≥0.02m above contact points, measurable vertical support>1N, valid nonsaturated contact buffer, and wheel center s≥0.30m. The scorer touched flag is latched by the recorder.|
|STTW handoff|Alive jump-phase lane with observed far contact resumes ground control at the next control boundary. No1s stable-support gate is applied.|
|Goal continuation|Ground-continuation/bypass phase, root x∈[16.5,18.5]m, |y|≤0.5m, world vx≥0.5m/s, no physical/ground-control failure and no historical road violation, before original10s horizon.|
|Auxiliary clearance|Recorder latches scorer stage≥2. Both wheels must have been observed in the interior20–80% of the gap while unloaded, clearance≥5mm and in-strip; both centers then pass the far lip during a valid attempt. Recontact with near bank, unsupported obstacle contact, strip departure or reversing after clearance can abort a current attempt. Clearance is logged but excluded from Q's probability product.|

Near-bank arming in rough-support mode permits intermittent wheel contact: both wheels must have been observed on near support within80ms, remain before the lip, within strip, and have conservative upper clearance≤5cm; qualified support accumulates for17 samples. The scorer rejects physically discontinuous wheel position changes exceeding12dt+0.002m per sample. These details matter when interpreting the label 'flight'; a raw support-mask change alone is not the fitted target.

## Failure and horizon rules

The verified adapter declares failure for nonfinite state/action/contact values, body contact, root z<−0.15m, |root y|>3.9m, root x<0.5m, |upright-frame roll|>0.610865rad (35°), or |pitch|>1.308997rad (75°). Ground controller domain exit or numerical failure also terminates execution. No additional yaw-angle termination is used in the pinned jump physical-limit specification. The task horizon is10s from original episode start; it is not renewed at query/publication.

Distance from root XY to the current reference polyline greater than0.6m latches a road violation at physics frequency. This flag can invalidate eventual goal arrival without immediately terminating the episode. Physical validity, scorer attempt validity, corridor validity and final task success are distinct predicates.

The model's world-quaternion-derived Euler roll is not the upright-frame roll used for failure limits. Its nominal asset offset must not be interpreted as robot lean. Predicted motion error figures use normalized quaternions and circular differences.
