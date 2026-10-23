# Frozen prospective context-transfer results

[Actual nominal XY, all four policies and alphas](analysis/actual_xy.png)

| Geometry | Parent task | Primary918501 | Repeat918502 | Repeat918503 | Parent domain | Each taught domain |
|---|---:|---:|---:|---:|---:|---:|
|70°R1.9|77/96|79/96|79/96|79/96|10|5|
|130°R2.7|83/96|86/96|86/96|88/96|5|0|
|Total|160/192|165/192|165/192|167/192|15|5|

Primary remains918501 as locked before evaluation. No switch to918503. All models retain32/32 unique nominal tasks and32/32 nominal working successes; all8 geometry/direction/speed contexts preserve strict speed-error decrease across alpha0,.2,.6,1. These are fixed-alpha tests, not online switching.

Primary paired outcomes:12 task improvements and7 regressions. Working performance94→91/192. Remaining domain exits leave the later physical horizon UNKNOWN; do not claim a global physical feasibility boundary or universal survival.

## Regressions and limits

Four regressions occur with later pulses at2.8s and smallalpha: two70°cases at2.3m/s alpha0 have tail path0.230–0.285m, heading0.258–0.285rad and speederror0.127–0.134m/s. Two right-turnalpha.2 cases have acceptable path0.077–0.080m but heading0.141–0.145rad. These are meaningful task-recovery failures.

Three130°/2.7m/s alpha1 regressions have tail path0.102–0.105m while speed/heading/goal are normal. They are small threshold crossings, retained in counts but interpreted separately. All seven raw paired records are in analysis/primary_regressions.json. Broadly better mean recovery does not make every context better.

## Decision

Retain original918501 as the leading research recovery candidate, with parentu325 as immutable baseline/fallback. Evidence supports a reproducible improvement on previous DEV and a modest net improvement on prospectively locked new contexts, while preserving the tested preference behavior. It does not support unconditional replacement, certified safety, physics/reset generalization, or arbitrary onlinealpha switching. The three optimization seeds share parent/data/reset.

Do not immediately retrain on these evaluation cases or tune to their thresholds. Close this evaluation round and consolidate the task-level evidence and the separate tolerance-model negative results. Any later targeted study of late smallalpha recovery must explicitly retire this panel from independent evaluation and use another locked panel afterward. No new reward, low-level controller change, tolerance predictor, or hand-coded selector is justified by these results alone.

The formulas were not used in this teaching update. Their previous diagnostic information and failed training-utility comparisons remain separate. Claims of analytic tolerance improving training are not supported by this round.

768 fresh evaluations used1500856 active lower ticks,0new training. All policy and protocol hashes were committed before outcomes. Per-panel full reports and raw trajectories remain in the run directories; compact publication retains hashes and key figures.
