# TRAIN coordination feasibility probe

Question: do fixed two-second speed-off or steer-off targets produce reusable recovery behavior in training-domain contexts where the unchanged parent fails?

72 deterministic cases: 24 paired contexts, three arms (control/speed_off/steer_off), two directions, speeds 2.25/2.4 m/s, alpha 0/.125, forces 0/±24 N. Radius 2.4 m, angle 115 degrees, pulse at 1.7 s for 0.1 s; complete state copied at 1.8 s; intervention ends 3.8 s; original deadline 10 s. Same parent u325, V3, local300 and all control limits. Maximum 144000 lower ticks, zero PPO. One panel, no duration/force sweep, no auto-retry.

Use all control successes/failures and each arm's paired changes; report no-force retention, domain exits, continuous path/heading/speed and XY. A single reset snapshot is shared: 72 lanes are not 72 independent replicates. Geometry/speeds/force/onset differ from DEV, but no claim of statistical independence or broad generalization. Entire pilot is TRAIN-only and cannot later become final test evidence.

Decision: if intervention gives no useful recovery pairs, stop imitation route. If useful pairs exist, inspect physical/performance regressions and availability of deployable observation/action targets before designing training. Successful forced rollouts alone do not prove a learned actor can reproduce them. Existing screen saves diagnostic traces, not full observation targets; additional collection would need an explicit dataset contract. No actor update or automatic adoption in this run.
