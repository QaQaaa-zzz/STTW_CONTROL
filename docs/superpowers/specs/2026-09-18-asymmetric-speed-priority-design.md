# Asymmetric speed priority and base-output scaling design

This document records the user-approved design. It is an implementation contract, not evidence that the resulting policy will learn or satisfy the task.

## Task semantics

Alpha zero prioritizes the fixed original geometric path and may accept temporary underspeed. Alpha one strictly prioritizes the active external speed reference and may temporarily enlarge the turn radius, but it must return on time. Overspeed is expensive at every alpha. Ordinary feasible cases should permit all alphas to track both objectives without forced trajectory separation.

The vehicle, ECBC+ESO, 200 Hz control, actuator limits, tanh action mapping, and residual limits stay fixed. Alpha never changes authority. A configurable `base_output_scale` multiplies both base commands before residual and external command disturbances are added. Its default is 1.0. Learning uses 0.8; original baseline remains 1.0 and zero-residual 0.8 is reported separately.

## Reward and recovery

The shared reward implementation separates underspeed and overspeed using the pre-action active external reference. The priority ratio is configurable: rho 34.2 is the main synthetic-prior candidate and rho 10 is the controlled comparison. Overspeed weight is fixed at rho/(1+rho); only underspeed and path weights vary with alpha. The Huber-like cost, directional tolerance bands, three-second recovery tightening, final asymmetric speed band, geometric path/heading bands, roll/roll-rate conditions, action costs, and zero recovery bonus follow the user's numerical specification.

Command-conflict recovery starts when the already published active reference enters its recovery segment. Disturbance recovery retains observable departure timing. New commands cannot erase an existing recovery debt. Raw request and active reference, directional reward terms, directional peaks/integrals/band durations, hold, timeout, and physical failure are logged. Old configurations keep their exact legacy reward behavior when the asymmetric switch is disabled.

## Training distribution and preparation

Every 10-second task starts from a real closed-loop straight preparation that preserves vehicle, ESO, actuator, history, and clock continuity. The main mixture is 40% feasible nominal tracking, 40% one finite acceleration-turn conflict followed by a feasible recovery segment, and 20% one 0.5-second lateral force of plus or minus 2 N on a feasible gentle bend. Left/right directions are balanced. Alpha is sampled once per episode.

The fixed core pair begins from stable 2.3 m/s. At task time 1.00 s the raw request becomes 2.5 m/s and plus or minus 1.8 rad/s; at 1.95 s yaw request returns to zero. It uses 1.0 m/s2 speed-reference slew and 2.4 rad/s2 yaw-reference slew. Reference integration, total turn angle, and remaining recovery time are audited. Excluded recovery samples retain reasons and become pressure-test inputs rather than being silently discarded as infeasible.

## Optimizer and selection

Both training candidates use 1024 environments, 128 rollout steps, 48 updates, four epochs, minibatch 32768, learning rate 0.0003 fixed, gamma 0.9995, GAE 0.99, clip 0.2, entropy 0.001, and initial standard deviation 0.15. Actor and critic are three-layer 128 ELU networks, both observing alpha. Mean full-update KL above 0.02 rolls back policy, value function, optimizer, and update-side state; the learning rate is not adapted.

Checkpoint selection compares initialization, update 1, a declared middle update, and update 48 using four fixed full-episode development scenes: ordinary acceleration, core left conflict, core right conflict, and gentle-bend lateral disturbance. Each uses baseline, alpha 0/0.5/1, identical complete prepared state, and overspeed peak/duration gates in addition to existing tracking/failure criteria. The original five-scene historical panel remains a regression check. Later checkpoints are not forced to win.

## Evidence boundary

Reward contract tests establish formulas, compatibility, reconstruction, and selected synthetic candidate ordering. CPU/JAX equality establishes implementation agreement. GPU smoke establishes that the pipeline executes. Formal development panels determine whether a trained policy meets the declared conditions. None of these alone proves real-vehicle safety or a hard no-overspeed guarantee.
