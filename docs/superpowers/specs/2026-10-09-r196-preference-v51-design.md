# R196 Preference V5.1 design

Implement the user supplied V5.1 contract as an explicit overlay on the V5 code at commit `6a264f7`. Preserve the frozen R196 lower controller, governed-centered interface, physical model, action mapping, upper network, and original six-case evaluation protocol.

The reward adds one independently capped `primary_excess` component and replaces `yaw_damping` with independently capped `yaw_recovery`. The former uses post-physics speed/steer error against the original limited command and is selected by the fixed endpoint alpha. The latter uses the world yaw-rate obtained from unwrapped-yaw differencing and is reward-only. The sum of component caps becomes 2080 and drives finite-task failure scoring; there is no total-cost clip.

Training is a single fresh 16-second campaign. Reset samples nominal, conflict, random, and heading-recovery-start families with probabilities 0.30/0.40/0.10/0.20. Only the fourth family offsets the upper virtual reference yaw. It leaves the physical snapshot unchanged and initializes the frozen lower controller from the actual pose. Evaluation resets keep zero initial heading offset.

The two endpoint learners start independently with fresh Actor, Critic, Adam state, and standard deviations `[0.30, 0.10]`. The bounded engineering check uses 8 environments, 16 policy steps, and 2 updates in a separate run. Official training uses 512 environments, 128 policy steps, at most 100 updates, with evaluations at updates 25 and 100 and no automatic extension.
