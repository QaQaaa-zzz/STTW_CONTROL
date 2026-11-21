# Prospective training comparison

Compare three independent initializations of the shared PCRP middle actor and critic (7101, 7102, 7103) with three dedicated networks (alpha 0, .5, 1, each initialized using 7101). The shared model uses all five preferences with one sampled per episode. Use the same per-network interaction budget, architecture, route bank, local balance controller, randomization and training code. Retain each final model regardless of outcome. Dedicated models each receive the full training budget for one preference. The three dedicated runs supply one initialization per preference.

Each stage runs one network for 150 updates of 4096 environments by 24 steps (14,745,600 transitions). Stop at update 150 or a training error. The first stage runs shared seed 7101. The 50 Hz policy and 200 Hz lower loop retain the 0.2 ms physics step. PPO uses the Go2 Spring Jump runner at gamma .99 and lambda .95 per 20 ms. Train from a common warm vehicle start on the archived TRAIN geometries with flat and wave ground, signed lateral pulses, and the existing Go2 domain randomization. Explicitly reinitialize both trainable networks and the optimizer. The interface loads the frozen historical actor bundle for reference shaping and observations; the sampled actions come from the newly initialized trainable actor. This prospective Isaac/Go2 protocol differs from the original 77-update H75 training.

Evaluate the final model on the frozen six-condition matrix used for the analytical governor, five preferences for the shared actor and its designated preference for each dedicated actor. Treat the archived physical groups as matched development evidence. Report physical completion, final hold, A/B/C and common-window speed/path RMSE individually, then summarize between shared-model seeds. The historical H75 weights do not enter the three-seed calculation.

## Interpretation of the matched comparison

The historical H75 results establish a placement-dependent reversal for one frozen actor. The new Isaac/Go2 runs test how a freshly trained actor performs under a different training protocol. Report variation across 7101–7103 using the same physical groups for each seed. A gap between H75 and a new actor combines several training differences and cannot be assigned to initialization alone.

At each of the three dedicated preferences, compare the dedicated actor with the shared actor initialized from the same seed (7101) and evaluated on the same groups. Report completion, final hold, A/B/C and common-window speed/path errors separately. If both policies have zero passes, report the observed errors and termination causes; the experiment supplies no equivalence margin. The three dedicated actors provide one observation per preference for this comparison.

## Shared seed 7101

Training reached update 150 in 13,780 s with 14,745,600 transitions. Use the final checkpoint for matched evaluation. The archived training scalars describe episodes collected during optimization. The final logged ended-episode failure statistic was 0.6303, speed RMSE was 0.2030 m/s and path RMSE was 0.5377 m. The three quantities use the training runner's per-step ended-episode averaging. [Training curves](training/shared7101.png) · [150 per-update records](training/shared7101.csv).

The final evaluation actor has SHA-256 `3852518acfff9a7f7aa401ba5e50b5fff71a433d389aa17c6ff86dcd88ede3cf`. Native and exported mean actions differed by at most 8.95 × 10⁻⁸ over 1,280 CPU inputs covering all five preferences. The fixed evaluation uses 40 layout–physical-group pairs and five preferences, with archived matched Zero and governor trajectories. Seed 7102 uses the same training settings with fresh initialization and a profiler on the first regular sampling step.


Seed 7101 completed 31/200 and met final hold in 0/200 on the matched development matrix. [All conditions, paired paths and termination reasons](training/shared7101/README.md).

The first regular policy sampling step of seed 7102 used the native Torch CUDA profiler. The PhysX triangle–triangle collision kernel accounted for 3.029 s and 93.35% of self CUDA time on that step. Aggregate-pair collision took 11.698 ms (0.36%). These are device times for one profiled step with 4096 environments. [Original profiler table](training/shared7102_first_step.txt).
