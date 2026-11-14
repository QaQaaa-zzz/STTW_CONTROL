# Training convergence and checkpoint comparison

Descriptive training trends, not proof of convergence; training episodes and windows are correlated. KL is the logged first-rollout-step estimate, not full-batch KL.

| Updates | Mean step reward | Value loss | KL mean / max | Success / failure / timeout flags | Done |
|---|---:|---:|---:|---|---:|
|1–25|-0.011529|8.89505|1.72401 / 38.17600|[3.0, 25432.0, 0.0]|25435|
|26–50|0.023556|0.26234|0.00932 / 0.02078|[6.0, 24462.0, 0.0]|24468|
|51–75|0.025850|0.25582|0.00903 / 0.01679|[5.0, 24302.0, 0.0]|24307|
|76–100|0.027460|0.26439|0.00906 / 0.02244|[5.0, 24099.0, 0.0]|24104|
|101–125|0.028336|0.24832|0.01011 / 0.02049|[8.0, 23760.0, 0.0]|23768|
|126–150|0.029471|0.24235|0.00961 / 0.01651|[3.0, 23419.0, 0.0]|23422|

Last 50 updates reward slope: 4.055286333912988e-05 per update. Terminal flags may overlap; done is the union, not their sum.

The sampled training-reward best is not the fixed-route best. Reward is associated with the pre-update model. Do not compare rewards across reward configurations.

[All actual/reference XY](checkpoint_comparison_xy.png)

| Checkpoint label | Goal / evaluated | Per-route termination |
|---|---:|---|
|dynamics25|7/9|{'0': 'goal', '1': 'timeout', '2': 'goal', '3': 'goal', '4': 'goal', '5': 'corridor_or_domain_exit', '6': 'goal', '7': 'goal', '8': 'goal'}|
|dynamics50|6/9|{'0': 'goal', '1': 'timeout', '2': 'goal', '3': 'goal', '4': 'timeout', '5': 'timeout', '6': 'goal', '7': 'goal', '8': 'goal'}|
|dynamics75|6/9|{'0': 'goal', '1': 'physical_failure', '2': 'goal', '3': 'goal', '4': 'timeout', '5': 'goal', '6': 'goal', '7': 'timeout', '8': 'goal'}|
|dynamics100|6/9|{'0': 'goal', '1': 'timeout', '2': 'goal', '3': 'goal', '4': 'goal', '5': 'timeout', '6': 'goal', '7': 'corridor_or_domain_exit', '8': 'goal'}|
|dynamics125|5/9|{'0': 'goal', '1': 'timeout', '2': 'timeout', '3': 'goal', '4': 'goal', '5': 'timeout', '6': 'goal', '7': 'timeout', '8': 'goal'}|
|dynamics150|5/9|{'0': 'goal', '1': 'timeout', '2': 'goal', '3': 'goal', '4': 'timeout', '5': 'timeout', '6': 'goal', '7': 'corridor_or_domain_exit', '8': 'goal'}|

Nominal DEV candidate selection only. Every candidate uses all declared routes; per-route winners cannot be combined into a single successful controller.
