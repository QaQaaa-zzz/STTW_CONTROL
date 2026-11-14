# Training convergence and checkpoint comparison

Descriptive training trends, not proof of convergence; training episodes and windows are correlated. KL is the logged first-rollout-step estimate, not full-batch KL.

| Updates | Mean step reward | Value loss | KL mean / max | Success / failure / timeout flags | Done |
|---|---:|---:|---:|---|---:|
|26–50|-0.055030|34.40659|0.01635 / 0.06619|[1.0, 32359.0, 1.0]|32361|
|51–75|0.012552|0.38429|0.01006 / 0.03481|[4.0, 26890.0, 0.0]|26894|
|76–100|0.021919|0.37205|0.01366 / 0.03146|[6.0, 25237.0, 0.0]|25243|
|101–125|0.027790|0.30145|0.01162 / 0.02925|[8.0, 23844.0, 1.0]|23853|
|126–150|0.032176|0.26268|0.01023 / 0.02149|[13.0, 23183.0, 0.0]|23196|
|151–175|0.034123|0.24525|0.01282 / 0.03097|[18.0, 22789.0, 0.0]|22807|
|176–200|0.034700|0.22203|0.01010 / 0.02530|[5.0, 22697.0, 0.0]|22702|
|201–225|0.035738|0.21962|0.01146 / 0.03756|[13.0, 22627.0, 0.0]|22640|

Last 50 updates reward slope: 5.143622801799027e-05 per update. Terminal flags may overlap; done is the union, not their sum.

The sampled training-reward best is not the fixed-route best. Reward is associated with the pre-update model. Do not compare rewards across reward configurations.

[All actual/reference XY](/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/rb_adaptation/shared_plant_checkpoint_comparison/evaluation/xy.png)

| Checkpoint label | Goal / evaluated | Per-route termination |
|---|---:|---|
|shared35|0/9|{'0': 'physical_failure', '1': 'physical_failure', '2': 'physical_failure', '3': 'physical_failure', '4': 'physical_failure', '5': 'physical_failure', '6': 'physical_failure', '7': 'physical_failure', '8': 'physical_failure'}|
|shared75|0/9|{'0': 'physical_failure', '1': 'physical_failure', '2': 'corridor_or_domain_exit', '3': 'physical_failure', '4': 'physical_failure', '5': 'physical_failure', '6': 'corridor_or_domain_exit', '7': 'physical_failure', '8': 'physical_failure'}|
|shared115|4/9|{'0': 'goal', '1': 'timeout', '2': 'timeout', '3': 'goal', '4': 'timeout', '5': 'timeout', '6': 'goal', '7': 'timeout', '8': 'goal'}|
|shared155|3/9|{'0': 'goal', '1': 'timeout', '2': 'timeout', '3': 'corridor_or_domain_exit', '4': 'goal', '5': 'corridor_or_domain_exit', '6': 'goal', '7': 'timeout', '8': 'timeout'}|
|shared195|5/9|{'0': 'goal', '1': 'timeout', '2': 'timeout', '3': 'goal', '4': 'goal', '5': 'timeout', '6': 'goal', '7': 'timeout', '8': 'goal'}|
|shared225|6/9|{'0': 'goal', '1': 'timeout', '2': 'goal', '3': 'goal', '4': 'goal', '5': 'timeout', '6': 'goal', '7': 'timeout', '8': 'goal'}|

Nominal DEV candidate selection only. Every candidate uses all declared routes; per-route winners cannot be combined into a single successful controller.
