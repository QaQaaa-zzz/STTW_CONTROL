# Full-chain DEV spatial support calibration

[All18 mechanism-map actual/reference/I/C XY](../trigger_fix/xy.png) · [all72 calibration DEV XY](../dev_fullchain/README.md).

Calibration uses maps06/07 only, before applying margins to known DEV map48. Same existing per-region/per-axis maximum spatial support miss +0.02m for both models; no obstacle outcome used to tune margins. Fixed task, state/domain and goal-hold gates are unchanged.

|Predictor|Margin|True admit /4|False admit /14|False reject /4|True reject /14|
|---|---|---:|---:|---:|---:|
|C|raw|1|1|3|13|
|C|own|0|0|4|14|
|C|shared|0|0|4|14|
|C|cross|0|0|4|14|
|I|raw|4|1|0|13|
|I|own|0|0|4|14|
|I|shared|0|0|4|14|
|I|cross|0|0|4|14|

Every expanded system has no plan on either root. Removing false admissions by rejecting every candidate is not successful task planning or a safe fallback policy. No new physical execution.

|World x region (m)|C margin x/y/z (m)|I margin x/y/z (m)|
|---|---|---|
|-5.0..8.3|0.029/0.080/0.047|0.029/0.080/0.047|
|8.3..10.3|0.072/0.104/0.090|0.059/0.080/0.072|
|10.3..10.6|0.074/0.116/0.084|0.063/0.088/0.077|
|10.6..12.6|0.071/0.242/0.117|0.053/0.254/0.063|
|12.6..25.0|0.083/0.677/0.159|0.040/0.243/0.041|

C needs0.677m lateral expansion after x12.6; I needs0.243m. Shared takes the axiswise maximum; cross applies the other model’s margin. No uniform shrinking is adopted. Calibrating occupancy cannot repair a wrong center trajectory or a missing predicted goal hold.

Spatial supports are compared at equal world x after stable sorting, not equal time. This reuses the earlier spatial-rule convention and can hide timing differences; event errors are reported separately in the trigger-fix report. Full saved prediction horizon is used for calibration, whereas candidate checks stop at predicted goal or horizon. Outside-support actual frames are recorded instead of silently clipped. Sampled axis-aligned box supports and +2cm allowance are empirical, not a continuous collision/safety proof.

C: 72 whole trajectories, 0 actual samples outside predicted x support, 0 decreasing-x predicted steps (including independent interface resets).

I: 72 whole trajectories, 0 actual samples outside predicted x support, 48 decreasing-x predicted steps (including independent interface resets).

[Per-case/per-stage support errors](support_errors.csv) and [all geometric rejection witnesses](rejections.json) retain stage, model geometry index, obstacle/bound index, expanded clearance and applied margin. Native geometry ordering is the frozen terrain.mjb/BodyEnvelope order. P4 still requires the matched direct/residual comparison and a viable, validated decision system; calibration alone is not acceptance.
