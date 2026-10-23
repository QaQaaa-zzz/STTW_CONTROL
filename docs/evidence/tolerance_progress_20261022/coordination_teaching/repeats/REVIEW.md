# Optimization-seed repeat: numerical results complete

[All-alpha nominal XY](nominal_review/actual_xy.png)

| Seed | Moderate /200 | Strong /200 | Low-speed /60 | Target /8 | Unique nominal /30 | Nominal working /30 | Monotone turn speed preference /4 |
|---|---:|---:|---:|---:|---:|---:|---:|
|918501|200|195|58|6|30|24|4|
|918502|200|194|58|6|30|24|4|
|918503|200|195|58|6|30|24|4|

Parent:moderate200,strong189,low53,target1,nominal30,nominalworking30,preference4. All three taught actors have4 strong domain exits versus9 parent. Seed918502 has one paired task regression beyond the six common strong gains: tail path0.100544m (0.544mm above threshold), correct speed/heading/goal and no domain exit. Both new seeds retain the five low-speed gains with zero low-speed task losses.

The two new seeds completed940 fresh cases and1870528 active lower ticks. Both400-step teaching runs completed; actual TensorBoard fit scalars verified at http://localhost:6016. No new PPO, no reward or controller change, no seed selection. All numerical panels and per-panel figure reports are complete; status.json records reports_ready.

This establishes repeatability under minibatch-seed changes with the same parent, dataset and initial snapshot. It is not independent data/environment replication. Recovery benefit and work-roll cost both repeat. Do not pool940 cases as independent trials or report only the best seed.

Decision: freeze original918501 final400 as primary candidate, preserve918502/918503 as supporting repeats and parentu325 as fixed baseline. Do not tune further on these panels. Next declare new geometry/speed/interior-alpha/pulse-duration contexts, evaluate parent and all three candidates without further training, and report all cases. This will test prospective context transfer, not online alpha switching or reset/physics generalization. Tolerance shaping remains a separate negative/auxiliary result.
