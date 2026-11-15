# P4 full-chain DEV diagnostic

[All36 actual/reference/predicted XY](xy.png).

|Method|True admit|False admit|False reject|True reject|
|---|---:|---:|---:|---:|
|C0_4300|33|0|3|0|
|C1_4225|17|0|19|0|
|C2_3525|11|0|25|0|
|F0_4425|32|0|4|0|
|F1_4375|25|0|11|0|
|F2_3550|9|0|27|0|

C propagates predicted states/history from real initial state; I uses route/speed-conditioned TRAIN interfaces. All future actual data are only offline labels. Same frozen skills, original task gates and raw envelopes; no new calibration or physical episodes. These are known DEV conditions, not independent TEST. Empty forbidden-box terrain still checks world bounds; absence of negatives cannot establish obstacle rejection quality. P4 remains incomplete.
