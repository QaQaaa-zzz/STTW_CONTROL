# P4 full-chain DEV diagnostic

[All18 actual/reference/predicted XY](xy.png).

|Method|True admit|False admit|False reject|True reject|
|---|---:|---:|---:|---:|
|C0_4300|4|1|0|13|
|C1_4225|4|1|0|13|
|C2_3525|0|1|4|13|
|F0_4425|2|1|2|13|
|F1_4375|4|1|0|13|
|F2_3550|0|1|4|13|

C propagates predicted states/history from real initial state; I uses route/speed-conditioned TRAIN interfaces. All future actual data are only offline labels. Same frozen skills, original task gates and raw envelopes; no new calibration or physical episodes. These are known DEV conditions, not independent TEST. Empty forbidden-box terrain still checks world bounds; absence of negatives cannot establish obstacle rejection quality. P4 remains incomplete.
