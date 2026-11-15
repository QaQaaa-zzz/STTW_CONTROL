# P4 full-chain DEV diagnostic

[All36 actual/reference/predicted XY](xy.png).

|Method|True admit|False admit|False reject|True reject|
|---|---:|---:|---:|---:|
|C0_4300|33|0|3|0|
|C1_4225|21|0|15|0|
|C2_3525|6|0|30|0|
|F0_4425|33|0|3|0|
|F1_4375|24|0|12|0|
|F2_3550|15|0|21|0|

C propagates predicted states/history from real initial state; I uses route/speed-conditioned TRAIN interfaces. All future actual data are only offline labels. Same frozen skills, original task gates and raw envelopes; no new calibration or physical episodes. These are known DEV conditions, not independent TEST. Empty forbidden-box terrain still checks world bounds; absence of negatives cannot establish obstacle rejection quality. P4 remains incomplete.
