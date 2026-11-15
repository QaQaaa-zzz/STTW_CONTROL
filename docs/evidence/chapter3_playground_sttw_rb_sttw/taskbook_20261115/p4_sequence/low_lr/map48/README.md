# P4 full-chain DEV diagnostic

[All18 actual/reference/predicted XY](xy.png).

|Method|True admit|False admit|False reject|True reject|
|---|---:|---:|---:|---:|
|S0_3475|2|1|2|13|
|S0_3425|4|1|0|13|
|S1_2725|2|1|2|13|
|S1_3450|4|0|0|14|
|S2_2625|0|0|4|14|
|S2_3250|0|0|4|14|

Same frozen skills, original task gates and raw envelopes; no new calibration or physical episodes. C propagates predicted states/history from real initial state; I uses route/speed-conditioned TRAIN interfaces. These are known DEV conditions, not independent TEST. All future actual data are only offline labels. Empty forbidden-box terrain still checks world bounds; absence of negatives cannot establish obstacle rejection quality. P4 remains incomplete.
