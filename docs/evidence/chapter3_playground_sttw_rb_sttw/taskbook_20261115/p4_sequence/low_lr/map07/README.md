# P4 full-chain DEV diagnostic

[All36 actual/reference/predicted XY](xy.png).

|Method|True admit|False admit|False reject|True reject|
|---|---:|---:|---:|---:|
|S0_3475|34|0|2|0|
|S0_3425|33|0|3|0|
|S1_2725|34|0|2|0|
|S1_3450|27|0|9|0|
|S2_2625|6|0|30|0|
|S2_3250|8|0|28|0|

Same frozen skills, original task gates and raw envelopes; no new calibration or physical episodes. C propagates predicted states/history from real initial state; I uses route/speed-conditioned TRAIN interfaces. These are known DEV conditions, not independent TEST. All future actual data are only offline labels. Empty forbidden-box terrain still checks world bounds; absence of negatives cannot establish obstacle rejection quality. P4 remains incomplete.
