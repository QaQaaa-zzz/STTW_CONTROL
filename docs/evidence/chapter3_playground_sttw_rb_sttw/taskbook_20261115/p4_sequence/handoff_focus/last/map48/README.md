# P4 full-chain DEV diagnostic

[All18 actual/reference/predicted XY](xy.png).

|Method|True admit|False admit|False reject|True reject|
|---|---:|---:|---:|---:|
|C0_4475|2|1|2|13|
|C1_4450|4|0|0|14|
|C2_3625|4|0|0|14|
|F0_4475|0|0|4|14|
|F1_4450|0|0|4|14|
|F2_3625|1|0|3|14|

C propagates predicted states/history from real initial state; I uses route/speed-conditioned TRAIN interfaces. All future actual data are only offline labels. Same frozen skills, original task gates and raw envelopes; no new calibration or physical episodes. These are known DEV conditions, not independent TEST. Empty forbidden-box terrain still checks world bounds; absence of negatives cannot establish obstacle rejection quality. P4 remains incomplete.
