# P4 full-chain DEV diagnostic

[All36 actual/reference/predicted XY](xy.png).

|Method|True admit|False admit|False reject|True reject|
|---|---:|---:|---:|---:|
|C original|0|0|36|0|
|I original|0|0|36|0|
|True original|0|0|36|0|
|C displacement|0|0|36|0|
|I displacement|0|0|36|0|
|True displacement|0|0|36|0|

Same frozen skills, original task gates and raw envelopes; no new calibration or physical episodes. C propagates predicted states/history from real initial state; I uses route/speed-conditioned TRAIN interfaces. These are known DEV conditions, not independent TEST. All future actual data are only offline labels. Empty forbidden-box terrain still checks world bounds; absence of negatives cannot establish obstacle rejection quality. P4 remains incomplete.
