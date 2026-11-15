# P3 handoff interchange: decision diagnostic

[All12 actual/reference/predicted XY](xy.png). Source: `/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p3_world_handoffloss/continuation`. No new physical episodes; actual labels reuse the frozen map48 Oracle. Known DEV, not independent TEST or full-chain prediction.

|Input|Margin|True admit|False admit|False reject|True reject|
|---|---|---|---|---|---|
|measured|raw|4|0|0|8|
|measured|frozen_M1|2|0|2|8|
|seed0|raw|4|0|0|8|
|seed0|frozen_M1|0|0|4|8|
|seed1|raw|1|0|3|8|
|seed1|frozen_M1|0|0|4|8|
|seed2|raw|0|0|4|8|
|seed2|frozen_M1|0|0|4|8|

Each row is12 conditions:2 upstream approaches x2 roots x3 departures, with4 actual feasible and8 failed. Raw geometry is a diagnostic; frozen M1 margin is unchanged, including its false rejections. Neither is a safety certificate. `state_only` replaces physical/previous state and event time while keeping real history; `history_only` replaces controller history while keeping measured physical state/time. This is an offline error-isolation experiment, not a deployable use of true future state.
