# Offline entry interchange diagnostic

[All12 actual/reference/predicted XY](xy.png). Actual future entry is deliberately used only in these interventions; none is an online planning result. The six approach0 failures have no real RB entry and remain in the full18-case report, not silently dropped from its denominator.

|Entry intervention|Feasible admitted /4|Infeasible admitted /8|
|---|---:|---:|
|C|2|0|
|physical|0|0|
|history|2|0|
|both|4|0|

Physical-only replacement loses both baseline admissions; history-only retains two; jointly replacing physical state and history admits all four. Thus these observations do not isolate one scalar or one component as the sole cause. The pair matters; a history inferred from predicted motion can be inconsistent with independently substituted physical state. Predicted entry time is unchanged in every intervention. No training, controller updates, success-threshold change or new physical episodes.

This experiment used the pre-fix predictor trigger8.60m; source resolved execution uses8.55m. Retain the diagnosis as historical evidence; do not combine it with trigger-corrected scores. The trigger correction is evaluated separately before considering additional training.
