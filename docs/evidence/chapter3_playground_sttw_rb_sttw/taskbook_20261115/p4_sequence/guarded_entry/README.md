# P4 corrected-gate entry intervention

All actual/reference/predicted XY: [map06,36 conditions](map06/README.md), [map07,36](map07/README.md), [map48 I,18](map48_I/README.md), [map48 true-entry,12](map48_true_entry/README.md). The six mechanism candidates failing before RB are excluded only from the true-entry intervention because no real entry exists; they remain in the original18-case C/I results. Unsupported max-horizon departure extensions in plots are diagnostic, not plans.

The original RB and displacement S1/2550 were frozen. All predictions use the same corrected state-event guard as the previous C query. True-entry substitutes measured physical and full controller/RB history only at RB entry, retains the predicted approach clock, and is an **offline future-state intervention**, not deployable planning. I uses coherent TRAIN-selected typical entry/departure states, not measured future test states.

|Model|True-entry events /84 eligible|I events /90|Previously measured causal C events /90|
|---|---:|---:|---:|
|Original RB|0|0|0|
|Displacement S1/2550|0|0|3|

All new queries admit zero candidates. Thus the strict event failure persists even when upstream physical/history estimation errors are removed; RB modeling is independently insufficient. This does not prove upstream prediction is harmless or that replacing it can never help. The three causal C events sharing a prefix are not evidence that estimated entries are generally superior.

Twelve GPU queries, zero new training or physical rollouts, known DEV only. Old ungated I successes do not transfer to this corrected event protocol. P4 remains incomplete. Evidence supports stopping further bridge adjustments and addressing the independently inaccurate RB model. The separately declared low-learning-rate continuation changes only optimization rate, not architecture/loss/data/task; its results are not part of this diagnostic report.

[Per-query counts and timings](summary.json), [frozen contract](contract.json). Commands and full arrays: `STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p4_guarded_entry`. Actual trajectories are reused; repeated prefixes are not independent samples.
