# P4 departure-input interchange

Actual/reference/predicted XY, with every eligible condition shown: S0/3475 [map06](seed0_update3475/map06/xy.png), [map07](seed0_update3475/map07/xy.png), [map48](seed0_update3475/map48/xy.png); S1/3450 [map06](seed1_update3450/map06/xy.png), [map07](seed1_update3450/map07/xy.png), [map48](seed1_update3450/map48/xy.png).

**RB exit physical-state error dominates the current downstream center-error diagnosis; replacing reset history alone does little.** True physical inputs reduce average spatial lateral RMSE from about8cm to3cm. Event timing explains much of the longitudinal exit error, but does not remove the lateral error. No new model or planner is adopted. P4 remains incomplete.

## Protocol and results

Freeze both previous candidate models and the same ground predictor. Replace only (a) physical qpos/qvel and previous20ms physical velocity, (b) controller history, or (c) both with the actual first phase2 record. Retain predicted approach/RB trace, handoff event and clock, domain/geometry/goal gates. Actual history is the recorded event-before-control interface. A physical-only swap deliberately leaves history based on the predicted state; the combination is an offline attribution intervention, not an executable state reset.

Twenty batched GPU queries completed in65.43s under the shared job lock:18 interventions plus2 baseline-parity queries. Zero new physics episodes, optimizer updates or control changes. Each method has72 original DEV conditions and12 mechanism conditions with actual handoff. The6 mechanism approach failures have no such interface and remain in the original18-case causal result; neither their failures nor S0's original false admission is erased.

For original DEV maps06/07, error summaries use unchanged predicted-event support: S0 has72/72 and S1 has54/72. All supported cases have spatial overlap. Rows outside support are still retained in [cases.csv](cases.csv), with missing spatial overlap explicitly null rather than zero.

|Model|Input intervention|Mean per-case spatial y RMSE(m)|Worst spatial y error(m)|Mean relative-time XY RMSE(m)|Raw true admit /72|
|---|---|---:|---:|---:|---:|
|seed0_update3475|baseline|0.0794|0.1945|0.2256|68|
|seed0_update3475|physical|0.0314|0.1411|0.0501|72|
|seed0_update3475|history|0.0742|0.1991|0.2189|67|
|seed0_update3475|both|0.0282|0.1528|0.0414|72|
|seed1_update3450|baseline|0.0821|0.3112|0.2935|53|
|seed1_update3450|physical|0.0304|0.1164|0.0477|54|
|seed1_update3450|history|0.0803|0.3127|0.2857|48|
|seed1_update3450|both|0.0278|0.1331|0.0393|54|

Spatial metrics align predicted y by actual world x over the recorded phase2 trajectory; sorting, uncovered samples and nonmonotonic steps are recorded. Relative-time metrics reset diagnostic time to each departure interface, so they intentionally isolate downstream dynamics rather than global handoff timing. All actual traces stop at their real failure/goal. No smoothing, goal redefinition or failed-case deletion is applied.

On mechanism map48, S0 has6/12 supported events and admits2/4 actual successes in every intervention; S1 has12/12 and admits4/4 in every intervention, with0/8 false admissions. These are raw, uncalibrated **offline** interventions. Unchanged gates prevent true-state replacement from laundering missing predicted handoffs into valid plans. No improved calibrated task-completion result is claimed.

## Event time versus physical-state error

Using the saved full125-step RB trace, compare its event-selected state with the nearest20ms sample at the actual handoff's absolute time. This additional lookup reads the real future event time and is only an attribution diagnostic; it is not a proposed stopping rule. Selection remains inside the saved horizon for all supported cases below. It can use trace samples beyond the model's selected event, which are not a committed execution plan.

|Model / maps|Supported n|Handoff time MAE(s)|Event x MAE(m)|At real-time-grid x MAE(m)|Event y MAE(m)|At real-time-grid y MAE(m)|
|---|---:|---:|---:|---:|---:|---:|
|seed0_update3475 / map06,map07|72|0.1049|0.1884|0.0221|0.0582|0.0581|
|seed0_update3475 / map48|6|0.2125|0.3804|0.0302|0.0362|0.0489|
|seed1_update3450 / map06,map07|54|0.1515|0.2754|0.0245|0.0722|0.0492|
|seed1_update3450 / map48|12|0.1513|0.2785|0.0108|0.0312|0.0256|

This distinguishes two remaining problems: event timing and lateral state prediction. A ground-only retraining campaign is not justified by these results; first improve and evaluate RB event timing together with lateral/yaw exit state, preserving the state-consistency guard and existing calibration. Merely supplying actual time or shrinking margins is not a fix. Native stage3/4 execution history is still not fully reconstructed by the model and remains a documented limitation.

## Verification and scope

Default-query arrays exactly match saved baseline for both36-case map06 batches. Across the three interventions, approach/RB arrays are identical. Map48 uses12 eligible cases versus the original18-case inference batch: numerical differences are recorded (largest qpos trace difference4.58e-5, history-coordinate difference0.00238); event indices are unchanged. This is not claimed bitwise parity across different batch shapes. Physical/history substitutions independently match the recorded inputs, including previous velocity; ten existing skill behavior tests passed. [Detailed input checks](intervention_verification.json), [default parity](parity.json).

The analyzer initially encountered an unsupported trajectory with no spatial overlap; it now retains null metrics and uncovered counts. This required no GPU rerun; the error record is retained. [Run commands and timing](status.json), [contract](contract.json), [per-case metrics](cases.csv), [event-time samples](event_time_diagnostic.csv), [summary](summary.json). All maps are known DEV, with shared prefixes; none is independent TEST and method-condition counts are not independent physical rollouts.

Sources: `STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p4_departure_interchange`. Reproduce queries with `evaluate_skill_sequence.py --diagnostic-departure physical|history|both`; analyze with `analyze_skill_departure_interchange.py --run ... --baseline .../taskbook_20261115_p4_low_lr --output ...`; render with the existing `report_skill_sequence.py`. The committed status lists exact per-query arguments. Images are provisional research diagnostics, not a journal-format submission.
