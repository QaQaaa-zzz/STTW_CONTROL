# Pulse-induced response relative to each policy nominal trace

Posthoc diagnostic: same-time trajectory difference over the first 2 seconds from pulse onset. This includes changed path progress and closed-loop response; it is not an identical-state policy comparison. Common complete, valid-prefix contexts across the three trained policies only. No new simulation.

| Alpha | Policy | Matched n | Roll delta RMS (rad) | Path delta RMS (m) | Speed delta RMS (m/s) |
|---|---|---:|---:|---:|---:|
| 0 | baseline_update50 | 112 | 0.02331 | 0.05018 | 0.03650 |
| 0 | static_update50 | 112 | 0.02354 | 0.04892 | 0.03616 |
| 0 | dynamic_update50 | 112 | 0.02515 | 0.04964 | 0.03907 |
| 1 | baseline_update50 | 128 | 0.01879 | 0.05090 | 0.00659 |
| 1 | static_update50 | 128 | 0.01943 | 0.04854 | 0.00873 |
| 1 | dynamic_update50 | 128 | 0.01951 | 0.04952 | 0.00866 |

Alpha0 has 128 nonzero-force contexts per policy; 16 contexts excluded jointly because dynamic has invalid pre-pulse prefixes. All first-2-second windows are complete. Alpha1 uses all128. Exclusion is diagnostic only; full-task failure counts remain in the original report.

Conclusion: dynamic shaping has no overall pulse-response benefit on this panel; alpha0 roll response worsens about7.9%, alpha1 about3.8%. Static path response improves about2.5%/4.6%, with no consistent roll improvement. Do not interpret smaller nominal peak roll as improved disturbance rejection.

Decision: retain completed results and baseline models. Do not extend identical training. Next audit the reward mechanism before a new experiment: the implemented min(plus,minus) reserve penalizes absolute divergent coordinate |z| and can impose a nominal-turn penalty. This is an algebraic property; whether it caused the observed policy changes is still a hypothesis. Inspect nominal state/ESO-dependent reward activity, saturation and directional behavior, then consider a task-conditioned or nominal-relative shaping comparison. Do not claim the underlying formula is disproven by this reward experiment.
