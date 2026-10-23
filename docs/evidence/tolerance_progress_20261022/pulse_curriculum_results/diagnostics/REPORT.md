# Practical recovery and own-nominal response

Posthoc practical tail sensitivity: path .15m, heading .05rad, speed .1m/s, roll .33rad; complete goal and no physical/domain failure. Same-time first2s response minus own nominal, shared valid-prefix complete contexts across ALL compared policies. Includes progress differences, not identical-state causal comparison. Used development panel; original labels unchanged.

| Alpha | Policy | Original | Practical | Shared response n | Roll delta RMS | Path delta RMS |
|---|---|---:|---:|---:|---:|---:|
| 0 | parent | 133/144 | 142/144 | 128 | 0.023289 | 0.047903 |
| 0 | uniform_update25 | 133/144 | 143/144 | 128 | 0.023694 | 0.050631 |
| 0 | guided_update25 | 133/144 | 134/144 | 128 | 0.024834 | 0.056314 |
| 1 | parent | 141/144 | 144/144 | 128 | 0.018795 | 0.050899 |
| 1 | uniform_update25 | 0/144 | 144/144 | 128 | 0.019225 | 0.049188 |
| 1 | guided_update25 | 72/144 | 144/144 | 128 | 0.019410 | 0.049534 |
