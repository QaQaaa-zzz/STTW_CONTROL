# Local boundary verification

Original prediction: 4.721686 N. No training. Same complete state, held-out amplitudes only.

| force N | predicted pass | actual pass | peak roll rad |
|---|---|---|---|
| -4.0 | True | True | 0.291745 |
| -4.5 | True | True | 0.295691 |
| -5.0 | False | True | 0.299756 |
| -5.5 | False | False | 0.303587 |

Small probe predictions: [{"probe_N": 0.25, "secant_bound_N": 4.6027445793151855}, {"probe_N": 0.5, "secant_bound_N": 4.638138771057129}]

![XY and roll](xy_and_roll.png)

No continuous boundary or cross-scene guarantee. Review summary.json for baseline replay differences.