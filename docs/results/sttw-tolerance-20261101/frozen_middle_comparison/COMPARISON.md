# Four frozen middle policies on matched conflict routes

All policies use local200, wheel friction 0.8, 10 Hz reconnection and the same prepared banks1/3. U170 traces were already inspected when the relaxed thresholds were chosen; this is model-selection development evidence.

| model | alpha | original goal | H10 survival | narrow task | wide task | mean turn speed m/s | mean pre-goal maximum cross-track m |
|---|---:|---:|---:|---:|---:|---:|---:|
| U170 | 0 | 4/4 | 4/4 | 2/4 | 2/4 | 2.163 | 0.451 |
| U170 | 0.25 | 4/4 | 4/4 | 2/4 | 2/4 | 2.212 | 0.485 |
| U170 | 0.5 | 4/4 | 4/4 | 2/4 | 2/4 | 2.258 | 0.518 |
| U170 | 0.75 | 4/4 | 4/4 | 2/4 | 4/4 | 2.315 | 0.551 |
| U170 | 1 | 4/4 | 4/4 | 2/4 | 4/4 | 2.367 | 0.591 |
| H75 | 0 | 4/4 | 4/4 | 4/4 | 2/4 | 2.226 | 0.146 |
| H75 | 0.25 | 4/4 | 4/4 | 4/4 | 4/4 | 2.284 | 0.161 |
| H75 | 0.5 | 4/4 | 4/4 | 4/4 | 4/4 | 2.354 | 0.248 |
| H75 | 0.75 | 4/4 | 4/4 | 3/4 | 4/4 | 2.418 | 0.410 |
| H75 | 1 | 4/4 | 4/4 | 0/4 | 4/4 | 2.463 | 0.566 |
| W400 | 0 | 4/4 | 4/4 | 4/4 | 4/4 | 2.281 | 0.172 |
| W400 | 0.25 | 4/4 | 4/4 | 4/4 | 4/4 | 2.345 | 0.171 |
| W400 | 0.5 | 4/4 | 4/4 | 4/4 | 4/4 | 2.401 | 0.299 |
| W400 | 0.75 | 4/4 | 4/4 | 2/4 | 4/4 | 2.443 | 0.437 |
| W400 | 1 | 4/4 | 4/4 | 0/4 | 4/4 | 2.473 | 0.577 |
| u325 | 0 | 4/4 | 4/4 | 2/4 | 2/4 | 2.214 | 0.223 |
| u325 | 0.25 | 4/4 | 4/4 | 4/4 | 4/4 | 2.279 | 0.177 |
| u325 | 0.5 | 4/4 | 4/4 | 4/4 | 4/4 | 2.352 | 0.260 |
| u325 | 0.75 | 4/4 | 4/4 | 1/4 | 4/4 | 2.418 | 0.426 |
| u325 | 1 | 4/4 | 4/4 | 0/4 | 4/4 | 2.466 | 0.589 |

Task success uses the frozen relaxed buffered-anchor thresholds in [comparison.json](comparison.json). The saved source runs contain actual/reference XY panels and individual case outcomes. Prepared banks are paired initial states, not independent training seeds.
