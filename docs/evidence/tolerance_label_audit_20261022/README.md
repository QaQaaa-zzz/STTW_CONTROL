# Empirical tolerance label audit

No new simulation or training. Frozen identities and real snapshots are preserved in each group.

- `pilot_labels.json`: twelve directional groups across six DEV states. One sampled transition (3 N pass / 6 N fail); eleven right-censored at 6 N. This does not imply a continuous safe interval or a 6 N true boundary.
- `boundary_labels_v2.json`: alpha0/t1 negative-direction transition 5 N pass / 5.5 N fail. Positive direction UNTESTED_DIRECTION: this local panel includes only zero force on that side.
- `boundary_labels.json` is superseded by v2: the initial exporter called zero-only evidence RIGHT_CENSORED at 0 N. An added failing test exposed this misleading category; it is now explicitly UNTESTED_DIRECTION. Original runtime data were not modified.

The common API preserves first failure and successes after it, stops the sampled success prefix at unknown outcomes, rejects duplicate amplitudes (repeated trials require their own protocol), and leaves continuous_margin_N null. Ten tests pass, including a nonmonotone synthetic success pattern and nominal/incomplete cases.

Training readiness: these labels are too sparse for a general state-to-tolerance regressor. Do not impute censored/untested cases as exact margins. Full 20-second probe rollouts remain an offline calibration tool, not an inexpensive per-step training feature. Right-turn transfer is running separately. Its result must guide the next calibration step; no reward changes, surrogate fit, switching, or deployment is performed by this exporter.

Reproduce with learning/cli/export_tolerance_labels.py --run COMPLETED_RUN --output NEW_JSON. The supplied run must be completed and have all declared rows. Each output binds endpoint, lower, physics, snapshot hash, route, requested speed, pulse and deadline; callers must compare this identity before joining runs.
