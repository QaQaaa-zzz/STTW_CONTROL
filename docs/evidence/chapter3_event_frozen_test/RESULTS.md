# Frozen event-model prediction on held-out roots

Physical trajectory evidence: [same executed TEST cases](../chapter3_offroad_new_roots/RESULTS.md). This supplement reuses saved trajectories; no new physics or policy training was run.

## Protocol

The frozen three-head event model and shared-head ablation were evaluated on 240 actually published source-policy segments from four TEST initial-condition roots on the same map. These roots were previously inspected for other comparisons: this is a frozen out-of-training-root evaluation, not newly collected or previously unexamined confirmation data. Both checkpoints were selected at update 600 using DEV loss. The TRAIN-only PCA codec is identical (hash recorded in JSON). No parameters, PCA components, normalizers, thresholds, or checkpoint selection were updated.

Event metrics use the observed preceding event state (teacher forcing). State/time errors include only positive target events. Approach and ground-continuation prefixes every 0.5 s increase record counts but are correlated within trajectories; the independent-root count remains four. Initial-state completion scores use 32 autoregressive particles for the actually executed candidate, not future measured states. They do not constitute a new online planning comparison.

## Conditional event prediction

|Target|Records / hits|Three-head Brier|Shared Brier|TRAIN-rate Brier|Time MAE, three/shared (s)|XY error, three/shared (m)|
|---|---:|---:|---:|---:|---:|---:|
|Jump entry|1133 / 768|0.04798|0.04821|0.22776|0.1776 / 0.1448|0.4829 / 0.4915|
|Confirmed flight|93 / 89|0.01743|0.01621|0.04122|0.0675 / 0.0805|0.1404 / 0.1956|
|Far-bank contact|89 / 79|0.04385|0.04590|0.09998|0.0402 / 0.0694|0.1042 / 0.1777|
|STTW handoff|79 / 79|0.03824|0.03515|0.00001|0.0342 / 0.0238|0.1520 / 0.1793|
|Successful goal continuation|940 / 331|0.08880|0.10268|0.23453|0.2912 / 0.3517|0.7839 / 1.8112|
|Auxiliary clearance|93 / 67|0.18480|0.19137|0.20324|0.0704 / 0.0902|0.1960 / 0.2478|

## Initial autoregressive completion prediction

|Model|Parameters|Completion Brier|
|---|---:|---:|
|Three-head event model|35770|0.09570|
|Shared-head event-conditioned ablation|27710|0.11455|
|TRAIN completion-rate constant|—|0.12160|

Actual completed candidates: 34/240. The three-head model has lower aggregate completion Brier and smaller far-contact time/position errors than the shared-head model on this frozen set. This is a descriptive comparison over four roots, not a significance or map-generalization claim. Both models underperform the near-certain event-rate baseline for handoff occurrence (all 79 observed far-contact cases hand off), and shared-head flight Brier is slightly better. The comparison does not isolate parameter count: the three-head model is larger. Shared-head retains the event index and event targets; it is not a conventional unconditioned state-transition model.

## Checks and reproducibility

The prior-road flag was read at each original publication boundary: 0/240 TEST segments had prior violations. A separate check of all 1,199 source TRAIN/DEV segments also found 0 prior violations, so the fitting code's false initial flag does not mismatch these datasets. Models and codecs are identified by SHA-256 in the JSON files. Per-root metrics and all 240 completion predictions are retained. Event count/label construction follows the fitting entry point; frozen normalization is reused.

```bash
PYTHONPATH=learning/src /home/qy/miniconda3/envs/env_isaaclab/bin/python learning/cli/evaluate_ditch_event_model.py \
  --model /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_offroad_distance_training/model \
  --collection /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_offroad_new_roots/source/segments \
  --output docs/evidence/chapter3_event_frozen_test/event.json
```

For the ablation, replace `--model` with `/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_shared_event_head/model` and output with `shared_head.json`. No simulator, GPU, optimizer, or data-refitting step is invoked.
