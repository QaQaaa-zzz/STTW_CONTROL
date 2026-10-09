# RTX 4090D V5.1 execution performance

Audit baseline: `071fc6f1366736ea1824389bda3686f58b1232ba`. Implementation base: `77f715f` (retains completed-run evaluation budget repair). Performance branch: `performance/r196-v51-4090d`. This is execution engineering, not new control evidence or formal training.

The original experiment completed both 100-update endpoints. Its source worktree and run artifacts are unchanged. Main checkout document edits are preserved. No DVGC/JIT files or existing training processes were changed.

## Protocol

`learning/cli/benchmark_smooth_performance.py` reuses SmoothCampaign, the physical environment, RSL storage/GAE and the existing PPO. Modes: env, rollout (no optimizer), PPO (frozen storage), evaluation, short end-to-end (both endpoint engineering batches plus checkpoints). Each measurement restores complete environment state, learner, storage, episode statistics and RNG. Two warmups, three samples, then a separate synchronized diagnostic repetition. Initialization/compilation, measured execution and diagnostic instrumentation are distinguished. Raw artifacts remain under ignored `runs/performance/`.

Defaults remain 512x128, independent endpoints, 100 updates, validation25/100. Dimension overrides are explicit engineering inputs. XML, physical and control intervals, lower identity, reward, safety boundaries, full-rollout KL and rollback contracts remain fixed. No dependency upgrade or device preallocation change is claimed.

## Measurements so far

RTX4090D, driver580.159.03, Torch CUDA12.8. Fresh upper endpoints, prepared bank SHA256 `6bb5fc54499ecc4451251bc984c1d4459a32977ba92f09fb53d3382c5bf535d0`. Detailed versions, other processes and sampled GPU power/clocks/memory are in each run's hardware JSON and gpu.csv.

| Workload | Raw hot seconds | Median seconds | Interpretation |
|---|---|---:|---|
| 8x16 env baseline | 5.131330,4.338621,4.378705 | 4.378705 | 512 active ticks every sample |
| 8x16 env resolved dimensions | 4.245542,4.248834,4.255774 | 4.248834 | correctness fix; no attributed kernel gain |
| 512x128 rollout baseline | 28.918925,27.907847,31.868537 | 28.918925 | original full logging |
| 512x128 summary plus batch reset guard | 27.455516,27.382838,27.380626 | 27.382838 | about1% vs summary median; not robustly established |
| 512x128 rollout device summary | 27.643317,32.112284,27.648829 | 27.648829 | overlapping ranges; gain not established |

Baseline synchronized diagnostic: environment31.303574s, policy0.103284s, handoff0.016885s, storage0.033019s, episode-statistics0.027593s, GAE0.005234s. These are a separate instrumented run, not additive decomposition of the uninstrumented samples. PPO-only fixed-rollout baseline samples0.098963/0.103479/0.100336s versus candidate0.071958/0.068820/0.069801s (medians0.100336/0.069801,30.4% reduction). PPO is a small fraction of total time. Full evaluation, end-to-end and environment-size sweep: pending/unmeasured.

## Equivalence and adoption gates

Original regression96 tests passed. Dimension-focused17 passed. Device statistics preserve pre-reset attribution, persistent incomplete episodes, every reset event and40 consecutive5ms ticks. Boundary tests preserve NumPy's original float64 comparisons of float32 -.15/-.30 values.

The first physical full-state comparison **failed** rtol2e-4/atol2e-5 (observed violation4.334e-5). User subsequently clarified that the old implementation is a comparator, not ground truth, and the tiny difference alone must not block optimization. Contract/decision checks and A/A diagnosis continue; batch reset guard synthetic all/partial/single/no-done tests4 passed; no tolerance relaxation and no adoption. `training_summary` remains opt-in; default `evaluation_full` retains original logging. Do not interpret pure statistic tests as physical equivalence.

Optimized HLO retains f32[8,3200] global projection operations from frozen_lower_controller.py. Its isolated wall cost is not yet established, so no geometry-cache or local-window change is enabled. Handoff timing currently does not justify the conditional second-stage JAX rollout rewrite.

No model, cache, large trace or repeated run artifact is committed. Local commit history and shared research ledger record implementation progress; no push has been performed for this task.

PPO validation:30 tests passed. On the identical serialized512x128 rollout, normal updates, nonzero temporal weight (accepted count30), injected NaN and forced hard-KL rejection all produced bitwise-equal gradients, policy, full Adam and all RNG states. Metric differences were0 in this fixture. Snapshot/restore and KL/finite decision frequencies are unchanged. Nonzero temporal dynamic indexing is retained; no unmeasured dense-mask gain is claimed.

Evaluation review: sequential evaluator recreates/lowers its executable each repetition; candidate caches same-shaped compiled reset/chunk functions with dynamic parameters. Report compilation/cache reuse separately from hot physical execution, as well as actual total evaluation wall time. The first sequential warmup saved all18 trajectories: physics819.897549s, file output0.260435s, report15.574993s (independent reward audit4.384817s, plotting10.931829s), total871.661110s. These are warmup observations, not three-sample hot results. All12 endpoint audits passed; a separate six-case B0 alpha0 audit also passed. Original-vs-replayed-baseline trajectories have identical failure endpoints and lengths, but small B0 floating differences and some clipping/endpoint-extension diagnostic flag differences; all-field bitwise identity is not claimed. Candidate review found no concrete physics/reset/mask regression; actual18-lane numerical comparison remains pending.

Telemetry boundary: gpu.csv records whole-device sampled memory, not process-owned peak. The original512 baseline file reached21468MiB and its before/after process snapshots list different PIDs; this cannot establish benchmark-owned memory usage or a memory reduction. Subsequent benchmark instrumentation also records per-process GPU memory every second and wall-clock sample intervals. Torch allocator peak excludes JAX. Sequential evaluation A/A itself shows small continuous and clipping/endpoint-extension diagnostic differences (maxXY6.87e-5m across18 trajectories); all safety/terminal/clock fields and active lengths match. This diagnoses a non-bitwise reference, without automatically accepting future candidate deviations.
