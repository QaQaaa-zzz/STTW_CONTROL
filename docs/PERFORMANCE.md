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

Baseline synchronized diagnostic: environment31.303574s, policy0.103284s, handoff0.016885s, storage0.033019s, episode-statistics0.027593s, GAE0.005234s. These are a separate instrumented run, not additive decomposition of the uninstrumented samples. Full evaluation, PPO A/B, end-to-end and environment-size sweep: pending/unmeasured.

## Equivalence and adoption gates

Original regression96 tests passed. Dimension-focused17 passed. Device statistics preserve pre-reset attribution, persistent incomplete episodes, every reset event and40 consecutive5ms ticks. Boundary tests preserve NumPy's original float64 comparisons of float32 -.15/-.30 values.

The first physical full-state comparison **failed** rtol2e-4/atol2e-5 (observed violation4.334e-5). Same-kernel A/A diagnosis pending; batch reset guard synthetic all/partial/single/no-done tests4 passed; no tolerance relaxation and no adoption. `training_summary` remains opt-in; default `evaluation_full` retains original logging. Do not interpret pure statistic tests as physical equivalence.

Optimized HLO retains f32[8,3200] global projection operations from frozen_lower_controller.py. Its isolated wall cost is not yet established, so no geometry-cache or local-window change is enabled. Handoff timing currently does not justify the conditional second-stage JAX rollout rewrite.

No model, cache, large trace or repeated run artifact is committed. Local commit history and shared research ledger record implementation progress; no push has been performed for this task.
