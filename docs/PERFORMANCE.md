# RTX 4090D V5.1 execution performance

Audit baseline: `071fc6f1366736ea1824389bda3686f58b1232ba`. Implementation base: `77f715f` (retains completed-run evaluation budget repair). Performance branch: `performance/r196-v51-4090d`. This is execution engineering, not new control evidence or formal training.

The original experiment completed both 100-update endpoints. Its source worktree and run artifacts are unchanged. Main checkout document edits are preserved. No DVGC/JIT files or existing training processes were changed.

## Protocol

`learning/cli/benchmark_smooth_performance.py` reuses SmoothCampaign, the physical environment, RSL storage/GAE and the existing PPO. Modes: env, rollout (no optimizer), PPO (frozen storage), evaluation, short end-to-end (both endpoint engineering batches plus checkpoints). Each measurement restores complete environment state, learner, storage, episode statistics and RNG. Two warmups, three samples, then a separate synchronized diagnostic repetition (evaluation already includes phase timing and skips this duplicate). Initialization/compilation, measured execution and diagnostic instrumentation are distinguished. Raw artifacts remain under ignored `runs/performance/`.

Defaults remain 512x128, independent endpoints, 100 updates, validation25/100. Dimension overrides are explicit engineering inputs. XML, physical and control intervals, lower identity, reward, safety boundaries, full-rollout KL and rollback contracts remain fixed. No dependency upgrade or device preallocation change is claimed.

## Completed measurements

RTX4090D, driver580.159.03, Torch CUDA12.8. Fresh upper endpoints, prepared bank SHA256 `6bb5fc54499ecc4451251bc984c1d4459a32977ba92f09fb53d3382c5bf535d0`. Detailed versions, other processes and sampled GPU power/clocks/memory are in each run's hardware JSON and gpu.csv.

| Workload | Raw hot seconds | Median seconds | Interpretation |
|---|---|---:|---|
| 8x16 env baseline | 5.131330,4.338621,4.378705 | 4.378705 | 512 active ticks every sample |
| 8x16 env resolved dimensions | 4.245542,4.248834,4.255774 | 4.248834 | correctness fix; no attributed kernel gain |
| 512x128 rollout baseline | 28.918925,27.907847,31.868537 | 28.918925 | original full logging |
| 512x128 summary plus batch reset guard | 27.455516,27.382838,27.380626 | 27.382838 | about1% vs summary median; not robustly established |
| 512x128 rollout device summary | 27.643317,32.112284,27.648829 | 27.648829 | overlapping ranges; gain not established |

Baseline synchronized diagnostic: environment31.303574s, policy0.103284s, handoff0.016885s, storage0.033019s, episode-statistics0.027593s, GAE0.005234s. These are a separate instrumented run, not additive decomposition of the uninstrumented samples. PPO-only fixed-rollout baseline samples0.098963/0.103479/0.100336s versus candidate0.071958/0.068820/0.069801s (medians0.100336/0.069801,30.4% reduction). PPO is a small fraction of total time. Full evaluation is complete (see delivery below). Environment-size sweep was cancelled at user request; end-to-end comparison is unmeasured.

## Equivalence and adoption gates

Original regression96 tests passed. Dimension-focused17 passed. Device statistics preserve pre-reset attribution, persistent incomplete episodes, every reset event and40 consecutive5ms ticks. Boundary tests preserve NumPy's original float64 comparisons of float32 -.15/-.30 values.

The first physical full-state comparison **failed** rtol2e-4/atol2e-5 (observed violation4.334e-5). User subsequently clarified that the old implementation is a comparator, not ground truth, and the tiny difference alone must not block optimization. Contract/decision checks and A/A diagnosis continue; batch reset guard synthetic all/partial/single/no-done tests4 passed; no tolerance relaxation and no adoption. `training_summary` remains opt-in; default `evaluation_full` retains original logging. Do not interpret pure statistic tests as physical equivalence.

Optimized HLO retains f32[8,3200] global projection operations from frozen_lower_controller.py. Its isolated wall cost is not yet established, so no geometry-cache or local-window change is enabled. Handoff timing currently does not justify the conditional second-stage JAX rollout rewrite.

No model, cache, large trace or repeated run artifact is committed. Local commit history and shared research ledger record implementation progress; no push has been performed for this task.

PPO validation:30 tests passed. On the identical serialized512x128 rollout, normal updates, nonzero temporal weight (accepted count30), injected NaN and forced hard-KL rejection all produced bitwise-equal gradients, policy, full Adam and all RNG states. Metric differences were0 in this fixture. Snapshot/restore and KL/finite decision frequencies are unchanged. Nonzero temporal dynamic indexing is retained; no unmeasured dense-mask gain is claimed.

Evaluation review: sequential evaluator recreates/lowers its executable each repetition; candidate caches same-shaped compiled reset/chunk functions with dynamic parameters. Report compilation/cache reuse separately from hot physical execution, as well as actual total evaluation wall time. The first sequential warmup saved all18 trajectories: physics819.897549s, file output0.260435s, report15.574993s (independent reward audit4.384817s, plotting10.931829s), total871.661110s. These are warmup observations, not three-sample hot results. All12 endpoint audits passed; a separate six-case B0 alpha0 audit also passed. Original-vs-replayed-baseline trajectories have identical failure endpoints and lengths, but small B0 floating differences and some clipping/endpoint-extension diagnostic flag differences; all-field bitwise identity is not claimed. Candidate review found no concrete physics/reset/mask regression; the completed18-lane comparison and its numerical differences are recorded below.

Telemetry boundary: gpu.csv records whole-device sampled memory, not process-owned peak. The original512 baseline file reached21468MiB and its before/after process snapshots list different PIDs; this cannot establish benchmark-owned memory usage or a memory reduction. Subsequent benchmark instrumentation also records per-process GPU memory every second and wall-clock sample intervals. Torch allocator peak excludes JAX. Sequential evaluation A/A itself shows small continuous and clipping/endpoint-extension diagnostic differences (maxXY6.87e-5m across18 trajectories); all safety/terminal/clock fields and active lengths match. This diagnoses a non-bitwise reference, without automatically accepting future candidate deviations.

Statistics correction: within-run conservation localized the family-summary discrepancy independently of physical A/B differences. Identical512-environment synthetic float32 statistics on4090D gave DEFAULT dot max relative error6.26136e-5 against CPU float64; HIGHEST dot1.63702e-7. A GPU regression failed before the localized precision fix and all4 device-statistics tests passed afterward. Only the family reporting reduction changes; physics/reward/Actor arithmetic are untouched. Before/after receipts: runs/performance/statistics_reduction_precision{,_after}.json.

Sequential evaluation measured samples are751.663091/752.835059/848.134005s, median752.835059s. Two warmups872.003837/773.724593s. The optional sixth diagnostic duplicate was interrupted after all five complete panels were saved, since each operation already records every evaluation phase. Original error/KeyboardInterrupt status is retained alongside measurement_completion_receipt.json; this is not reported as a normally completed sixth repetition. No original training/evaluation process was signaled. Candidate harness avoids that redundant sixth evaluation.

## Delivery and adoption — 2026-10-09

User requested fewer tests and immediate implementation progress. The optional queue and its N256 benchmark were interrupted after verifying their PID/cwd/command; no original training or other-project process was signaled. Receipt: `runs/performance/reduced_testing_receipt.json`. N256/512/1024/2048 throughput sweep, GPU nonzero-temporal dense-mask trial and further repetitions are **未测 / not pursued**. No formal training or budget change.

V5.1 now selects batched evaluation by default through the existing training entry. Explicit `evaluate_preference(..., batched=False)` retains sequential evaluation; the benchmark selects its mode explicitly. Other preference versions keep sequential evaluation. Resolved dimensions, PPO synchronization and trusted shared compilation cache are enabled. Device-summary logging and batch reset guard remain opt-in because their speed benefit is not robust and strict continuous-state comparison fails; they are not silently promoted. Geometry implementation and Torch/RSL rollout remain unchanged.

| Complete fixed update100 evaluation | Sequential | Batched |
|---|---:|---:|
| Hot sample seconds | 751.663091 / 752.835059 / 848.134005 | 239.625945 / 239.373127 / 238.758176 |
| Median total seconds | 752.835059 | 239.373127 |
| Median physics seconds | 729.868574 | 222.482510 |
| Median file output seconds | .264066 | .264337 |
| Median report seconds | 15.370926 | 16.615104 |
| Median numerical audit seconds (within report) | 4.362646 | 5.337096 |
| Median plotting seconds (within report) | 10.766887 | 10.994466 |
| Actual active5ms ticks, every sample | 39600 | 39600 |

Total reduction68.2%, speedup3.15x. Candidate includes six additional B0 audits inside each report; sequential B0 audits were checked separately. Sequential executable rebuilding and candidate cache reuse are part of the actual entry-point wall time. Baseline older phase instrumentation did not separately capture compilation; do not invent that split. Candidate initial evaluation compilation35.169936s; subsequent cached lookup is negligible. Preparation outside these measurements is recorded separately in timing.json. Both paths retain six cases,18 trajectories,10s main and fast_turn16s extension, actual5ms evidence and failure masking.

All18 reward reconstructions pass per candidate panel. Safety/terminal/clock arrays and active lengths agree with the reference comparison. **Strict full numeric equivalence does not pass**: first-panel maximum XY difference is0.0211668m in fast_turn alpha0, already present inside10s; maximum speed difference0.00947785m/s and tick reward difference0.000524583. Some clipping/projection diagnostic flags differ. Sequential A/A itself differs by up to6.87e-5m; this does not explain away the larger batch difference. The first B0 action matches exactly, while the first physical step differs at float32 precision before lower actions diverge. Batch numerical execution is a plausible source, not proof the original was wrong. The compared physical report conclusions do not flip. Adoption is an execution-engineering decision under the user's instruction to proceed despite small discrepancies, not a claim of bitwise equivalence or improved control.

Cache smoke: preparation49.566802s on first run and17.998257s on reuse; hot medians4.204583/4.181844s. This is startup-cache evidence, not a physical kernel speed claim. Explicit user cache settings are honored. Latest summary smoke completed all five restored-state trials and saved JAX/Torch traces; contract decisions agree while strict continuous-state tolerance fails (`full_state_equivalence.json`). Statistics-only precision was independently repaired; no physics arithmetic changed.

Evidence remains local and ignored under `runs/performance/`: `baseline_evaluation_100`, `batched_evaluation_100`, `evaluation_AB_first.json`, `evaluation_AB_fast_windows.json`, `baseline_evaluation_AA.json`, `ppo_equivalence.json`, `summary_verified_smoke`, `cache_first_env_smoke`, `cache_reuse_env_smoke`. Each benchmark retains raw timings, identity/config/state snapshots, hardware/telemetry and applicable traces. Allocator peaks are distinct from sampled whole-GPU usage; no end-to-end memory reduction is claimed. Prior broad regression131 passed; subsequent localized GPU statistics4 passed and default-evaluation helper3 passed. No repeated broad suite was launched after the user's scope reduction.

### XY evidence (all methods and original reference)

- [straight_hold,10s](../runs/performance/batched_evaluation_100/trial_003/evaluation100/straight_hold_10s_xy.png)
- [gentle_negative,10s](../runs/performance/batched_evaluation_100/trial_003/evaluation100/gentle_negative_10s_xy.png)
- [steer_reversal,10s](../runs/performance/batched_evaluation_100/trial_003/evaluation100/steer_reversal_10s_xy.png)
- [speed_changes,10s](../runs/performance/batched_evaluation_100/trial_003/evaluation100/speed_changes_10s_xy.png)
- [gentle_positive,10s](../runs/performance/batched_evaluation_100/trial_003/evaluation100/gentle_positive_10s_xy.png)
- [fast_turn,10s](../runs/performance/batched_evaluation_100/trial_003/evaluation100/fast_turn_10s_xy.png)
- [fast_turn,16s extension](../runs/performance/batched_evaluation_100/trial_003/evaluation100/fast_turn_16s_xy.png)

Final engineering check: `adopted_end_to_end_smoke` completed5/5 restored trials (2 warmups +3 samples), each both independent endpoints8x16 with PPO and checkpoint. Raw hot seconds: [9.786585035006283, 10.422119882001425, 10.246228732008603]; median10.246229s. This excludes full evaluation, which is measured above; no matched old end-to-end run was made, so end-to-end speedup is **未测**. Current endpoint reward scalars were verified through TensorBoard HTTP6015; receipt retained. No formal training continuation.

## Geometric path 执行接线（2026-10-10）

用户要求立即将剩余主线迁接已采用的优化。`path_command_evaluation.py`按路径数组形状分组vmap，用50个上层步的scan分块；保留每个有效5ms日志、域失败独立冻结与工程故障停止，不填充或改变路径。训练接入共享JAX编译缓存，跨端点复用评测编译函数，已完成端点跳过训练advance编译。既有PPO同步优化和实际512×128维度继续使用；未正式采用的device summary/conditional reset guard未启用。

有界实际检查：alpha0 task best100，两速度直线各1秒，任务horizon20秒保持；串行与批量共800 lower ticks。执行+传输26.93895→10.65899秒，短窗口2.527倍；编译20.60→36.07秒，不能外推为完整训练或完整六场景倍数。步数/终止完全一致，XY最大差异9.54e-7m，每步reward最大差异1.99e-7。逐字段rtol2e-4/atol2e-5及布尔严格比较**未全部通过**，保留执行器浮点差异与final_command_clipped标志差异，不宣称逐位或全程数值等价；原始逐字段结果见`runs/path_eval_optimization_check_20261010/summary.json`及配对NPZ。采用执行组织不改变奖励或物理公式，不重跑旧物理面板。

恢复保留原run、checkpoint及四条已完成final轨迹，记录`optimization_boundary_interrupt.json`。缓存检查验证六B0、四final、三次继承DEV，错误actor身份拒绝。继承数值证据放`inherited_history`，避免下次resume当成本地有checkpoint的候选。5项既有终止/排序/缓存检查通过；本次只扩必要检查。根AGENTS要求后续默认复用已采用且适用的优化。
