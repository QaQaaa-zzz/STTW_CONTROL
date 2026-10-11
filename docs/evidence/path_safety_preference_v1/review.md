# Final review through 4720f56 and scoped runner-cache diff

Reviewed committed integration `07760c6`, recovery `f889431`, reward `8bb1c40`, repeated-resume fix `4c5657d`, recovery provenance fix `4720f56`, and the small subsequent `arm_runners` cache-sharing diff against `2b0896a`. Read-only; no training, simulation, or GPU check launched. The recorded CPU tests and sole 8x16 discarded GPU check are coordinator evidence, not new reviewer executions.

**Final assessment: no unresolved blocker or concrete correctness finding in the reviewed scope.** Both follow-up findings below are resolved. The tiny runner-cache diff reuses environments by full config identity; source/physical identity is checked before reuse, all use terminal v2 and the same snapshot, and the shared compiled key includes config/terminal/shapes/horizon/capture mode. No conflicting captured environment or reward function is introduced. `git diff --check` passed during this scoped review. The coordinator's final full CPU test rerun was still running when this assessment was written.

## Follow-up findings resolved

1. **P1, FIXED in 4c5657d: second exact resume derived nonexistent historical checkpoint paths.** `rescore_saved_candidates` previously derived `parent/alphaN/checkpoints/update_XXXX.pt` from each validation directory although inherited DEV source receipts referenced the original immutable checkpoint. The reviewed follow-up now resolves and audits `source.json` checkpoint/actor SHA, source update, full identity, and serialized schema/identity. Missing source receipts fail closed for the safety schema; only historical protocols retain the legacy fallback. The coordinator added a second-inheritance regression and reports 24 focused CPU tests passing. Source fix checked by this reviewer.

2. **P2, FIXED in 4720f56: recovery CLI omitted requested-arm provenance binding.** `validate_recovery_arm` now checks control/candidate role and reward-version consistency, exact complete source identity, v2 in both manifest and plan, completed run and declared endpoint counts, and exact equality between selected and final source receipts. The CLI invokes it before any tails and additionally checks selected weight SHA and live physical identity. The reviewed CPU tests exercise swapped roles, changed bank source, both terminal fields, wrong final receipt, incomplete endpoints, alpha mismatch, and reward-version mismatch. The coordinator reports 15 focused recovery tests passing.

## Previously reported findings checked as fixed

- Cached reward component field names now match flattened `raw_cost_*`/`effective_cost_*` fields.
- First-resume DEV preservation is connected and the new safety pointer is used during rescoring.
- Reward-fork plans reject v1 termination, non-error HTTP policy, or changed coverage-monitor settings.
- Parent final panels are frozen by trace/path SHA and checked against B0 paths; reuse also checks the expected original geometry.
- Preference bins start at the declared 3.0 m turn boundary.
- Legacy serial evaluator keys include config and terminal identity.
- Safety reports link their predeclared preference-qualification result.

## Recovery review

Complete PathState leaves, shape/dtype/weak-type signatures, raw prefix, immutable route, source identity, checksums, original clock, and original horizon are serialized. Cloning changes only alpha. Natural capture searches actual complete upper-boundary states and saves the first qualifying boundary, with no artificial qpos change. The oracle requires a full 0.5 s low-roll hold completed within 1 s and preserves original goal/final hold, initial violation accounting, later overlimit rejection, and real termination. Per-arm rewards use the corresponding frozen config; identical actor+alpha tails are physically reused and rescored. Generated reports put equal-aspect original/actual XY first and preserve actual prefix/tail endpoints. No separate concrete defect found in these mechanisms during this review.

Formal training and natural recovery remain NOT_RUN. Source review and CPU/engineering evidence do not establish recovery ability or policy qualification.

## Coordinator final verification

Final all-path CPU suite:110 passed in13.92s (fresh after both fixes and shared runner cache). No additional GPU or physical execution. Current Coverage stays isolated; formal parents still require completed pipeline/final receipts.
