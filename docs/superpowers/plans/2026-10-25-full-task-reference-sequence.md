# Full-task reference sequence implementation plan

> For agentic workers: execute inline using executing-plans; preserve existing isolated worktree and unrelated changes.

Goal: build and run the bounded pure-task opportunity test in FULL_TASK_SEQUENCE_OPPORTUNITY.md; no new learned model or safety claim.

Architecture: target generator selects two legal correction targets from actual path progress. It calls the existing make_live_reference_step once per upper interval, retaining all physical/internal states. Existing Actor runtime stays unchanged. Sequence and constant controls share full-horizon reference rights.

Tech stack: Python, NumPy, JAX/Flax, existing frozen MuJoCo/Warp lower kernel.

Global constraints: local300/ECBC/ESO frozen; original path/request/user alpha; 5ms/20ms; 10s; 32 random six-parameter sequences seed20261025 + zero + four constants; alpha0/.5; maximum74 actual episodes/148000 lower ticks; 0optimizer; unchanged filter/slew/limits; contact/fault/horizon stop; no future pulse parameters to target generator.

- [x] Task1. Create learning/tests/test_full_task_reference_sequence.py testing exact knot/end targets, smooth interpolation, asymmetric map roundtrip, full-domain legal reference vs physical stop/zero, and 37 reproducible bounded candidates. Run CPU red before implementation. Implement full_task_reference_sequence.py; CPU green.
- [x] Task2. Add prepare_full_task_sequence and CLI: authenticate completed80 source, exact physical core/initial/path/model/cache identities; write37 parameter bundles,74 cases,10 inherited nominal baselines, candidate array and source snapshot. Test case composition and constant rights before implementing. Never treat inherited rows as new physics.
- [x] Task3. Extend explicit runner stage sequence_nominal/runtime_kind reference_sequence; single common vmap/scan50 and live kernel. Keep old stage semantics and raw fields intact; add target/sequence validity receipts. Add CLI stage, colors by family in analyzer without omitting cases.
- [ ] Task4. Run related CPU tests; actual eval_shape groups with full initial state,0physics. Before physical launch audit source/config/limits and duplicate intended/mapped candidates; record declarations/actual number in shared ledger and Bark registration. Commit only designated code/docs. GPU process same existing cache/preallocatefalse,15min monitoring. No automatic retry or library expansion.
- [ ] Task5. After all74 complete, analyze every actual/inherited trajectory, original-alpha costs and full endpoints; show XY and failure cases; compare task_only64 and matched constant/zero. Mark only actual direct/reused visual review scope. Update ledger with benefit/negative/unknown and next decision; RA-L goal stays active until method and independent evidence established.

Pre-execution self-review: explicit initial source is completed anticipatory_reference_pilot_v2 (same one-context DEV). First iteration does not deduplicate acrossalpha;74 separately scored episodes remove unverified alpha-independence assumptions. Four constants fixed before collection: speed targets−0.5/+0.24m/s with steer0; steering−0.08/+0.08rad with speed0; smooth first/last knot0, three identical interior values. Full random sequence internal values use original legal scale ranges[-1,.25]m/s and±.2rad, no outcome-based resizing. Baselines inheritparent/task_only/physical/floor各2alpha+ESOalpha0+uniquezero=10; these are context-specific comparisons, not10 independent starts. No separate Gaussian/search optimization or eventwise teacher selection.

Execution checkpoint:49 relatedCPU tests passed2.23s; new target/protocol tests first failed on absent modules. Prepared source roles yield10 inherited traces, not11 (ESO onlyalpha0). All complete initial source lane comparisons passed. Actual GPU-environment eval_shape traced batches16/16/16/16/10 at50×4,0physics. Fixed inherited progress/reference interface replay retains37 distinct profiles, no candidate rejected or re-sampled; this is not candidate physical performance. Task4 physical launch andTask5 analysis pending.
