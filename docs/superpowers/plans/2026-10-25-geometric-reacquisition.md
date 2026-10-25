# Geometric reacquisition bounded screen implementation plan

> For agentic workers: execute inline using executing-plans in the existing isolated worktree. Preserve unrelated edits.

Goal: test2 existing healthy t10 states under3 same-authority geometric/zero reference rules, without resetting clocks or reapplying force.

Architecture: pure point selector returns a reference offset and telemetry. A wrapper turns that current offset into a constant5-knot input to the existing full-sequence single physical step. A dedicated suffix runner loads COMPLETE states, advances task2000→4400 with zero force and original floors, and appends saved source prefix arrays only for22s analysis. Original10s results remain immutable.

Tech stack: JAX/NumPy/Flax, existing contact-only frozen local300 lower kernel, vmap6/scan50, existing compilation cache.

Constraints: original world path/request/statealpha; dynamic reference corrections[−1,+.25]m/s and±.2rad via original mapping/filter/rates; floor source1.5 or1.0, not a new actuator bound; no old Actor query; 12s suffix/22s total;6 actual suffixes/≤14400 lower ticks/0optimizer; contact/fault/horizon stop; no second pulse or path reanchoring; original prefix only copied, not simulated.

- [x] Task1. Write failing tests for geometric_target(path,progress,pose,speed,nominal,mode,config): nearest forward point, shared bounded heading fallback, exact zero mode, numerical fault and legal map clip. Implement geometric_reacquisition.py. Test retained complete prefix clock in join_prefix_suffix and reject reset/missing clocks before implementation.
- [x] Task2. geometry_reacquisition_run.py prepare(): authenticate2 extracted carries, raw prefixes/initial XY, config/path/source andlocal300 Actor/checkpoint/config; runtime cfg differs only episode_s22. Seal6 fixed cases and all current source snapshots. execute(): batch6 actual carries, maintainphysical2700+ andtask2000+, mode/floor arrays, explicitzero pulse; scan50 for12blocks, save suffix-only arrays and unmodified-common-field concatenated full episode. Counter counts only new suffix ticks. shape-only traces50×6×4 without physics.
- [x] Task3. Make existing paired_changes accept an explicit baseline role (defaultparent preserved), use live_zero for this plan. Optional plot handover XY/time from actual saved state, without inferring positions. NewCLI phases prepare/shape/execute/analyze, Bark-only status for each. Test baseline matching before change.
- [ ] Task4. Run relevant CPU tests, prepare6 and actual full-state eval_shape; review primitive authority, model/source identities and no clock reset/no force construction. Commit only named code/docs, preregister ledger and Bark. Then execute and verify exactPID/session. Next regular check+900s.
- [ ] Task5. At completion analyze6 full22s episodes, original prefix10s labels separate, allXY/control/cost traces. Report fixed versus adaptive paired gains, contact, goal/hold, whole and suffix costs, intervention dose and clipping. Count two related DEV states, not independent starts. No tuningL/.18 or training if no useful opportunity; if fixed coversgain keep as common baseline, not novelty.

Point selector details locked before physics: L=[current clipped original lookahead,2,4,6,10,16,24]m, adaptive picks smallestL with body_x>original preview_body_x_min_m. Fixed chooses6m if its point is forward. Both use desiredsteer=.18*sign(wrap(original path heading−actualyaw)) when their point set has none. All targets convert desired absolute steer minus original nominal steer to the SAME offset interface. Zero mode returns exactzero. No roll/reference setpoint or original nominal publisher is changed. A forward point is not a physically feasible trajectory certificate.

Verification2026-10-25: current10 focused CPU tests passed1.08s; earlier57 related checks recorded in handoff. Actual6-carry GPU eval_shape completed with50×6×4 logs, taskticks all2000, physicalticks all2700, explicit6×5 zero pulses,0integration/optimizer. Preparation succeeded before context continuation; preserved rather than repeated. Physical execution and Task5 remain pending.
