# Minimum-loop draft completion audit

This audit covers the requested simulation-based minimum loop and English research draft. It does not approve journal submission or claim a superior planner. No further experiment is scheduled for this deliverable.

| Requirement | Inspected evidence | Finding |
|---|---|---|
| Preserve frozen driving/jumping and physical constraints | Three joint-panel configs; `ditch_ground_handoff.py`; full execution logs | H75/source jump retained; no PPO or physical/goal relaxation |
| Learned upstream state distribution, not only scenario selection | `interface_world_model.py`; I1 model fitting/output evidence | Valid-handoff probability and conditional 39D state/time distribution; actual supervised training documented |
| Downstream task-conditioned outcome | `handoff_model.py`, `handoff_task_context.py`; I10 | 58D state/geometry model, grouped 234 TRAIN/132 DEV labels; negative ablation retained |
| Integrate predicted states in decisions | Joint records and I12 `score_replay.json` | All 180 upstream score vectors and planned downstream routes reproduce exactly; route maximization follows expectation |
| Reconsider using measured state while moving | Joint handoff query/publication records | 44 revisions; 259 actual publications verified; no state reset at publication |
| Actual full-task positive simulations | I12 complete 270-episode records and XY | Feedback 30/90, open plan 28/90, fixed 28/90; eleven completions after revision and four successful non-straight final choices |
| Preserve failures and credible limits | I2, I7, I10, I11, I12 and main text | All denominators retained; no causal-rescue, superiority, equivalence, learned margin or map-generalization claim |
| Professional English manuscript with modeling/training/outputs | `MANUSCRIPT.tex`, PDF, claims/source manifests | Eight-page research draft: task, model, losses, planning, grouped experiments, figures, negative results and limitations |
| Reproducible artifact and integration checks | Build receipt, source files, raw-run paths, related tests | PDF compiles with four resolved figures and no overflow; ten relevant tests pass; source paths resolve |

## Boundaries that are not completed contributions

- The joint extension covers fifteen straight approaches and three downstream roads. Earlier 60-candidate S/90-degree/detour results remain separate and include substantial failure.
- All joint approach choices are 1.5 m/s, alpha 0. Available actions are not evidence of diverse state-dependent approach choices.
- Feedback versus open-plan mean difference is 0.0222 with root-bootstrap interval [-0.1111, 0.1556]. Only 1/89 paired handoff descriptors match exactly. Eleven goals after revision are not eleven rescued counterfactual failures.
- Empirical rescue margins, a trained short correction, true-history warm starts, new widths/maps and hardware execution are not demonstrated. They are not required to claim this bounded minimum executable loop and are not used to support it.
- LLM remains deferred; it is not the planner or a contribution of this draft.
- Authors, affiliations, funding/conflicts, human source/citation verification and submission approval remain pending. `submission_ready` stays false. Draft delivery does not imply publication readiness.

## Evidence entry points

- [Joint actual/reference XY and result table](../evidence/chapter3_joint_handoff_online/RESULTS.md)
- [Full joint records](../evidence/chapter3_joint_handoff_online/results.json)
- [Frozen protocol](../evidence/chapter3_joint_handoff_online/declaration.json)
- [Recorded-input score replay](../evidence/chapter3_joint_handoff_online/score_replay.json)
- [Model learning and physical predictions](../evidence/chapter3_interface_model/RESULTS.md)
- [Task-conditioned fitting](../evidence/chapter3_task_condition_model/RESULTS.md)
- [Matched historical task dependence](../evidence/chapter3_handoff_routes/RESULTS.md)

Historical reviews and superseded status statements are retained in Git history rather than mixed with this final audit.
