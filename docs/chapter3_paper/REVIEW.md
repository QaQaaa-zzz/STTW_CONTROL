# Current execution-outline audit

**当前验收。** T1–T4 已执行，空间转移与偏好消融均已写入八页英文稿。原始独立盲测、公开任务的配对复验与三种新墙布局分别报告。[证据与章节映射](README.md)。

The current manuscript follows the supplied Chapter3 execution outline. The older two-interface critique and its numerical evidence are retained below as historical work, not current acceptance criteria.

| Required stage | Current evidence | Remaining boundary |
|---|---|---|
| P0 shared physical execution and STTW/RB ownership | Shared adapter and complete chains; same physical plant, measured relock, retained histories | Validated domain only; no hardware claim |
| P1 ground dynamics and path update | Calibrated dynamic36/36 versus geometric31/36 on12 DEV maps | No independent new-map claim; calibrated-system comparison |
| P2 full-chain planning | Greedy3/12, backtracking12/12, joint12/12 | No independent success benefit of joint optimization |
| P2 rolling publication |200ms queries,40ms charged delay, preserved histories; physical12/12 in each of two panels | State coverage11/12; one actual replan followed by periodic checks; full coverage unresolved |
| P3 command domain |1.8 and1.9m/s each72/72,1.7m/s65/72 | Both ordered1.8/1.9 phase pairs also72/72; only change at measured RB handback, arbitrary ground switching remains unvalidated |
| P3 structured model comparison | Same-data ARX plus M1/M2 with terrain/history; three initializations each;600 updates per model completed; 25 evaluated candidate checkpoints per seed | All best at600 and still improving; M1 1s mean0.0194–0.0200m but worst whole-body support0.190–0.252m; two-speed follow-up108TRAIN/36DEV yields M1 mean0.0182–0.0192m; multi-map/planning validation remains |
| P4 independent comparison | Original50new-layout x4initial-state requirement retained | Not performed; existing DEV panels cannot substitute |
| Manuscript and traceability | Editable main LaTeX/PDF and evidence/claim registries, old prose retained inGit | Development draft, human verification and final literature/contribution work pending |

Neither physical survival, lower prediction mean, nor a weak comparator establishes the full research contribution. Failures, same-identity trajectory reuse, and different horizon coverage remain explicit. Prior flat-only prediction losses and lack of joint advantage against backtracking are retained.

## Historical review-directed audit

# Review-directed completion audit

This audit concerns the user's four concrete corrections and the latest request to test saved RB candidates. Expected numerical gains in the critique are hypotheses, not acceptance labels. Original experiments remain in Git and raw run directories.

| Requirement | Current evidence | Assessment |
|---|---|---|
| Shared remaining-route/time admission | I14; `route_horizon.py`; query and publication checks | Implemented. Development retest43/90 versus35/90 direct and36/90 distance. A nominal requested-speed screen, not a physical infeasibility bound. |
| Compact action-conditioned residual handoff model | I13; TRAIN-only PCA16,55D input,8D action skip, speed/time baseline | Implemented and refitted on unchanged split. Time RMSE.462→.128s and lateral-position.209→.172m; forward-velocity.450→.511m/s worsens. The proposed universal prediction improvement is not supported. |
| Route-relative continuation features | I13;19D Frenet geometry/dynamics and0.4s ballistic offsets | Implemented and refitted. DEV Brier.10827→.10650 remains worse than state-only.10290. Historical source7 still ranks straight first; its measured margins do not support the critique's assumed boundary crossing. |
| State-dependent execution and measured-state revision | I15;270 physical trajectories,261 verified publications | Approach now varies over1.5/1.8/2.0m/s but89/90 choices use alpha1. Feedback26/90, open30/90, fixed27/90;41revisions and7subsequent goals. No demonstrated feedback advantage. |
| Analyze RB's delivered state distribution | I13 historical legal-entry audit |71exact recorded entries,10gains10losses forRB400; lower lateral/roll-rate magnitudes do not imply a higher learned continuation score. |
| Test existing RB candidates in the actual chain | I16;360 physical trajectories | Source11/90,RB30119/90,RB40013/90,RB50016/90. RB301 gains13/loses5, with11/3 among60exact recorded entries. Development screening only. |
| Combine RB with measured-state route revision | `STTW_runs/chapter3_rb_feedback/declaration.json` | Completed270episodes: Source-fixed15,RB301-fixed20,RB301-feedback16 of90; 8gains/12losses against RB-fixed. All64 publications verified. This is frozen predictor transfer, not RB-specific calibration. No result claimed until terminal records are analyzed. |
| English manuscript describing model, fitting and outputs | MANUSCRIPT.tex/PDF; source and claim manifests | Eight pages,three figures,no overfull boxes. Physical errors, paired losses and development reuse reported. Human authorship/citation/submission checks remain pending. |

The three phases remain ground preparation, frozen jump execution, and H75 continuation. The current work does not add jump training, an LLM, or a new architecture beyond the requested corrections. Missing RB2–15 weights are not recreated by training. No model has been deleted.

The requested corrections, saved-candidate tests, and RB feedback composition have now been executed and documented. The transferred source-trained predictor is not calibrated to RB. Prediction and feedback advantages unsupported by these experiments remain negative findings, not silently accepted promises from the critique. No further checkpoint or threshold tuning is scheduled for this revision.

[Review corrections and prediction plots](../evidence/chapter3_review_repairs/RESULTS.md) · [all repaired joint XY](../evidence/chapter3_repaired_joint_fast/RESULTS.md) · [all RB candidate XY](../evidence/chapter3_rb_candidate_handoff/RESULTS.md).

PG9 adds eight predeclared terrain maps (286/288 physical goals), six TRAIN/two DEV maps for matched M0/M1/M2 fitting, and full observed departure prediction on all 72 DEV conditions. Selected checkpoints include intermediate updates575/550; formal convergence and executed model-planner benefit remain unproven. Evidence: [multi-map report](../evidence/chapter3_playground_sttw_rb_sttw/p3_multimap_models/RESULTS.md) and its paired complete-departure XY figures. No independent P4 claim.

PG10 adds a static same-candidate model-planning DEV pilot: M0 10/12, M1/M2 12/12 with identical selections,24 unique physics runs. This avoids one no-plan fallback but does not show fewer false-feasible published plans or a structural advantage over M2. Measured-state integration and independent E1–E3 remain required.

PG11 adds the canonical approach subset (72TRAIN/24DEV) with full observed3.56–4.10s predictions. The three illustrative XY panels use fixed map/root/speed and all three approach directions; full24conditions remain linked. PG10 now explicitly records the missing predicted terminal-hold gate; physical success is retained without calling it full dynamic admissibility.

Current PG11 includes600→900 continuation and retained intermediate best checkpoints; PG12 adds explicit predicted terminal/domain screening with only2new physics runs. PG13 retains the rolling coverage limitation after108new TRAIN histories: physical12/12, coverage11/12 with a different failure. Independent P4 E1–E3 remains pending; no full-coverage or structural-dominance claim.


PG14已加入：50新布局×4根，M0 156/200，M1/M2各200/200；收益来自减少无计划，非假可行下降或M1胜过M2。完整逐例/XY见p4_model_test；静态组件TEST与既有DEV分开，E1/E2和滚动验收仍待完成，作者核验待办。
