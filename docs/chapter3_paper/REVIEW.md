# Independent agent review and paper closure

Three delegated agents examined (1) the implemented mathematical model, (2) the argument/evidence gap, and (3) outputs and visualization. The primary agent integrated their findings and inspected sources and rendered figures. This is an internal methodological review, not external peer review.

## Current-stage acceptance

The user explicitly accepts an operational minimal loop without outperforming a baseline. Existing trained-model predictions, online candidate selection, physical bypass/crossing and STTW continuation satisfy this engineering-stage criterion. No further superiority experiment is required for this delivery. Performance, robustness and causal-benefit studies below are optional future extensions, not blockers.

## Contribution that the current evidence supports

An event-level predictive planning interface composes frozen ground and jump controllers for continuous forward traversal of a known off-road map. It predicts intermediate physical consequences, ranks finite route/approach candidates, accounts for motion during inference, and executes selections through landing handoff to a common goal. The contribution is not a general-purpose world model, arbitrary route generation, or a demonstrated completion-rate improvement.

|Reviewer finding|Action completed|Evidence/status|
|---|---|---|
|World-model definition was underspecified|Added exact state/action/map dimensions, event table, architecture, residual distribution and loss|English Sections IV–V, checked against source|
|Training was described as an outcome rather than a reproducible method|Added whole-root split, labels, missing-event handling, PCA, prefixes, optimizer/budget/checkpoint rule, actual loss figure|Sections IV-D and VI|
|Actual model outputs were absent|Added all2666 DEV predicted/observed transitions, occurrence plots including failures, time/XY plots, velocity/attitude and online event means|chapter3_event_outputs|
|Model metrics used only two DEV roots|Evaluated both frozen models on240 saved out-of-training-root executions; no refit|chapter3_event_frozen_test; previous inspection of these TEST roots disclosed|
|Teacher-forced metrics could be confused with model rollouts|Separated conditional previous-event inputs from32-particle initial autoregressive forecasts|Sections IV-D, V-A, VI-B, VII|
|Three-head effect lacked a comparator|Shared-head event-conditioned ablation evaluated on identical inputs and TEST trajectories|Capacity35770/27710 disclosed; not a pure structure-only effect|
|Route-switch case could imply learned capability boundary|Explicitly attribute late bypass exclusion to geometric admission; retain failed middle case|Section VII online case|
|Recovery and mode claims exceeded code|Replace learned mode-transition/stabilization claims with prescribed event index and goal-continuation proxy|Sections III–V|
|Score could be called calibrated probability incorrectly|Document overlapping goal/road semantics and heuristic ranking interpretation|Ranking-score equation and Section V|
|Initial road flag in fitting was always false|Count original publication histories:0/1199 TRAIN/DEV,0/240 TEST|No observed mismatch in these frozen datasets|

Final review additionally requested explicit contact/failure predicates and a controller/α definition. Both were added to the manuscript and `IMPLEMENTATION_DETAILS.md`; the reviewer confirmed model equations, training, output figures and denominators against actual evidence.

## Optional future work beyond the completed minimal execution loop

1. **Prediction-to-decision benefit is not established.** Current online outcomes5/12 versus rule6/12 are valid closed-loop evidence; they do not demonstrate that intermediate predictions improve decisions. A counterfactual choice panel or matched ablation is needed only if the paper claims such a benefit. Do not tune the inspected TEST batch and continue calling it untouched TEST.
2. **The score and state are imperfect.** Explicit remaining horizon is omitted, and goal/road probabilities overlap. Correcting them requires a new version with new frozen model/evaluation identity; it must not silently replace the current reported method. The manuscript describes the implemented estimator precisely.
3. **No ordinary fixed-step dynamics or no-continuation ablation.** Shared-head still uses event labels. These missing comparators limit claims about event formulation and downstream recovery as causes of better decisions. They are not disguised as completed experiments.
4. **Limited scope and sample size.** Four TEST roots, one map, one optimizer seed, failure of right-angle routes, and no hardware validation preclude broad robustness claims. A follow-up should add grouped new-root/new-map testing if generalization is claimed, rather than repeatedly optimizing isolated failures.
5. **Publication preparation.** Author verification, project-method references, declarations and final venue/layout approval are pending. The eight-page source is a professional research draft, not a submission certificate or an assertion that RA-L must accept the contribution.

## Deferred extension; not a prerequisite for this delivery

Only if the user later requests performance improvement, first make the completion-score semantics and remaining-time input explicit, keeping the same map/skills/actions and a frozen new-root evaluation. Then compare a direct classifier, event model and an event-free transition alternative at the same observed-state inputs. Preserve the original method as control. No further bottom-level PPO campaign is required to expose this predictive-planning question.

Follow-up completed: saved TEST ranking diagnostic connects model scores to branch choices (three-head4/4, shared0/4, fixed detour4/4). Small candidate-state differences preclude a causal common-state claim; the need for matched online decision comparison remains. See `../evidence/chapter3_event_frozen_test/RANKING.md`.

Online head ablation now executed: three-head5/12, shared1/12, TRAIN rule8/12 across four previously inspected roots at three positions. This closes the absence of actual shared-head execution, not capacity control, untouched confirmation or superiority over the rule.
