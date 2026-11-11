# Chapter 3 English manuscript

**Current draft:** [PDF](MANUSCRIPT.pdf) · [editable LaTeX](MANUSCRIPT.tex). The main paper now describes the implemented two-interface model, rather than the earlier serial event-chain prototype. This is a research draft, not a submission-ready or superior-planning claim.

The scope is frozen STTW/jump composition on a known off-road map: query state and candidates → valid handoff/state distribution → conditional full-goal continuation → actual moving selection and execution. Ground routes have a separate output head sharing the upstream trunk. The extension adds task-conditioned downstream road choice at measured handoff. No empirical recoverability margin, unseen corridor-width generalization, or true-history warm start is claimed.

## Evidence and reading order

- [All 270 actual/reference XY trajectories and online results](../evidence/chapter3_interface_online/RESULTS.md): 30 independent roots, three repetitions; interface15/90, direct33/90, distance41/90. The 33 interface detours all time out.
- [Model inputs, outputs and training curves](../evidence/chapter3_interface_model/RESULTS.md): 582-dimensional input, conditional39-dimensional handoff descriptor, grouped1379TRAIN/540DEV executions. DEV completion Brier0.11520; conditional handoff position RMSE0.1317m.
- [Labels and all-route handoff evidence](../evidence/chapter3_interface_labels/RESULTS.md), [earlier continuation pilot](../evidence/chapter3_continuation_model/RESULTS.md), and [stage/init audit](../evidence/chapter3_handoff_audit/RESULTS.md) establish what the labels do and do not measure.
- [Review and remaining gaps](REVIEW.md), [controller/event definitions](IMPLEMENTATION_DETAILS.md), and [current research design](../CHAPTER3_EVENT_WORLD_MODEL.md).

The paper separates abstract, motivation, prior work, task, mathematical model, supervised training, candidate execution, experimental protocol, results, discussion and conclusion. Figures show actual handoff predictions, optimization curves and all online XY outcomes. Earlier event-chain manuscript versions remain in Git history; immutable experimental evidence remains linked from the source manifest.

## Build and traceability

```bash
python3 docs/chapter3_paper/build.py
```

The local IEEE two-column template is retained from the earlier draft; current venue submission requirements must be rechecked at submission. The build receipt records actual page count and layout checks. Figures are versioned; no simulation or weights are needed to compile. Source and claim manifests distinguish current evidence from historical evidence and keep human verification pending. The review-directed revision includes new physical development retests and supervised model refits; no new policy training.

## Author-review and tooling notes

Author identity, affiliations, funding/conflicts, formal citations for the project's STTW/RecoveryBridge methods, repository disclosure scope and publisher-required AI disclosure remain for accountable human review. No author, funding, ethics approval, submission date or acceptance claim has been fabricated. The methods are implemented simulator methods, not a hardware validation.

Scientific-writing and scientific-visualization skills guided the revision, and three delegated agents audited implementation, evaluated gaps and produced frozen prediction evidence. The manuscript text and analysis were AI-assisted and remain subject to author verification. Procedural skill attribution: T. Kassis, V. Agarwal, Y. He, D. Patel, and A. M. Brueckner, *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, 2026, [arXiv:2609.00065](https://arxiv.org/abs/2609.00065), [DOI](https://doi.org/10.48550/arXiv.2609.00065). Metadata was checked against the arXiv record; this is a tooling citation, not robotics evidence.

Current measured-state extension: 270 frozen executions, task-conditioned 13/90, state-only 15/90, fixed straight 14/90; one successful selected left continuation. Evidence I11 preserves unmatched incoming states and all failures. This isolated panel is downstream-only feedback; the joint panel below evaluates both interfaces.


## Review-directed revision

[Four requested corrections and prediction plots](../evidence/chapter3_review_repairs/RESULTS.md) · [shared timing-screen XY/results](../evidence/chapter3_horizon_screen/RESULTS.md) · [repaired joint XY/results](../evidence/chapter3_repaired_joint_fast/RESULTS.md).

Shared timing screening yields43/90 versus35/90 direct and36/90 distance. Compact residual prediction improves time/y errors but worsens velocity errors; Frenet Brier.10650 remains worse than state-only.10290. The repaired joint feedback completes26/90, open30/90, fixed27/90; 41revisions with7subsequent goals. All are development retests. The original30/28/28 panel remains archived and is not pooled. No new jump-policy training was performed. RB candidate evaluation is separate and does not change these source-skill results.

[Saved RB candidate integration screen](../evidence/chapter3_rb_candidate_handoff/RESULTS.md): Source11/90, RB30119/90, RB40013/90, RB50016/90. RB301 gains13/loses5; conditional exact-entry subset gains11/loses3. This is a separate preassigned-approach development screen, not a new joint-planner result. No model was deleted. Training did not save separate updates2–15; that requested sweep was canceled after identity clarification.

[RB301 plus feedback transfer](../evidence/chapter3_rb_feedback/RESULTS.md): Source-fixed15/90,RB301-fixed20/90,RB301-feedback16/90. Feedback vsRB-fixed gains8/loses12; all64 publications verified. Source-trained continuation model frozen; no new calibration or training.
