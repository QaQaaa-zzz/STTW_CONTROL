# Chapter 3 English manuscript

**Current draft:** [PDF](MANUSCRIPT.pdf) · [editable LaTeX](MANUSCRIPT.tex). The main paper now describes the implemented two-interface model, rather than the earlier serial event-chain prototype. This is a research draft, not a submission-ready or superior-planning claim.

The scope is frozen STTW/jump composition on a known off-road map: query state and candidates → valid handoff/state distribution → conditional full-goal continuation → actual moving selection and execution. Ground routes have a separate output head sharing the upstream trunk. No empirical recoverability margin, new downstream corridor generalization, true-history warm start, or repeated feedback replanning is claimed.

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

The local IEEE two-column template is retained from the earlier draft; current venue submission requirements must be rechecked at submission. The build receipt records actual page count and layout checks. Figures are versioned; no simulation or weights are needed to compile. Source and claim manifests distinguish current evidence from historical evidence and keep human verification pending. No new physics or policy training was performed for this manuscript revision.

## Author-review and tooling notes

Author identity, affiliations, funding/conflicts, formal citations for the project's STTW/RecoveryBridge methods, repository disclosure scope and publisher-required AI disclosure remain for accountable human review. No author, funding, ethics approval, submission date or acceptance claim has been fabricated. The methods are implemented simulator methods, not a hardware validation.

Scientific-writing and scientific-visualization skills guided the revision, and three delegated agents audited implementation, evaluated gaps and produced frozen prediction evidence. The manuscript text and analysis were AI-assisted and remain subject to author verification. Procedural skill attribution: T. Kassis, V. Agarwal, Y. He, D. Patel, and A. M. Brueckner, *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, 2026, [arXiv:2609.00065](https://arxiv.org/abs/2609.00065), [DOI](https://doi.org/10.48550/arXiv.2609.00065). Metadata was checked against the arXiv record; this is a tooling citation, not robotics evidence.
