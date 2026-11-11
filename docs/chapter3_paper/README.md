# Chapter 3 English manuscript

**Current paper:** [PDF](MANUSCRIPT.pdf) · [editable LaTeX](MANUSCRIPT.tex). This is the authoritative English research draft; `docs/CHAPTER3_MANUSCRIPT.md/.docx` retain the earlier Chinese working report.

The manuscript follows the official RA-L initial-submission two-column US Letter template. The downloaded `ieeeconf.cls` comes from the [template linked by IEEE RAS](https://ras.papercept.net/conferences/support/files/ieeeconf.zip); current guidance was opened on 2026-11-11 UTC at [RA-L Information for Authors](https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/). Formatting is a drafting choice, not a claim of acceptance or submission readiness. The current eight-page draft uses the published maximum rather than the six-page standard length; final venue choice and author approval remain outstanding.

## What is now in the paper

- Sections III–V: task/control interface, 30-D state and 11-D action encoding, terrain, six prescribed event transitions, three-head residual network, every output, masked training objective, training settings, particle rollout score and actual publication protocol.
- Sections VI–VII: grouped TRAIN/DEV/TEST roles, conditional versus autoregressive predictions, training-loss figure, predicted/observed state and time figures, frozen TEST evaluation, physical route coverage, online outcomes and costs.
- The main paper contains six figures: architecture, map, optimization, state outputs, event-time outputs, online forecasts/execution. Velocity/attitude and positive/negative probability plots remain in the [output evidence](../evidence/chapter3_event_outputs/README.md).
- [Claim and gap review](REVIEW.md) distinguishes the now-supported contribution from stronger claims that still need experiments.

[Operational event/controller definitions](IMPLEMENTATION_DETAILS.md) specify contact thresholds, flight debounce, handoff, failure limits and α semantics.

## Build

From this worktree root:

```bash
python3 docs/chapter3_paper/build.py
```

All included figure files are versioned under `docs/evidence/`; no simulator or model is needed to compile the draft. Raw physics and frozen models remain on the experiment disk. [Source manifest](source_manifest.json) records experiment inputs, citation URLs and verification status. [Claim map](claims.csv) binds central claims to source locations. Software checks are not accountable human verification; author verification is pending, and `submission_ready` is false.

## New evidence produced for this revision

- [World-model output figures and 2666 DEV predictions](../evidence/chapter3_event_outputs/README.md): architecture, six recorded loss pairs, occurrence probabilities, time/state outputs, root3 early/middle/late forecasts including failure.
- [Frozen event-model TEST evaluation](../evidence/chapter3_event_frozen_test/RESULTS.md): 240 executed candidates from four out-of-training roots, no refitting, no new physics. Main/shared autoregressive Brier 0.09570/0.11455; TRAIN-rate baseline0.12160. Both models' handoff occurrence errors are worse than a nearly certain constant.
- [Previous online execution](../evidence/chapter3_offroad_distance_online/RESULTS.md): 36 actual trajectories, event/direct/rule5/12,5/12,6/12. No new online benefit claim is introduced by the frozen prediction evaluation.

## Author-review and tooling notes

Author identity, affiliations, funding/conflicts, formal citations for the project's STTW/RecoveryBridge methods, repository disclosure scope and publisher-required AI disclosure remain for accountable human review. No author, funding, ethics approval, submission date or acceptance claim has been fabricated. The methods are implemented simulator methods, not a hardware validation.

Scientific-writing and scientific-visualization skills guided the revision, and three delegated agents audited implementation, evaluated gaps and produced frozen prediction evidence. The manuscript text and analysis were AI-assisted and remain subject to author verification. Procedural skill attribution: T. Kassis, V. Agarwal, Y. He, D. Patel, and A. M. Brueckner, *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, 2026, [arXiv:2609.00065](https://arxiv.org/abs/2609.00065), [DOI](https://doi.org/10.48550/arXiv.2609.00065). Metadata was checked against the arXiv record; this is a tooling citation, not robotics evidence.

Supplement: [frozen candidate rankings and all selected XY trajectories](../evidence/chapter3_event_frozen_test/RANKING.md). Three-head/shared-head select 4/4 versus 0/4 successful stored branches; fixed detour also 4/4. Candidate publication states differ, so this is not an exact common-state or new online comparison.

[Actual shared-head online ablation](../evidence/chapter3_shared_head_online/RESULTS.md): 36 Isaac trajectories; three-head/shared-head/rule5/12,1/12,8/12. Distinct from the earlier direct-classifier comparison; no training.
