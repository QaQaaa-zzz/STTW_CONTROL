# Research design: preference-conditioned disturbance preparation

Updated 2026-10-30. Working research design for the active English, double-column, 12-page TCST manuscript objective. Proposed claims below are hypotheses until their listed evidence is complete. The user prioritizes one coherent contribution combining preference-conditioned single-track control and disturbance tolerance; three unrelated new algorithms are not required. RA-L is an alternative positioning, not a replacement of the current manuscript objective.

## Central argument and three contribution statements

**Central question:** How should a single-track robot temporarily execute a different tracking preference to prepare for uncertain disturbances, and subsequently restore its user's preference while completing the original task?

1. **Preference-conditioned control behavior:** A frozen middle policy exposes a usable path–speed tradeoff through a scalar preference. Characterize its nominal task performance and the resulting state/reference histories in low-speed turns. Evidence: five-alpha nominal and mixed U170 development panels. This is an application/method contribution; do not equate scalar conditioning itself with a new learning algorithm.
2. **Preparation-dependent closed-loop disturbance tolerance:** Evaluate candidate preferences jointly with the available preparation time and the specified continuation/return policy. The scientific object is the outcome of an evolving closed loop, rather than an instantaneous equilibrium margin. Evidence: matched-state preference/time branches; new experiments must separate changes in preparation from changes in where the gust meets the turn.
3. **Task-preserving preference coordination:** Select a preference using the predicted disturbance and original-task outcomes, then restore the requested preference using current feedback and geometric reconnection. Evidence: paired held/return outcomes plus matched basic baselines on a compact track panel. Continuous repeated neural selection is a proposed extension, not a current implemented result.

These are three connected components of one control contribution. Figures, numerical comparisons, and equations should develop this argument directly; detailed provenance remains in companion records rather than repeated qualifications in the abstract.

## Existing evidence and the next discriminating question

Start with actual/reference XY in the [published results](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/reports/sttw-tolerance-summary-20261029/docs/results/sttw-tolerance-20261029/EVIDENCE.md). The paired return demonstration and successful frozen-model cases remain the main examples; the left 24 N failure is a boundary case.

A read-only audit of `u170_preparation_time_coverage_v1/analysis/results.json` found a particularly useful mechanism question. For the same source `left90_R1p3_v2p0_source_t1p6_a0`, candidate alpha=1 survives and reaches the original goal with preparation 0.3 s, contacts with 0.4 s, and survives/reaches with 0.5 s. For source history a1, the same candidate survives all three, but misses the goal at 0.4 s. These are held-candidate development branches, not feedback-return validation. All 26 rows, including omitted reused grid entries, must remain distinguishable.

**Inference:** more preparation is not uniformly better in these records. Since the gust time changes with preparation, turn phase changes too. Do not present this as a monotone minimum preparation-time threshold or as an isolated causal effect of preparation. A stronger next experiment fixes the actual gust time/location and varies the decision start using saved complete states along the same nominal prefix. The online controller still receives no actual future gust timestamp; the timestamp belongs only to the paired experimental protocol.

## Minimal baseline and evidence plan

Use the same frozen U170, lower controller, original task, friction, actuator permissions, initial states, force schedules, and 10 Hz geometric reconnection for all methods.

- B0: execute the user's alpha throughout.
- B1: fixed alpha=0.5; retain the existing five-alpha sweep as context, not as a hindsight per-test oracle.
- B2: a simple predeclared curvature/speed rule with the same feedback return and candidate authority. Tune on development sources only; no gust-time input.
- Proposed: tolerance-guided preference choice plus the same return mechanism.
- Mechanism ablations: omit preparation-time input; omit the original-task prediction head. Run only if needed to establish contributions 2–3, using matched data and executable policies.

First reuse existing exactly matched baselines. Then freeze a small panel containing a left/right turn pair, an S connector, and a two-turn track segment. Preserve the previous 10 s episode contract for the first panel; constructing a whole-lap evaluation requires an explicitly recorded new horizon, rather than silently extending all old trials. Paths have a fixed global goal and declared corridor width; do not choose widths after inspecting tracking errors. Do not add repeated neural decisions until labels and evaluation cover their continuation policy.

Report actual/reference XY, contact, original-goal completion, path error, arrival time, preference deviation integral, restoration time, and actual speed. Methods need not win every case. No experimental run is launched by this design document; executable configs, source identities, episode budget, data roles, and stopping conditions are recorded before each new stage.

## Literature anchors checked 2026-10-30

- [TCST official scope and page requirements](https://www.ieeecss.org/publication/transactions-control-systems-technology): regular papers use double-column IEEE format, maximum 16 pages. The user's 12-page target is a manuscript requirement, not the journal maximum.
- [Li, Arslan, Atanasov: Fast and Safe Path-Following Control using a State-Dependent Directional Metric](https://arxiv.org/abs/2002.02038): abstract describes directional metrics and a virtual reference governor. This motivates distinguishing our preference-conditioned single-track behavior and preparation/return outcomes from generic reference governance. Full-method comparison remains to be completed.
- [CycleRL](https://arxiv.org/abs/2603.15013), published DOI [10.1109/LRA.2026.3704013](https://doi.org/10.1109/LRA.2026.3704013): abstract and publication metadata checked; robust learned bicycle control is a relevant contemporary comparator in related work. No unperformed performance comparison is implied.

## Manuscript assembly

Planned 12 pages including references: introduction/related work 2; system and task 1.5; preference/tolerance method 2.5; experimental protocol 1; simulation results 3; discussion/conclusion/references 2. These are editorial allocations, not blank padding. Write the canonical manuscript in English, with three contribution statements and results organized by the questions above. Keep one canonical manuscript and reproducible double-column export. Local Tectonic 0.17.0 is installed under `runs/paper_toolchain/tectonic`. An IEEEtran journal-class smoke document compiled successfully on 2026-10-30; the log identifies IEEEtran V1.8b. Archive source and SHA256 are in `runs/paper_toolchain/provenance.json`. The canonical Markdown is now fully English and exports to an IEEEtran double-column PDF through a generated LaTeX file. The current build has nine pages. Source evidence IDs remain in Markdown and the evidence manifest; the printed PDF uses numbered literature citations. The first two pages have been visually inspected, and the final build has no missing-character, undefined-reference, or overfull-box warnings. Remaining work includes complete page inspection, publication figure/table treatment, inline mathematical notation, and substantive evidence needed for the twelve-page target.

Scientific Agent Skills supported independent ideation, a separate adversarial read, and evidence-bound writing. Its reference is verified against the arXiv record: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner, *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, 2026, https://doi.org/10.48550/arXiv.2609.00065. Candidate suggestions and reviewer opinions are not experimental evidence.
