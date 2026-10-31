# Research design: preference-conditioned disturbance preparation

Updated 2026-10-30. Working research design for the active English, double-column, 12-page TCST manuscript objective. Proposed claims below are hypotheses until their listed evidence is complete. The user prioritizes one coherent contribution combining preference-conditioned single-track control and disturbance tolerance; three unrelated new algorithms are not required. RA-L is an alternative positioning, not a replacement of the current manuscript objective.

## Current-method reconciliation — 2026-10-31

The existing manuscript and its12-page PDF describe the earlier U170/local300 continuation-conditioned return43 experiment. They are a historical manuscript snapshot, not a report of the newly improved lower controller and online pilot. The following evidence map governs the next substantive revision; current decisions remain in the shared project ledger.

| Element | Existing manuscript | Current frozen experiment | Revision evidence |
|---|---|---|---|
| Lower controller | ECBC+ESO+local300 | ECBC+ESO+STTW_LOCAL_DOMAIN_U200 | E286 |
| Predictor |43 inputs, four outcomes, ELU |41 inputs,16–16 ReLU hidden layers, contact-avoidance output | E287–E289,E292 |
| Supervised experiment |Specified feedback-return continuation through task end |Held candidate; preparation0.2/0.6s, hypothetical pulse0.15s, observe through pulse end+3s | E287–E288 |
| Fitting |Three members,5000 updates each |Three fixed-last members,1000 updates each; BCE plus paired ranking weight0.5 | E287,E292 |
| Deployment |One source-boundary query then prescribed return |Repeated50Hz candidate queries,0.2s conditional dwell,10Hz geometric reconnection | E286 |
| Task treatment |Learned goal/path/speed outputs |Nonlearned curvature/reference checks plus original-task evaluation | E286 |
| Current effect |Old low-speed return comparisons |Original DEV gains, prospective24 null result, and one low-speed recovery gain under common4m reconnection | E290–E291,E294,E297–E299 |

The new contact predictor evaluates a held-candidate counterfactual, not the full future sequence of online switches. Repeated querying is part of the controller and must be evaluated in closed loop. Its binary target is neither full-task goal completion nor a certified force threshold. The old return43 target, architecture figure, equations and successful24N examples cannot be relabeled as current41-feature evidence.

### Three connected contribution targets for the revised study

1. **Preference-dependent control and disturbance response across speed regimes.** Use the frozen scalar-conditioned middle policy and the improved lower controller to characterize executable path–speed choices. Bind nominal task and same-complete-state disturbance comparisons to one physical/controller identity. Application evidence supports the single-track setting; conditioning on alpha itself is not claimed as a new learning algorithm.
2. **Preparation-conditioned counterfactual tolerance prediction.** Predict contact avoidance after candidate commitment, preparation and a specified hypothetical disturbance. The expanded-state fit improves Brier0.562549→0.065211 on the unchanged160-row DEV set (E289). DEV has no discordant-alpha groups, so this result supports outcome discrimination rather than independent candidate-ranking accuracy. The matched speed-state information ablation is complete(E293): full-state DEV Brier0.065211 versus0.071052, while AUC0.911 versus0.912545 does not favor full state. Preparation-information ablation is now complete(E302): removing preparation increases DEV ensemble Brier to0.075847 and reduces AUC to0.907455; one of three seeds improves without it. Ten preparation-discordant DEV pairs come from two source-direction groups; full predicts five correctly and ties five. This is limited positive information evidence, not strong independent ranking evidence.
3. **Task-constrained repeated preference execution.** Combine prediction with geometric eligibility, user-priority restoration and conditional dwell. The risk-disabled24-case ablation removes both observed outcome gains (E291), supporting the incremental contribution of risk qualification inside this controller. The first frozen prospective24 comparison has no paired outcome advantage(E294). A subsequent common4m low-speed engineering comparison yields one additional arrival and survivor versus geometry-only(E297); its77.35% pre-goal user0 occupancy and pre-force intervention are measured(E298–E299). The adjacent/mirror24-case campaign is complete(E303): fullq/noq/fixed0 each4/6arrivals andsurvivors; fixed1 has4arrivals/5survivors. The isolated low-speed gain does not transfer to this panel. Broader reliability and restoration qualification remain explicit evidence questions.

These are contribution targets, not three established originality claims. The current minimal comparisons include fixed user preference and the same geometric coordinator without risk qualification. The larger completed dataset and new lower-controller evidence must not be pooled with old local300 results as if all controllers shared one identity.

### Evidence-to-contribution audit — 2026-10-31

| Connected contribution | Current positive evidence | Discriminating result or missing evidence | Next decision |
|---|---|---|---|
| Preference-conditioned closed-loop capability | Five-alpha response differs in the36N low-turn envelope; improved lower command tracking and nominal/domain results exist | Fixed-from-start alpha differences do not establish same-state switching benefit;±48N low-turn cases all fail(E295) | Describe measured speed/turn/force scope and preserve the common physics identity |
| Preparation-conditioned tolerance estimation | Frozen960-label dataset and retained DEV score; old-DEV risk-enabled control gains | Full-state information only modestly improves Brier over speed-state; near-fall probabilities extrapolate beyond observed TRAIN feature ranges(E293,E300); preparation ablation yields a modest ensemble advantage and mixed per-seed results(E302) | Evaluate prediction and online control separately; add disturbed source coverage if frozen validation warrants it |
| Temporary preference execution with task reconnection | Under common4m permission, full coordinator survives and reaches the low-speed original goal at8.565s; geometry/fixed0 fail, fixed1 reaches9.605s(E297–E299) | One post-hoc recovery condition; previous new-scene panel has zero advantage(E294). Length permission itself contributes to task completion(E296) | Finish frozen adjacent/mirror check before choosing a final controller and revising the manuscript |

The current paper-level narrative is a capability–prediction–execution chain. Preparation and restoration are temporal control decisions; their claimed value must be separated from simply fixing alpha1, changing speed, or allowing a longer reconnection path. Common planning permissions make the control comparison interpretable. The data support a useful working example and a concrete operating boundary. Completed adjacent/mirror validation(E303) directs the next improvement toward disturbed-state prediction and restoration qualification rather than another threshold sweep. No three-component novelty claim follows solely from naming these modules.

E293–E300 bind both positive and negative current results. The canonical manuscript/PDF remains the historical local300/return43 study until one final method identity and its evidence are chosen. This internal design audit uses the scientific-writing evidence workflow locally; human source verification remains unrecorded.

### Recovery qualification: mechanism review — 2026-10-31

Focal question: does false restoration arise from missing disturbed-state training, a mismatched prediction target, or insufficient executable recovery authority? The present review uses scientific-brainstorming locally, with a separate non-originator AI critique. The candidates were generated before the literature check; the accountable user has authorized research iteration but has not verified these hypotheses. Simulation only; no unpublished data were sent to external services.

Decision criteria, in order: distinguish the observed failure mechanism; reuse complete-state evidence at low simulation cost; produce an actionable preference decision; preserve broad task performance. No numeric novelty scores were assigned. Candidate A is matched-protocol disturbed-state coverage and calibration of the current single predictor. Candidate B separates contact-free continuation without another gust from preparation for another gust, initially through the same conditional predictor with d=0. Candidate C improves the lower controller if no candidate can recover the relevant state. A new second network is not currently justified.

The non-originator critique identifies two confounds: the360new branches still add a gust, and their fixed-source histories differ from the actual online restoration states. High predicted probability before a fall does not prove that retaining the previous alpha would avoid that fall. Disconfirming observations are concrete: if better held-label calibration leaves online false qualifications unchanged, coverage alone is insufficient; if zero-gust and perturbed queries induce the same eligibility, separate recovery qualification adds no control information; if all five held candidates fail without another gust, scheduling at that state lacks demonstrated recovery authority. An equal-data single conditional network is the first alternative to any two-head proposal.

The bounded primary-source check found existing task/recovery policy separation in [Recovery RL](https://arxiv.org/abs/2010.15920), and state-dependent acceptance/modification of proposed actions in [predictive safety filtering](https://arxiv.org/abs/1812.05506). Abstracts and metadata were checked; a complete method comparison was not performed. These findings challenge treating a recovery gate alone as original. The proposed scientific focus remains the measured coupling between preference execution, preparation and restoration in single-track dynamics. The search does not establish a new research gap.

Immediate decision: finish the360matched-protocol branch collection and audit the frozen predictor before fitting or changing online thresholds. Reuse the original fixed trajectories for their own-alpha, zero-additional-gust continuation labels. A later, separately declared small branch test at actual online restoration states can determine whether returning versus retaining alpha changes the outcome; it is not launched by this review. No result is inferred for a second network or new controller.

### Measured recovery release result — 2026-10-31

A same-condition six-method consolidation now binds the basic baselines to this result: fixedalpha0, fixedalpha1, geometry-only, originalshortdwell and fixedcommitment all complete4/6; measuredrecovery completes5/6 with the originalfour retained. Fixedalpha1 andcommitment survive5/6, confirming that the incremental result is taskcompletion after survival. Full36raw-record independent checks and six all-method XY panels are in `u170_lower200_recovery_release_online_v1/baseline_comparison`. This comparison does not substitute for the ongoing frozen-rule onset transfer or broader speed/geometry validation.

The same6-case diagnostic panel now produces5/6original-goal arrivals andsurvivors, versus4/6arrivals/5survivors with fixed3.75s commitment. Actual right+36N support1.42→3.16s followed by user0 reachesgoal9.940s; all prior completed tasks remain. Release uses measured roll/rate settling for0.2s sampled at50Hz, minimum0.6s support age and qualified user eligibility. There is no input of the true future pulse time. This closes one online preparation–recovery–return example, with only0.060s horizon margin and left−36N failure retained. The candidate should be frozen for broader validation rather than tuned further on this panel. q labels remain held-action counterfactuals; this result does not validate their calibration for a sequence of future returns. See `u170_lower200_recovery_release_online_v1/RESULTS.md` and all6paired XY/reward figures. Main manuscript remains historical pending final method/evidence selection.

### Online commitment result — 2026-10-31

The one declared3.75s horizon-linked support commitment was tested on allsix adjacent/mirror nominal/±36N conditions. Survival increases4/6→5/6 with no binaryloss; originalgoal arrival remains4/6. Both nominal cases complete, but right−36N arrival is1.465s later. Right+36N now survives yet missesgoal (closest1.811m); it returns toalpha0 at5.18s and later re-entersalpha.75 at7.78s. This supports the importance of continuing a support action, while rejecting this fixed-window variant as sufficient to reproduce the successful diagnostic task return. Future progress requires a measured-state return decision or a prediction target matching the actual return policy; no fixed-duration grid was launched. Pre-event event-lane alpha/deadlines agree; no realfuturepulse timing is an online input. Evidence: u170_lower200_commitment_online_v1/RESULTS.md and allsix paired reports.

### Bounded related-work update — 2026-10-31

Two focused searches checked policy switching/dwell and preference-conditioned robotic control after the independent commitment proposal. Primary records checked: [de Priester and Sanfelice, RLJ2025](https://rlj.org/papers/priester2025timer) describe a timer-based hybrid supervisor with adaptive dwell, stability conditions and robustness to measurement noise; [Chemingui et al., CAPS, AAAI2025](https://arxiv.org/abs/2412.18946) select reward-maximizing policies subject to learned cost constraints; [Mousa et al., PROMO,2026preprint](https://arxiv.org/abs/2610.01260) expose preference inputs for quadrupedal tracking/stability/efficiency. PROMO is reported as submitted to T-RO, not accepted. Publisher/author/arXiv abstracts and bibliographic records were checked; OpenReview direct PDF access encountered a browser challenge. Full theorem/method replication was not done.

These sources rule out presenting a timer, a preference input, or risk-filtered policy selection alone as our new contribution. The current experimental question remains single-track preparation and recovery timing with a fixed user preference, actuator/reference dynamics, and original-goal restoration. The horizon commitment experiment is an implementation candidate for matching held predictions to actual execution; novelty is not established by that repair. The current six-case diagnostic cannot substantiate broad robustness or an independent validation claim.

### Recovery-return capability result — 2026-10-31

The recoveredalpha1 state at task3s admits allfive contact-free held continuations throughH10. Returning to originaluseralpha0 reaches the originalgoal at9.905s; the other choices miss it, with minimumgoal distances1.173/1.499/1.792/2.010m foralpha.25/.5/.75/1. Together with the1.62pre-gust intervention, this gives a support-then-return trajectory witness under a common plant and originaltask. It does not establish an online timing law: the two intervention boundaries were selected from inspected diagnostic evidence.

A separate non-originator scientific-brainstorming review proposes matching minimum support commitment to the current held-label horizon3.75s, with existing feasibility interrupts and normal coordination afterward. This needs a matched nominal/disturbed online test: task3 return success does not imply success after that commitment expires. A no-oracle policy must time from its own intervention or use measured state, never the actual future pulse. Do not introduce a time sweep or call a scheduling compatibility repair a new prediction method. An alternative is to label the actual executable return policy; this changes the target and requires a new declared fit. Evidence and reviewer provenance: u170_lower200_recovery_return_v1/RESULTS.md and independent_review.md.

### Pre-gust decision intervention result — 2026-10-31

At the recorded1.62s alpha1→.5 decision, holdingalpha1 survives throughH10 while holding0/.25/.5/.75 fails around2.24–2.27s. The complete common state, original+36N event and10Hz planner phase are preserved. In contrast all five alternatives from1.82s fail. This is direct evidence that the time at which preference execution changes matters in this trajectory. It is not yet a learned preparation-time effect or proof of a deployable improved coordinator. Alpha1 still misses the originalgoal (closest2.010m), identifying recovery-to-task return as the remaining control problem. One-source diagnostic alternatives cannot establish generalization.

Decision: retain the preference/preparation research direction. Investigate an observable commitment-and-return policy and matched-protocol risk calibration rather than concluding that alpha has no useful authority or merely tuning the existing probability threshold. Do not expose actual future gust time to the online rule. Evidence: u170_lower200_pregust_branch_v1/RESULTS.md and allfive XY/reward traces. No new online gain or publication novelty is inferred from this result alone.

### Actual restoration intervention result — 2026-10-31

The precise online restoration is1.82s, not1.9s. At that complete native state, allfive heldalpha continuations fall atapproximately2.24s while retaining the original remaining.13s,+36N pulse and absolute10Hz planning phase. Previousalpha.5 andalpha1 also fail. This directly weakens the proposed explanation that merely withholding this late restoration would save the trajectory. The original online policy switches again later, so the counterfactual changes its future held continuation, not just one action. The failed late branches do not establish that preference scheduling is ineffective earlier. The next discriminating source is the actualpre-gust1.62s change1→.5. Fullreport: u170_lower200_restoration_branch_v1/RESULTS.md. No new mechanism or publication-level gain is claimed from this negative diagnostic.

### Revision completion checks

- Freeze one final middle/lower/predictor/coordinator identity and rebuild the main architecture, method equations and experimental table around it.
- Report all declared conditions by speed regime and actual measured speed, with originalXY/goal, contact, task costs and preference intervention/restoration. Preserve raw endpoints and failures.
- Complete frozen prospective validation before making new-scene claims; keep train, reusedDEV and new validation roles distinct.
- Use the existing12-page export as a layout starting point only. Rebuild and inspect all pages after the method/results revision; current page count does not imply current scientific completeness.
- Keep the old experiments and their verified provenance available as historical studies. Sources E286–E292 were checked by the AI against local artifacts; accountable human verification is not recorded.

## Historical research design through 2026-10-30

The following sections document the previous return43/local300 study. Their references to the then-current method, next experiments and page count retain that historical scope.

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

Planned 12 pages including references: introduction/related work 2; system and task 1.5; preference/tolerance method 2.5; experimental protocol 1; simulation results 3; discussion/conclusion/references 2. These are editorial allocations, not blank padding. Write the canonical manuscript in English, with three contribution statements and results organized by the questions above. Keep one canonical manuscript and reproducible double-column export. Local Tectonic 0.17.0 is installed under `runs/paper_toolchain/tectonic`. An IEEEtran journal-class smoke document compiled successfully on 2026-10-30; the log identifies IEEEtran V1.8b. Archive source and SHA256 are in `runs/paper_toolchain/provenance.json`. The canonical Markdown is now fully English and exports to an IEEEtran double-column PDF through a generated LaTeX file. The current build has twelve pages. Source evidence IDs remain in Markdown and the evidence manifest; the printed PDF uses numbered literature citations. All twelve pages of revision83b8048 were visually inspected. The review found missing table captions, code-style mathematics, and undersized multi-panel plot labels. Table captions and inline mathematics are now implemented; three matched-baseline figure families have been redrawn at7.16-inch width with explicit7-point labels. Their raw identities, endpoints and all numeric costs match the original analyses. The revised build has no missing-character, undefined-reference, or overfull-box warnings. Remaining work includes redrawing the older plot families, inspecting the revised full layout, organizing results by scientific question, and auditing prediction evidence against the claimed contributions. Two repetitive main-text plots were removed while their numerical findings, source data, and evidence links remain available.

Scientific Agent Skills supported independent ideation, a separate adversarial read, and evidence-bound writing. Its reference is verified against the arXiv record: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner, *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, 2026, https://doi.org/10.48550/arXiv.2609.00065. Candidate suggestions and reviewer opinions are not experimental evidence.
