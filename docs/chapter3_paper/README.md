# 第三章任务书执行与文稿状态

**原生高台开发试验。** 把第二章 0.16 m 高台接入第三章粗糙弯道和中央墙后，8 个共同初态、两种接近和两种冻结策略共 32 条轨迹均出现真实腾空、台面双轮承重和锁摆交回。分别实跑三个驶离选择后，Source 和 Full RB 在近、中、远三档均为 0/8 个初态双向反转。交接已足以开展规划对照，左任务仍受交回时真实侧向运动和随后的路径跟踪约束。[实际轨迹、整程录像与逐合同判定](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/native_platform/approach_alignment/paired_tasks/RESULTS.md)

在上述 96 条物理后缀上，使用相同车体包络与冻结裕量，地面动态预测将 Source 误放行从 3/207 降至 0/207，误拒从 45/81 升至 72/81。Full RB 的误放行同为 0/198，误拒从 54/90 升至 81/90。每条实际驶离承担六项合同评分，有效样本按根和物理路线分组。此结果支持讨论模型迁移造成的取舍，H2 的净规划收益仍待建立。[共享裕量与零裕量的逐根对照](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/native_platform/approach_alignment/ground_predictions/RESULTS.md)

第三章考察一个具体决策：同一辆车从同一初态出发，若远岸改为必须通过另一侧检查点，是否需要在沟前就选择不同的接近路线？第一章的锁摆 STTW 执行地面接近和驶离，第二章提供冻结的 Source 与 Full RB。第三章以实测接管状态衔接两段控制，以整车碰障、检查点与终点保持共同评价整程。原 Full RB 在 map00 的 36 条中首触远岸 30 条，稳定交回 0 条；当前主矩阵使用冻结 Source。[任务书](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/STTW_第三章_Codex任务书.md) · [技能失败漏斗](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/phase1_mechanism/rb_funnel_diagnosis.md)

**截至 11 月 19 日的可写结论。** 在 0.55 m 宽沟的一个开发初态，两条沟前右弯路线均产生核验飞越，远岸双轮承重并锁摆交回。两种接近各实跑直行、左绕和右绕三条后缀，同一轨迹分别按左右必经合同计分；三档均未观察到双向反转。远档只有一种接近能完成右侧任务。两条实际右绕均避开墙却因终点保持不足而超时，说明只检查碰撞会漏掉任务失败。[实际与参考 XY](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/phase2_task_reversal/task_reversal_xy.png) · [合同矩阵](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/phase2_task_reversal/oracle_task_reversal.csv)

同一开发根、相同占据裕量下，复用地面动态模型将误放行由 3/31 降至 0/31，将误拒由 5/5 降至 2/5；三次纠正来自同一条物理超时轨迹。冻结两条接近移至两个新初态，原图和新布局各 0/4 稳定交回。随后八个开发初态的 144 个接近候选也未产生合格交回。当前证据支持在实测交回状态上研究短窗预测，另一个中心正向起步的开发初态已取得单条真跳、稳定交回并完成保持的实测（卸载 557 ms，9.885 s 完成）。中心初态的三种后缀中左绕撞墙、右绕保持 0.405 s 后超时。相同裕量下两种模型均误放行 3/16、误拒 2/2，开发根 3 的预测收益尚未复现。在 0.45 m 两个开发初态的六条后缀中，三条超时、一条离开跟踪域、两条碰墙。共享裕量下运动学均拒绝，动态模型错放一条右绕超时在三个时间档的评分。3B 训练缺少跨初态的预测收益依据。已测试的两个独立迁移初态仍未能交回；双向任务反转及在线整程收益仍待验证。[逐例结果与迁移轨迹](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/phase2_task_reversal/RESULTS.md)

保持原来成功的两个起点与三条接近命令，换到此前冻结的 19033 布局后，三条中一条仍有核验飞越，两条首触远岸，均在锁摆交回前终止。技能接口的名义比较也已核对到接管帧：两个沟宽共 168 对，Source 与原 Full RB 的起步至接管位姿、速度与地面控制器历史均逐元素相同。0.45 m 下，换成 Full RB 新增 7 次核验飞越、保留 7 次、丢失 14 次，两者都没有稳定交回；0.55 m 下 Source 唯一有合法后续的交回在换成 Full RB 后丢失。数据仅限该场景和候选库。[同起点布局迁移 XY](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/phase2_task_reversal/matched_layout/matched_layout_xy.png) · [同入口技能对照](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/phase1_mechanism/rb_funnel_diagnosis.md)

已完成同根 3、同车道 71、两种技能在两个独立施力窗口的 4 条配对实测。起跳准备受扰时仅 Source 锁摆交回，原直行在终点保持 0.390 s 后超时。空中窗口的 50 个子步，Source 与 Full RB 分别有 0 和 0 个子步受地面支撑。Source 在承重后、交回前越出跟踪域；Full RB 飞越未通过，无双轮承重，未交回。[实测技能与扰动对照](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/phase4_contributions/RESULTS.md)

## 3.2 冻结跳跃技能的零样本迁移

原 map00 上冻结 Source 与 Full RB 分别只有 1/36、0/36 次锁摆交回。Full RB 的 30/36 次远岸首触均未带来整车越过远岸的结果。在 0.55 m 沟的八个开发初态上，对两个冻结接近分别只改变沟前速度请求。1.8 与 2.0 m/s 两档下，Source 的核验飞越分别为 3/16、2/16，双轮承重各 1/16；Full RB 两档均为 0/16 次核验飞越。两种技能两档均无合格交回。原第二章 XML 是连续地面加台阶，迁移到本章真沟还增加了跨岸与落地交回要求。提高实测入口速度在这一有限筛查中未能恢复交回。后续同预算接口适配须获批准，并独立报告适配后的权重和结果。[八个初态实际与参考 XY](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/phase1_mechanism/domain_alignment/speed_xy.png) · [原始逐例、失败时序与适配提案](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/phase1_mechanism/domain_alignment/RESULTS.md)

> **2026-11-18结论撤回：** 用户指出成功视频没有明确主动跳跃。已复查视频：该案例不能作为主动起跳跨沟成功证据。旧goal/airborne标签仅保留作历史执行记录；P2机制解释及依赖它的后续跳跃结论需重新验收。不能将短暂无接触或到达目标替代真实起跳。


当前执行依据为[新版任务书](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/STTW_第三章_Codex任务书.md)。旧DOCX不再指导实验。下方LaTeX/PDF是此前执行大纲下的文稿快照，尚未吸收新版任务书的全部实验；不能把编译完成视为P0–P6完成。

| 历史阶段 | 当时的证据与边界 |
|---|---|
| P0–P3 | 已有冻结、快照续接、Oracle和分段预测的有界实验；仍保留数值重放差异、预测误差和失败，不能称通用可靠技能世界模型。 |
| P4 | 规定的全链实现和G/I/C/Oracle等对照已有记录；标定后C/I在机制案例均无计划，可靠规划能力未达到。见[阶段审查](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p4_sequence/stage_review/README.md)。 |
| P5 | 条件地面模型两结构三种子均续至1200；条件串联与同预算α选择已完成已知DEV诊断，尚无α成功率增益。原source/完整RB首张TRAIN地图各36条均0最终成功，交接仍未解决。见下方最新证据。 |
| P6 | 实测落地更新C-R与任务重配置、新地图族泛化尚未完成。 |
| P7 | 可选；当前不接LLM。 |

P5已知DEV地图06/07共648条件：399最终目标成功，249失败；所有条件触发了旧交回逻辑，主动腾空跨岸需以新物理记录重验。固定α0/0已72/72，不能据此声称自动α选择增加成功率。TRAIN地图00/01共648条件：409目标成功，239失败；TRAIN仅作模型数据，不能与DEV混成泛化成功率。两组失败均保留。[DEV全部XY](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_alpha_grid/README.md) · [TRAIN全部XY](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_alpha_train_collection/README.md)。两张TRAIN采集已全部纳入，不能与648条DEV混成一个泛化分母。

当前shared225源于登记为rb_run3的PPO-only分支并经过本项目适配；工程名“RB225”不等于完整RecoveryBridge贡献证明。方法对照必须保留真正来源、各自匹配预测器及相同预算，不能直接用shared225与source之差归因完整RB方法。参见[谱系核对](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_snapshot_alpha/jump_lineage_audit.json)。

上述进度绑定source_manifest中的TB_P4_REVIEW、TB_P5_DEV、TB_P5_TRAIN、TB_RB_LINEAGE；代理已检查本地证据，人工核验保持pending。当前不主张可靠C优于I、自动α收益、完整RB因果收益或P6泛化。

条件跳前模型首组seed0已完成同预算600更新：残差1s/3s平均最大XY为0.02913/0.06640m，直接模型为0.04233/0.10367m。两者按1s开发标准选到末轮，末段仍有改善或波动，未认定收敛；单种子预测误差不等于规划收益。[全部逐例XY、存档曲线和限制](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_alpha_models/phase0_seed0/RESULTS.md)。

三种子首轮完整结果已补齐：[全部预测XY及曲线](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_alpha_models/phase0/RESULTS.md)。固定DEV末段误差仍改善或波动，续训保留原best，不把600更新视为收敛。

三种子600→1200续训已完成，按每25更新的固定1s DEV标准保留各自最佳。残差三个种子的1s平均最大XY误差为0.025898/0.024915/0.024919m，3s为0.058819/0.053635/0.05784m；直接模型对应0.037864/0.037031/0.040508m和0.101263/0.103404/0.092517m。DEV216例，1s3115窗、3s955窗；重叠窗口不是独立样本。直接seed2的3s均值相对首轮变差，残差最坏3s误差仍约0.18–0.20m，不能将均值改善解释为可靠全链规划。[全部XY、逐例与曲线](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_alpha_models/phase0_continue600/RESULTS.md)。证据TB_P5_CONDITIONAL_CONTINUE_3SEEDS，人工核验pending。

落地后条件模型两结构×三种子首轮已完成：[全部实际/预测XY](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_alpha_models/phase2/RESULTS.md)。残差三个种子的1s均值为0.025435/0.025027/0.024991m，3s为0.061066/0.062666/0.065937m。seed0续至1200后，固定1s标准选残差1175、直接1200，1s/3s均值分别为0.018819/0.044153m与0.022154/0.047914m；[完整续训XY及尾部误差](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_alpha_models/phase2_continue600_seed0/RESULTS.md)。有效DEV1s8398窗/618例、3s2511窗/553例，重叠窗口不独立；尚未认定收敛、独立TEST泛化或全链规划收益。证据TB_P5_DEPARTURE_3SEEDS及TB_P5_DEPARTURE_CONTINUE_SEED0，人工核验pending。

## 此前执行大纲文稿快照

Current editable draft: [PDF](MANUSCRIPT.pdf) · [LaTeX](MANUSCRIPT.tex). This revision follows the supplied execution outline: fixed physical map and goal, continuous STTW→RB→STTW, ground dynamics, joint planning, and structured prediction. It replaces the prior two-interface framing in the main text; historical prose remains in Git through0c485653 and all old evidence is retained below.

The current draft binds ground36/36 versus31/36, greedy3/12 versus joint12/12, the stronger backtracking12/12 counterexample, and rejected long-horizon history prediction to PG1–PG5 evidence IDs. Both the original rolling panel and a54TRAIN-prefix support expansion are physical12/12 but coverage-qualified11/12. PG6 adds discrete speed validation (1.8/1.9 each72/72;1.7=65/72). PG7 adds the same-data M0/M1/M2 comparison, three learned seeds each through600 updates, and full geometry-tail evaluation. No independent-test, general joint-planning advantage or formal training-convergence claim is made. PG8 adds a matched two-speed refit (108TRAIN/36DEV) with all six models through600 updates. Multi-map/planning model validation and P4 new-layout evidence remain required.

The source manifest and claim registry retain historical records with explicit roles. Three related-work entries were checked against primary abstracts/author PDF; broader literature coverage and all human verification remain pending. Compilation/layout checks establish document rendering only. The scientific-writing skill guided this revision; its existing tooling attribution was rechecked at the primary arXiv record on2026-11-14 and appears in the draft bibliography. No external unpublished content was uploaded.

## Historical manuscript and evidence


**Historical two-interface draft (through0c485653):** [PDF](MANUSCRIPT.pdf) · [editable LaTeX](MANUSCRIPT.tex). The main paper now describes the implemented two-interface model, rather than the earlier serial event-chain prototype. This is a research draft, not a submission-ready or superior-planning claim.

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

PG9 adds eight predeclared terrain maps (286/288 physical goals), six TRAIN/two DEV maps for matched M0/M1/M2 fitting, and full observed departure prediction on all 72 DEV conditions. Selected checkpoints include intermediate updates575/550; formal convergence and executed model-planner benefit remain unproven. Evidence: [multi-map report](../evidence/chapter3_playground_sttw_rb_sttw/p3_multimap_models/RESULTS.md) and its paired complete-departure XY figures. No independent P4 claim.

PG10 adds a static same-candidate model-planning DEV pilot: M0 10/12, M1/M2 12/12 with identical selections,24 unique physics runs. This avoids one no-plan fallback but does not show fewer false-feasible published plans or a structural advantage over M2. Measured-state integration and independent E1–E3 remain required.

PG11 adds the canonical approach subset (72TRAIN/24DEV) with full observed3.56–4.10s predictions. The three illustrative XY panels use fixed map/root/speed and all three approach directions; full24conditions remain linked. PG10 now explicitly records the missing predicted terminal-hold gate; physical success is retained without calling it full dynamic admissibility.

Current PG11 includes600→900 continuation and retained intermediate best checkpoints; PG12 adds explicit predicted terminal/domain screening with only2new physics runs. PG13 retains the rolling coverage limitation after108new TRAIN histories: physical12/12, coverage11/12 with a different failure. Independent P4 E1–E3 remains pending; no full-coverage or structural-dominance claim.


PG14已加入：50新布局×4根，M0 156/200，M1/M2各200/200；收益来自减少无计划，非假可行下降或M1胜过M2。完整逐例/XY见p4_model_test；静态组件TEST与既有DEV分开，E1/E2和滚动验收仍待完成，作者核验待办。

## P5最新配对证据

[原source与完整RB的全部XY和逐例结果](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_source_full_rb/RESULTS.md)：22/36与30/36远岸接触，1/36与0/36交回，均0/36最终成功。不能将接触增加称为整程能力收益。

[同预算α选择](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p5_alpha_models/alpha_selection/README.md)：4个已知DEV任务，固定α0与完整枚举均4/4；18查询候选排列保留全部9种，部分选择退步。无独立TEST、α收益或完整RB因果收益。主稿仍未达到提交条件。
