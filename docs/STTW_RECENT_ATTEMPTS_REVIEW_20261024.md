# STTW 最近尝试、结论与创新定位

截至2026-10-24。本文件是一次研究复盘；当前运行与决定仍以[共享台账](/home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md)为准。范围是STTW，主要覆盖10月21日至24日的单网络、容扰度与抗扰工作，历史双网络作为起点；不混入RecoveryBridge/JIT结果。

先看已有实测：[名义五α与零中层的完整轨迹](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/zero_middle_nominal_full_analysis_v1/REPORT.md)、[当前H75初始96条实际XY](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/online_initial_baseline_v2/all96_xy.png)、[同状态因果参考干预XY](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/causal_reference_full_task_v1_attempt2/analysis/REPORT.md)。不同面板的条件、时间窗和标签不同，不合并成一项总体成功率。

本次核对冻结模型登记、对应结果报告、共享台账、当前PID及训练状态；没有重新运行全部历史仿真或逐个重算所有历史NPZ。下表数字为已定位结果报告的记录，当前训练另列。成功样本与失败样本均保留。

## 目前到底在做什么

已有结构是：冻结local300底层，用中层输出有界、滤波的速度和转向参考修正；用户通过α选择路径优先或速度优先。近期实际研究的是：这种取舍如何影响受扰后的不摔倒能力，能否提高它，以及付出多少速度、路径和任务代价。

容扰度最初承担解析训练信号的角色；数轮比较后没有建立稳定增量，现已降为解释和评价基线。当前正在训练的是“原任务续训”与“原任务＋真实有限窗防摔结局”的对照，没有把旧解析公式作为核心奖励。它不是牛顿值容扰度预测网络。

必须分别看三类结局：完整观察窗不触地；继续并完成原任务；按用户偏好取得有意义的速度—路径取舍。第一项有局部提升就值得保留，不要求无敌；第一项也不能自动替代后二项。路径域退出后保持最后参考的物理尾，属于明确的复合控制合同，不能写成中层一直在自主恢复。

## 已有网络与任务功能

|尝试及目的|具体做法与结果|结论|
|---|---|---|
|双网络→单网络：保留偏好，简化部署|历史Coverage两个端点分别运行固定α；用它们做8000步监督蒸馏，再用V3奖励与课程训练400轮，保留u325开发候选及u400终点|单网偏好接口已建立；蒸馏、奖励和课程同时变化，不能把全部效果归因于V3。u325不是独立测试选出的最终最优模型|
|检验“只是训练不足”|完整尾段PPO由25延至75；强扰动任务186→191/200，低速52→56/60，中等与直线保持。按原记录汇总448→457/470，9改善0退化|充分普通续训确实有用，不能用短训练负结果否定单网/PPO；这些主要是任务恢复改善，不是9次防止摔倒|
|回头检查双网络|18例小面板：旧双网原生20s路径端9/18、速度端18/18；共同前10s路径端8/18。对应原单网路径端9/18、教学单网16/18，速度端均18/18|没有证据需要主线切回双网。旧奖励、权限与时域不同，这不是纯架构消融|
|检验名义偏好是否有用|H75五α四名义工况20/20原任务保持；零中层4/4到达且不摔，末段航向约4–6°超过原严格阈值。高速存在不同α的效用用途，部分较低速度工况被零中层覆盖|偏好调节和姿态/出弯恢复有有限用途；不能用20/20对0/4夸大成“没有网络就不能走”。部分中间α路径误差优于端点，不宣称严格单调或Pareto最优|

谱系不是单调变好的直线：双网→BC→u325；u400、短恢复、完整尾段H75及示范教学是不同分叉。9个模型、56个内容寻址对象及代码/配置已冻结，见[冻结登记](/home/qy/STTW_CONTROL/runs/model_registry/middle_lineage_20261024_v1/README.md)。

依据：[75PPO长度对照](evidence/tolerance_progress_20261022/full_tail_length75/FINAL_RESULTS.md)、[双网回访](evidence/tolerance_progress_20261022/dual_endpoint_revisit_v1/RESULTS.md)、[完整名义对照](evidence/tolerance_progress_20261022/baseline_provenance_audit/CONCLUSIONS.md)。

## 容扰公式、采样与奖励：做过什么

|尝试及目的|具体做法与结果|结论|
|---|---|---|
|解析方向余量：用物理量描述恢复难度|推导不稳定模态坐标、有限脉冲折算及最小L2控制努力；做局部标定和带符号同状态扫描|局部模型下代数成立。它描述理想控制输入下的恢复潜力，不直接给出冻结参考闭环的真实防摔阈值|
|同状态扫描：先看公式是否含信息|32状态×13力档共416例，任务恢复408/416；另60状态×9力档共540例，任务恢复394/540。后者检查组任务失败排序AUC约.854，单看力幅约.748|有局部任务难度/方向信息；前一组无已见物理失败且有6个物理结局UNKNOWN，因此不能由此建立摔倒阈值|
|容扰度直接进奖励|匹配续训、静态和动态余量成本。α0普通133/144，静态134/144，动态103/144；α1加入余量成本后任务保持明显退化|没有稳定额外收益；该组未观测物理摔倒，不能把任务0/144说成全摔。停止把直接奖励塑形作为已验证方法|
|完整任务中的均匀/引导扰动训练|双端点各训练25轮，864例开发评价。α0父/均匀/引导原任务均133/144，但引导域退出增多；α1末段路径偏移影响原阈值标签|引导没有一致超过均匀；部分差异是任务偏移，并非防摔差异|
|短恢复片段、critic预热与名义锚|4s片段，2轮critic＋25PPO；α0父133→均匀138、引导136/144；α1均141/144|联合训练协议改善了任务保持，均匀还修复2个域退出；容扰引导没有超过均匀。组件一起改变，未隔离哪项导致改善|
|校准引导强度并加打乱对照|将采样关注区间移至risk2–3，保留同父模型/力档/α。强面板父/guided/shuffled/uniform分别189/189/191/186，各200；guided与父逐例0改善0退化|不能用采样确实改变来替代控制收益。停止追采样中心或公式小数，不采用guided|
|由短片段改完整尾段|25PPO阶段暂未显示标准面板净收益；随后用预声明长度对照得到上表H75改善|“尾段更长”本身不是充分条件，训练量和效果须实际核验；H75作为更强对照保留|

依据：[416扫描](evidence/tolerance_progress_20261022/train_margin_scan_results/CONCLUSIONS.md)、[540扫描](evidence/tolerance_progress_20261022/preference_recovery_pilot/train_difficulty/CONCLUSIONS.md)、[奖励塑形](evidence/tolerance_progress_20261022/reward_shaping/REPORT.md)、[真实脉冲续训](evidence/tolerance_progress_20261022/pulse_curriculum_results/diagnostics/CONCLUSIONS.md)、[短片段](evidence/tolerance_progress_20261022/short_recovery_results/CONCLUSIONS.md)、[校准/打乱](evidence/tolerance_progress_20261022/preference_recovery_calibrated/README.md)。历史名义重复行、同快照多个力档及单训练种子不能计成独立复现。

## 示范搜索、教学和更强对照

|尝试及目的|具体做法与结果|结论|
|---|---|---|
|干预搜索：确认是否真有能学的救法|对现有中层分别关闭速度修正/转向修正等；120条搜索中，control21/40任务完成、speed_off27/40、steer_off13/40，得到6个有效TRAIN见证|存在可利用的任务恢复示范；不是普遍关闭速度修正更好|
|用公式排序省搜索|同一搜索中，预先指定16查询预算找到3个见证，均匀期望也3个；24查询为次要较好结果|没有证明主要查询预算下公式能提高搜索效率|
|示范教学与跨几何迁移|两个跨几何面板，原父160/192、教学165/192；加入更强75PPO对照后为164/192|教学有局部任务收益，但相对更充分普通训练仅净1例，不能夸大方法优势|
|“有效见证”对随机成功示范|三组成对示范/优化种子：强任务合计净+4，低速净−4，整体8改善8退化|该筛选机制没有整体增量；不能把任何教学收益都归于见证筛选|
|经验方向恢复图|新105°/R2.4、144条，父任务62/72→教学70/72，8改善0退化；3条策略—方向曲线出现失败后更大力又通过|方向分解是有用评价；这里是任务恢复，不是已确认物理摔倒边界。不能直接用最大通过档或全局二分替代完整离散图|

依据：[搜索](evidence/tolerance_progress_20261022/tolerance_teaching_search_pilot/RESULTS.md)、[强PPO迁移对照](evidence/tolerance_progress_20261022/long_baseline_transfer/RESULTS.md)、[见证筛选对照](evidence/tolerance_progress_20261022/witness_selection_v1/final/RESULTS.md)、[经验恢复图](evidence/tolerance_progress_20261022/empirical_tolerance_v1/RESULTS.md)。以上正收益保留，但不归因于未获增量的解析排序。

## 改为真正“不摔倒”之后的机理与控制试验

|尝试及目的|具体做法与结果|结论|
|---|---|---|
|低速、转向及参考权限|固定完整状态，扫描方向和力，比较保持、小幅加减速、转向调整。若干低速工况小幅加速能救回；150分支扩检面板又出现保持已足够、提前加速额外触地的情况|参考确能改变防摔结局，但“始终加速更安全”不成立，速度不是唯一状态|
|修正模型：加入真实反馈与约束|低阶ECBC/ESO模型纳入部分滤波/限幅/历史；24例后扩检54例，仍存在保守误判及排序打平，遗漏local300与完整非线性|比自由输入模型更贴近实际，但没有建立可靠参考选择增量；不靠调整预算继续追旧标签|
|直接学习闭环能力|整理198条真实稳定/瞬态轨迹，比速度、原始状态、物理增广、通用增广及固定RBF；稳态到瞬态外推弱，后续56条新固定参考后缀中速度与RBF判断相同|物理增广没有独立优势；当前数据不证明复杂预测器比简单状态/力幅更必要。模型分数也未转化为新控制收益|
|物理特征进入偏好网络|普通PPO、六维物理特征、同容量通用特征匹配75PPO，含H75共880例。原低速物理标签三新模型逐工况相同；新几何物理特征的任务与完整无触地数不优于普通/通用对照|这一组物理特征没有额外防摔收益。contact→UNKNOWN不算救回；停止同类特征追调|
|连续两次推扰A3|270条，比较保持与±.24m/s速度偏置。非零二推完整无触地：保持67/72、加速64/72、减速63/72；两个干预均0救回，分别额外3/4次触地|首推历史与等待时间影响再次抗扰；固定加减速未改善，停止该单通道库扩展|
|偏好与物理边界的关系|左右弯共80分支，固定原H75。α1在部分方向优于α0；加零中层后，旧几何受扰无触地zero4/16、α0为0/16、α1为6/16；新几何zero与α1均4/16且通过条件相同|偏好确实伴随不同进入状态与方向性能力，但网络对简单跟随器的抗扰增量只在部分旧工况成立，未稳定迁移|
|中间α是否同时兼顾|新几何α0/.5/1/zero受扰完整无触地分别2/3/4/4，各16；中点有一个高速左弯名义效用用途，也有右弯防摔代价|存在有限任务—防摔取舍；没有得到普遍无损或普遍更抗扰结论|
|B1无扰切换与受扰瞬时切换|无扰弯中/出弯fixed/step/ramp共24条均任务成功。另同一α0受扰状态比较继续α0、立即α1、零latent，共72记录，非零每种均0/16无触地|原网络可切偏好，但这些起扰状态下临时换α没有救回。保留差异，停止该时点调度，不把提前规划和瞬时救回混为一谈|
|新几何公式排序确认|32受扰中公式AUC1，但原80N阈值有10个危险乐观误判；联合普通状态基线AUC约.968，各α内部也可达1|排序好不等于牛顿阈值准确或保守，更不等于接入控制有用|
|因果ESO参考规则|不读取未来真实力/时刻；24条全任务的8个非零配对，不触地5→7，2救回0丢失。16/32/48N补充后同一56轨迹集合仍2救回0丢失；交换左右速度的12条另有1个局部救回|这是实际局部防摔正证据，允许有路径代价；当前为简单基线、相关开发状态，未证明连续α协调/独立泛化。没有使用旧容扰公式选择动作|
|跟随父目标、两段时序参考|父目标跟随12条对强锁定规则无防摔增量且损失一个联合任务结局；时序库64条实际后缀，对均值常值右弯1救回/2丢失，左弯无防摔增量且有任务损失|没有支撑更复杂的目标合成/时序库，保留锁定基线，不继续增加段数找正例|

物理结局常涉及共同domain-hold；各策略自身前缀与同一完整快照是不同实验单位。上表“α1比α0更强”不自动推出同状态改变α有效。无触地的轨迹可能绕离路径，这仍是物理正结果，但任务代价必须明列。

依据：[低速参考权限](evidence/tolerance_progress_20261022/low_speed_reference_authority/REPORT.md)、[扩检](evidence/tolerance_progress_20261022/reference_authority_expansion/REPORT.md)、[反馈模型](evidence/tolerance_progress_20261022/reduced_proxy_expansion/REPORT.md)、[能力数据/模型](evidence/tolerance_progress_20261022/baseline_provenance_audit/CLOSED_LOOP_DATASET_FINDINGS.md)、[三臂880例](evidence/tolerance_progress_20261022/reserve_feature_expansion/CONCLUSIONS.md)、[A3](evidence/tolerance_progress_20261022/repeated_disturbance_recovery/CONCLUSIONS.md)、[偏好与零基线/同状态切换](evidence/tolerance_progress_20261022/baseline_provenance_audit/CONCLUSIONS.md)、[新几何](evidence/tolerance_progress_20261022/baseline_provenance_audit/NEW_GEOMETRY_ZERO_FINDINGS.md)、[中点](evidence/tolerance_progress_20261022/baseline_provenance_audit/MIDPOINT_PHYSICAL_FINDINGS.md)、[B1](evidence/tolerance_progress_20261022/preference_switch_phase/RESULTS.md)、[公式确认](evidence/tolerance_progress_20261022/baseline_provenance_audit/PROSPECTIVE_FINDINGS.md)、[因果规则](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/causal_reference_full_task_v1_attempt2/analysis/REPORT.md)、[力档补充](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/causal_empirical_force_sets_v1_attempt2/tolerance_curve/REPORT.md)、[左右速度迁移](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/causal_turn_transfer_v1/analysis/REPORT.md)、[父目标跟随](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/parent_following_comparison_v1/analysis/REPORT.md)、[时序库](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/temporal_reference_opportunity_v1_attempt2/analysis/REPORT.md)。

## 目前正在做的事情

先核对了训练/评价语义：任务域退出与非轮触地是两件事，旧训练与后来的hold续行评价有合同差异。由此建立完整物理尾采集、合法决策mask、独立物理价值头和真实尾结局信用；这些是正确学习工具，不是新的策略梯度理论。

首批更新后的12条新轨迹：原H75、任务续训和物理信用三者都3/4不触地，无额外防摔gain/loss。名义左例两种续训都明显减少路径偏差，说明普通任务训练已能解释局部改善。见[首批实际结果](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/dual_survival_post_update_v1/REPORT.md)。

当前online_preference_survival_v1_attempt3沿原64轮进行，两个匹配臂task_only/physical_credit，每臂每轮48条真实回合，包含α0/.5/1、左右低速紧弯、0/有符号32N及4个动作噪声draw；前2轮critic-only。最终结果PENDING。四个draw不是四个独立训练种子，物理价值头没有力幅查询输入，不解释成F*预测器。

已有工程成果：488条固定终点评测清单、确定性/带噪声/零修正/速度下限采集器、native权重及导出认证。五分支共80个实际lower ticks的短工程检查完成，41项相关CPU测试通过；本轮又补7项批量分组/动态参数及V3重评分/标签测试。短工程检查无外扰、每支仅.08s，不能算新策略抗扰成果；488完整评测尚未执行，批量driver与最终完整绘图仍在实现。已封存的训练源没有因这些辅助模块改变。

训练先前两次启动/数值检查错误已保留。首批96条及一臂已完成的critic更新经身份核验复用，没有重复执行已完成物理/优化。Bark完成/错误通知、TensorBoard真实回报和状态入口保留。当前具体完成轮数以台账/状态文件为准，不在本复盘维护另一份实时状态。

## 创新应该如何定位

**已经有实验支持、可复用的内容**：有意义的单网速度—路径偏好功能；普通续训及示范带来的局部任务改善；偏好/参考与方向性防摔能力的耦合现象；简单因果参考调整的局部真实救回；完整物理结局与任务代价分开测量的证据链。这些不因容扰公式没有控制增量而作废。

**值得发展的候选贡献**：在冻结学习底层与真实有界参考权限下，研究偏好可调的两轮机器人抗扰任务控制，明确何时可以保留用户偏好，何时需要付出减速、路径或参考变化代价，并用强简单对照验证实际用途。它可以形成任务/系统研究问题，但当前尚未证明一套稳定优于普通训练和简单规则的新协调方法。

**目前不成立的创新主张**：解析容扰公式给出可信真实防摔阈值；容扰度引导训练稳定优于均匀；物理特征独有增益；见证筛选稳定优于随机成功示范；瞬时切α普遍救回；新增物理信用已经提高成功率；方法具备安全认证或已达到RA-L投稿证据完整度。

原始文献边界已再次核对：[Leng等的偏好条件多目标控制](https://arxiv.org/abs/2510.10851)已有单策略跟踪—外力顺应取舍；[SafeExplorer](https://arxiv.org/abs/2607.08925)已有恢复接管下的合法决策score项及尾部回报信用；[Recovery RL](https://arxiv.org/abs/2010.15920)已有任务与恢复目标分工。因此α输入、mask、安全critic或恢复尾信用本身不能承担首次性。这里是有界近邻核对，不是全面文献查新；不同平台/任务也不自动证明方法增量。

我建议保留一个中心问题：**小车在存在速度—路径冲突时，怎样通过现有中层参考权限，更好地兼顾偏好、抗扰与任务？** 将容扰度作为可选能力评价，而非必须证明有效的公式模块。先完成已经授权的匹配训练与终点评价；依据额外收益及代价决定方法去留。允许失败和小误差，不要求100%成功；同时保留零中层、固定α、简单速度规则和普通续训这些强对照。

实机尚未部署，也没有现成传感/电机控制程序；依用户最新要求先做好仿真，本复盘不启动硬件工作、不索要硬件信息。此前部署契约/接口核对是准备资产，不计作实机实验或方法增益。

方法分析沿用Scientific Agent Skills的experimental-design与scientific-critical-thinking，用于区分实验单位、假设与证据，并非人类同行评审。软件参考：[Kassis等，Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。
