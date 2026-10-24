# 论文证据链与路线取舍

2026-10-24，有界原始文献核对＋现有报告/产物核对。不是全面新颖性检索、录用预测或新实验结果。当前运行事实仍见共享台账。

**后续结果更新**：本文下方“zero24运行中/待裁决”是制定矩阵时的状态。现已[完整完成](NEW_GEOMETRY_ZERO_FINDINGS.md)：zero与α1同为4/16且通过条件完全相同，α1相对zero零新增/零丢失。按预定规则停止新几何抗扰增量主张；跨策略内部公式与联合状态模型均AUC1，独立预测增量也未建立。保留下方原判断与停止逻辑，不将历史待定冒充仍在运行。

## 最新裁决补充（2026-10-24）

[固定中间偏好判别](MIDPOINT_PHYSICAL_FINDINGS.md)现已完成24新/11694ticks＋72复用，96逐例/24图。非零力α0/α.5/α1/zero无触地分别2/3/4/4（各16）；中点对zero/α1零新增一丢失，对α0一新增零丢失。高速左弯保留已有名义效用用途且本组防摔结局相同；高速右弯有一例防摔代价。保留有限联合取舍，停止普遍抗扰增益/全局无损主张；不继续补α/力/相位。下面“待独立审查”已被本项覆盖，属历史计划。


[完整名义迁移](NOMINAL_TRANSFER_FINDINGS.md)已完成24条/48000ticks，H7520/20原任务保持，zero4/4到达不摔但尾航向超阈值。高速偏好效用用途存在，低速zero覆盖；不能泛化名义优势为抗扰优势。

[固定参考低通对照](REFERENCE_FILTER_FINDINGS.md)完成4条/7584ticks，3条提前任务域退出；四条保持失败、无已观察触地，逆映射/原slew/幅限0截断，低通递推误差小于3.4e-8rad。该固定简单平滑没有解释H75收益；不扫参数，也不声称排除了全部传统控制。24PNG/140评分完整，低通未改变原偏好效用结论。

[历史信用分配核查](TRAINING_CREDIT_SCOPE_AUDIT.md)证明完整回合弱力准备训练已做，不能包装为从未尝试的新方法；缺准备导致固定bank阴性没有成立。实机用户已明确暂无传感/电机控制程序，不再寻找已存在入口。

下面“正在运行/未启动”的描述是历史决策上下文。当前参考低通已结束；中间偏好的任务—防摔折中仅待独立审查，不自动授权扩α扫描/强扰动训练。即时切换、六特征和A3阴性保持。RA-L目标仍未达到，保留系统价值与容扰诊断，方法增量仍需新证据。

## 不能把哪些部件本身当新颖性

### 2026-10-24 原始文献补查与新的机制缺口

本轮限定检索bicycle/steering/recoverability/control limits、preference/force compliance和early termination/recovery；不是全面综述。直接读取与出版方索引读取分开记录，未访问到的全文不冒称已读。

- Lovato、Lot、Massaro，2026，[Standstill Balancing of Two-Wheeled Vehicles via Steering Control](https://doi.org/10.3390/app16146887)。出版方直接访问429，但其索引返回第2–3节公式及第4/6节内容。本次核对：Whipple转向角输入、转向加速度项、二阶执行器、发散模态及有界转向的可恢复区域均已存在。故“不稳定模态＋转向限幅恢复边界”不构成我们的新方法。其恒速线性模型与我们的ECBC/ESO/local300闭环不同，不能直接替代闭环容扰边界。
- Reijne、van der Helm、Schwab，2026，[Maximum Allowable Handlebar Disturbance](https://tsr.international/TSR/article/download/28055/24882?inline=1)。本次直接读方法/讨论：用车把转矩脉冲定义防摔能力，区分实验实际摔倒与模型阈值，并提出重复试验估计概率边界。该篇为框架论文，明确本研究未采新数据；不能借它声称实测验证已完成。其0.3s车把转矩与我们的0.1s车体横向力不可数值替换。“测最大可扛扰动”的想法也不能单独当首次贡献。
- Leng等[偏好控制原文](https://arxiv.org/html/2510.10851v2)，本次复读III与IV-D：力—速度统一奖励、历史特征重建、偏好输入与瞬态推扰比较均存在。我们的速度—路径取舍和单轨参考权限有区别，但当前没有超过简单控制的普遍防摔增量，不能只换平台宣布新方法。
- [BeamDojo作者论文](https://why618188.github.io/beamdojo/static/BeamDojo.pdf)，第IV-C：阶段学习中将误步惩罚与吸收终止分开处理已有明确先例。它的地形映射不是本项目可直接采用的物理修改方案。这里仅借鉴“训练终止如何影响探索/信用”的问题，不移植其改地形训练。
- [Recovery RL原始记录](https://arxiv.org/abs/2010.15920)与[作者机构原文第9章](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-77.pdf)：任务策略、风险估计与恢复策略分工已存在。因此“加安全critic/恢复策略”或把目标改名为风险/CVaR不是新的机制证据。

**有待检验的不同问题**来自本项目代码而非检索空白：[训练终止/学习信号审计](TRAINING_CREDIT_SCOPE_AUDIT.md)新增完整231条指标与96既有分支核对，0物理/训练。原强扰动训练在domain_exit或roll>.7时吸收终止，后来评价允许domain-hold继续观察真实接触；两种对象不同。所有75PPO minibatch接受，不能把阴性归因于更新没执行；advantages未保存，风险被淹没仍UNKNOWN。该发现不是新算法，也不推翻六特征等阴性。

为说明待研究目标，可将真实接触、任务域退出分别记作τC、τD。给定控制模式m（必须包含实际handover）、剩余观察窗h，物理风险可定义为

\[
V_C(s,m,h)=\Pr(\tau_C\le h\mid s,m),\qquad
V_C(s,m,h)=c(s)+(1-c(s))\mathbb E[V_C(s',m',h-1)]
\]

其中h>0，非接触终点h=0取0，接触取1。域退出本身不是c=1，但可能改变后续控制模式；原任务return仍须独立报告。上述是标准有限视界命中概率递推，不是新定理；没有真实续行数据时不得从reset后的状态拼接bootstrap。现有13条受扰无触地均含fallback，只证明该复合控制器的结局，不证明原Actor可自行恢复。

**当前决定**：保留这个可证伪的目标合同/信用截断假设，先明确可操作的同合同比较，再决定是否值得新训练；不立即增加网络、改γ/T或复活旧扫描。用户已明确容扰度先以不摔倒定义，原任务表现独列，无需重复询问该目标。硬件清单问题仍待回复，但不影响本轮只读审计。RA-L新价值仍需实际机制收益及现实任务证据支撑。

- Huang等，2026，*RRT*-Guided Dual-Layer PPO Robust Control and Planning for Autonomous Bicycles in Rugged and Constrained Terrain*，[出版方](https://doi.org/10.3390/s26165087)。本次出版方索引返回摘要及方法/实验/结论段，直接网页访问仍受限，未逐式审全文。上层速度/转向补偿、下层平衡学习已有直接自行车先例。其平台含动量轮且任务不同；不同平台本身不建立我们的方法优势。
- Liu等，2026，*Coordinated Trajectory Tracking and Self-Balancing Control for Unmanned Bicycle Robot Against Disturbances*，[出版方](https://doi.org/10.3390/act15010049)。本次出版方索引摘要明确外环速度/航向规划与内环平衡、饱和速度规划及RBFNN/FSMC；直接页面429，未核对全部条件。因此“有界速度协调兼顾轨迹和平衡抗扰”已有近邻，不能以未读到某细节推断对方没做。
- Leng等，*Preference-Conditioned Multi-Objective RL for Integrated Command Tracking and Force Compliance in Humanoid Locomotion*，[arXiv原始记录](https://arxiv.org/abs/2510.10851)，v2日期2026-03-06。本次重读摘要；先前定向方法阅读见reserve_feature_expansion/CONTRIBUTION_DECISION.md。偏好输入、跟踪/外力顺应取舍与实机已被覆盖。顺应与抵抗侧向推倒不是同一个目标，但“偏好＋外力”这一组合不足以单独构成新意。
- Fahmi等，*Bicycle Acrobatics with Reinforcement Learning*，[arXiv原始记录](https://arxiv.org/abs/2608.00880)，2026-08-01。本次只读摘要：已展示自行车RL多行为编排与实机。因此不以“自行车上学习复杂任务”单独承担新意，也不把其不同硬件表现当我们系统基线。

上述为located evidence；检索不完整。检索词围绕bicycle/velocity/steering/coordination/disturbance与preference-conditioned；不进行缺席性首次主张。条目1/2的摘要足以限制宽泛主张，不足以断言我们的完整协议已被相同工作覆盖。

## 已有内容分别证明什么

|候选贡献|已有证据|必须保留的反例/替代解释|决策|
|---|---|---|---|
|单α中层可调速度—路径取舍|旧完整名义五α轨迹；H75在四工况20/20终点保持|零中层也4/4到达且不摔，仅尾航向约4–6°超原阈值；部分α减速大而路径收益小|保留系统功能；需实际问题与基线差异，不能只靠网络结构|
|恢复教学带来方法收益|旧parent160/192→教学165/192|较强75PPO为164/192；见证对随机成功示范整体净0，均为任务标签|不能主张教学普遍优于普通续训/筛选必要；历史正结果保留，不能换成防摔收益|
|物理容扰特征改善策略|三臂880例完整|没有额外实证防摔救回，部分contact变UNKNOWN不是成功|停止该特征细调/追加种子，不包装正贡献|
|临时改α实现防摔救回|同状态72例，包括48新分支|16受扰/模式全部触地，两个干预0新增|停止该时点反应式调度，不扫时机追正例|
|固定公式解释状态相关能力|旧DEV48及新32受扰排序均AUC1|新联合状态基线.96795，仅5个跨α错序；新F≥80仍10乐观误判|候选解释性排序表征；非可信阈值/物理信息必要性/闭环增益|
|偏好塑造进入状态并改变方向性防摔|旧16受扰/模型，zero4、α0零、α1六通过；新α0二/α1四通过|自身前缀不同；无触地均含fallback；旧α1相对zero仅2新增|当前最值得裁决的系统主张；新几何zero24正在运行，不能提前填结果|

原始入口：本目录CONCLUSIONS.md、PROSPECTIVE_FINDINGS.md；long_baseline_transfer/RESULTS.md；witness_selection_v1/final/RESULTS.md；reserve_feature_expansion/CONCLUSIONS.md；runs/formula_prospective_confirmation_v1/paired_risk_audit/REPORT.md。旧HANDOFF_ZH.md早于较强PPO与见证对照，属于历史交接，不能单独当当前贡献结论。

## 当前创新假设与三种后续选择

**候选问题（idea）**：在无额外平衡机构、冻结底层和真实有界参考接口下，偏好会同时改变任务代价及可抵抗的扰动方向；名义姿态更平稳并不保证更抗冲击。用可解释状态排序及真实完整分支界定这种取舍。这是问题/系统贡献候选，不是新控制算法已成立。

1. **保留系统路线**：仅在新zero对照显示实际增量或有清楚任务代价收益时，围绕取舍与方向性防摔组织确认及实机。需说明兜底的贡献，不能把偏离路径的不摔叫完成任务。当前未证明达到RA-L标准。
2. **闭环容扰估计研究**：将解析量作为先验，估计控制器专属、方向/时长条件的真实防摔区间，需与简单联合状态模型和保守低阶闭环模型比较；目前只是idea。单一±80N标签无法给出F*，不得用本组新标签拟合阈值再称前瞻验证；也不能因想要新方法而恢复已停止的gamma/T扫参。若后续采用，必须另立真正新信息来源和一次性预测协议，当前不启动。
3. **停止新增控制/预测方法，转系统或硬件证据**：若zero覆盖现有抗扰优势且无任务代价优势，则接受当前容扰方法增量未建立；已有名义偏好/稳定性、公式误差、负结果与部署契约仍可复用，不再靠小面板找正例。

当前只执行zero24，不同时新增名义实验/模型拟合/训练。其结束后按预定规则作去留决定；B2/B3/B4不开展。硬件尚未部署，实际传感/控制入口待用户信息，离线等价核对不代表实机验证。

Scientific Agent Skills的scientific-brainstorming/critical-thinking用于问题、证据、替代解释与停止条件分离；另一AI reviewer已独立反审zero优先级，不是人类同行审稿。[软件引用](https://doi.org/10.48550/arXiv.2609.00065)见既有贡献文档，本次未新增或修改该书目元数据。
