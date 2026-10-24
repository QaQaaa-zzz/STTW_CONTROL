# 论文证据链与路线取舍

2026-10-24，有界原始文献核对＋现有报告/产物核对。不是全面新颖性检索、录用预测或新实验结果。当前运行事实仍见共享台账。

## 不能把哪些部件本身当新颖性

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
