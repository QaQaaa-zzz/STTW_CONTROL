# 已授权短恢复分支训练：设计与执行计划

用户在外部评审核查后明确要求“开始”。本轮执行此前提出的真实TRAIN前缀分支、4s恢复片段和名义能力保持。复用现有隔离worktree，不改冻结local300，不启动独立修复agent。使用brainstorming、writing-plans和TDD技能组织实施；用户已授权整体方向，工程参数在本计划具体化，不再增加确认轮。

## 设计取舍

选择原双端点baseline u50为父模型，保留旧任务奖励。相比同时切换V3/update325，这一选择减少架构/奖励变化；相比再跑完整20s回合，集中真实恢复事件。结论只针对本轮组合干预，不单独归因于critic预热或KL。

1. 从每端点512个新随机TRAIN路径采集父策略真实无扰轨迹（route episode100，沿用已冻结demand_exposure_v2，避免使用DEV快照）。每条在0.8–3.5s的控制对齐随机时刻保存完整状态，包含物理、ESO、local300、中层历史/滤波/路径/时钟。父轨迹延长至8s，均匀采集名义观测/父动作均值，用作保留锚；故障状态不入可训练bank但保留计数/来源。按实际活动tick计全部采集成本。
2. 训练片段为分支后4s。80%片段施加一次0.1s的±2/4/6/8N侧向力，其余20%为无扰片段。方向在起扰时冻结。uniform/guided复用原力选择规则；不在本轮增强引导或加容扰度奖励。每次reset从该端点合法bank独立抽样，原闭环时钟不归零，副本不互相污染。
3. 原任务截止时间保留20s；4s结束是采样截断而不是原任务成功/失败。Critic仍看原任务剩余时间，截断处用最终观测V bootstrap并阻断跨reset GAE；物理/跟踪域失败按原终止逻辑，不bootstrap。全任务失败罚仍按原剩余时间，未改变奖励含义。
4. 原父actor/critic/logstd weights-only＋freshAdam。2轮critic-only预热验证actor/logstd字节不变，再25轮PPO。Actor LR3e-5、critic3e-4；名义观测上KL(parentGaussian||learnerGaussian)系数0.05，每minibatch512条确定性循环锚样本，父网络与锚无梯度。标准PPO自身KL监视继续保留。
5. 每组512env×128步×(2+25)＝1,769,472策略步，4组共7,077,888；warmup和PPO分别计数。最多2worker。单seed首轮，不把它当多seed结论。采集、GPU smoke、评测另计，耗时不设默认上限；无隐式续训。
6. 先CPU行为测试，再小GPU bank＋后缀/冲量/timeout smoke，再生产bank和训练。所有长任务登记Bark-only、TensorBoard新logdir和effective_config；检查真实reward标量。最终先复用同864例DEV框架对照parent/uniform/guided，保留完整20s、XY、原始阈值与连续指标。无预设优越；若稳定保持和恢复有收益，再声明独立场景/训练seed确认。

## 实施顺序与验证

- 新bank采集模块与CLI：复用load_environment、PathCommandActor、随机路径和真实policy_step；vmap/scan缓存；写完整bank、名义锚、逐状态身份、实际prefix/suffix成本和reject状态。
- 短恢复wrapper：复用PathBatch.advance_one(auto_reset=False)，只增片段计数/脉冲/截断/合法reset；测试完整状态复制、一次施力、工程故障不reset、timeout最终观测和终止掩码。
- 扩展现有DirectCommandPPO的可选父KL锚，默认关闭不影响旧run；复用value_only_update。测试KL方向/零点/actor梯度、父锚无梯度和关闭兼容。
- 在现有tolerance_training入口增加显式short_recovery模式，保留父身份验证、原reward与20s时钟、freshoptimizer和有限步数；新增bootstrap语义和暴露/路径×力实际活动样本日志。
- 冻结具体plan与effective_config后启动采集/训练；核对新checkpoint和实际进度，不覆盖旧run。

本计划不是已完成实验结果；阶段状态在共享PROJECT_STATE维护。实现时发现合同不一致必须先更正再启动，不因测试通过宣称控制效果有效。

## 已完成实现与预检（启动快照）

代码644a591及其父提交已实现：TRAIN前缀bank/名义锚采集、4s wrapper、原20s时钟、超时最终观测bootstrap、两轮critic-only、父高斯KL、联合路径×力档活动tick统计、effective_config、有限调度和最终评测。37项相关CPU测试通过（后续调度补充23项复核亦通过）。

首个bank smoke在加载父record时KeyError(actor)，发生在仿真前，0物理步，错误及通知保留；改为复用已有完整父模型解析器后，8个GPU TRAIN前缀全部采集成功（1920lower ticks）。随后均匀/引导各8条4s真实GPU片段通过，共12800lower ticks，两组各8次片段截断、5次非零推力，最大完整冲量误差0.0008211Ns，原时钟保留。这里不是性能结论。

实际有限运行：`runs/short_recovery_train_v1`，准备控制器PID482564；正式bank采集alpha0/1 PID485649/485650各512contexts，状态见`prepare_status.json`及`banks/alpha{0,1}/status.json`。采集后自动启动最多2worker训练和独立TensorBoard；首reward产生后逐job核验HTTP。当前发布快照仍在采集，尚无训练曲线或控制效果结论。邮件/自动修复禁用，Bark已登记。

## 最终checkpoint错误与恢复（原始失败记录保留）

原alpha0两组均完成27次learner更新，但旧保存条件仅`iteration%5==0`，最后保存u25，缺失u27；调度器据此报错，alpha1尚未启动。已修复为每5轮、最后一轮或停止时保存；新增27/25/2轮回归和新旧checkpoint恢复测试，共32项相关CPU测试通过。后续checkpoint保存父KL锚循环位置，避免恢复后重置保留采样进度。

显式恢复入口`learning/cli/recover_short_recovery_campaign.py`启动新attempt `runs/short_recovery_train_v1_recovery1`，旧bank/anchors复用，旧失败状态不覆盖。alpha0从经hash验证的u25恢复policy/optimizer/随机数/拒绝计数，按原accepted_minibatches日志推导并验证KL游标；从已存route counters+1的完整bank状态开始，明确不是仿真逐帧接续。只补26/27两轮，alpha1按原配置从零完成27轮。

计费：旧attempt实际3,538,944策略步；保留模型血缘3,276,800；丢失未保存的262,144步保留为成本。恢复attempt新执行上限3,801,088步，合计实际7,340,032，而最终模型血缘仍为原声明7,077,888步。不会把重算费用隐藏或计作新独立证据。

恢复TensorBoard独立日志 `http://localhost:6012`，实际第26轮reward已通过HTTP核验（uniform −.03955267，guided −.04086924）；历史6011保持。Bark仅通知，自动修复与邮件仍禁用。恢复后自动衔接864例完整任务评测。

## 结果后续：同状态TRAIN方向扫描（2026-10-22）

当前864例DEV已完整结束。α0 parent/uniform/guided原口径133/138/136，α1均141；短回合+critic预热+名义锚联合协议有效迹象来自α0，不归因于单个组件。容扰度引导仍未优于均匀。固定公式在父策略α0非零扰动上的失败排序AUC约0.450（相关DEV描述性），不因其不精确反复调小数，也不盲目增强同一排序。

下一步先测真实闭环容扰度，新增 `runs/train_margin_scan_v1/declaration.json`：只取原α0 TRAIN bank中预先固定的32个等间距index，冻结uniform u27作suffix，不训练。每状态同一完整父策略前缀，固定世界方向施加0.1s脉冲，力档0及±2/4/6/8/10/12N；416例，最多1,664,000lower ticks。原20s时钟与任务截止不变，逐环境到期停止，不auto-reset。先用2状态×3力档工程核对，最多24,000ticks，之后只执行一次正式网格。

三层标签明确区分：physical_survived_horizon（域退出提前停时为UNKNOWN）、task_recovered（无物理/域失败，达到原目标与末1秒保持）、performance_success（再加原工作侧倾上限）。完整父前缀已在bank采集时核验，后缀换Actor不宣称单Actor全程可达。离散方向边界复用empirical_tolerance.summarize_direction；允许非单调、UNKNOWN和右删失，不把最大通过档称精确/连续安全边界。

每第4个选中状态保留为validation，其余calibration，暂不拟合。此阶段仅收数据与判断代理能否解释真实任务恢复；不新增Actor特征、不改奖励、不解冻底层。状态/配置/hash/实际物理量落盘，使用现有Bark-only中文项目通知。扫描后根据任务/性能边界差异与排序再决定是否需要轻量校正或经验辅助目标，不先承诺解析引导有效。
