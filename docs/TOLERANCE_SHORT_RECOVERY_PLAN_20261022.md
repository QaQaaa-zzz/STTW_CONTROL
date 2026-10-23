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
