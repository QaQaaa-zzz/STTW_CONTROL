# Live Preference Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. No delegated agents are required.

**Goal:** 为原alpha策略建立与已验证live参考/合法交回相同的fresh on-policy采样与完整任务信用接口，先完成零优化工程验证，再决定匹配普通PPO对照。

**Architecture:** 新增独立模块，不改旧held-tail采样和旧task-prefix GAE。合法Actor决策之间的固定回退期间，真实任务奖励按实际upper步数折扣聚合；价值只在合法决策状态查询。复用现有50Hz/four5ms live参考内核、16clean-block交回与raw Gaussian采样。

**Tech Stack:** NumPy、JAX/Flax、现有Torch模型导出和原控制内核；不安装依赖。

## Global Constraints

- 冻结ECBC/ESO/local300、物理.0002×25/.005/.02、351D Actor/352D Critic、原alpha/V3奖励和参考幅度/滤波/slew。
- 原collector/GAE保持历史语义；域退出对新任务是累计遥测，只有contact/fault/10s结束物理。两训练臂将共享同一新任务合同，不能把全部新收益归于支持规则。
- 不在无合法Actor动作的时刻制造训练动作/密度；奖励也不能因此丢弃。未知不能训练为失败。
- raw Gaussian动作的logp在tanh/支持映射之前保存；支持是MDP内确定性参考映射，不改raw动作分布密度。
- 采集冻结一个Actor/critic快照；原alpha不偷换，未来力/时刻只进入物理积分。
- 新接口工程检查最多12个真实10s回合/≤24000实际5ms步、0optimizer：两旧route×alpha.5×0/±32N×profile0/7。没有自动追加/长训练或硬件。
- 采用vmap12、scan50、共享编译缓存；900秒或退出事件检查，中文Bark完成/错误。

### Task 1: 跨合法决策间隔的任务信用

**Files:** Create learning/src/sttw_control/live_task_credit.py; Test learning/tests/test_live_task_credit.py.

**Interfaces:** `live_task_gae(rewards, old_values, actor_valid, active, terminal, *, gamma, gae_lambda)`，输入同一完整episode的upper控制步数组；返回仅合法决策处的advantages/returns、discounted_transition_rewards/duration_steps/continuation_discounts。

- [x] 手算测试：合法动作在0与3，回退1/2的奖励必须进入第0个动作，折扣gamma^3；lambda按实际upper步数lambda^3定义并明确为一种标准时间折扣SMDP估计选择。
- [x] 测试lambda=1退化到完整discounted Monte Carlo，最后一次合法动作之后仍有回退奖励不能丢失。
- [x] 测试连续合法动作与旧complete_task_gae一致；未知终点/跨reset active段/无合法决策/非有限奖励拒绝。
- [x] 实现并运行相关CPU测试。不得将此标准估计器称新算法、消除偏差的定理或容扰证书。

### Task 2: 保护观测的随机live采样器与batch组装

**Files:** Create learning/src/sttw_control/survival_live_collection.py, survival_live_batch.py; tests/test_survival_live_collection.py, tests/test_survival_live_batch.py.

**Interfaces:** `LiveSurvivalCollectionState(state,key,gate,clean_blocks)`; `make_live_survival_collection_step(env,actor,*,horizon_s,task_critic)`与现有动态params/log_std/path/speed/pulse/profile参数匹配。`assemble_live_arm`仅选择真实合法动作，但从完整实际reward构造Task1转移。

- [x] 先测试：失效观测仅用零dummy输入，决策valid=false且不保存伪造action/value；合法交回后保存新真实观测和当前tick，不复用旧输入。
- [x] 外部先采样raw Gaussian和pre-action critic值，再由现有make_reference_step的合法执行/支持映射推进同一四tick内核；保留全部state/history/ESO/key。
- [x] 保留raw_action/old_mean/oldlogp、实际issued_latent、profile、legal/fallback/reentry、完整lower奖励和接触/域遥测；RNG只按声明规则推进，不混称相同行为。
- [x] batch组装核完整physical outcome、raw density、真实tick与Actor351+剩余时间重构；Task1接回退奖励。不把域退出当task terminal，也不将unknown作为负例。
- [x] 与原collector相关CPU测试一起验证旧接口未改；不写新的预测器或改奖。

### Task 3: 独立零优化GPU工程检查

**Files:** Create learning/cli/collect_live_preference_smoke.py及独立runs/live_preference_collector_v1下plan/status/source快照。

- [x] 从认证task64端点及左右原完整task0/physical700源初始化，原log_std/normalizer不改；精确case/模型/源码SHA落盘。
- [x] 先shape，再登记台账/中文Bark，一次12回合0optimizer；错误保留并停止，不覆盖旧run或自动重试。
- [x] 检查真实采样密度/值、reward后缀、合法查询与交回、完整物理/history/clock；结果可能包含摔倒，摔倒不是工程失败。未实际出现的分支覆盖标未验证，不凭单测冒充GPU已验证。
- [x] 输出全部case XY与逐20ms奖励/累计/有符号分量、真实失败终点、采样计数及原状态。只验证接口，不作控制改进或训练充分性结论。

## 后续研究判断

工程完成后，才单独封存有/无同一profile7支持的匹配普通PPO训练设计，原alpha奖励相同、共同live/交回合同、相同初始Actor/critic与预算。新训练必须提供实际reward scalar已加载的TensorBoard HTTP链接和进度，不靠event文件存在声称可见。先核match的普通训练是否能消除任务损失，再讨论能力条件控制增量；本计划不启动64轮长训练，不承诺收益。

## 实际收尾

2026-10-25：首次import失败及0积分保留，显式attempt2完成12回合/22562实际5ms步/0optimizer；全部六共享XY已看，真实交回9次（1475为交回后的决策步），MC与密度审计通过。详见[runs工程裁决](../../../runs/live_preference_collector_v1_attempt2/analysis/DECISION.md)。这些勾选只表示本工程计划完成，不表示匹配训练已启动或论文目标完成。
