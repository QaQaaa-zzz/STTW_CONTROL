# 校准采样尺度：容扰度＋单网络偏好恢复训练

> 2026-10-23 完整回收：[任务复盘与XY图](completed_review/REVIEW.md)。新增1240例全部完成，2,451,416底层tick；400例父策略按完整身份复用。四策略唯一名义任务均30/30。强扰动u325/guided/shuffled/uniform恢复189/189/191/186（各200）；guided与父策略逐例0改善0退化，不采用guided，停止自动代理参数搜索。shuffled的小幅改善仅为单种子开发观察。以下保留启动时的协议与历史进度快照。

目标不变：冻结 ECBC＋ESO＋local300，通过一个α条件中层策略保留速度—路径偏好，同时提高受扰任务恢复。公式提供方向性难度先验；现在检验更合适的训练关注区间，不改公式的γ/E_b，也不把经验采样中心称为安全边界。

## 为什么做这一轮

[540例TRAIN诊断](../preference_recovery_pilot/train_difficulty/CONCLUSIONS.md)表明风险有排序信息，但旧risk≈1采样偏容易。校准数据中风险[2,3)失败27/82，故选探索性的target=2.5、bandwidth=.5；已看过的检查数据与旧DEV面板都不称未见最终测试。前一轮真实引导未胜过打乱对照的负结果保留。

## 哪些变化，哪些条件相同

新的guided/shuffled均从完整u325父Actor/Critic/log_std开始，fresh Adam；同一320上下文TRAIN题库、V3、原始10秒任务、4秒片段、local300、物理0.0002×25/底层.005/中层.02、351/352维输入和2维参考修正。学习种子918301、脉冲918203，力±8/16/24/32N、0.1秒、80%启用；每组2轮Critic预热＋25轮PPO，固定最终u27。

采样中心1→2.5，带宽.2→.5；引导强度.7与α内打乱种子918204保持。实际α和力档在相同reset计数下匹配，提前终止后的真实样本占比继续记录。[4096抽样审计](sampler_intervention_audit.json)证明：risk2–3占比uniform30.7%、guided44.5%、shuffled30.2%；三组启用标志、力档、实际α与名义状态匹配。这是分布改变证据，不是训练收益。

## 对照复用与成本

uniform训练使用已完成的原u27，验证父权重、配置、题库、种子、更新数及全部相关学习合同，逐参数核对最终导出与learner。实际320状态概率表在修改target/带宽后逐项相同。[复用证据](uniform_reuse_receipt.json)。这不是新训练种子或重复实验。只允许复用uniform，改变种子/力档/奖励/父策略等则拒绝。

新增两组训练共3,538,944策略转移、最多14,155,776底层tick；另有512环境×8策略步工程验证16,384tick，约41.92秒，完整非零脉冲432、未记录工程/物理/域失败。保留旧训练成本，不把复用对照记成新的零成本获得。

只用一个GPU训练进程：此前本机两个并行进程约88秒/组/轮，单个约42秒/轮，没有聚合吞吐收益。本轮按轮数执行，不设墙钟截断；不自动加训。

## 训练结束后自动做什么

已启动独立的等待/评测进程。认证两个新最终checkpoint和复用uniform后，执行同一完整10秒的moderate/strong/straight面板；1240例新评测、最多2,480,000底层tick，400例原父策略结果按完整身份复用。uniform checkpoint会随候选重新评测，不把这当新训练重复。

分别报告任务新增/退化、域退出、工作侧倾、连续路径/速度/姿态误差和名义30个唯一工况的偏好行为。约厘米级阈值擦边保留原标签、补连续误差；训练回报更高不作为胜出依据。新候选未采用，最终测试封存。

## 实时进度

- TensorBoard：[http://localhost:6009](http://localhost:6009)，新运行独立日志；guided实际`train/mean_step_reward`已通过HTTP核验至至少step4，前两轮Critic预热Actor逐位不变，随后已执行PPO。每轮采样约41秒，shuffled尚未启动。此页是发布快照，后续以实时文件为准。
- 训练：`runs/preference_recovery_calibrated_v1/status.json`；逐组`guided/status.json`、`shuffled/status.json`与`metrics.jsonl`。
- 后处理：`runs/preference_recovery_calibrated_post_v1/status.json`，当前等待训练。
- 源码2a1ced8，训练控制器1599712，首组1599744，等待/评测1599776；PID仅表示此启动记录，不代替完成收据。
- 中文Bark已登记训练、逐组、工程与后处理状态；邮件和自动Codex修复保持关闭。正常完成及报错分别通知，不声称能检测所有静默进程死亡。

```bash
cat /home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/preference_recovery_calibrated_v1/status.json
cat /home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/preference_recovery_calibrated_post_v1/status.json
```

原始协议：[训练声明](training_declaration.json)、[实际训练配置](training_config.json)、[后处理声明](post_declaration.json)、[工程结果](engineering_receipt.json)、[启动记录](launch.json)。本轮实现与相关回归34项CPU检查通过；实际512环境工程验证完成。CPU通过不等于新策略已提升。
