Read /home/qy/STTW_CONTROL/research-hub/AGENTS.md and PROJECT_STATE.md.

Current authorized task: Codex_Lower_Interface_Repair.md. Protect live V5.2: read-only, no stopping or hot edits. Isolated R196 semantic diagnosis and at most one B0 fast_turn12s replay, optionally one alpha0 if needed. Reuse saved audit. Prepare fresh210D local-command candidate using existing lower_command/trainer, physical fixed-protocol best and existing PPO rollback. Formal lower training NOT authorized; no auto launch, no installation, no automatic push. Preserve physical/ECBC/ESO/action contracts and immutable R196.

2026-10-10: User explicitly authorized committing and pushing the lower diagnostic delivery to origin/repair/r196-local-lower-interface. This does not authorize training or deployment.

2026-10-10 current task override (user-authorized): execute Codex_Path_Feedback_V1.md using STTW_Path_Feedback_V1.json from ab195e7, isolated experiment/path-feedback-v1. A existing four paired NPZ only; B exactly straight10s and left90_R2 20s zero-upper frozen local300, maximum6000 lower ticks; C preparation only, no training. Preserve original upper best/local300/all old results, physics XML/.0002x25/.005/.02 and lower210 interface. No automatic push, best changes, registration, DVGC/JIT writes, other process changes or old full audits. Shared state rules: /home/qy/STTW_CONTROL/research-hub/AGENTS.md and /home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md, incremental locked updates.

2026-11-04 user task: execute STTW越野偏好控制研究规划书 first; defer continuous-terrain planning and RB integration until STTW offroad completion. This isolated worktree is experiment/sttw-offroad. Read /home/qy/STTW_CONTROL/research-hub/AGENTS.md and /home/qy/STTW_CONTROL/research-hub/STTW_PROJECT_STATE.md; use the existing exclusive ledger lock. Preserve frozen H75/LOCAL300 and original model; terrain assets are explicit new physical domains. No hardware execution; declare each simulation/training budget before execution. Prior dated task scopes above are historical and do not define this task.

2026-11-06 用户要求所有GPU训练/Isaac评测/采集排队：统一持research-hub/.gpu-job.lock且启动前检查全机CUDA计算进程，存在其他任务时等待，不并发挤占，不终止其他会话任务。此要求适用于本工作树全部会话。

## 长时间等待必须结束当前回复（用户明确要求，2026-11-07）
- 当剩余工作需要长时间等待仿真、训练、采集或资源排队时，必须先告知用户“开始等待”、等待对象/原因、预计完成时间或合理时间范围，并结束当前回复；不得持续轮询、反复 sleep/wait 或用无进展的状态回复消耗会话。
- “终止”指结束当前助手回复，不是杀死已授权的后台任务。后台任务保留阶段状态并按既定预算运行；不为等待扩大实验或启动额外任务。
- 用户再次要求推进时，集中查看一次状态；若已完成，直接分析并推进下一项；若仍需长等，通知后结束回复。只有运行异常或必要的短启动核查才再次检查。
- 等待期间若有可立即推进的独立工作，先完成该工作，再通知并结束回复。不得以此延长无实际产出的等待。

## 等待阈值修订（2026-11-07，用户最新明确）
- 只有预计剩余等待超过一小时，才因等待结束当前推进并通知用户；一小时以内继续完成分析和后续工作。此条覆盖此前长等待即暂停的较宽解释。
- 短仿真直接收取完整结果，不以几十秒到几分钟的等待暂停目标。继续避免无意义高频轮询，优先并行完成独立分析。

## 第三章结果台账范围（用户最新要求，2026-11-08）
- CHAPTER3_PROJECT_STATE.md只记录已有结果的实验及结论、证据；不记排队、PID、启动预算、测试计数和待办。运行前合同/运行状态/依赖写各run目录，方法与后续方案写现有设计文档。仍按共享锁维护结果台账。
- 后续原两路线验证必须使用新跨沟跳跃网络和同一张非结构化地面地图；旧update2/平台队列不得恢复。新模型与地图未就绪时等待，不静默回退旧技能。
