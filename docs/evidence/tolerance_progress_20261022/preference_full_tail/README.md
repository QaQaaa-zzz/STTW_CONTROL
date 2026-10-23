# 完整恢复尾段训练

2026-10-23已启动，PID1844537；状态以服务器runs/preference_full_tail_v1/uniform/status.json为准。单组uniform，2critic+25PPO，1,769,472策略转移；只改4秒片段上限为10秒，保持原任务10秒截止。原320状态库、u325、V3/local300/均匀采样/KL锚不变。旧4秒uniform作匹配交互量历史对照，非独立重复。真实脉冲暴露量不要求相同，需随结果报告。

[设计](DESIGN.md) · [实际声明](training_declaration.json) · [工程收据](engineering_receipt.json)。512lane8步工程检查已完成，16,384ticks，0工程/物理/域失败；不等于训练提升。独立TensorBoard http://localhost:6013 已HTTP200，首轮奖励标量待核实。本页为启动快照，不自动更新。
