# 同状态等幅正负推力：容扰度方向信息审计

same complete state, equal force magnitude, opposite directions; accuracy conditional on discordant task outcomes; correlated magnitudes within states; descriptive posthoc development audit, not physical guarantee or causal learning evidence

| 数据角色 | 有效状态 | 名义失败排除 | 方向结果不同的力对/状态 | 公式方向正确 | 校准多数方向正确 |
|---|---:|---:|---:|---:|---:|
| calibration | 35 | 5 | 51/23 | 39/51 | 37/51 |
| validation | 19 | 1 | 29/13 | 29/29 | 16/29 |

同一完整起扰状态比较±相同幅值；排除本身无法完成无扰任务的状态。仅在两个方向任务结果不同的力对上判断哪个方向更危险；两个方向都成功或都失败不能区分方向。

固定方向基线只由calibration多数失败方向决定：-1。不按validation结果选择方向。

这些是已使用TRAIN题库的事后诊断，validation是状态拆分名称，不是独立训练种子或封存最终测试。不同力档共享状态，不能把每个力对当独立重复。方向预测有效不等于引导训练有效；后者仍保留负结果。

来源：/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/preference_train_difficulty_v1/analysis/summary.json；原始results哈希匹配。新增训练与物理交互均为0。
