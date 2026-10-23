# 单网络负结果：训练长度与有效暴露审计

读取7组原始metrics.jsonl及effective_config；每组连续27次更新=2次critic预热+25次PPO，1,769,472策略转移。计数经源码tolerance_training.accumulate_rollout_totals核对：pulse.fired为fired_now事件累加，不能当独立初态数或成功恢复数。全部force_histogram之和与fired一致。

| 试验/组 | 实际施扰次数 | 前5→后5 PPO平均奖励 | 前5→后5平均域退出 |
|---|---:|---:|---:|
| preference_recovery_train_v1/guided | 7388 | -0.852 → -0.822 | 11.0 → 10.2 |
| preference_recovery_train_v1/shuffled | 7610 | -2.107 → -1.685 | 36.4 → 28.0 |
| preference_recovery_train_v1/uniform | 7589 | -1.795 → -1.590 | 30.6 → 26.8 |
| preference_recovery_calibrated_v1/guided | 7660 | -1.998 → -1.747 | 34.6 → 30.0 |
| preference_recovery_calibrated_v1/shuffled | 7603 | -1.843 → -1.499 | 31.6 → 25.2 |
| preference_coverage_recovery_v1/uniform | 4020 | -0.758 → -0.774 | 13.0 → 13.0 |
| preference_full_tail_v1/uniform | 4016 | -0.842 → -0.917 | 13.8 → 15.4 |

所有组预热Actor保持不变，已有parent KL锚；25次PPO均未记录hard-KL或nonfinite停止。不能将旧早期试验约960次暴露的解释推广到这些后续组。不同组采样分布不同，奖励/退出的跨组高低不是公平策略能力比较。

## 判断

1. 25次PPO不证明收敛，也不证明训练不足。若干组后期指标改善，继续学习可能有空间；coverage和full-tail没有一致改善趋势，不能预期简单加轮数必然成功。
2. 预热、KL锚已实施，不能当作尚未做的新方法。Critic损失仍大也不自动证明critic失配；目标尺度与分布会影响它。
3. 采样曲线不是固定场景成功率，初末窗口对比仅为描述，不做收敛显著性检验。重复使用状态库的数千施扰不等于广泛状态覆盖。
4. 本轮机会排序实验没有Actor更新，其负结果与本报告的PPO训练长度问题分开。

## 下一项有判别力的对照（尚未启动）

选择已完成且具备完整优化器/checkpoint的单网络分支，固定状态库、奖励、采样、父锚与评测，仅将25次PPO延伸至75次；保存50/75边界，用固定开发面板和无扰保留检查学习曲线，主要比较预声明75与25，不按最佳点报告。恢复优化器、RNG和锚游标须核验；若只能weights-only则明确不是严格续训并另设对照。先结合正在运行的双网络回访决定分支，不同时延长所有旧组。

若任务成功率随长度改善，原负结果只能限制短训练设置；若训练指标改善而固定评测不改善，优先处理覆盖/任务目标差距；若无扰退化，需要审视保留约束而非无限加轮数。任何结果都不能直接推断整个单网络架构不可行。

论文主线仍为偏好条件下的经验恢复能力与有效示范教学。容扰度解析特征是否能帮助节省获取成本尚未证实；该审计用于辨认训练混杂因素，不把额外训练本身包装成创新。

[训练诊断曲线](training_curves.png)；[逐组可核对数据](analysis.json)；[复算脚本](analyze.py)。
