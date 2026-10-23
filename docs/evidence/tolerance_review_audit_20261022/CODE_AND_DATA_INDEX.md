# 审阅代码与原始数据入口

以上一次已验证远端提交 `a7036d99e2488fd2cf74ccbc253c3c9986a935bb` 为历史代码锚点；下列文件在收到评审前已经存在于远端。之前报告没有把全部入口集中列出，本次补齐导航和逐例数据。无需复制第二套源码。

| 内容 | 固定版本源码 |
|---|---|
| 真实外力：子步力臂、重叠时间和冲量 | [teleop_env.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/src/sttw_control/teleop_env.py) |
| 随机训练路径与环境槽配对 | [path_command_scenarios.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/src/sttw_control/path_command_scenarios.py) |
| 起扰时刻、方向冻结、力采样和回合状态 | [tolerance_pulse.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/src/sttw_control/tolerance_pulse.py) |
| 本轮权重继承/优化/遥测主循环 | [tolerance_training.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/src/sttw_control/tolerance_training.py) |
| PathBatch、reset、环境装载和模型保存 | [path_command_training.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/src/sttw_control/path_command_training.py) |
| tick奖励与4tick策略步累加 | [path_command_env.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/src/sttw_control/path_command_env.py) |
| 实际路径族、reset和成本样本统计 | [path_command_statistics.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/src/sttw_control/path_command_statistics.py) |
| PPO目标、GAE接口与优化 | [direct_command_ppo.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/src/sttw_control/direct_command_ppo.py) |
| 完整状态分支及原始标签评测 | [evaluate_tolerance_comparison.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/cli/evaluate_tolerance_comparison.py) |
| 实用恢复及自身零力差分 | [analyze_tolerance_recovery.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/cli/analyze_tolerance_recovery.py) |
| 物理外力与脉冲schedule单测 | [test_tolerance_pulse.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/tests/test_tolerance_pulse.py) |
| 训练路径采样单测 | [test_path_coverage_sampler.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/tests/test_path_coverage_sampler.py) |
| 奖励组件与终止上界单测 | [test_tolerance_reward_integration.py](https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/a7036d99e2488fd2cf74ccbc253c3c9986a935bb/learning/tests/test_tolerance_reward_integration.py) |

## 本次新增的可审查资料

- [第一轮1152例原始结果](reward_shaping_cases_1152.json)
- [第二轮864例原始结果](pulse_curriculum_cases_864.json)
- [标量能量模型逐例预测、标签与误差](scalar_model_energy_details.json)
- [等效输入幅度计算](equivalent_input_audit.json)
- 每组全部25轮原始metrics，含路径族实际样本占比、失败计数、力档事件数、成本和PPO优化指标：
  - [uniform_alpha0](uniform_alpha0_metrics.jsonl)
  - [guided_alpha0](guided_alpha0_metrics.jsonl)
  - [uniform_alpha1](uniform_alpha1_metrics.jsonl)
  - [guided_alpha1](guided_alpha1_metrics.jsonl)

本次补充 [策略步时间尺度测试](../../../learning/tests/test_tolerance_policy_step_accounting.py)，覆盖baseline/static/dynamic的真实policy_step聚合路径。全部28项相关测试通过，未改变历史训练算法。

逐例JSON不含完整物理轨迹或可加载模型；全量NPZ/完整快照/权重仍在原机。数据是开发证据，重复零力分组和相关上下文不能当独立样本。
