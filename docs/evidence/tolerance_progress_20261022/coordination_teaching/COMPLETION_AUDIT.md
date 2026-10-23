# 已授权研究任务的完成核对（2026-10-23）

按用户最新要求，目标为完成容扰度路线的推导与实证检验、推进单网络α恢复能力并解释任务及结果。允许失败，以总体改善为核心；不要求公式一定产生正结果，不要求所有场景成功。本核对不把实车验证、发表论文或任意在线α切换加成已完成事项。

| 要求 | 已检查证据 | 判断 |
|---|---|---|
| 容扰度公式正确性与实际含义 | TOLERANCE_EXTERNAL_REVIEW_AUDIT；本次重跑代数/方向/能量及时间步测试 | 简化模型下核查完成；非真实闭环安全证明 |
| 实际检验公式训练价值 | calibrated/completed_review/REVIEW及交接中匹配guided/shuffled/uniform结果 | 已完成，未证明额外收益，负结果保留 |
| 单网络α任务继续推进 | teaching代码；固定Actor SHA fe8b4cc0…；standard_dev与repeats报告 | 获得恢复收益且保留所测偏好的研究候选 |
| 评测与重复性 | 七阶段实时status全部completed且与成本receipt SHA一致；前瞻paired重新计数160→165/192；三优化seed完整报告 | 声明评测完成；不是独立环境重复 |
| 不回避代价与失败 | working指标、12改善7退化、八源轨迹诊断 | 保留原标签，不追求逐例全胜 |
| 任务解释、结果及下一步交接 | HANDOFF_ZH含任务例子、证据映射、候选身份、负结果与适用边界 | 已交付 |
| 代码与精简结果推送 | experiment/tolerance-calibration-20261022；已有各逻辑提交push成功 | 已发布；原始权重/轨迹留服务器 |

本次CPU检查21项通过（tolerance_algebra、directional_tolerance、tolerance_model_energy、tolerance_policy_step_accounting、recovery_teaching、recovery_intervention）；它们验证公式/数据/干预相关行为，不代替物理评测。七阶段状态文件SHA与cost_receipt逐一一致，主Actor完整SHA重新匹配，前瞻原paired条目重新计数。

研究结论：单网络恢复教学在所测任务中带来总体成功率提升，固定α偏好保留；容扰度诊断有信息但训练增益未建立。按用户最新要求，局部反例不是追加训练的必要条件。没有把未证明的首次性、物理安全、在线切换、实车部署或论文录用算作成果。更广泛论文贡献验证属于后续研究问题，当前代码、实验、分析和交接均已有对应产物。
