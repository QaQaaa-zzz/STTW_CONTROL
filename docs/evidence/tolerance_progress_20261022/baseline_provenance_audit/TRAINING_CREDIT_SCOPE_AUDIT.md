# 阴性训练结果能否归因于缺少受扰前准备？

结论：**固定bank三臂没有准备前缀的直接信用分配成立；以前从未训练过准备前缀不成立；缺失准备是阴性原因尚未建立。** 本核对0新训练/物理，不把新的解释自动当成新方法。

## 固定bank三臂确实优化了什么

`reserve_feature_expansion_v1/{training_declaration,extension_declaration}.json`声明4s片段、原任务10s，力为±16/24/32/48N、固定low_speed_train_bank、uniform/physics/polynomial。`short_recovery.py::PreferenceRecoveryBatch.reset()`直接取固定bank状态及历史，令pulse起点为该状态当前物理时钟；下一个策略步即施扰。没有当前策略真实执行的受扰前prefix。retention anchor约束共享网络动作，但不是接受扰动回报的准备轨迹。

若写作

\[
J_{\rm snippet}(\theta)=\mathbb E_{x\sim\mu_{\rm parent}}V_\theta(x;F),
\]

该重置分布不因当前Actor的准备动作而改变。片段策略仍会改变受扰后的状态，且训练产生的函数泛化也可能改变完整回合行为，不能说它在任何意义上都无法影响进入状态。

与之相比，完整回合目标可写为

\[
J_{\rm full}(\theta)=\mathbb E_{\tau_{\rm pre}\sim p_\theta}
[R_{\rm pre}(\tau_{\rm pre})+\gamma^kV_\theta(x_k;F)].
\]

这里同时有准备动作/进入状态分布变化与接续控制的作用，且必须保留前缀任务代价。仅写EμθV会遗漏该代价。增加这一通路不保证收益，更不是新的策略梯度理论。三臂阴性只适用于已测固定起扰分布/权限/奖励/训练协议下的特征增量，不是所有提前协调方法不可能的证明。

## 反证检查：完整回合弱脉冲训练已经实际做过

选取真实完成链`runs/pulse_curriculum_train_v1/uniform_alpha0`，并核对四臂曝光汇总；不是把标定面板误当训练。

- manifest代码`d404ca4ac524ccbffe602009f6b811ed47ab1d46`；`PulseBatch.reset()`调用`base.reset()`从完整任务起点开始，当前可训练Actor运行到起扰。
- `tolerance_pulse.py::schedule/activate/advance_one`：50%回合施扰，0.8–3.5s且20ms对齐，持续.1s，力±2/4/6/8N，在起扰时锁定世界方向。
- 旧执行链PPO每步act→process_env_step，128步rollout后compute_returns/update；起扰前动作没有被剔除。γ=.997、GAE λ=.99允许同rollout内跨起扰传播，跨rollout依赖bootstrap/价值学习，不能说每次准备都得到完整扰动回报。
- uniform_alpha0实际完成25更新、1,638,400策略转移；metrics连续1–25，曝光958次，训练物理失败0。四臂起扰次数958/960/978/969，全部物理失败0。guided不是未执行，但没有一致响应优势。
- 冻结local300身份与当前一致（d408a94e…），但旧父策略baseline-u50端点、历史奖励（reward_version缺省）、20s、1.8–2.6m/s、弱力和25PPO，与当前H75/V3/4s固定bank/强力75PPO不同，不能称单变量消融。
- 旧864例评价无观察到物理触地，不能用于证明真实防摔改善；任务/域退出负结果保留。`pulse_pilot_v1_attempt2`明确training_steps=0，是标定而非准备训练。

证据入口：该臂plan/manifest/effective_config/metrics/status；`pulse_curriculum_train_v1/training_exposure_summary.json`；`pulse_curriculum_post_v1/evaluation/{manifest.json,report/REPORT.md,recovery_diagnostics/REPORT.md}`。独立AI reviewer核对旧版本训练/GAE链；主agent另核对manifest、完整25条metrics、实际曝光、reset源码与三臂声明。不冒称重新执行了训练或NPZ全量复算。

## 对下一步的限制

历史证据只说明：“完整准备信用通路”与“真实防摔挑战”尚未由这两组协议合并隔离验证。它不说明补足这一点必然有效，也不构成新颖性证明；不能为了保留喜欢的路线忽略旧完整回合阴性。

当前不启动新full-episode强扰动训练，不恢复原六特征/即时切α/固定速度A3。若以后考虑准备导向研究，必须先回答可实现参考权限、与简单快速行驶/参考整形的差距、真实防摔标签及公平曝光；新论断不能只靠“之前训练不足”。现有名义迁移收益与物理抗扰负结果分别保留，硬件暂无程序不是数学或仿真结论。

本审查采用Scientific Agent Skills的critical-thinking/brainstorming并由独立AI反审，不是人类同行评审。
