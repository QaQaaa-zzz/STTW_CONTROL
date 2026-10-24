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

## 补充：训练终止语义与真实防摔目标的差别（2026-10-24）

这是另一个机制问题，不替换上方准备前缀的历史结论。新增审计`audit_failure_learning_signal.py`读取每臂完整77条指标（2次critic预热＋75PPO），两阶段更新1–77无缺口/重叠；只读复用96条分支，0物理/优化。结果在`runs/failure_learning_signal_audit_v1/{audit.json,contract_authentication.json}`。

|完整训练链|physical标记数|domain标记数|完成片段计数|实际起扰数|接受/尝试PPO minibatches|
|---|---:|---:|---:|---:|---:|
|uniform|328|7811|31471|26040|1200/1200|
|physics|337|7878|31495|26053|1200/1200|
|polynomial|320|7808|31453|26014|1200/1200|

**计数不是互斥分类**：physical标记与domain可能同一步重叠，不直接相加为失败数或算互斥占比；physical也不是实际触地计数。全部PPO更新的hard-KL/nonfinite stop为0，接受仅说明未被优化守卫拒绝，不能证明恢复梯度充分或已学到。

具体合同已核对：原manifest Git5872461的teleop_env/path_command_env/path_command_reward与现文件字节相同；load_environment、termination_flags、save_checkpoint函数AST亦相同，训练short_recovery/tolerance_training与声明SHA相同。标准训练加载原TeleopPhysics，其physical.failed包含真实非轮接触、峰roll>.7或非有限；terminal_contract_v2的domain_exit也吸收终止。path_command_env对physical.failed或domain_exit都用failure_reward替换该策略步奖励；形式为原10s剩余折扣成本尾项（extra5、scale.1、rate_bound8980.2、gamma.997）。剩余时长不同，实际惩罚不同；公式相同不代表逐例优势相同。4s片段超时会bootstrap，domain/physical终止不会因片段处理自动变成可续行状态。

因此，此前训练与后来“contact-only＋共同domain-hold”的物理观察合同有差别。复用最新96条：α0/α.5/α1/zero的非零力完整无触地为2/3/4/4，其中全部都经历domain-handover；它们在旧任务合同下会停止获得后续物理观察。**这些不是训练片段的反事实续行**，不能说训练中的328等事件里有多少本可获救。当前96条没有越.7后仍无触地的样本，不能扩大该反例；既有独立单状态17例contact_continuation中确有转向*.85/−72N越.7后4s无触地并回区间，见原REPORT，不重跑。

保存的标准checkpoint包含策略/优化器/RNG等，不包含rollout storage或逐例advantages；指标也只是聚合。故“风险被平均奖励淹没”“失败得到错误优势”均为UNKNOWN，不能从平均reward、value loss或接受率补推。

**机制假设（未验证）**：现行任务吸收终止可能截断部分仍有紧急防摔价值的续行，使学习不能区分“偏离任务但存活”与“真实触地”。这不同于缺少准备前缀，也不证明它是阴性原因，更不证明改终止必有收益。独立AI反审支持将其保留为待检验问题，反对据此立即开训练。

下一步若检验，必须保持所有方法相同的目标合同：用户已明确容扰度先以不摔倒定义，故将原任务表现与物理存活分列；不得把路径离域后换控制器存活包装成原Actor完成任务。若同完整状态续行多数依旧触地，则假设减弱；若仅靠放弃任务/fallback存活，只支持应急防摔价值；若普通PPO/简单规则在同合同下同样改善，则不能归因于新容扰学习方法。当前不改生产终止/奖励、不扩旧网格、不启动新训练。

## 冻结基线后的实现起点（2026-10-24）

用户要求先冻结双网络→蒸馏单网络→后续分支，已保存9模型及真实继承关系，见[FROZEN_MIDDLE_NETWORKS](../../../FROZEN_MIDDLE_NETWORKS.md)。未来新试验不修改这些模型或重指向别名。

新增独立`failure_contract.py::split_failure_events`只负责区分任务结束、有限窗物理结束、真实接触与工程未知；roll>.7保留遥测，不直接当摔倒。域退出后物理目标仍可继续，但这不自动提供合法续行状态或critic输入。5项有针对性的测试通过：域退出与接触分离、roll越界不冒称接触、非有限不当负样本、固定观察窗不越窗bootstrap、混合batch逐lane语义。

用已保存96条完整分支做0积分事件回读，物理标签96/96保持原样，13条受扰在任务结束后继续无触地的记录得到正确区分。`runs/failure_contract_replay_v1/receipt.json`保存逐例/源SHA/模块SHA。该结果是事件记账验证，不是成功率提升、Actor恢复或新训练已运行；现有训练入口/生产终止/奖励完全未接入新模块。

下一实现阶段须明确定义domain退出后的控制模式与合法观测、有限窗物理回报，以及进入/离开模式时的critic目标；不能仅把done改False，让无效路径观测进入原Actor，或把重置后的状态当续行bootstrap。只有这些条件具备，才声明普通PPO与候选机制的同合同仿真对照。
