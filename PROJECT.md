# Preference V5 Stage2 completed, not qualified (2026-10-09)

User explicitly overrode the failed Stage1 gate to try Stage2. The separate run `runs/preference_v5_stage2_override_20261009` restored both complete update40 learners and finished update120 plus matched update80/120 six-case evaluation. Alpha0 brakes and limits fast-turn roll while retaining more steering; alpha1 retains more speed but weakens steering and still exceeds the common roll bound. Final joint speed/steer/heading hold is alpha0 0/6 and alpha1 1/6 at10s; no method passes fast_turn heading hold at16s. The result is complete, unqualified and not adopted; no250/500 extension. Full actual-control-first and XY evidence: `runs/preference_v5_stage2_override_20261009/analysis/REPORT.md`.

Remote-review evidence including Stage1 update20/40, Stage2 update80/120, compact 5ms trajectories, numerical CSV/JSON, control/XY/reward figures and TensorBoard scalar export is under `docs/evidence/r196_preference_v5_20261009/README.md`.

---

# User amendment — total250 and no wall-clock cutoff (2026-10-08)

The user extended each upper endpoint to250 total updates and then explicitly cancelled additional compute/wall-clock budget limits. Preserve the250-update endpoint, metered elapsed time, original safety/nonfinite checks and no automatic retries/extensions. Stage150 evaluation is disabled through the already-supported runtime_control; reuse stage60 and evaluate at250. Automatic successor: runs/smooth_v4_extend250_20261008, launched by learning/cli/continue_smooth_command.py after the current worker exits. Restore full upper learner (Actor/Critic/Adam/std/RNG/counter) from the current fresh run's last checkpoint, never physical-best for training. Prepared physical/ESO/history states reset because old checkpoints do not serialize them; first unused episode keys avoid recycling training schedules. No further value-only warmup.

15 relevant tests passed including exact learned-state/Adam restoration and disabled wall cutoff with continued time metering. Shared research-hub/AGENTS.md records the STTW future default. This supersedes earlier150/wall-cap requirements below; historical attempts and reports remain immutable.

---

# User amendment — fresh upper training 150 each, 2026-10-08

Supersedes the warm-start60 plan below. Both endpoint Actors/Critics/Adam/std initialize fresh (seed81); no policy weights transferred. Frozen R196 lower remains fixed alpha1. Authorised budgets: train4500s/endpoint, compile600s, evaluation1200s, total10800s. Two initial value-only rollouts remain separate, then at most150 PPO batches. Checks at60/150; user cancelled stage20, original six10s cases at60/150 with separate16s fast_turn tail. No automatic extension. New run: runs/smooth_v4_fresh150_20261008_run03; live research state in /home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md.

Warm-start run cancelled by user after4 alpha0 policy batches, no alpha1 batches; last saved policy checkpoint0 (warmup Critic updated). No warm-start model is used in the fresh run. Related12 focused checks passed, independent saved-rollout reward reconstruction error<1e-8. Control improvement remains unverified pending actual fixed cases.

---

# R196 SmoothV4 implementation plan — 2026-10-08

Source implementation: 6cdb3ce. Numerical authority: docs/smooth_v4/attachment/*alpha[01].json. Independent endpoint actors warm-start at 250/143; frozen R196 lower_alpha=1; reset critic/Adam/std. Preserve all physical/control/action interfaces.

- [x] Locate true implementation and run saved NPZ audit, zero new physics.
- [ ] Implement component-protected reward, causal 20ms offset differences, same-episode Actor regularization and specified command distribution.
- [ ] Check against independent NumPy reference, pair/reset semantics and initialization; no full repository or lower necessity rerun.
- [ ] Value-only 2 rollouts then at most 60 policy batches/endpoint; report accepted epochs separately. Evaluate at 20 and 60; six original 10s cases plus fast_turn16s without replacing 10s conclusions.
- [ ] Deliver actual physical/command comparisons, timing, failures and limits; update shared research state and local commit.

Compute ceilings: compilation600s, training1800s per endpoint, evaluation1200s, overall5400s. Existing prepared bank reused. No new seeds or automatic extension. TensorBoard must expose actual formal reward.

---

# 2026-09-28 Direct command V3 approved execution plan

Numeric authority: learning/configs/STTW_Direct_Command_V3.json. Approved design: docs/direct_command/STTW_Codex_Direct_Command_V3.md. This isolated task replaces V2; no Phi/kpsi learner or analytic action allocator is imported. Parent 92afde6; reuse teleop physics, exact reference integration, same-state ECBC previews, actuator and RSL internals. Local commit only, no push.

- [ ] Pure action/observation/reward and causal schedule modules, tests against supplied numerical reference and fixed clock semantics. 345/346 observations, alpha in context, 2 raw Gaussian latents, original-command-relative corrections only.
- [ ] Independent shared RSL adapter, 69186 Actor parameters, per-group LR/gradients, full and per-alpha KL, epoch rollback at >.03/nonfinite, complete checkpoint identity and RNG.
- [ ] Integrate 4x5ms wrapper with existing physics-only kernel. Eight declared prepared full states, fixed per-episode slew and family PRNG folded with env_id/episode_index. Preserve finite terminal and final pre-reset observation.
- [ ] Meter all new compute: compile300s, preparation/interfaces/smoke120s, pilot1200s, review180s, total1800s. Additional supervision, incremental per-second review writes. No implicit expansion or prior model load.
- [ ] Interface zero-policy checks, 8x16x2 smoke, fresh512x128x<=20 pilot, last-completed main and seed88001 each B0/alpha0/alpha1. One-second compiled chunks for review avoid per-20ms host tree copies, retaining 5ms evidence and true endpoints.
- [ ] Same-state alpha sensitivity without physics, independent reward reconstruction, physical PNG/PDF/CSV/report, limits/failure/cap/grad/KL/timing. Update current validation and methods ledger, local logical commit.

Source-reading/code-writing time is separate from compute. V2 is preserved and already stopped at its own single-episode review limit after 20 updates; no V2 continuation is part of this task. V3 has a fresh bounded budget and fresh learner. Skills: using-git-worktrees, writing-plans, test-driven-development and subagent-driven-development for independent modules, verification-before-completion.

---

# 当前实施更新：不对称速度代价与固定几何路径训练（2026-09-18）

新入口为 `asymmetric_priority_rho34.json`（主候选）和 `asymmetric_priority_rho10.json`（单变量对照），共同使用 `ppo_asymmetric_priority.json`。α=0允许有限欠速以贴原始几何路径，α=1优先速度；超速权重和0.05m/s过程带对所有α固定，最终速度带为[-0.20,+0.05]m/s。奖励方向逻辑集中在`tracking_reward.py`的显式新objective，旧objective与冻结重评分身份保持兼容。

每回合正式任务10s；任务前3.5s以base scale 1.0真实闭环准备稳定直行，任务时钟归零时保留车辆、ECBC/ESO、执行器和10帧历史，学习任务切换到base scale 0.8。训练reset复用预计算的完整准备状态，716,800次准备计算转移单列，不计入每组6,291,456正式训练转移。40/40/20任务分别为可行名义、有限加速转弯后回归、缓弯中单次±2N/0.5s侧向扰动；核心左右场景的生效yaw参考约2.695s归零、积分转角约1.71rad。

RSL保持三层128 ELU、固定3e-4学习率；全采样高斯平均KL超过0.02时回退Actor、Critic、log-std和Adam状态。固定开发比较覆盖普通加速、核心左右冲突、缓弯侧扰动，在初始化、更新1/24/48比较原始scale1.0 ECBC+ESO、scale0.8零残差消融和α=0/.5/1策略，并记录欠/超速峰值、积分和越带时间。工程短跑已通过，正式运行状态以新运行目录为准；训练和存活不代表任务合格或安全保证。

# 当前实施更新：固定学习率、可比选模与精简日志交付（2026-09-16）

用户上传bc390691的64轮运行：更新实际执行，但训练片段best为update_0001，12标准回合均未完成最终跟踪。当前新增入口为`ppo_geometric_stable.json`；几何reward/task不改，alpha继续入网，RSL3.2保留。固定LR3e-4、更新KL回退、1024×128×48，相同6,291,456训练预算。训练中不增加验证；阶段末用`select_best.py --fixed-evaluate`比较声明候选（含zero/first/last），不强制后期胜出、不覆盖原训练best。`reward_review.py --selection ... --compact`输出精简包。工程检查与运行命令见learning/README与VALIDATION新增节；尚无新正式训练或实车证据。下方旧当前段落为历史快照。

# 当前实施入口：几何路径／速度奖励修复（2026-09-16）

本节覆盖下方历史时间位置任务的当前入口；用户明确授权实现并推送，正式训练由用户执行。
α直接输入Actor；`timed_reference.mode=geometry`仅复用随机指令生成固定全局曲线的代码，不要求按时到达原参考点。
原始曲线只在reset积分一次；连续局部投影给出几何横向/航向误差，速度仍跟外部时间指令。原始yaw-rate不再作为速度组奖励。
`tracking.objective=geometric_huber`使用归一化速度/路径Huber代价，保留10:1有限偏好、过程容忍、共同roll要求与固定残差权限。
恢复正奖金取消；第一次3秒超时扣5，超时且待回归另扣2/s；超时标志不因晚恢复清除。零残差仍是同新参考/执行器配置下的ECBC+ESO。
新入口：`geometric_reward_recovery.json`、`ppo_geometric_reward.json`、`geometric_reward_panel.json`。新Actor280维，旧timed310维不可续训。
保持RSL-RL 3.2.0三层128 ELU、4096×24、64轮，共6,291,456转移，额外64×1400预热；按采样更新前模型的训练奖励选best，无训练内评估。
训练结束由用户显式运行`learning/cli/reward_review.py`：4场景×3alpha=12残差回合，4条基线物理轨迹跨alpha重评分。
自动生成逐步/累计奖励、分项、真实/估计/请求速度、全alpha叠图、交叉评分、恢复门槛与`review.zip`，具体命令见learning/README.md顶部。
本地83项纯数学/JAX测试通过，完整物理测试交由仓库CI。未运行正式训练、目标GPU性能或实车；不宣称任务已学会或安全保证。
下方旧配置、历史结论和不可变运行证据保留，不能混比不同奖励的回报。

---

# STTW_CONTROL

## 当前新增实施入口：α输入网络的几何路径恢复（2026-09-15）

用户明确确认：**α必须输入残差Actor**。本轮未采用外部偏好参考管理器；α只改变速度/原始路径的有限偏好和过程容忍，不改变两路残差权限或roll/最终回归条件。

新实验入口：`learning/configs/path_priority_recovery.json`、`ppo_path_priority.json`、`path_priority_standard_panel.json`。旧command、T150/T250、门控和固定侧倾实验及其冻结配置不修改；下文旧“当前”段落均按日期追溯，不覆盖本节的新实现说明。

- 保留ECBC＋ESO、200Hz、原XML物理子步与执行器；残差前轮角速度±1.5rad/s、后轮轴速±10rad/s。无α权限门控；零残差回归同配置基础控制输出。没有把前轮改成转角动作。
- 原始全局平滑弯道＋可见分段速度参考＋随机转向/侧向力/后轮负载。未重定位、平移或重定义原路径来制造回归；不把yaw-rate跟踪冒充路径跟踪。当前配置是左弯第一阶段，左右转/不同曲率/多训练种子泛化未验证。
- Actor仍256→128 LeakyReLU、两输出tanh。新任务为27字段×10帧＋10mask＝280维，包含原15字段、3路径字段、α、上一步两路残差和6个回归状态字段。无真实外力幅值/结束时刻输入；定位和前向速度估计接入实车仍须验证，轮速代理不是真速度。
- 奖励见`tracking_reward.py`：宽窄两档正向速度/几何路径跟踪、Huber尾部、有限容忍、独立roll与roll-rate、残差幅度/变化、回归时间与每回合一次完成奖励；普通项乘0.005s，不做总奖励非负截断；失败整步替换为−100。
- 回归时钟从初始化1s以后**可观测离开共同跟踪带**起算，扰动持续期间也计时，不使用仿真扰动结束标签。3s总偏离预算；最终横向0.1m、航向0.15rad、真实速度误差0.2m/s、roll≤0.3rad和roll-rate≤0.3rad/s连续0.5s。曾恢复与末段保持分开，超时标志保持；无离带是维持跟踪，不是恢复成功。
- 训练仍是本项目JAX/Flax/Optax PPO，不是RSL-RL。新配置1024×128×32＝4,194,304训练转移，4epochs、minibatch32768（每轮最多16个优化小批次），保留LR3e−4、gamma.9995、lambda.99、std.15、KL.01。额外预热预算64×1400＝89,600计算转移，单独记录；未用改变物理步长来提速。
- 新开发选模先检查失败、共同末段保持、回归超时、名义性能退化、过程容忍超限比例；仅合格候选参与best。流水线无合格候选时仍可对last作诊断，但不宣称合格。每个α/seed/场景保留真实终点和奖励重建。

本轮为实现与有界工程验证，**没有启动正式训练、没有得到新训练模型、没有证明恢复成功率/训练提速/实车安全**。测试命令和CI结果见`docs/VALIDATION.md`；启动命令、奖励公式及限制见`learning/README.md`新增节。运行长训练需另行记录预算和实验身份。


## 当前唯一实施入口（2026-09-14重整）

- 源码已整合回主工作区learning/，新任务/训练/面板唯一默认入口为learning/configs/command_recovery.json、ppo_command_recovery.json、command_standard_panel.json；旧worktree仅保留历史溯源，新阶段使用主工作区。
- 当前范围：保留ECBC+ESO和实际执行器，10帧条件残差；command是速度—偏航指令恢复诊断，不等同几何路径恢复，尚无任务成功证据。
- 新配置：10秒窗口，alpha端点10:1；1024×256×8，epochs4/minibatch2048/KL.01，固定温和/急转弯开发面板，末段.5秒双误差保持，累计回报与全程误差均对照基线。新结果目录runs/command_balanced_20260914。
- command_continuation旧配置已停止于已保存update35，不是32轮预算完成。新128步GPU全链预检已通过。
- 历史证据和负结果见docs/METHODS_AND_RESULTS.md末尾“当前重整”，测试范围见docs/VALIDATION.md。以下日期段落是历史快照，不能把旧“运行中”或“通过”作为当前状态或任务成功。


跨方法导航：[方法与结果总表](docs/METHODS_AND_RESULTS.md)。当前最新完整分析为2026-09-14五臂实验，见VALIDATION末节；下方旧日期结论为历史快照。

## 当前结论与入口（2026-09-12）

当前任务是保留模型1状态反馈ECBC＋ESO，以有界前轮转向/后轮驱动残差恢复连续接地转弯中的扰动。论文主线是生存要求下的速度—路径取舍及执行器响应，PPO作为训练工具；不整体切换Flow/Diffusion-MPC。

最新 `runs/priority_fixed_roll_20260912` 已complete：32更新、67,108,864训练步，90条标准评估、12组配对视频/状态图、45条残差的15组奖励组成图。三档α各9/12联合恢复、3/12物理失败，失败均为正向转向3秒扰动；共同ECBC＋ESO为12/12恢复、零失败。当前固定目标不是已验证更优主方法。末次近似KL约1296.95，32次更新中11次无受扰转移；训练稳定性和采样覆盖仍待处理。

- [论文中文草稿与研究方案](docs/paper/MANUSCRIPT.md) · [PDF](docs/paper/MANUSCRIPT.pdf) · [HTML](docs/paper/MANUSCRIPT.html)
- [可编辑控制框图SVG](docs/paper/figures/control_architecture.svg) · [框图PDF](docs/paper/figures/control_architecture.pdf)
- [当前验证与历史证据](docs/VALIDATION.md) · [代码与运行说明](learning/README.md)
- [最新运行状态](runs/priority_fixed_roll_20260912/pipeline_status.json) · [完整媒体](runs/priority_fixed_roll_20260912/complete_media/INDEX.md) · [奖励组成图](runs/priority_fixed_roll_20260912/analysis/reward_breakdown/INDEX.md)

本文件只维护当前决策、接口和后续计划；按日期的实现/实验历史集中保留在VALIDATION.md，原始runs及冻结声明不改写。旧记录中的“尚在运行”“下一轮”等只对其记录日期有效。

## 当前实际结构

几何路径跟随器生成转向角参考和速度参考；ECBC＋ESO根据轮速估计与状态反馈输出基础前轮角速度，后轮基础量为速度参考除以0.1m半径代理。条件Actor输出两维归一化直接残差，缩放后与基础命令叠加，经统一执行层约束进入仿真。

- 200Hz控制；MuJoCo步长0.0002s，每控制步25个物理子步。
- 前轮接口rad/s，后轮轮轴速度rad/s；不是前轮角度动作或电机转矩动作。
- 转向残差±1rad/s，后轮残差±5rad/s；最终命令±3/±60rad/s，转向位置±0.8rad。
- 默认额外延迟为0，无额外命令变化率约束；XML速度伺服与力矩限制保留。限幅不是稳定性保证。
- 当前使用直接残差，`action_mapping=None`。动力学响应映射/指令余量分配属于已有实验分支，不是当前默认控制链。
- ECBC内部左倾为正，ROS右倾为正，符号转换必须明确且只做一次；轮速代理不是真速度。

## 学习观测、目标与α

当前Actor为200维输入：19字段×10帧＋10有效掩码，MLP 256→128、LeakyReLU，两输出经tanh。历史首末跨度45ms；包括过去命令及α，不是几秒长历史。19字段顺序以`observation.py`为准：侧倾误差/角/角速度/速度代理，转向角/角速度、机体z轴角速度、前后轮速，路径转向/速度参考、基础转向量，上一最终两路命令、ESO估计，三项路径量、α。无显式姿态风险、无俯仰输入，Actor无真实扰动力标签。

`learning_roll_reference=0.11938052083641214rad`，约6.84°。用户明确选择只固定学习网络、姿态奖励及训练恢复加分目标；ECBC仍使用动态侧倾参考。未来trace中的reference_roll表示学习目标。此常数仅为当前左转圆诊断，右转/8字不得直接当作通用平衡角。默认None仍保持动态学习目标，与旧checkpoint兼容；显式固定目标参与身份绑定。

α训练逐回合U[0,1]随机，回合内不变；开发/标准评估均为0、0.5、1。没有已训练的动态α或上层α估计网络。外部α进入观测及奖励，不等于将某个执行器硬分配给某一个任务。

## 当前奖励与恢复标准

生存奖励5/s；失败转移整步替换为−100，无当步生存奖励。当前跟踪系数为速度100、横向20、航向1、越界80，归一化分母分别0.3482045463855167和3.0783592369780886；新待训配置改为wv=0.1×10^(2α−1)、wp=0.1×10^(1−2α)，三档速度/路径权重为(.01,1)/(.1,.1)/(1,.01)。已完成fixed_roll运行仍是冻结的旧线性权重，不追改历史结果。姿态项10eθ²＋θdot²、转向参考误差平方、归一化动作平方×.01不变。全部常规项乘dt=.005；当前公式与变更见VALIDATION；论文第4节所述旧线性权重对应已完成实验。

训练恢复+5：扰动结束后侧倾误差/角速度/速度/转向误差连续0.5s达标，仅首次一次；没有路径/航向恢复条件。标准联合恢复另检查绝对横向0.2m、同策略名义额外横向0.05m、航向0.15rad、真实速度0.2m/s，要求曾离开额外带且最后0.5s保持、无失败。两个定义目前不同，不允许写成相同。

每轮条件评估必须按冻结配置重建每个α、场景、种子的奖励组成并逐帧核对，标出扰动持续区间、真实失败终点，保存分项NPZ、PNG/PDF及INDEX。当前三α×五场景×三种子是45残差轨迹和**15组**图，每图三α；早期排队记录误写9组已在当前状态纠正。

## 训练与评估协议

最新任务圆R=4m、左转、v_ref=2.1m/s、30s回合。随机训练事件4–7s开始、持续1–3s，转向偏置±0.1–0.4rad/s或整车质心航向侧力±1–3N，20%名义、常值/半正弦随机；每回合当前一种扰动。标准事件6–9s常值转向±.4rad/s或侧力±3N，加名义共五情况。

当前配置入口：`priority_conditioned_learning.json`、`ppo_priority.json`、`priority_standard_panel.json`。上一完整预算8192×256×32=67,108,864训练控制步，seed63；开发初始及1/8/16/24/32、三α×两seed×五情况×配对，共2,160,000步上限；标准90回合540,000步上限。此预算已结束，写在这里不表示授权脚本自动重复启动。新阶段先声明预算、配置、角色、停止条件。

标准seed47001/47002/47003已用于开发，不是独立holdout；它们不等于三个独立训练种子。条件实验固定末次32作为端点，不能临时换中期16后称末次成功。一般门槛选出的best只作既有开发候选。

原排队脚本因DVGC失败退出；用户后来明确直接启动，manual_launch单独记录，最新训练现已结束。队列status中的error是历史，不代表最新训练失败。不得重启旧watcher或重复启动同一output；不修改DVGC文件。

## 论文证据与决策

早期同预算直接残差/响应映射对照支持直接残差在指定面板改善恢复（12.583→5.888s，径向.17920→.12321m），不支持映射优于直接残差。此前余量筛查没有充分证明即时命令竞争是主要原因，不能为突出分配而人为缩小执行器限制。

当前应称“偏好条件残差候选方法”。双通道、历史、α或解析分配本身不足以证明创新；必须用公平基线、单因素消融和跨训练种子证据说明机制作用。已有仿真尚不足以声称优先级机制可靠、硬实时、实车有效或稳定性保证。

能量动机保留：反复速度变化可能增加机械功，但“减速必定最好”“保速必定节能”均未证明。当前只有200Hz采样正/负机械功与J/m，非电池能耗；无能量奖励。恢复空间当前为根参考点沿路径左右占用，不含完整车体包络。失败/短窗口不可与完整任务作节能或RMSE排名。

## 下一阶段顺序（尚未执行）

1. 审计近似KL、概率比尾部与各α优势分布；实现可验证的更新停止/回退。现已实现可选target-KL候选拒绝及全批回退，正在做有界工程筛查。
2. 设计真实滚动预热或完整错峰reset，声明额外交互预算；不只改tick，不伪造ESO/历史。
3. 固定奖励做有界短开发对照，先看失败与联合恢复，再看α物理取舍，不自动无上限加训。
4. 同奖励动态/固定学习目标、单转向/双通道、普通残差/条件策略消融。传统协调控制需合理独立调参预算。
5. 稳定后才做至少三训练seed、独立初态、左右圆/速度半径/8字。自动α与能量目标是后续条件，不立即堆叠。
6. 实车先标定执行器/传感器/定位并测200Hz推理时延，实车执行仍需单独授权。

## 文件与验证

运行库在`learning/src/sttw_control/`，薄入口在`learning/cli/`，参数在`learning/configs/`，行为测试在`learning/tests/`；ROS和原始模型留原目录。`docs/paper/`保存唯一论文草稿、导出及可编辑科研图，不创建版本后缀副本；统计源数据仍在runs，图表输入摘要在evidence_manifest.json。维护文档为本文件、AGENTS、learning/README及VALIDATION，各自负责当前计划、规则、操作和验证记录。

此次文档任务不运行训练、不改控制源码；源码最近已有完整CPU88通过1跳过及排队脚本3项检查。本次另核验最新运行90行、12媒体、45奖励重建和15图，论文图表及PDF按源数据生成。Git大任务提交/push按AGENTS执行。

固定学习目标训练已核验：90评估、12媒体、45奖励重建/15组图完成；三alpha均9/12恢复，但正向转向3s全失败，末次不可作为优于基线的结果。共同无失败场景路径/恢复时间仍劣于基线，alpha速度单调性未成立。中期16开发三alpha均8/8，后期退化及KL异常持续。建议先中期16九回合事后经典筛查（上限54000步，未启动），再处理PPO更新约束/真实错峰采样；保持奖励与目标固定以隔离原因。完整结果见VALIDATION最新节。

## 指数权重与训练诊断（已实现，未新训）
新增weight_schedule/weight_base/tracking_weight_scale，默认旧线性兼容checkpoint，当前配置指数base10、整体.1。独立整体缩放不会改变相邻alpha的10倍权重比；不能保证受扰轨迹上两项惩罚仍等比。训练在每次保存验证checkpoint后刷新training/diagnostics/curves.png及PDF，显示policy/value loss、近似KL、每步reward、熵、受扰覆盖与开发恢复/失败/路径/速度。全部21份历史日志及分页叠加在runs/training_diagnostics/INDEX.md，包含工程短跑，不作跨奖励排名；较长条件实验另列，缺失量显示未记录。

Actor和Critic独立200→256→128网络，LeakyReLU(.01)，Actor两高斯均值+独立可训练log_std，经采样/tanh输出两维，Critic输出单值；10帧19字段+10mask。Adam lr3e-4、gamma.9995、GAE.99、clip.2、熵系数.001、梯度范数clip.5、初始std.15、log_std截断[-4,0]；8192×256每更新2097152样本、minibatch2048、epochs4即4096次小批次优化/更新，预算仍32更新但本轮未启动。原实验没有KL停止；新约束仅在stability配置启用，曲线和有限观测KL均不保证全状态稳定。

当前已批准最多两组PPO稳定性筛查：第一组8192环境、KL.01约束、真实256初态池预热6000步、8更新/16777216训练步，先做；通过即不跑第二组。第二组恢复1024/2048小批次/4epochs、64更新同训练步预算，不加新约束/预热，固定奖励/网络/seed。每组训练墙钟上限45min，不重试扩预算；仅开发一seed三alpha经典3s事件，最后候选18条小记录面板并保留奖励图。判据与完整预算见VALIDATION最新节；预热额外1536000步独立计费，初态复用明确标注。配置已实现，实际进展以runs/ppo_stability_screen_20260912/status.json为准。


最新筛查已停止：ppo_stability_screen_20260912触发45min上限，第一组仅记录7/8更新，开发update1/4均12/12恢复零失败，第二组未启动。精确KL均<.01且2/5/6轮确实拒绝超限候选；受扰覆盖5.115%。update4速度/路径RMSE改善约10.1%/3.0%，恢复时间却延长约6.0%，不能宣布问题解决。现有best为update1；原末次判据未完成。分析与七轮曲线见本运行analysis，详情见VALIDATION最新节。建议先用已存候选作小经典记录诊断，不重跑大预算或自动启动第二组；此次未新增训练。

## 已批准的8192完整短跑重启

用户批准从头重新完成第一组8更新，不启动1024组，不加载旧checkpoint。实施顺序：训练增加final-only开发验证选项（默认保留旧日程），每更新在验证前保存参数/optimizer/RNG与进度，验证完成后补写结果；筛查允许无强制墙钟上限、2700s仅提醒，保留显式有限超时的旧计划行为；行为测试及小GPU预检通过后冻结新输入并启动。固定8192×256×8=16777216训练控制步、256×6000=1536000预热计算步，epochs2/minibatch8192/KL.01/seed63，任务奖励网络不变。初始基线和末次开发共两次、上限360000控制步；末次18条经典配对记录上限108000步并生成三组奖励组成/误差图，无大套视频。只做既有开发种子，不是长期稳定或holdout证据。新目录ppo_stability_completion_20260912，原超时运行保持不变。无自动续训/第二组；非有限或子进程错误仍停止。参数检查点不含完整物理状态，不宣称无缝恢复。


2026-09-13：8192完整8轮运行已complete且通过短工程判据，第二组未启动。开发12/12恢复、CPU经典6/6受扰残差恢复，均无失败；路径/速度/恢复时间较基线改善且alpha有小幅一致取舍，侧力横向占用反而增大，尚无广泛偏好或节能证据。18配对记录与9条奖励重建/3图完成，训练约32.9min（不含CPU记录）。保持当前方法，先复现稳定性和公平条件消融，不自动加训，完整数值见VALIDATION最新节。


固定速度诊断（2026-09-13）：reward_breakdown生成时同步输出speed_comparison.png/PDF，每alpha一行，左侧目标/真前向速度/后轮估计同轴叠加，右侧估计减真值；沿用扰动区间与真实终点。NPZ保留三速度原值。当前轮三场景九条残差使用已录轨迹补图，无新仿真；不修改奖励或观测。

速度源对照已按用户补充加入同场景ECBC＋ESO基线（真速度与轮速估计均叠加、估计误差并排对照），原图路径原位更新；估计误差应相对基线解释，不能仅凭存在误差判异常。


## 2026-09-13进度诊断与独立训练种子复现（已批准）

实施：从已有配对轨迹计算圆周展开角×参考半径的沿路径进度、(目标进度−实际进度)/目标速度的有符号时程落后、事件开始至结束记录的横向占用；后者只计根节点。增加左右转跨圈行为检查，不用实际绕行弧长冒充进度。现有数据只作探索诊断，不设置事后通道阈值；下一阶段宽/窄通道+时间要求应事先定义，不将几何时程落后当实际到达延误。固定6.84度目标仍只用于当前左圆。

直接启动ppo_stability_replication_20260913，seed64替代63，其他训练参数/任务/开发及记录种子保持；独立训练种子但不是独立holdout评估。8192×256×8=16777216训练步；预热256×6000=1536000计算步；初始+末次开发≤360000步，18经典配对记录≤108000步。三alpha0/.5/1，保持网络、指数奖励、物理和KL.01，末次8为比较端点；无第二组，无自动加训，45min提醒而非硬停止。非有限或执行错误停止，不等待、不修改、不终止其他项目进程。完成后自动奖励/速度基线对照/进度图；本阶段不加上层alpha或能量奖励。

2026-09-13 seed64复现已完成并通过，开发12/12与CPU受扰6/6均恢复零失败；两训练seed的抗扰/真速度改善复现，但alpha路径取舍方向不一致，沿路径时间收益未跨seed复现。seed64增大alpha反而增加三场景的进度落后；seed63全部比基线落后，seed64全部比基线提前。先审计径向偏移/切向速度/相对名义的额外进度损失，再做固定alpha=.5同预算两seed对照，不自动堆上层网络或扩训练。详细数值见VALIDATION最新节；当前无新训练。

## 平滑变曲率弯道与后轮负载（2026-09-13，用户已批准训练）

新增BendConfig：10m直线、16m正弦平方曲率弯段、55m出口直线，峰值曲率.18/m、总转角1.44rad，lookahead2.5m，v_ref2.1m/s。基于最近线段投影生成路径误差和前视转向，不改ECBC/ESO和物理。学习侧learning_roll_reference=None：使用ECBC当下动态参考，网络误差、奖励及恢复加分共同一致，保留绝对侧倾失败判定；不是固定6.84度，也不是把姿态真值当目标。

事件schema增加第6项rear_load_torque_nm，前5项顺序不变；正Nm加载原始负向前进轮轴，模拟前进工况的阻力负载。不改轮速状态/指令、不减少执行器能力，CPU/MJX均通过qfrc_applied加载并在事件外清零，trace记录实际广义外力。随机训练增加后轮负载事件，Actor不读取扰动标签。默认新配置None/0剔除身份字段维持旧checkpoint配置哈希，新增弯道/非零负载绑定新身份；旧模型不直接复用。

基线筛查最多4个30s回合（nominal、.24Nm、1.2Nm、3N侧力）=24000控制步，seed47001；另GPU预检128训练步+32预热计算步，验证后轮通道/动态侧倾和真实JIT。先验证名义不摔及轻载幅度，再确认更强负载安全。平滑参考动态目标在名义约-1.02至4.88度，直道靠近0；侧力时允许因路径纠偏产生相反参考，不声称风险保证。

通过后直接启动bend_training_20260913：8192×256×8=16777216训练步、256×6000=1536000预热计算步、seed63、epochs2/minibatch8192/KL.01，alpha逐回合随机，评估0/.5/1。保持其他奖励数值，不加能量/上层alpha。初始+末次开发单seed46001，名义+后轮+侧力3情况配对，最多216000步；CPU经典记录seed47001三alpha×3情况×2策略最多108000步。后轮幅度以基线筛查冻结结果为准；随机起点6–10s、时长1–3s、常值/半正弦，20%名义，其余60%后轮、剩余转向/侧力。45min仅提醒；单阶段，无1024/无自动扩训练，出错停止。完整图/奖励/速度基线输出，圆进度图不套用该弯道。

冻结负载训练幅度0.4–1.2Nm；标准后轮8–11s为1.2Nm、侧力3N。1.2Nm基线额外降速约.043m/s且未离开额外路径恢复带，故不能把该工况“未离带”称为恢复成功或明显优先级冲突。本轮为探索，仍保持原评价阈值并报告无恢复资格。

新弯道8轮已完成：侧力三alpha有离带后恢复，10.905s→6.025/6.110/7.375s；后轮三alpha始终未离开额外路径带，状态6/6仅末端保持，不能写成六次成功恢复。受载全程速度RMSE降低，但相对自身名义的额外降速均约.043m/s，与基线基本不变，主要改善名义速度偏差。建议先差分审计后轮响应、再小预算校准更强可恢复负载，不盲目续训；详细范围与机械功负结果见VALIDATION。本次无新运行。


## 2026-09-13后轮负载筛查（用户批准直接运行）

新增同策略名义速度差分、缺速积分、速度单独回稳诊断，进入以后奖励诊断固定输出。速度资格阈值.05m/s、真实目标误差.2m/s、保持.5s；仅速度诊断，不修改原路径联合恢复标准。未偏离速度带时回稳时间null且标记不需恢复。已有轨迹只读重算。

直接运行rear_load_screen_20260913：同现有弯道/动态参考/执行器、seed47001，名义0与后轮2.4/4.8/7.2Nm逐级共最多4回合×30s=24000控制步。事件8–11s常值，未改奖励或物理能力。不等待其他项目。每级若摔倒或末端未保持速度带则不升更高负载；无自动重复、无自动启动训练。目标是校准明确可恢复的速度偏离，不以通过次数包装性能。

后轮筛查实际停于load_0绘图：原始配置缺默认controller，已修复为完整解析配置供后处理。保留原error，rear_load_screen_continuation_20260913显式复用load_0并核验身份，继续尚未执行的三个负载；新增最多18000步，原总预算24000不变，无训练。结果待实际运行。

负载continuation已按物理失败停止，非程序异常：2.4Nm完整且速度回稳.53s，4.8Nm于8.655s侧倾超过.7rad，跳过7.2。实际后轮驱动力矩上限3Nm，4.8超过能力且导致轮轴反转；固定力矩此时不再代表通用阻力。该选档过粗，不通过改执行器能力解决。建议后续先用已筛查1.2–2.4Nm，不自动升级或重跑；详见VALIDATION。

## 2.4Nm标准负载短训练（用户批准，2026-09-13）

不再幅度搜索，使用已筛查完成的2.4Nm作标准后轮8–11s负载；随机训练范围1.2–2.4Nm，其他路径、动态学习参考、奖励、随机时长/起点/通道概率、seed63、PPO不变。从头8更新8192×256=16777216训练步，预热1536000计算步，初始+末次开发216000、CPU经典18配对记录108000步上限。新运行bend_load_training_20260913，旧记录不覆盖。无续训、无第二组、45min仅提醒，非有限/执行错误停止；不等待其他项目，不重复GPU预检或负载筛查。

完成后输出奖励/速度基线及同策略名义速度差分，重点检验额外降速、缺速积分、速度单独回稳是否改善，并报告联合路径恢复资格，不能以名义偏差改善代替抗负载改善。只用已有开发seed，不称holdout或安全域；本轮不改上层alpha或能量。

2.4Nm训练已complete：侧力恢复与误差改善保留，但后轮最大额外降速、缺速积分与基线几乎相同，回稳均.53s。策略受载追加后轮指令均值仅约.0011rad/s，主要改善名义偏差，不能声称负载抗扰提升。不原样续训；下一步先小预算检验额外后轮控制是否有实际恢复效果，再决定学习诊断或固定alpha对照。本次无新运行，详见VALIDATION。

## 2.0Nm冻结模型对照（2026-09-13，用户明确要求）

区分后轮±5rad/s学习残差指令范围、最终轮轴速度指令限幅与伺服实际力矩/转速响应；不能据残差未达边界推断执行器有足够输出。用现有bend_load_training的update8，不训练，seed47001、三alpha0/.5/1，事件8–11s改为2.0Nm，其他任务/物理不变。只名义与后轮两场景、两策略，共12回合≤72000控制步，输出rear_load_2nm_20260913及奖励/速度/自身名义差分；与既有2.4Nm数据比较，不重跑侧力。不增加幅度搜索或新训练。

2Nm评估已complete：额外降速/缺速积分仍与基线相同，降载使两者同比减少；200Hz实际力矩峰值2Nm场景约2.045、2.4Nm约2.453，均未记录到3Nm饱和。速度伺服增益3与稳态负载速度差的量级相符。停止原样训练/负载搜索，下一步优先短指令响应诊断；只分析未新启动，数值见VALIDATION。

## 奖励比例对照重训（2026-09-13，用户批准直接启动）

当前弯道配置生存奖励5→1/s，tracking_weight_scale .1→.3，使速度/路径整组惩罚同状态下均乘3，相对生存比例提高15倍。α指数形状、两组归一化、姿态/动作惩罚、恢复加分5与失败替换-100不变。不据正累计回报断言常数生存奖励压制策略梯度，本轮仅检验奖励比例假设。

从头启动bend_reward_balance_20260913：同seed63、8192×256×8=16777216训练步，预热1536000；初始/末次开发最多216000步，CPU三alpha×名义/2Nm后轮/3N侧力×两策略共18回合最多108000步。训练随机负载仍1.2–2.4Nm、起点6–10s、持续1–3s；标准后轮改2Nm持续8–11s，其他物理/控制/网络/PPO保持。旧训练和rear_load_2nm记录不改，以同2Nm面板比较实际速度、相对自身名义额外降速、缺速积分、路径和失败；跨奖励版本累计回报不直接比优劣。单阶段8更新，无自动续训，45min仅提醒，执行错误/非有限停止。完成自动生成loss/KL/reward及分项、误差和配对速度图；数据角色仍为开发诊断。

## 五臂环境配置对照（2026-09-14，用户批准全部执行）

按顺序执行exp1_history1_8192、exp1_history20_8192、exp2_rear_command_8192、exp2_rear_command_1024、exp3_rear_load_1024。实验一严格复用dynamic_recovery_speed/history_20260909的任务/观测/无alpha奖励，8192×256×4各8388608步对齐历史1024×256×32；seed42、4轮均开发验证、原选模规则best或last回退。更改批量8192/epochs2/KL.01，256×2400真实相位预热；更新频次不同，所以是训练配置对照而非环境数单因素消融。历史奖励生存1/s已核查原提交源码，不静默套当前alpha奖励。

实验二当前10帧/随机alpha/当前奖励，后轮事件第6项通过rear_disturbance_mode=command解释为正的减速指令幅值rad/s，注入基础后轮命令后与残差共同限幅；外部广义力严格为零。训练.4–.8rad/s，标准2/3rad/s于8–11s；并非2Nm动态等效。event.json同时记录模式、实际指令偏置、外部负载零；trace增加injected_rear_rate。模式默认torque保持所有旧配置/身份兼容，物理能力不变。两种配置各16777216训练步：8192×256×8/epochs2/minibatch8192/KL.01/预热256×6000；1024×256×64/epochs4/minibatch2048/无KL限制/无预热，seed63。实验三同后者但原2Nm外部负载和当前奖励，对照bend_reward_balance_20260913末次模型。

共67108864训练步，2764800额外预热计算步。实验一评估沿用3个旧回归seed×8受扰+名义×两策略，各54条，实验二/三各18条三alpha配对；共162条最多682560记录控制步。均开发数据，不是holdout。五臂全部完成，不因一臂通过跳过；执行异常停止保留原始记录，不自动重跑或扩大预算，无固定45分钟墙钟终止。每臂保存PPO曲线/奖励重建/基线三速度对照及配对差分，末尾统一索引分析。条件评估0/.5/1是在同策略上改变真实输入的干预，不用各自加权回报比较优劣。计划与哈希冻结于runs/environment_comparison_20260914_inputs，输出runs/environment_comparison_20260914。先做真实GPU极短全流程预检，通过后直接启动；不等待或改动DVGC。

五臂对照已全部complete，162条评估无物理失败，但不等于恢复成功。旧无alpha任务1024恢复/路径优于8192；当前1024负载出现KL尖峰与名义路径退化，不能全盘回退旧配置。后轮指令偏置与外力矩在未饱和线性伺服下近似等效，所有组额外降速仍约.071m/s。当前无新训练；建议先检查中期模型/有界指令响应，再做保留KL约束的1024有限对照。每次完成方法实验须同步维护docs/METHODS_AND_RESULTS.md，保留负结果和证据边界。

## KL单因素对照与反向后轮负载（2026-09-14）

用户指出8192使用KL停止、1024未使用，不能归因环境数。本轮kl_matched_load_20260914仅在既有exp3_rear_load_1024上将target_kl从null改.01；任务、奖励、seed63、1024×256×64、epochs4、minibatch2048、无预热均不变。预算16777216训练步，开发最多216000步，末次24条评估最多144000步；64更新结束或执行/非有限错误停止，不自动续训。训练仍随机正负载，标准面板保留旧名义/+2Nm/侧力并加-2Nm。与旧1024的公共面板才是KL单因素比较，与8192仍有其他超参数差异。

另用已有8192 update8评估rear_assist_20260914：alpha0/.5/1、seed47001、名义与后轮-2Nm持续8–11s、两控制器，12回合最多72000步。负力矩帮助原始负轮速方向加速，目标仍2.1m/s，伺服/能力不改；此为未训练助推扰动诊断，不预设必然形成速度/路径冲突。姿态、路径、速度耦合，生存优先是任务目标，当前加权奖励并非严格优先保证。新增相对自身名义的额外升速、超速积分和绝对速度差积分，保留掉速指标及所有奖励/配对速度图。准备检查已通过，运行状态以各status.json为准。

## 2026-09-14 KL与助推结论更新

两运行均complete，详见VALIDATION末节。KL1024仅增加target_kl=.01，压制更新尖峰（最大近似KL .005184、每轮最终精确KL≤.009560），但速度改善伴随路径进一步退化，且名义状态就有.189/.379/.451m路径RMSE；best=None。8192仍是当前弯道面板较好的综合参考。反向-2Nm的额外升速约.0715m/s、速度回稳保持完成.52s，基线/网络几乎相同，不主张抗助推收益。保留KL，不再原样重复外力矩训练；已启动的独立指令条件残差（feat/command-recovery隔离工作区）继续检验状态相关优先级必要性。

## 2026-09-14 独立指令任务完成结论

command_recovery_20260914已complete，代码仍在feat/command-recovery隔离工作区（1d58cfa），本轮只分析不改运行/训练。独立速度/yaw-rate随机请求、10帧动态alpha条件下层、8192×256×8，seed64；自动alpha上层尚未训练。24标准评估基线/残差均3/12失败，均为急转弯；其余三场景均完成但网络速度误差更大。急转弯仅延迟失衡.10–.12s，不能称恢复成功。KL未触发限制、无回退，开发生存时间略退化，未建立不同状态下优先级选择的必要性。下一步建议先短预算可恢复性与简单指令平滑对照，再决定安全目标及固定/动态alpha训练；本次未启动任何新仿真。详见docs/VALIDATION.md末节与runs/command_recovery_20260914/analysis/INDEX.md。


## 2026-09-14：9度姿态阈值与完整指令范围残差

用户授权的代码修改已在现有隔离分支feat/command-recovery提交17f6dee，源码目录runs/worktrees/command-recovery，当前配置为该目录learning/configs/command_recovery.json。主工作区旧训练源码未混入新动作语义。新配置超过9度(0.15707963267948966rad)开始100倍平方越界惩罚，其余奖励不改；0.7rad失败阈值及物理能力不改。

新增full_range指令叠加：先把基础输出截入[-L,L]得到b，a>=0时u=b+a*(L-b)，a<0时u=b+a*(L+b)，当前strength=1。前轮L=3rad/s、后轮L=60rad/s；网络0保留基线，正负端点可覆盖甚至反转基础输出；仍受转向位置/伺服力矩等约束。full_range不使用旧固定残差尺度，并绑定新checkpoint身份；旧additive默认行为与身份保留。没有动力学分配或执行器扩容。

相关测试54项通过，新增新配置CPU零残差物理回归与奖励重建后motion_commands文件7项通过，共55项不同测试；仅CPU工程验证。本次未重训，不能复用旧checkpoint宣称新机制有效。后续需要冻结新配置从头训练，并保留同场景基线/奖励组成/速度与偏航误差图。


## 2026-09-14 full_range重训执行

用户批准按上一轮训练。冻结配置runs/command_full_range_20260914_inputs；8192环境×256步×8更新=16777216控制步，seed64、epochs2、minibatch8192、KL.01，256×2400预热计算步。训练JSON和四标准测试面板与旧轮逐项相同，任务仅改变9度阈值/full_range。先执行command_full_range_20260914_smoke（64训练步、128预热计算步、.25s指令切换全链路），成功后正式command_full_range_20260914从头训练和自动三alpha配对评估/奖励组成/速度图/视频。只开发对照，不称holdout；非有限或执行错误停止，8更新后不自动续训，无45分钟硬上限。代码在runs/worktrees/command-recovery，声明提交f68f810；实时进度查看各运行status.json，本文不替代实时状态。


2026-09-14 full_range训练和24条评估均complete，结果见VALIDATION/METHODS_AND_RESULTS最新段。开发seed46001三alpha均存活12s，但标准急转弯仅从4.895s延后到9.125–9.465s失败，普通场景通过减速降低姿态且速度/yaw误差均劣于基线；不声称总体提升。不原样续训，优先离线检查回正失稳及动作增益/探索对照。本次仅分析，未启动新训练。


2026-09-14用户要求撤回9度阈值：feat/command-recovery提交a2d871d已将当前任务恢复.3rad，保留full_range，7项相关测试通过，旧实验不改，未重训。核对旧1024速度恢复训练为32轮/epochs4/minibatch2048/8388608步/16384个优化minibatch，best update24；新8192为8轮/epochs2/minibatch8192/16777216步/4096个优化minibatch，command任务固定末次、best为空。总采样更多不等于策略迭代充分，且任务不同不能单因子归因。当前8轮是探索预算非收敛判据；后续需command专用开发选模和有上限的平台停止机制，当前未实现，不能宣称最后checkpoint最佳。


## 2026-09-14 执行开发选模与32轮上限训练

实现位于feat/command-recovery提交1c43ca1，当前任务0.3rad/full_range。17项CPU相关测试通过；command_convergence_20260914_smoke进行GPU全链路验证。正式计划command_convergence_20260914为8192×256×最多32=67108864步，seed64/epochs2/minibatch8192/KL.01；从头训练，初始/1/每4轮开发验证seed46001和alpha0/.5/1，最少16轮且连续4次无改善停止。前两级（失败数、初始最多1秒的速度/yaw相对基线退步计数）相同，平均回合回报需提高>.1才更新best；名义容差.05m/s和.05rad/s，不代表整个名义场景无退步。best可能仍失败，不作安全保证。原标准面板只在选模之后评估24条，自动完整诊断/视频。

旧9度command_full_range中间checkpoint1–8审计已启动：runs/command_checkpoint_review_20260914，旧冻结任务/开发seed46001，24个策略回合与3个共享基线≤64800步，不训练，不用标准面板选模。保留各checkpoint奖励组成/速度图和完整轨迹，属于事后开发对照。新训练实时状态以runs/command_convergence_20260914/status.json为准，工程预检不能当训练充分或恢复证据。


2026-09-14 GPU早停/选模预检complete：第2轮提前停止，last update2、best update1，六条CPU配对评估实际使用best并完成图与视频。正式command_convergence_20260914已启动（源码声明a39419f），阶段以status.json为准；旧checkpoint审计继续独立运行。未声称新策略性能或收敛。


## 2026-09-14 旧checkpoint开发审计完成

证据runs/command_checkpoint_review_20260914/status.json、summary.json、INDEX.md。旧9度/full_range冻结任务，单开发seed46001、alpha0/.5/1，8个checkpoint×3策略回合和3共享基线；非新0.3rad训练结果，非独立测试。各轮存活数（分母3）为0/0/0/0/2/3/2/3；平均完整回报-276.200/-231.999/-195.623/-225.144/-227.629/-102.412/-129.163/-102.287；初始直行速度RMSE均值.061/.123/.183/.240/.326/.368/.364/.398m/s。按失败数、初始退步数、平均回报预定排序选update8，但它较update6平均回报只高.125，不是显著优势；三个alpha中8相对6仅alpha0回报改善，.5/1退步。

结论：没有发现明显全面更优的早期模型，不能把旧结果解释成仅末次选错；存活随训练改善伴随正常直行降速，6–8轮有局部平台/反复，但不足以证明收敛，不能证明继续训练必好或必坏。保留新0.3rad/32轮上限训练，不改运行参数。


## 程序故障桌面报警

learning/cli/watch_run.py（逻辑learning/src/sttw_control/run_watchdog.py）独立监视--status和--launch中的进程，--output保存监视状态与alert.txt。默认10秒检查、连续3次异常确认；捕获phase错误、未完成时进程消失/僵尸/PID复用，以及持续不可读状态文件。complete为正常结束；物理failed数组/小车摔倒不当程序错误。没有默认无日志超时，不检测仍存活但卡死的进程；不自动重启/停止训练。使用zenity错误弹窗，失败时尝试notify-send；依赖当前登录桌面会话，桌面服务不可用会记录监视器error。弹窗发送不代表用户已阅读。

已运行：runs/command_convergence_20260914_watchdog/status.json，监视当前训练主进程915580，保存进程starttime防PID复用；首次测试弹窗明确标注测试，关闭不影响后续监视。4项行为测试通过，实际桌面后端已发出测试，监视状态monitoring/0异常。源码放主工作区，未改运行训练工作树的源码。后续新运行需启动对应监视器，不会自动监视其他项目或所有进程。

启动示例（从项目根目录，输出目录应尚不存在）：
```bash
PYTHONPATH=learning/src /home/qy/mujoco_playground/.venv/bin/python learning/cli/watch_run.py --status runs/某次运行/status.json --launch runs/某次输入/launch.json --output runs/某次运行_watchdog
```
launch.json需包含流水线主进程pid；应监视流水线根状态而不是提前完成的子训练状态。正常完成或发出一次确认故障报警后监视器退出，避免重复弹窗。


2026-09-14用户追加正常完成弹窗：监视器在流水线根状态complete后显示“STTW 仿真完成”信息窗口，记录completion.txt和completion_notified；错误仍使用故障弹窗。新增--resume重用已停止的监视目录，核对目标状态路径并避免正常完成重复通知。已仅重启当前监视器使新行为生效，训练主进程915580未动。5项行为测试通过，包含完成弹窗及resume不重复发送；未发送伪造完成弹窗。全流水线完成（包括评估/出图）才通知，非训练子阶段结束。


## 2026-09-14 训练奖励分项日志补齐

新增日志功能位于隔离工作树runs/worktrees/reward-logging、分支feat/reward-component-logging、提交585e684。当前正在运行的command_convergence代码不修改、不重启，已经发生但未保存的训练采样分项无法补回，不能拿评估分项冒充训练统计。后续启动需使用新日志实现。

command任务新增reward_components_mean_step：alive、attitude、roll_rate、speed、yaw、action、failure，均为含alpha/控制周期的有符号每采样步贡献；失败替换其他项。每轮同时记录分项和及重建绝对/缩放误差；GPU内部先归约，不保存全量环境轨迹。diagnostics/reward_components.png/pdf/json展示分项和总奖励核对。14项CPU相关测试通过；64训练步GPU预检输出runs/reward_logging_20260914_smoke，状态以实际产物为准。旧checkpoint评估奖励图仍独立标记为评估，不修改旧原始日志。


奖励日志GPU预检已complete：64训练步，七分项和与总奖励逐样本最大绝对差3.7253e-9，PNG/PDF/JSON生成，14项CPU相关测试通过；单点图标记另经2项绘图测试验证。实现与验证提交585e684/2eb7b30（feat/reward-component-logging）。当前正式训练保持原进程，不声称其既有日志已补齐分项。


2026-09-14 command_convergence已complete：32轮预算结束，best28用于全部最终评估；策略四标准场景×三alpha12/12存活，急转弯通过但速度与yaw请求被明显放宽。普通场景仍两项均劣于基线，不能声称全面提升；同训练8→28开发指标支持训练不足曾是因素但不证明收敛。详细数据见VALIDATION/METHODS_AND_RESULTS及运行analysis/INDEX。完成弹窗已发出。下一步建议冻结best28测试alpha独立/联合切换与正常工况干预，不原样加训。本次未启动实验。


## 2026-09-14 追加32轮与急转弯轨迹核查

用户授权从command_convergence末轮update_0032追加32轮，输出runs/command_continuation_20260914；轮次33–64，额外67,108,864训练步，累计134,217,728。8192环境×256步，epochs2/minibatch8192/LR3e-4/targetKL.01/gamma.9995/GAE.99保持不变，奖励、动作权限和随机命令分布冻结不改。command_patience=0以完成用户指定32轮，仍保留非有限停止、KL保护；无自动继续。预热256×2400=614400计算步另计。恢复参数、Adam、PPO RNG（预热后恢复），仿真/ESO/历史重新初始化，非逐步无缝续跑。保留旧best28作为同一开发面板重新验证的选模候选，并另记新阶段best。开发seed46001/alpha0,.5,1；验证新阶段1/每4轮/末轮；最终沿用四标准工况seed47001，固定alpha0,.5,1，配对基线，最多24条12秒。此为继续训练诊断，非独立训练种子或holdout。

新增恢复快照SHA/身份/归一化/Actor一致性检查，全局轮次和累计步数；新训练打印真实随机采样的七奖励分项，并在每次验证刷新本阶段及前32轮对照loss/KL/reward/开发曲线。alpha_0/alpha_1/alpha_2是列表索引，对应0/.5/1，新评估保存alpha_values.json，旧目录不重命名。

三轮旧实验标准指令面板一致，9份基线time/qpos/qvel逐元素相同。急转弯指令3秒变为v=2.7m/s、yaw rate=1.4rad/s，8秒变为2.1m/s、0rad/s；基线4.895秒失败，未执行8秒回正。新best28满12秒存活，但10–12秒平均yaw rate在alpha0/.5/1下为.19746/.31204/.32828rad/s，末帧.17047/.30112/.32053，不能称为准确回正或返回原轨迹。该任务没有独立几何参考路径。轨迹/视频及来源见runs/command_convergence_20260914/analysis/trajectory_comparison/INDEX.md。

续训前工程验证：CPU训练/诊断/选模/命令行为测试；4环境×16步恢复快照并追加1轮，验证全局update2/累计128步及绘图/奖励恒等式，工程预检不作性能证据。正式启动状态以runs/command_continuation_20260914/status.json为准。

续训预检已完成：20项CPU行为测试通过，GPU从update1追加到update2，新增64步、累计128步，奖励分项和lineage曲线生成成功。参数/Adam/RNG恢复有哈希与行为测试支持，未保存物理状态。

正式续训已启动：runs/command_continuation_20260914/status.json为training，launcher PID1252724，监视PID1252725；源码提交bcd48a1（feat/reward-component-logging已push），奖励分项与续训代码GPU预检通过。当前处于初始化，尚无第33轮性能结论。


## 2026-09-14 逐工况分析与任务成功纠正
- 每次结果分析必须逐实验、逐标准工况、逐alpha和种子报告；不可仅展示总体存活率或挑选最好视频。分别说明指令/扰动前、持续阶段、撤销后表现，对照同场景ECBC+ESO，列出速度、偏航/路径、姿态、失败终点、奖励分项和局限。
- 每个评估情况必须有实际每控制步总奖励对steps曲线，以及每步有符号奖励分项（重点速度、偏航/路径）和误差时序；与基线配对，标记指令/扰动区间和真实失败终点。累计回报只能作为附图，不能替代每步奖励。保留PNG/PDF、逐步数据和全覆盖索引。
- 存活不等于任务成功。请求回正后持续偏航属于跟踪未完成；best只表示开发排序最优，可能不合格。成功需要预先声明的跟踪容差、保持时间和安全条件；未定义或未计算时报告未验证，不事后编造成功率。alpha代表取舍，不允许据此默认完全放弃被降权任务。


## 2026-09-14 有限偏好与快速诊断重整（当前实施）

用户否定过大的端点奖励比和小时级无效反馈。停止command_continuation旧权重续训，保留last快照/日志，不将停止视为预算完成；仅终止该STTW进程组与关联监视器，不动DVGC。历史最可靠正面证据为无alpha直接残差的三初态种子受扰恢复15/24→24/24；映射、后轮负载、共享alpha策略尚无一致优势。command是速度—偏航指令恢复的诊断任务，没有几何路径，不能替代原论文的路径抗扰证据。

新配置原位维护command_recovery/ppo_command_recovery/command_standard_panel。alpha两项权重采用q^(alpha-.5)、q^(.5-alpha)，q=10，端点相对10:1、中点1:1；共同scale=.3不动，默认旧q=100用于旧身份/日志兼容。10秒episode，不截断旧记录；gentle/tight指令3–6秒，之后4秒恢复；straighten与reversal保留前阶段，reversal回正由9秒提至7秒。仍200Hz，保持真实执行器/控制器/10帧输入/网络。随机训练命令分布保持（时刻、大小和alpha随机），只缩短窗口。

新开发panel=2固定序列（gentle/tight）×alpha0/.5/1×seed46001，每条≤10秒。补上名义全程比较和最后.5秒速度/yaw双误差均≤.2的保持量；它只叫末段跟踪保持，不宣称全任务完成。新排序依次失败数、基线存活工况全程速度/yaw退步数（各+.05容差）、末段保持缺失数、负平均配对累计回报差；另输出开发门槛是否通过。best仍只候选，不把不合格模型当成功。配对累计reward/各分项总和/差值和同窗结果均保留，负奖励不用倍数推断。

性能证据：8192旧32轮87.07min，采样3573.9s、开发验证1171.0s、优化9.99s；非minibatch计算占主导。新有界首阶段1024×256×8=2,097,152训练步，epochs4/minibatch2048（每轮最多512次小批次更新）、LR3e-4/KL.01，预热128×1800=230400计算步另计，验证1/8及baseline，6条×2000×3=36000开发步，最终四工况×三alpha×配对基线=24条≤48000步。只绘验证点/末点的曲线、每轮仍记录和保存checkpoint。用户短诊断优先：8轮固定终止，不自动无上限扩训、无45分钟墙钟停止，错误/非有限立即停止。

工程预检：1024与8192各16步、2次测速重复+首次编译运行，最多442368控制步，零残差同起始状态，测同步wall、失败/有限状态，不作为策略性能。GPU PPO smoke4×16×2=128步，.25秒窗口和2×3短开发序列，验证新奖励/选模/日志/出图，最终短CPU评估；通过后启动上述8轮诊断。

资源决策更新：已有历史逐项同步耗时足够支持1024短反馈试验，不另运行1024/8192零残差benchmark，不消耗该预备预算；实际吞吐从新8轮前两轮metrics取。只运行128步GPU PPO全链预检后启动正式8轮。

新10:1奖励GPU全链预检complete：2×4×16=128训练步，6开发回合正确返回末段保持/完整误差，累计奖励差入日志；最终3alpha×配对基线共6条.25秒CPU回合，3组每步奖励PNG/PDF/NPZ和配对累计奖励JSON均完成。48项核心CPU测试通过；此为工程验证，非性能结论。

新有界8轮已启动：runs/command_balanced_20260914/status.json=training，训练pipeline PID1360655、监视PID1360657，报错/完成弹窗监视monitoring、0异常。主工作区合并提交f5477b2，85项相关CPU行为测试通过，GPU预检声明的全部训练源码SHA与主工作区逐个一致。当前初始化中，无新性能结论。旧12秒best28完整12对累计回报表和每步图已补齐，见其analysis/INDEX.md及paired_rewards.json。


## 2026-09-14 奖励实际影响、积分参考与评估瓶颈

用户澄清10:1目标是两任务对奖励的实际影响，不只是系数。当前实现仅保证系数端点10:1，不能保证实际贡献比；须按alpha/工况/阶段统计归一化误差与每步贡献。当前开发温和基线speedRMSE=.0436447m/s、yawRMSE=.0908064rad/s，两者scale均.2，yaw归一化MSE约为speed的4.33倍，alpha1实际speed/yaw代价率约2.31而非10。首轮训练平均speed=-.00517874、yaw=-.0992997、failure=-.102234/step，yaw/speed约19.17；这些是混合随机alpha统计，不能冒充单个alpha贡献。

建议下阶段采用独立训练校准样本的固定Mv=E[(ev/sv)^2]、Mr=E[(er/sr)^2]，以Cv/Mv、Cr/Mr形成可比较尺度后施加alpha偏好。校准需覆盖训练分布、保持明确样本权重，冻结正数下限及尺度，不用最终测试反调奖励，不按当前每步误差强行除自身使惩罚常数化。只能在校准分布平均上定义偏好，不宣称新策略任意状态都严格10:1。当前8轮奖励未中途改变，该校准尚未实施。PPO优化长期联合回报，不是简单选择数值较大的一个奖励项。

确定v/r命令可积分生成参考：psi_dot=r_ref，x_dot=v_ref*cos(psi_ref)，y_dot=v_ref*sin(psi_ref)。已由冻结10秒面板生成四工况PNG/PDF/NPZ及来源哈希，见runs/command_balanced_20260914/analysis/reference_plan/INDEX.md；直行和完整圆解析自检通过。参考在x0=y0=psi0=0坐标，叠实际运动时需按实际初态平移旋转。属于指令运动学参考，未证明动态可行，未加入观测或奖励。r=0保持当前航向，不返回初始方向。新gentle/tight/straighten/reversal累计参考航向分别51.57/240.64/154.70/51.57度。若以后加入几何路径误差，应采用连续进度约束投影的横向误差，并独立报告沿程/时序误差，避免把减速造成的时序落后重复当路径误差。

第1轮新1024采样39.7515s、优化.1778s、开发验证141.2334s；初始化reset11.589s、预热174.783s、基线验证152.085s、初始化共341.386s。开发验证已经MJX/GPU六条批量scan；最终完整trace是CPU串行24回合。故直接说开启GPU即可提速不成立。CPU并行候选先在相同10秒gentle、两个初态seed47001/47002做1-worker/2-worker对照，预算4回合8000控制步，检查time/qpos/qvel/reward/terminated逐元素一致；不把基线并行验证当策略性能或CPU/GPU等价证据。

CPU并行工程核验完成：相同两种子10秒gentle，1-worker=24.5248s、2-worker=13.3823s，端到端加速1.833倍（含子进程启动/导入/保存，不含出图视频）；两种子的time/qpos/qvel/reward/terminated逐元素完全一致。源脚本和summary在runs/command_balanced_20260914/analysis/evaluation_benchmark。当前为外部分析预检，生产评估尚未切换；支持有限CPU进程并行方向，不支持未验证的GPU/CPU选模替换。


2026-09-14 command_balanced已完成训练与24条最终评估，best8不合格：残差10/12未摔、0/12末段跟踪保持；九条温和/回正/反转奖励和两项RMSE全退步。8轮曲线不足以判断已收敛，KL未卡更新，详见VALIDATION/METHODS_AND_RESULTS最新完整报告。本次未追加训练。


## 2026-09-14 用户明确要求累计100轮后再评估

用户取消第2/3/8轮对比，审计进程已停止，部分审计不作结论。当前8轮没有达到任务要求，但不能由8轮推断最终能否学会，先前“未收敛”等表述应理解为没有足够收敛证据，不是已证明训练无效。

立即从command_balanced update8恢复参数、Adam和PPO RNG，续92轮至累计100，输出runs/command_balanced_100_20260914。奖励/1024环境/256步/epochs4/minibatch2048/LR3e-4/KL.01/10秒episode/动态alpha均保持原冻结配置；新增24117248交互，累计26214400。仿真闭环重新初始化与同配置预热，不是无缝物理续跑。command_selection=False、validation_final_only=True、command_patience=0、plot_interval=4，每轮保存模型和日志，每4轮刷新奖励/损失/KL及前8轮叠加图；不做中间策略性能比较，不提前停止（程序错误/非有限保护仍生效）。初始化仅固定基线验证，跳过初始策略/旧best比较，第100轮末做固定开发与标准面板评估，使用last100，不能称其为全程最优。继续训练结束与全部评估结束分别弹窗。无自动第101轮。


累计100轮续训已实际启动：runs/command_balanced_100_20260914/status.json=training，pipeline PID1458325、watchdog PID1458327，启动源码960ba55。日志已进入MJX reset编译；这是启动证据，尚无第9轮及以后性能结果。监视器分别覆盖训练完成、整个评估完成和运行错误；不继续第2/3/8轮审计。


## 2026-09-14 累计100轮完成，当前判断

command_balanced_100训练和评估complete，训练/最终完成均有zenity发送记录。平均训练奖励明显改善并在后20轮趋缓；标准模型12/12存活、4/12末段保持，基线9/12和9/12。急转弯防摔有效但欠转严重，普通三工况9/9奖励与速度/yaw RMSE均退步，当前不是合格的通用跟踪控制器。第100轮被KL整轮回退，Actor与99相同，使用100是声明预算终点、不是best。详见docs/VALIDATION.md和docs/METHODS_AND_RESULTS.md最新段，全部图与逐阶段分析在runs/command_balanced_100_20260914/analysis/training_review/INDEX.md。

本次仅分析、核验原始24条轨迹、补齐100轮训练图并更新台账；未改奖励/物理配置、未追加训练。下一步建议少量后期checkpoint开发审阅及单因素名义行为保持对照；不能继续把降低转弯幅度后的存活当成任务成功。


## 2026-09-14 用户要求收缩残差权限

实施计划：只原位修改learning/configs/command_recovery.json，将full_range改为已有additive，strength=1，前轮残差±1.5rad/s、后轮±10rad/s；早期默认±1/±5，因此保留更强补偿而不再直接覆盖全指令范围。前轮总角速度±3rad/s、角度±.8rad、后轮总轴速±60rad/s及XML力矩/伺服参数不变，奖励、观测、alpha、ECBC/ESO不变。公式u=clip(u_base+[1.5,10]*clip(a,−1,1),执行器指令边界)，再走原有位置/延迟约束；这不是能量或稳定性保证。

验证计划：更新已有当前配置行为测试，检查中等权限的零残差基线一致、可用范围内不能完整接管、实际总限幅保持；运行残差和命令环境相关测试。不改旧freeze/checkpoint，不把旧full_range模型静默应用新动作语义；新任务需新训练身份。本次只修改和验证，不擅自启动长训练。

实施完成：上述中等权限已设为当前command_recovery配置。30项残差/命令行为测试通过；逐字段确认仅动作合成与两路残差范围改变，新旧策略身份不同。当前阶段为已配置、未训练，不能声称解决了欠转弯。详见VALIDATION/METHODS_AND_RESULTS最新段。

## 2026-09-14 TensorBoard接入计划

用户批准接入过程可视化。复用每轮metrics.jsonl记录，在训练函数外层用上下文管理的事件写入器，每轮JSON落盘后同步导出标量并flush；成功或异常退出都关闭写入器。训练reward/分项、loss、近似与最终KL、回退/最终保留更新数、耗时/吞吐/样本数、固定开发各场景alpha种子的回报与误差分别分组；缺失项不补0，不伪造中间评估。全局PPO轮次作step。原位模块learning/src/sttw_control/tensorboard_logging.py，薄导入CLI learning/cli/tensorboard_logs.py，依赖写入learning/pyproject.toml。

历史导入仅读取声明的metrics文件，要求轮次严格递增，拒绝重复覆盖已有事件目录；保留来源SHA。将原8轮与续92轮合成单一100轮事件序列。训练事件放各运行training/tensorboard；统一服务索引runs/tensorboard下使用目录链接，不复制原始实验。启动本机TensorBoard并验证HTTP和事件中的100轮原值；不运行新训练、不修改PPO/奖励/物理参数。测试事件读回、缺失值、KL回退计数、重复导入保护及异常关闭。


## 2026-09-14 TensorBoard过程日志已接入

实现tensorboard_logging模块与tensorboard_logs历史导入CLI；train外层上下文管理关闭写入器，原metrics.jsonl落盘后每轮同步flush事件。训练参数、网络、奖励、采样与物理不变。全局PPO轮次、奖励/分项、loss、近似与最终精确KL、整轮回退/最终保留更新数、开发逐样本评估/基线差和耗时均可查看；缺失项不补0。命令评估标签绑定声明case/alpha/seed顺序。依赖tensorboard>=2.18,<3（当前2.21.0），未安装TensorFlow，未改JAX/CUDA依赖。

已有1–100轮日志回放到runs/command_balanced_100_20260914/training/tensorboard，来源SHA记录sources.json；原freeze/metrics/checkpoint保持不变。事件按运行目录存放并自动链接runs/tensorboard索引，当前服务http://127.0.0.1:6006，PID1808332，日志/启动记录runs/tensorboard_service。服务--samples_per_plugin scalars=100000；本版本0会返回空曲线，已改正并通过HTTP核验100点。

核验：新增6项事件读回、缺失值、回退保留计数、异常关闭、重复导入保护、声明样本顺序和运行索引行为测试通过；另8项既有训练/诊断测试通过。真实100点总奖励、速度/yaw分项及价值loss逐点匹配JSON，139个标量tag，开发评估仅1/8/100轮；HTTP200且服务完整返回100个奖励点。保留原图表和完成/错误通知。本次未启动新训练；新中等残差权限仍为已配置未训练。使用说明在learning/README.md TensorBoard节。

## 2026-09-14 中等残差权限100轮训练声明

用户批准开始新一轮，主要检验限制残差干预。运行command_medium_authority_20260914：前轮±1.5rad/s、后轮±10rad/s的additive残差；其余任务配置与command_balanced_100冻结任务逐字段一致（奖励、观测、动态alpha、模型与执行器物理等不变）。TrainingConfig沿用上一轮1024×256、epochs4、minibatch2048、LR3e-4、KL.01、seed64、预热池128×1800，只设从头训练100轮，无resume/incumbent，最终评估、不提前选模、每4轮出图。新增26,214,400交互步；每轮checkpoint和TensorBoard，100轮后开发6条×10s（baseline初始化另6条），最终原四工况×三alpha×seed47001×两控制器24条≤48,000步，附逐步奖励/速度估计/误差图及视频。warmup最多230,400计算步另计；无额外性能筛查。停止为100轮预算完成或错误/非有限保护，无45分钟墙钟停止、不自动101轮。

数据角色为开发对照，已有标准面板已参与多次反馈，不称独立holdout。旧full_range训练为8+92分段且中途物理重置，本次从头连续100；同seed/总预算/PPO参数不保证完全相同随机经历。只声明动作权限是本轮有意改变的控制机制，不能凭一个训练种子证明因果。新旧动作身份不兼容，不导入旧Actor或优化器。配置和预算已核验并保存runs/command_medium_authority_20260914_inputs；按用户已授权方案执行，不新增网络/奖励机制。待实际启动后记录PID与状态。


## 2026-09-14 中等权限训练已启动

运行runs/command_medium_authority_20260914，主进程1824265，监视器1824267，源码f48238c；主状态training，日志已进入MJX reset与物理阶段池预热编译。冻结输入位于同名_inputs，训练从头100轮，1024×256=每轮262144交互，共26214400；前轮±1.5rad/s、后轮±10rad/s，中等additive权限。奖励/模型/观测/alpha分布与旧100轮相同，PPO优化参数相同，无旧模型resume，最终原四工况三alpha面板。旧8+92与本次连续100的重置差异已声明，单种子开发对照，不是严格因果或泛化结论。

启动前39项命令环境、TensorBoard、PPO约束和通知测试通过，配置加载与预算/物理/奖励逐字段检查通过，TensorBoard HTTP200。每轮指标写入后自动出现在本机6006页面，首次曲线需等待初始化、基线验证与首轮采样完成。训练完成/全部评估完成/运行错误均由独立监视器通知。不动其他项目进程，无额外长实验、无自动101轮。当前只有启动证据，未产生新性能结论。


## 2026-09-14 中等权限100轮已完成：当前进展与限制

command_medium_authority_20260914 complete，通知已发；对旧full_range12/12标准回报改善，对基线6/12改善。持续减速大幅修复；仍欠转，alpha1在请求yaw0后稳定+.161rad/s，旧.2末段指标虽12/12通过但不可称任务成功。训练后期仍改善，100轮全更新、51200次小批次，无KL回退。建议保留当前权限并预声明续50轮，在125/150做小开发评估；若回正偏置仍在再设计单因素任务保持修改。此为建议，本次未启动新训练。完整逐工况分析/图在runs/command_medium_authority_20260914/analysis/review/INDEX.md，方法台账与VALIDATION已同步。

## 2026-09-14 容忍带奖励与新训练计划

用户要求alpha偏好不能放弃另一任务：速度优先时yaw角速度误差容忍.1rad/s，yaw优先时速度误差容忍.5m/s，超限强罚。采用保留原平方跟踪成本+新增容忍带成本：zv=max(|ev|−.5,0)/.1，zr=max(|er|−.1,0)/.05，H(z)=z²/(1+z²)，Cextra=10[(1−alpha)H(zv)+alpha H(zr)]，单位reward/s。alpha0只启用速度超限项，alpha1只启用yaw超限项，alpha.5各一半。阈值内新增项为0但原跟踪罚保留；连续有界额外成本≤10/s，控制步.005s额外扣分≤.05。不设宽限或新增失败终止；指令切换即按误差计算，暂态按持续时间累计，不能保证不倒或严格满足约束。当前yaw=.16、alpha1额外约5.902/s；速度误差.6、alpha0额外5/s。未设无限或突跳罚，避免单个瞬态新引入无界代价。

实现：MotionCommands新增可配置阈值/增长尺度/权重，默认关闭以保留旧奖励；新command配置启用。CPU/MJX共用reward_terms；新增speed_tolerance/yaw_tolerance分项自动进入训练JSON、TensorBoard和逐步图，诊断显示容忍线及超限时间，开发验证额外报告容忍带保持，不改变原.2末段指标或冒称任务成功。旧关闭功能配置身份兼容，启用或调整容忍带会改变模型身份。保持中等残差±1.5/±10、原网络/物理/PPO不变。

验证预算：边界、符号、alpha端点/中点、惩罚上界、旧身份/旧奖励、CPU重建与JAX一致性测试；工程GPU预检2环境×8步×1轮=16训练步，.04秒窗口、短固定开发与3alpha配对CPU图（6条，共48控制步），仅检验数据链路。通过后正式从头1024×256×100=26214400步训练，seed64，开发46001、最终47001×四工况×三alpha，每条≤10s；训练100轮后统一评估，不提前停或自动101轮，错误保护不变。奖励改变，因此不直接续旧checkpoint。最终与基线按本轮新奖励评分；跨轮总回报不能直接比，另比物理误差/姿态/超限时长。保留弹窗和TensorBoard。本轮是开发试验，不是独立holdout。

工程预检完成：53项测试通过，GPU短流水线complete，6条CPU轨迹重建最大误差7.621e-8，新增TensorBoard项和容忍图核对通过，真实旧checkpoint身份保持兼容。即将按上述冻结预算启动正式100轮。


## 2026-09-14 容忍带100轮训练已启动

运行runs/command_tolerance_20260914，源码dd12af9，主进程2225800、独立监视器2225802，状态training，已进入MJX reset/阶段预热编译。冻结输入和预算在runs/command_tolerance_20260914_inputs。1024环境×256步×100轮，从头训练；不续用旧奖励模型、不自动追加轮次。唯一任务差异为新增容忍带成本，原中等残差权限/物理/网络/训练参数保留。新奖励下重新计算配对基线，最终四工况×三alpha×seed47001，逐条≤10s并保留全部奖励图/误差图/视频入口。训练完成、最终完成、错误分别由监视器弹窗；TensorBoard本机6006服务在线。当前仅有启动和工程验证证据，正式改善尚待结果。


## 2026-09-14 容忍带100轮完成：改善偏置但尚未达标

运行command_tolerance_20260914已complete，update_0100为100轮预算终点，非best。1024×256×100、seed64；训练和流水线完成均有zenity通知。训练100轮保留、50824次小批次更新、无整轮KL回退；平均每步奖励五个20轮区间依次−.116343/−.099631/−.081380/−.076219/−.068799，最终−.065623，最终精确KL.003831。失败与原yaw罚改善，新增yaw超限罚均值却从−.012711变为−.014388，不能据总奖励上升声称容忍带学会或已收敛。训练墙钟84.76min，采样75.47min、优化19.52s；瓶颈主要在仿真采样，不能归因为minibatch优化耗时。

标准4工况×3alpha×seed47001模型12/12存活10s，基线急转4.895s失败。新公式同口径回报胜基线4/12（急转3+反向alpha0），胜旧中等权限按新公式重计分9/12；不是新旧日志直接比。冻结训练参数完全相同，任务仅新增五个容忍字段；新旧12条基线time/qpos/qvel逐元素相同。模型alpha1回正最终yaw约.1115–.1129rad/s，旧约.161，下降约30%但仍大于.1；新增放松目标末段保持8/12，alpha1四条全不通过。速度超.5仅急转短暂.07s，其余0，说明这面板对速度容忍分支检验不足。

完整逐工况/alpha/种子、指令前/持续/撤销阶段、配对奖励分项、超限时长和物理量见runs/command_tolerance_20260914/analysis/review/INDEX.md；全部逐步奖励图、PNG/PDF/NPZ和视频入口仍在analysis/INDEX.md。新训练总览training_summary.png/pdf，新旧alpha1偏航叠图yaw_comparison.png/pdf，重建数据review.json含48条新旧轨迹来源SHA。原terminal_tracking_hold按末0.5s每一步速度/yaw各≤.2，而非RMSE；此前中等权限文字中的RMSE说明有误，此处纠正，旧原始记录不动。该旧指标12/12通过也不是严格回正达标。

惩罚近阈值强度需如实解释：alpha1误差.1123时额外约.57分/秒，末段净奖励仍约+.35/s；当前H(z)在阈值附近平方起步，并非刚越线立刻极大惩罚。正奖励不等于策略因此停学，软约束也不保证达标。所有标准模型姿态罚0、峰值最高13.88°；随机训练姿态罚与失败罚非零。最大归一化动作前轮.2411/后轮.1740，无逼近残差边界证据。当前实验无外部扰动力/负载、无显式参考几何路径，仍属速度/yaw指令恢复验证。

建议（未执行）：先保持配置续50轮至150，在125/150小开发面板监视回报、超限时间和末段偏置，避免仅按总reward判断。若容忍/名义跟踪不改善，停止同配方续训，再单因素检验线性铰链等近阈值超限罚；不要同时改网络/PPO/动作范围。后续回到连续转弯持续扰动恢复主线，并补独立训练种子、固定alpha/普通残差消融及动态alpha测试。此次仅分析与报告，没有追加训练或修改控制代码。


## 容忍带续50轮与yaw正奖励150轮顺序实验计划

用户批准先续训练第一组再排队第二组。A从command_tolerance_20260914/update_0100恢复参数/优化器/RNG，新增50轮至150（物理/ESO/历史重新初始化，不是无缝物理续跑）；任务奖励不变，1024×256，每轮262144步，新增13107200步。B从头150轮、39321600步，唯一任务改动增加yaw_tracking_reward_rate=5.0、yaw_tracking_reward_scale=.1rad/s；正奖励B=5exp[-(ey/.1)^2]分/秒（用户要求加大，峰值从1调至5），固定α独立幅度，原有全部惩罚/生存/失败替换保留。失败步不发此奖励。默认rate0确保旧实验奖励和身份兼容。无物理/动作/观测/PPO学习率等改动，训练seed64。

A在阶段25/50即全局125/150开发验证；B在125/150开发验证，最终均原四工况×三alpha×seed47001配对ECBC+ESO，每条10s，开发种子46001；保留训练分项/TensorBoard/逐步图/视频及错误、训练完成、流水线完成弹窗。预算只含上述两组；每组预热池128×1800另计，无超时45min限制，不追加第151轮。不把不同奖励的原始回报直接比较，最终按同公式重计分及物理指标对照。A100+50与B连续150的物理重置差异须披露，单训练种子开发试验。

实现顺序：新增正奖励共用函数、环境计算与诊断/模型身份兼容；显式validation_updates替代为实现125/150而额外跑第一轮开发；原位扩展deferred_training为command流水线，输入哈希/单次认领/前置成功与进程退出确认，启动后独立监视器。测试奖励符号/单调/边界/失败替换/CPU重建/JAX一致性、旧checkpoint身份、指定评估轮次、队列失败与单次顺序启动。短GPU2env×8步×1轮后正式启动A，B队列等待A完整训练评估完成；队列不查询或等候其他项目GPU进程。


## 正向yaw跟踪奖励与两组顺序训练：启动前验证

按用户追加要求，正奖励峰值设5分/秒，B=5exp[-(ey/.1)^2]，独立正分项yaw_tracking；误差0/.1/.2分别+5/+1.8394/+.09158分/秒。原惩罚和生存奖励均不变，失败时仍仅−100、不发正奖励。新配置learning/configs/command_yaw_reward.json；默认关闭确保旧checkpoint兼容。

66项命令奖励、PPO约束、续训相关、队列和通知/TensorBoard测试通过；真实旧update100参数、优化器与RNG恢复及策略身份检查通过。GPU2环境×8步×1轮预检command_yaw_reward_smoke_20260914 complete，新增每步正奖励均值.02466745，奖励重建最大误差1.863e-9；6条CPU配对轨迹最大误差7.621e-8，TensorBoard正分项、PNG/PDF/NPZ已核对。工程验证不代表跟踪改善。

第一组command_tolerance_continuation_20260914按原奖励续50轮至150，新增13107200步；第二组command_yaw_reward_20260914从头150轮、39321600步，奖励峰值5。两组1024×256、其余PPO/中等残差/物理相同；第一组100+50物理重置与第二组连续150的差异须披露。均在全局125/150开发验证、150最终标准面板，不提前停、不追加。顺序队列仅等待第一组流水线完成且进程退出，两次确认；不查其他GPU作业，前置失败则停止排队并通知，冻结输入/源码哈希改变拒绝启动，独占锁和认领文件防止重复启动。等待队列上限24h只是等候期限，不是训练限时。


## 两组顺序实验已实际启动/排队

A：runs/command_tolerance_continuation_20260914主进程2581835，监视器2581837，源码4579f50；状态training，已恢复兼容update100并进入MJX reset/预热编译，新增50轮至150，奖励不变。B：runs/command_yaw_reward_20260914_queue/plan.json，队列进程2581838，等待A完整成功且进程退出后启动runs/command_yaw_reward_20260914；B当前未启动，没有结果。B正向yaw奖励峰值5分/秒，从头150轮。两个_inputs保存冻结配置与预算。

队列会校验输入/源码哈希、独占启动，前置失败不继续；A完成通知已配置，B实际启动时自动启动独立训练/完成/错误监视器，队列自身错误另弹窗。TensorBoard6006在线。第一组100+50物理重置与第二组连续150的区别保留记录，不能当完全同随机经历的因果对照。当前仅启动/排队证据，未宣称训练完成或性能改善。


## 2026-09-15 容忍带续50轮与yaw正奖励150轮综合分析

两组及队列均complete，训练/评估完成均有zenity通知。T100指原容忍100轮，T150指原奖励续50轮，Y150指增加5exp[-(ey/.1)^2]分/秒正奖励从头150轮。训练seed64，标准seed47001，4工况×3alpha×2控制器；实际审计三批72条轨迹，新旧配对基线time/qpos/qvel逐元素一致，奖励重建最大误差6.327e-6，全部标准逐步奖励/分项/误差/速度图PNG/PDF/NPZ与视频索引覆盖。两组任务差异仅正奖励两字段；T150为100+50中断物理重置、Y150连续150，不能当完全同经历的严格因果对照。

T150末段yaw容忍保持由T100的8/12升至12/12；alpha1末段yaw由+.112降至约−.0186rad/s。但末速度2.018–2.030（目标2.1）系统性偏低，标准速度RMSE全部12/12比T100大。T150原公式回报胜T100 5/12、胜基线4/12；普通任务未整体超过基线。T150急转roll峰值7.80/8.21/9.98°，虽安全余量更大但持续转弯均值yaw仅.432/.441/.507（请求1.4），不能把防摔或末段保持当全程成功。

Y150相对T150速度RMSE改善12/12、yaw改善6/12（回正/反向改善，缓转/急转RMSE退步）。其末速度2.072–2.104，更接近2.1，末段yaw三alpha约+.0125/−.0272/−.0409，12/12末段容忍保持。Y150急转均值yaw.741/.764/.855，比T150更接近请求，但roll峰值升至13.75/14.40/15.77°且撤销后振荡更大。所有最终标准模型均存活10s，基线急转4.895s失败。Y150按自己Y奖励仅3/12优于基线；把T150原轨迹重计Y奖励，Y150仅4/12优于T150，不能用新增bonus造成的回报抬高证明方法优势。按共同T奖励比较则Y150胜T150 6/12，口径差异明确保留。

训练T150续阶段101–125/126–150平均每步reward−.05546/−.05367，改善趋缓；Y150的76–100/101–125/126–150为−.06067/−.06071/−.06091，约100轮后平台且后期value loss有所升高。两组均无整轮KL回退，实际续阶段25494/Y组75472个保留小批次更新。Y125开发急转alpha0/1分别在6.855/6.340s失败，alpha.5 roll峰值.3057rad；150开发才全部存活。两组150为声明终点而非best，训练平台、KL受限、最后存活都不证明收敛/安全。

动作证据：缓转末基础后轮21rad/s，T150残差−.333/−.401/−.286，Y150残差+.471/+.291/+.145，速度改善确有指令变化；部分稳态转向残差近乎抵消基础输出。归一化残差峰值T150前/后.281/.249，Y150 .195/.205，未逼近限幅，不应再主要归因权限太小。Y150缓转速度RMSE随alpha0/.5/1为.0100/.0274/.0431，速度优先偏好未呈可靠单调行为。正奖励误差.3仅.000617分/秒、.5仅6.94e-11，窄高斯主要区分已接近目标的状态；这是公式性质，非唯一已证根因。

训练续50轮53.48min，Y150 137.28min；采样42.37/124.00min，优化14.85/35.16s。主要成本在仿真采样，无法仅据当前日志认定GPU竞争原因。完整报告/分项/阶段表和来源SHA在runs/command_yaw_reward_20260914/analysis/review/INDEX.md与review.json；training_comparison.png/pdf含两条150轮曲线，yaw_all_cases.png/pdf含12工况组合叠图。标准图分别在两组analysis/INDEX.md，开发125只有日志摘要，无完整逐步轨迹，不伪造视频或曲线。

建议（未执行）：暂停盲目续200或继续扩大正奖励；先审阅已有标准轨迹按alpha/指令段的奖励与基础/残差输出，确认普通可行命令被主动改变的原因。训练分布审计需要另声明少量采样，聚合训练日志不能重建逐环境分布。若检验奖励形状，应单因素改变正奖励宽度，不同时改峰值/网络/PPO；严格因果需匹配起点/重置流程。论文主线仍需持续转弯两类动态扰动、固定alpha/普通残差消融、多训练种子和动态alpha测试；当前只有指令任务，没有能量效率/实车/机制创新证明。此次不修改训练控制代码，不启动新训练。

## 2026-09-15 α权限门控与T150续训（已授权，实施计划）

目的：检查有限偏好门控是否减少持续残差偏置，并单独检查T150继续学习的趋势。原始基线提交b623a93已含ECBC调用±4rad/s及XML转向ctrlrange±3rad/s；不改变这两个物理/控制配置。

- [x] 共享执行合成增加可选基础指令投影，默认关闭以保留旧checkpoint；新command观测提供投影后的基础转向量，trace保留原始base。
- [x] command增加可选g_min=.2，g=.2+.8|2α−1|；两路动作同乘，训练/CPU评估/逐步奖励重建一致；动作成本使用门控后的归一化动作，原动作与实际残差分别记录。旧配置关闭门控、身份保持兼容。
- [x] 验证端点/中点/连续α、零残差一致性、饱和处反向作用、奖励重建、旧模型加载、短GPU训练检查。
- [x] A新门控从头200轮：沿用T150容忍奖励（无新增yaw正奖励），1024×256，4epochs/minibatch2048/KL.01，seed64；52,428,800控制转移。B原T150恢复模型/优化器追加100轮到250，26,214,400转移，不改变奖励、门控或投影；物理闭环重置如既有续训流程。
- [x] A先启动，完成全流水线后队列启动B；均10秒回合，开发每50轮及最终，标准4场景×3α×1seed×配对基线；最终checkpoint是预算终点，不自动称best或收敛。保留TensorBoard、逐步奖励组成/速度图/视频、训练结束及完整结束独立通知；出错停止队列。无自动超预算续训。

这两组同时包含训练长度/起点差异；不是用来单独证明门控的因果优势，最终需与同预算无门控对照区分。此次不修改DVGC，不修改实车执行链。

启动核实：A主进程351193、独立监视器351194，runs/command_alpha_gate_20260915/status.json为training，已进入MJX编译；B队列351195，runs/command_tolerance_extension_20260915_queue/plan.json等待A完整成功后启动。源码f869a26已push，输入/源码及T150快照哈希已冻结。TensorBoard原6006服务保留，正式训练初始化后自动加入runs/tensorboard索引。训练结束和流水线结束分别提示。

## 2026-09-15 门控结果与best补齐（最新分析）

G200正式训练/评估complete；T150追加100轮正在运行（本次检查到168，尚无T250最终结果），其训练/评估监视器正常。G200训练结束和全流水线结束均有zenity通知。

用户追问best后发现command_selection=false把固定预算端点评估与best保存混在一起，导致旧best_checkpoint=None。本次增加独立best_model和best_model.json：固定开发六组合平均累计奖励最大；仅从已评估checkpoint选，不用随机rollout平均reward替代；资格/失败/全程退化另记录，不把best当合格。50/100/150/200轮开发均值−78.830/−44.195/−63.126/−71.444，因此G100是已评估四轮中的best。原G200末次记录不覆盖。未来训练每次开发验证自动维护best_reward_checkpoint；当前运行的T续训不重启，轻量select_best监视器维护索引，兼容的恢复起点T150也参加比较。

G100补跑相同四工况、三α、seed47001的24条配对标准轨迹，全部奖励分项/速度/逐步图/视频已生成；best补评完成通知已送达。G100相对G200的12个累计回报均更高，但急转α=.5在5.180s失败，G200是5.630s，不能把回报更高当每项安全性能更好。末段双误差保持T150=12/12、G100=11/12、G200=7/12；均不等于全程成功。G100仅3/12回报超过T150（均常规α=.5），对基线仅4/12（3个基线提前失败的急转，1个反向α0小幅提升）。普通9组合8个未胜基线。G200普通9组合全部未胜基线，α1回正后持续约−.22rad/s。

报告runs/command_alpha_gate_20260915/analysis/review/INDEX.md，best标准图在best_evaluation/analysis/INDEX.md，指针training/best_model→checkpoints/update_0100。72条轨迹奖励重建最大6.3263e-6，跨运行基线time/qpos/qvel逐元素一致。G200基础转向11/12组合从未超±3，失败急转中点仅1步超界，普通退化不能主要归因限幅死区。G200训练后段平台、开发性能下降；优化器63,951个保留小批次、无整轮KL回退，不能因此认定性能稳定。

下一步：等已授权T250结束，比较各阶段best/last及配对基线，再决定下层候选。暂不启动上层α训练或新奖励实验；若继续研究门控，应分开偏好与权限，做固定权限/仅投影/门控的同预算消融，并检查1–3s动态α训练与10s固定α评估的分布差异。此次只修复best保存/索引，不改变正在运行的策略训练或预算。

## 2026-09-15 跨方法展示与精简监控
已将固定command场景的XY参考/实际轨迹和逐步总奖励叠加实现为command_comparison共享模块，command_diagnostics每次自动生成；显式manifest可比较多个冻结模型，不重新仿真。当前T150/G100/G200+ECBC四场景、三alpha、seed47001共四组轨迹图和四组奖励图位于runs/command_alpha_gate_20260915/analysis/trajectory_comparison。参考只是原始指令的理想积分，偏离参考与存活分开判断；急转弯失败端点保留。
TensorBoard 6006切换到runs/tensorboard_core/events，显示T连续训练、G门控、Y额外yaw奖励三条方法系列。精简投影持续追踪metrics.jsonl，原始事件和完整JSON不删。新训练直接记录core标签；当前已运行trainer不热改，由投影过滤。开发总回报均值与训练每步平均reward名称明确区分，不能当作同一统计量。

## 2026-09-15 T续训结束后的固定checkpoint复核
T150续训100轮已完成到T250，训练/流水线完成弹窗均已由原监视器送达。冻结task与T150完全相同，训练参数仅续训起点、轮数及开发验证时刻不同。固定开发均值T150=-24.217882、T200=-27.157631、T250=-27.160751，best_model仍指T150；继续优化随机训练reward不等于标准任务改善。
本次按用户要求补T200标准四工况×三alpha×同seed47001，24条配对回合，每条≤10s，共≤48000控制步；不训练、不改变物理/奖励。保存到checkpoint_evaluation/update_0200，独立监视器负责失败和完成提示。与已有T150/T250记录共同生成analysis/comparison中的轨迹、逐步奖励、分阶段表和训练曲线。此面板已经用于开发，不称独立测试。
T200补评现已complete；analysis/comparison报告48条不重复轨迹，重建最大误差6.3263e-6。T200/T250各3/12标准回报超过T150，普通9组合均不及基线；T250速度12/12改善但yaw仅2/12改善。保留T150开发候选，下一步建议冻结模型比较固定/分段alpha并核对转向补偿，不自动续训。具体逐阶段表、所有图和建议见该报告。

## 2026-09-15 RSL-RL替换与用户指定200轮预算
已快进同步远端2032491的显式alpha几何恢复实现；中断前本地原型仅保存于Git stash，不合并回已完成实现。新几何配置使用RSL-RL 3.2.0官方PPO，MJX物理/ECBC/ESO/280维alpha历史观测/奖励不变。DLPack设备交换，RSL存储pre-tanh高斯动作，物理收到tanh；超时加gamma*V(真实下一状态)，不使用RSL默认当前状态timeout捷径。ELU激活随导出元数据保存，旧LeakyReLU模型仍按原激活加载。
用户已授权4096×24、minibatch24576、epochs5、lr=.001 adaptive、200轮：每轮98304转移、20小批更新，总19,660,800转移及4000小批更新。gamma=.9995、GAE=.99、clip=.2、std=.15、KL目标=.01，RSL原生自适应LR不是原JAX拒绝/回退。rollout边界不reset，只有真实done清理全部状态。阶段真实warmup64×1400额外89600计算转移。开发1/25/50/75/100/125/150/175/200，标准阶段末只用经典三扰动+名义、alpha0/.5/1、seed49001；最多24条10秒回合。工程短测先核验ELU导出、timeout、真实GPU两轮链路，再启动200轮；不自动延长预算，不声称替换库必然提高性能。

正式流水线已启动：runs/path_rsl_4096_20260915，launcher PID1670321；冻结声明预算19,660,800，当前训练初始化。监视器PID1670322，状态在runs/path_rsl_4096_20260915_inputs/monitor，分别监控training/status.json与pipeline_status.json。TensorBoard既有6006服务新增RSL_4096_ELU系列；代码提交eacbaa8已推送。训练是否有效以完成后的逐工况配对结果为准。

## 2026-09-15 RSL200完成复核与强制图交付
训练200轮19,660,800交互完成；九次GPU开发评估均未全门槛达标，best_reward_model为R200（开发受扰平均45.568），best_checkpoint=null不是未保存best。CPU标准seed49001四工况×alpha0/.5/1，模型12/12存活并末段共同容差保持，严格完整任务11/12（含3无扰动；受扰8/9），配对基线存活3/12、严格0/12。α0侧向力恢复超时仍判失败，不能放宽标准。基线无扰动7.21s摔倒；新任务同时改动几何参考/奖励/网络/训练器，不能单独归因RSL。α1常同时改善速度/路径，尚无上层α或可解释取舍证据；后轮.4Nm实际注入但与nominal很接近，不能称强负载恢复。
报告runs/path_rsl_4096_20260915/analysis/review/INDEX.md含全工况全alpha逐阶段表、四场景轨迹、逐步reward和训练图；完整奖励分项/速度叠图保留于analysis/reward_breakdown。训练记录墙钟74.7min，rollout39.0min、开发27.8min，优化计时受异步影响不作为精确GPU性能结论。训练与完成弹窗均已确认zenity送达。下一步优先无扰动基线审计、同初态CPU/MJX核对和α0侧向力期限诊断，不自动续训。
用户再次要求每次先给图：tracking_diagnostics.generate_panel现自动生成analysis/comparison全声明场景/alpha/seed配对轨迹+每步总奖励PNG/PDF与证据哈希索引，AGENTS.md已明确几何任务同样强制。实际重建本轮12组24文件通过；相关行为测试14 passed、1 skipped（opt-in GPU）。原冻结实验记录未改，新增图属于事后审计。

## 2026-09-15 累计奖励与场景身份纠正
用户要求累计总奖励对steps已纳入tracking_diagnostics：逐步求和保留终止罚；每步奖励分项和累计分项同时包含配对基线。全12组合累计终值与记录episode_return核对一致，8项相关测试通过、1项opt-in跳过。新增总览analysis/review/cumulative_rewards.png，各组合在analysis/comparison和analysis/reward_breakdown。
本轮path_rsl使用新几何平滑弯道+nominal/force_right/steer_left/rear_load面板，与历史gentle/tight_turn/straighten/reversal命令四场景不同；前次“四场景”表述未强调任务切换，现明确纠正。旧四场景冻结文件保留，未声称本轮模型已完成旧面板评估；不能跨任务比较累计奖励或将改善单独归因RSL。未启动额外训练。

## 2026-09-15 保留当前几何奖励，混合四类参考轨迹
用户澄清并确认：保留path_rsl本轮全部奖励、280维10帧观测、显式alpha、ECBC/ESO、执行器残差权限及RSL超参数。只把参考任务更换为原gentle/tight_turn/straighten/reversal指令积分得到的固定全局轨迹，用同一策略混合训练；不恢复旧T150 yaw-rate奖励。错误方向的未提交适配和配置已撤回，正式训练没有使用过它们。
新增可配置reference_paths，reset均匀抽取四类；轨迹编号是环境状态、不作为Actor新增特征，回合内不重选，自动reset同时重置历史。按原指令速度与偏航积分生成参考，几何跟踪控制器仍使用原lookahead/转向界；不宣称参考在工作姿态范围内必然可行。保留原随机扰动分布及无扰动比例。新任务配置仅reference_paths和speed_schedule与上一轮不同，奖励/观测/权限配置逐项相等。
200轮×4096×24=19,660,800训练转移，20优化/轮，ELU，LR .001 adaptive；训练seed65，开发48001，标准49001，全部episode≤10s。沿用九个开发时点，四类参考×三扰动×三alpha并配对无扰动，共72个开发回合/次。结束标准4参考×4扰动面板（含nominal）×3alpha×2方法=96条回合、最多192000控制步。各参考分别展示，不能当作旧直接yaw-rate控制的同定义结果。工程短测后直接启动，不自动追加预算。

正式启动确认：runs/path_reference_rsl_4096_20260915，pipeline PID2116022，watchdog PID2116023。pipeline_status=training，training/status=initializing，监视器running且连续错误0。TensorBoard6006新增RSL_four_references。GPU短测、16条CPU面板及反向轨迹奖励图/视频均完成，重建误差2.06e-9；实现提交b95e3be已push。未宣称正式200轮已经完成或学会任务。

## 2026-09-15 四类参考R200结果与投影缺陷
path_reference_rsl_4096_20260915完成200轮及96条标准评估。R200累计回报0/48优于基线，末段共同保持及严格任务0/48；存活31/48对基线21/48，不能称成功。开发best_reward_model=R50（−62.20，受扰20/36失败、末段0/36）；已标准评估的是R200（开发−153.99），不是R50。全工况/alpha/seed与逐阶段结果、累计图、轨迹见analysis/review/INDEX.md。训练64.5min，pipeline约100min。
首要缺陷：全局最近点投影不适用于新交叉参考；alpha.5急转无扰动1.425s模型参考进度从2.74跳到17.44m，基线1.460s从3.00跳17.69m再返回，实际车体位移为厘米量级。发生在转弯/外扰前，原0.06s短测未覆盖。奖励重建max5.40e−6只证明公式一致，不证明正确路段选择。下一步先修连续进度/局部可达投影、补全程零残差测试，奖励/网络/权限/PPO保持，验证后再考虑重训；不能把本轮负结果仅归因RSL或reward。完整96条projection_audit.json保留。当前只分析，未续训。

## 2026-09-15 连续投影修复与同预算重训
用户已批准修复后直接训练。新增显式continuous_projection开关（旧冻结任务缺省false），当前参考库启用。EnvState保存path_progress；每步在上一进度±(0.05m+2×实际XY位移)区间内做线段投影，支持回退，reset归零。控制器lookahead、Actor路径特征、奖励及CPU/GPU评估均读取同一进度，轨迹/奖励图记录实际值。局部搜索范围是数据关联约束，不是稳定性或动态可行性保证。
配置逐项比对：除continuous_projection/projection_margin外，奖励、参考轨迹、观测维数、alpha、基础控制、残差权限、随机扰动及RSL超参数均等于上一轮。重新从零训练200轮×4096×24=19,660,800转移，warmup64×1400，9次开发、最终96条标准回合，沿用训练65/开发48001/标准49001。新输出runs/path_projection_rsl_4096_20260915，先通过四类10s零残差检查（失败保留真实终点）及GPU短链路再启动，不追加预算。
当前250 passed、2 skipped，包含交叉处不得跳支路、正常回退、区间边界测试。奖励公式文件未修改；历史任务/模型仍保留全局投影行为。

实际GPU RSL短测2轮32转移complete，采样/优化、跨参考开发验证、best索引、ELU导出及训练图通过；奖励重建最大误差3.725290298461914e-09。证据runs/projection_validation_20260915/training。准备启动相同200轮预算，短测不作为效果证据。

正式训练已启动：runs/path_projection_rsl_4096_20260915，PID2627237；监视器2627238已确认running、consecutive_errors=0，训练阶段initializing。TensorBoard6006新增RSL_continuous_projection，训练结束和全流水线结束分别通知。实现已本地提交cf6606f；两次push均遇gnutls_handshake TLS连接中断，保留本地提交，不声称远端已更新。

## 2026-09-16 连续投影RSL训练完成复核

`runs/path_projection_rsl_4096_20260915`完成200轮、19,660,800转移和96条标准回合。标准R200：累计回报20/48优于配对ECBC+ESO，存活36/48对21/48，但完整任务及末段联合保持均0/48。gentle全部回报退化；tight_turn全部失败；straighten 9/12回报改善且路径误差下降、速度误差上升；reversal 11/12回报改善但后段偏轨，不能算成功。各自失败观察窗不同，不用全程RMSE作等时长因果比较。

奖励最佳已保存R75（开发均值−16.29、9/36失败）；R200开发−58.33、9/36失败。R150开发0/36失败、6/36末段保持，但回报−44.05。标准面板只评了R200，尚无R75/R150同面板结果。没有合格候选时流水线回退last不等于reward-best；后续先补评R75和R150再决定优化，不盲目续训。

96条实际progress窗口检查0违规，最大单步.033902m；48条模型逐步奖励重建最大1.14858e−7。全场景/α/seed逐阶段表、轨迹、每步/累计总奖励和分项见 `runs/path_projection_rsl_4096_20260915/analysis/review/INDEX.md`（另comparison、reward_breakdown索引）。训练约64min，采样38.4min、开发18.5min、记录优化12.2s；这次未改代码、未追加训练。单训练种子开发证据，尚不支持泛化或论文性能结论。

## 2026-09-16 R75 best与R150配对补评
用户要求直接绘制best模型及R150。冻结原四参考×四设置×三alpha×seed49001，复用原CPU基线，新增96个残差回合、最多192000控制步、每回合≤10s；6个CPU评估进程，不训练。输出runs/path_projection_rsl_4096_20260915/checkpoint_comparison，统一叠加基线/R75/R150轨迹、每步及累计回报，并逐checkpoint重建全分项。失败保留真实终点；异常停止并弹窗，完成也弹窗。之后主图默认best_model.json记录的开发奖励最佳，不默认last；best未完成评估必须明确标记。

## 2026-09-16 R75 best及R150标准补评完成

同一冻结参考/奖励/物理/初态49001，四参考×四设置×三alpha；新增96个残差CPU回合，基线逐字节复用原面板并核验96/96副本一致。6进程评估，仿真276.7秒，含全分项及跨模型PNG/PDF绘图总399.6秒。无新训练，完成zenity弹窗已记录。
R75回报胜基线35/48（gentle11、tight0、straighten12、reversal12）；失败12/48全部急转，末段联合保持0/48、严格任务0/48。R150胜33/48（7、4、10、12）；失败0/48，末段保持7/48、严格任务0/48。不能由不摔判任务完成。R75是固定开发奖励best，不是R200，补评证明不能用R200代替best主图。
输出 `runs/path_projection_rsl_4096_20260915/checkpoint_comparison/INDEX.md`：13张跨模型PNG及PDF、全部48条件轨迹/每步及累计回报；各模型分别48条分项、误差和速度诊断。首页小图明确无外扰、alpha=.5、四参考；完整图覆盖alpha0/.5/1及持续外扰4–5s，失败打叉，不补齐伪造后段。R75奖励重建max1.16543e-7，R150 max9.56505e-8；哈希保存在source_hashes.json。一个训练种子，开发诊断而非泛化证明。

## 2026-09-16 alpha误差叠图与best覆盖纠正

新增共享tracking_diagnostics.write_alpha_error_overview，由generate_panel自动调用。每工况/seed同图叠加全部alpha的真实速度误差与几何有符号横向位置误差、配对基线、持续扰动区间和实际失败终点；PNG/PDF/NPZ及哈希。R75/R150各16工况已从原轨迹生成，不重复仿真，入口checkpoint_comparison/INDEX.md。
RSL保存全部200轮模型，但此前仅9轮固定开发评估，R75不是已证明的全程最佳。新增显式best_model_every_update模式；ppo_path_priority.json启用，未来每轮更新后在同一固定开发面板比较并维护单一best_model；不拿更新前rollout reward作为更新后模型分数。selection_coverage记录未评估轮次，续训跨阶段覆盖仍需单独核查。不自动重跑历史200轮审计、不改变冻结记录，本次未训练。全部开发评估将从9次增加到200次，增加显著成本；绘图仍按独立节奏，不能宣称效率不变。历史R75保持候选身份。

## 2026-09-16 三层128网络同预算重训

用户要求128→128→128三层结构并按昨日连续投影任务再训练一次。Actor和Critic同改；ELU、280维显式alpha历史、固定残差权限、奖励、连续投影、任务/扰动分布、训练seed65均保持。旧配置默认256→128，旧模型元数据加载保持兼容；隐藏尺寸改为TrainingConfig配置，贯穿RSL构建、Flax开发验证、导出与恢复检查。

- [x] 先补三层网络非零输出的Torch/Flax导出加载一致性测试，以及旧默认与配置合法性检查；再实现尺寸传递。
- [x] 标准流水线新增显式evaluation_reward_best，开启时严格使用已保存固定开发奖励best，资格另记录；测试不得回退last。
- [x] CPU相关回归与GPU两轮短测（32训练转移），通过后冻结原任务/标准面板及新训练配置。
- [ ] 从零200轮×4096×24=19,660,800训练转移；warmup64×1400=89,600计算转移；20小批更新/轮。每轮固定开发72回合×最多2000步，共最多28,800,000开发转移，另初始基线最多144,000；开发seed48001。每轮best验证沿用用户最新要求，相比昨日9次增加成本。
- [ ] 标准best评估4参考×4设置×3alpha×配对方法=96回合、最多192,000转移，seed49001，每回合≤10s；配对轨迹/每步与累计奖励/分项/alpha误差和视频自动交付。全部为开发诊断，不称独立测试。
- [ ] 预算终点停止，不自动续训；异常停止并通知。启动并确认独立监视器，训练完成与全流水线完成分别弹窗；检查Git diff后commit/push。

第二组用户确认：三层128网络基础上，所有alpha相关六项奖励统一乘3，tracking_rate=12、tail_rate=.3、budget_rate=6；其余奖励、容忍带、物理和残差权限不变。配置learning/configs/path_reference_reward3.json。它放大alpha评分差异，也放大跟踪相对共同奖励的强度，不能解释成纯偏好差异消融。第二组同200轮、同种子、同验证/标准预算，从零训练，第一组全部成功后才启动；失败停止队列。输出分别runs/path_mlp128x3_20260916与runs/path_mlp128x3_reward3_20260916。队列使用共享deferred_training的process依赖模式，仅等待声明的第一组进程与完成状态，不等待或停止其他GPU任务。

启动核实：第一组pipeline PID233196、监视器233197，pipeline_status=training、training/status=initializing；第二组队列PID233198等待第一组全流水线完成。两组输入/源码哈希和PID起始身份已记录。CPU全套258 passed2 skipped，最终相关24 passed，GPU两轮奖励重建最大3.72529e-9。独立只读代码复核无阻断项。两组正式训练均未完成，后续以各实时status与评估索引为准。

## 2026-09-16 随机指令与时间轨迹任务切换（用户批准）

旧A在已记录35轮/3,440,640训练步后取消，完整checkpoint与best（停止前update0020）及原始metrics均保留；中断轮可能存在未完成验证的checkpoint，不计完成预算。旧B未启动并取消队列。停止身份、原状态快照见runs/path_mlp128x3_20260916_inputs/stop_record.json。停止原因是任务定义变更，不是宣称模型已收敛或当前结果不佳。

新任务设计与实施计划：
- [x] 独立时间参考：reset随机连续采样速度1.7–2.5m/s、yawrate±.6rad/s，初始2.1m/s/0；三次切换时间分别U[1.5,2.5]、U[3,4]、U[5,6]s，速度/yaw变化率限制.5m/s²/.6rad/s²；最后目标保持至10s。参考只在reset与实际位姿对齐，之后精确积分前一步平滑请求，禁止随实际位姿重定位。
- [x] 同时观测和奖励速度/yawrate、时间参考沿程/右法向位置及航向误差；沿程actual-reference负为落后。保持ECBC/ESO与原物理，基础转向由参考yaw前馈及航向/横向反馈生成；Actor新增可定位观测的沿程误差及yaw请求/实测，尺寸绑定checkpoint。alpha1命令优先、alpha0位置优先，权限固定；共同最终沿程与yaw门槛.15m/.15rad/s，加旧速度.2/横向.1/航向.15/姿态门槛与.5s保持，离带起3s恢复期限。
- [x] 采用已批准三层128与3倍相关跟踪项（tracking_rate12、tail_rate.3、budget_rate6）；新增yaw项与速度平分命令奖励，位置正奖励联合沿/横/航向，旧几何配置缺省行为和身份保持。
- [x] CPU/JAX积分、奖励与参考重建、旧checkpoint兼容、CPU/MJX闭环及三层GPU短测通过后才正式启动。
- [ ] 新策略从零200轮×4096×24=19,660,800训练步；warmup64×1400=89,600计算步。固定开发随机种子48001–48004，三扰动×三alpha配对nominal，每轮72回合最多28,800,000开发步，另初始基线最多144,000。开发参考来自随机分布，绝不使用标准四参考选best。
- [ ] 标准四参考仅在结束后评估，使用旧指令表但同样经过新声明的slew限制独立积分；4参考×4扰动设置×3alpha×seed49001×配对方法=96回合、最多192,000步。旧标准已用于历史开发，不称全新独立holdout；新结果不能直接与旧奖励/参考回报比较。每条件轨迹、每步与累计分项、沿/横/yaw/速度和alpha叠图完整覆盖。
- [ ] 先短工程筛查，错误停止；正式预算不自动增加。新配置、源码、耗时/样本预算冻结，监视阶段/流水线完成及错误，Git逻辑提交后push。

正式启动核实：runs/timed_random_rsl_4096_20260916，pipeline567379，monitor567380，training/status=initializing。冻结输入/预算在_inputs/experiment.json，pipeline declaration包含开发/基线/warmup成本。最终全套311 passed2 skipped；新增开发gate后的GPU training_verified两轮32步complete，固定随机开发每轮验证、best索引、沿程/yaw gate接线通过。当前只证明启动，不宣称训练完成或任务提升。

## 2026-09-16 用户授权取消训练内评估并续训
- [x] 精确暂停当前 timed_random 流水线、训练器和监视器，保留原始记录。
- [x] TrainingConfig 增加 RSL 专用 training_reward_selection；开启后不构造验证器、不跑基线/中间/最终开发评估，保留旧模式用于旧配置。
- [x] 用本轮实际采样平均每步奖励选择产生该采样的更新前 checkpoint；最后更新没有后续采样时不冒称已计分。best_model.json 明确随机训练批次排名、无任务合格结论，原开发 best 保留在旧目录。
- [x] 单测验证禁用评估和奖励/checkpoint 对应；GPU 短续训验证 snapshot 恢复、无开发评估及 best 链接。
- [x] 从最新完整 snapshot 新目录续训至总计200轮，额外仅声明重建物理初态 warmup；不自动运行后续标准评估。保留原场景/网络/奖励/优化参数；更新文档、commit/push，启动监视器并核实真实进度。

最终验证315 passed2 skipped；GPU两轮32步续训无任何评估、reward/policy映射正确。正式续训目录runs/timed_random_rsl_4096_20260916_reward_only/training，trainer818060、monitor818061；TensorBoard6006并列显示旧开发阶段与新训练奖励阶段。旧update42完整snapshot保留，剩余158轮预算15,532,032训练步，warmup89,600计算步，所有自动评估预算0。

## 2026-09-16 reward-only训练完成与结果分析
已核实完整200轮、19,660,800训练转移；续训158轮32.63分钟（含初始化），稳定采样中位10.41s。训练采样best=policy192，在update193获得0.0496043；policy200已保存但未采样计分。最后20轮平均0.048711，主要是位置/yaw正奖励与超带惩罚改善，不能推出alpha有效或任务达标。没有执行额外评估/续训；建议以后单独标准面板诊断best。完整图/数据/局限见runs/timed_random_rsl_4096_20260916_reward_only/analysis/REPORT.md。

## 2026-09-16 后续日志与三类图片
- [x] 训练日志按alpha区间[0,1/3)、[1/3,2/3)、[2/3,1]记录采样数、平均奖励、物理失败次数、速度/yaw/沿程/横向RMSE和有符号均值；明确这是随机训练样本分组，不是精确alpha0/.5/1配对评估。预转移alpha/请求与后转移状态匹配。
- [x] 现有完整轨迹日志导出逐控制步CSV，包含参考/实际XY、速度、yaw、along/lateral误差、每步/累计总奖励与分项、终止标志；保持trace.npz原始证据。
- [x] 使用训练奖励best192单独标准诊断：4指令×4扰动×3alpha×baseline/residual×seed49001=96回合，各≤10s，总上限192,000转移，非训练内评估，不续训。生成XY/每步/累计奖励三联图与既有全覆盖诊断。
- [x] 行为测试、工程短测、逐步重建、完整索引；更新台账、commit/push。

完成：320 passed2 skipped，GPU日志两轮32步验证通过；标准best192完成96回合，48组三联图/16组alpha误差叠图/逐步CSV及全覆盖配对指标齐全。入口runs/timed_random_rsl_4096_20260916_reward_only/standard_best192/INDEX.md。残差与基线均12/48物理失败（急弯），末端保持7/48与0/48；位置/yaw跟踪有改善，alpha效果可测但速度偏好不单调，未宣称整体达标。

## 2026-09-16 补齐同场景跨alpha轨迹与结论证据
用户指出原主图按alpha拆分，缺少同场景XY叠加。已在共享write_alpha_error_overview原位增加16组XY全图/自动最大同时间分离局部放大、PNG/PDF/原始坐标与时间NPZ，含三alpha、基线、参考、扰动粗线与失败终点；不重跑仿真。固定初态/事件/配置和共同时间网格检查通过。16/16残差轨迹随alpha变化，基线跨alpha坐标差为0；事后以冻结共同最终容差归一化，13/16端点符合命令联合误差降低/位置联合误差增加，10/16三点单调；速度单项并非稳定单调，急弯仅共同失败前窗口，不能宣称普遍可控或达标。证据表/反例与哈希位于standard_best192/analysis/alpha_errors/EVIDENCE.md。相关7项测试通过，16组真实数据图生成并检查。

## 2026-09-16 用户授权新测试水平与直接全范围训练
- [x] 新面板timed_priority_inrange_panel.json保留原4类指令、扰动与时序，速度目标裁至[1.7,2.5]、yaw裁至±.6；旧面板不覆盖。best192固定seed49001、三alpha配对基线，共96回合≤10s/最多192,000转移，先运行并查看结果。
- [x] 用户追加边界余量后，最终新训练timed_reference_standard_range.json直接随机速度[1.7,3.0]、yaw±1.8，不采用逐级课程；原三次随机切换窗口和slew保持。原标准2.7/1.4处于范围内部，幅值覆盖不等于全部切换时序覆盖或可行性保证。模型/执行器/奖励/alpha回合采样及最终门槛不变。
- [x] 相关32项测试；最终配置GPU两轮32步training_margin完成，无评估。原较窄配置的预备短测32步单独保留，工程总64步，不用于正式训练权重。
- [x] 新任务从零200×4096×24=19,660,800步，warmup64×1400=89,600计算步；三层128、reward-only best、alpha分组日志。无训练内开发评估、无自动延长、无自动末端评估。新面板完成检查后立即正式启动并核实更新，监视器通知。

新面板96回合物理运行已完成并检查：残差0/48物理失败、10/48终态共同保持、41/48恢复期限违反；基线3/48失败、0/48保持、48/48期限违反。完整分项图后台继续。随后已启动runs/timed_random_standard_range_20260916，最终随机范围[1.7,3.0]m/s、yaw±1.8，从零200轮。用户纠正后续标准默认仅四场景×三alpha=12残差回合，不默认扰动笛卡尔积；相同条件复用基线轨迹、重算alpha奖励，本次完整96回合不删除。

正式宽范围训练已核实完成前4轮393,216步，状态training，训练范围3.0/1.8和每轮alpha分组98,304样本核对一致，无开发评估。trainer1314906/monitor1314907。

## 2026-09-16 奖励再放大/残差权限诊断（只分析）
用户指定inrange_best192 gentle__force_right跨alpha图。已用原三轨迹核对：最大XY分离.19979m；撤扰后alpha1速度RMSE.01250对alpha0 .01594，XY .25163对.13168，存在行为取舍但全程速度不单调。残差归一化输出最大转向.268、后轮.069，没有接近限幅或有效指令裁剪，不支持本工况先增输出权限。速度正奖励已达上限99.3–99.6%；固定轨迹交叉评分中即使评分alpha1，也选alpha0轨迹（110.496 vs105.007）。统一再×3、ratio20、speed_fine .05的离线重评分都没有改变最优轨迹列；不能据此预测重训，但不支持把简单放大当确定解决方案。报告inrange_best192/analysis/reward_authority/REPORT.md。未改正在运行的wide训练/奖励/权限，未新仿真。

## 2026-09-16 奖励设计表格远端交付
用户要求把gentle__force_right三alpha奖励分析数据制作表格并推送。版本化派生表位于docs/data/reward_design_20260916/README.md，包含6行配对指标、奖励分项、9行分阶段误差、动作权限、12行交叉评分（含离线变体）、冻结参数和同观测网络响应；ZIP含6×2000步CSV及来源哈希。未上传模型或完整训练产物，未改运行训练/奖励。数据核对分项/累计总和，ZIP校验及逐步行数通过。

## 2026-09-18 用户批准随机几何取舍与200轮训练（实施中）

目的：在相同ECBC＋ESO与残差权限下，让alpha表达速度/固定原始几何路径的有限取舍。用户本轮明确选择允许沿路径暂时落后，取消时间位置追赶与原始yaw-rate奖励/共同保持门槛；时间积分仍生成不可重定位原路径，along/yaw继续作诊断。与旧时间任务不直接比较回报。

设计及实施计划（writing-plans，现有文件原位维护）：
- [x] tracking_reward.py增加可选geometric/huber模式，保留旧默认/身份；尺度速度.30m/s、横向.10m、航向.15rad，航向成本乘.2，主成本率12/s，有限权重比20作为待验证起点，不宣称复现合成拟合34.2。取消恢复正奖金；共同姿态/动作约束保持。过程速度容忍[1.0,.2]m/s、横向[.1,.4]m，7s离带预算内在末.5s保持前线性收紧至共同.2m/s/.1m；首次超时罚10、仍超时离带成本2/s，超带率2/s。无提前知道扰动结束。
- [x] timed_reference.py生成80%随机持续目标→4s开始yaw收回至0的恢复机会回合，20%保留原随机压力流；速度1.7–3/yaw±1.8、slew .5/.6、10秒不变。恢复回合首次目标于U[.5,1]s到来，持续至4s，随后同速度/零yaw。区分提供低曲率收尾与已证明动态可行，压力流不冒称保证恢复；不新增从不完整快照拼接的偏离reset。
- [x] timed_env.py在固定原路径上局部连续投影；只允许已下发参考段，无未来随机指令预瞄。观测及基础路径反馈用投影切向/曲率；速度目标仍当前外部请求。沿时间位置/yaw保留诊断而不参与目标。310维10帧/三层128保留，特征语义绑定新模型。
- [x] 行为测试先失败后实现：端点成本排序、停滞/偏移反例、计时/失败罚、几何落后曲线投影、连续分支/因果、CPU/JAX一致、旧checkpoint兼容；逐步奖励与参考重建。
- [ ] 有界GPU工程测试2轮×8环境×8步=128训练步，预热8×1400=11200计算步；另CPU配对闭环最多4回合×10秒=8000步。正式从零200×4096×24=19,660,800步，预热64×1400=89,600计算步，原优化参数不变，训练奖励best，不执行训练内/末端自动评估。
- [ ] 通过后冻结task/training/源码与物理哈希，新目录启动训练及阶段完成/错误监视器，核实真实更新；相关检查、逻辑commit及push。预算终点停止，无自动续训。

成立条件：两种偏好所需动作在同一受限残差闭环中可实现；低曲率段及剩余时间足以让落后车体退出弯道和纠偏；部署可获得定位/速度估计及已收到的路径；优化收敛到有效策略。准静态v|yaw|筛查不能证明这些条件。普通可兼顾工况不要求alpha人为拉开差距。首次实训为设计筛查，不宣称上述条件已全面满足。

用户随后确认采用本轮手动编辑的XML：default/main geom摩擦.3→.8，前轮独立10/priority1、后轮独立.3/priority1保持。新训练冻结该版本，作为明确物理变化，旧结果不能用于单因素奖励消融。

本轮追加控制器核对：用户怀疑后轮未反馈，实际ROS computeControl调用updateSystemParams(rear_vel*.1)，Python _prepare同样向controller_step输入rear*.1。后轮外层基准是v_ref/.1，但MuJoCo velocity执行器kv3已含轮速误差闭环，实际mj_forward对−21目标与−20.5/−21/−21.5实测输出−1.5/0/+1.5Nm，限制±3Nm。没有证据支持按“缺轮速反馈”新增积分器；外层真实车速/滑移反馈是另一个设计，需可得速度估计和调参，本轮不冒充ECBC错误修复。审计保存在_inputs/controller_feedback_audit.json。

用户最新效率指令：后续仅保留冻结配置、模型/checkpoint身份所需校验，不重复计算大量源文件/派生图哈希；不额外重复扫描。

实现阶段已通过CPU全套346项（1跳过）及最终改动相关32项；配对真实轨迹与增量参考重放通过独立奖励重建。前两组GPU共256训练步完成，第三组128步验证增量参考，额外工程预热每组11200计算步。正式配置入口为learning/configs/geometric_random_recovery.json和ppo_geometric_random_recovery.json。控制器审计证明已有测速反馈和执行器轮速伺服，本轮未加独立速度PI。

最终GPU增量参考短测已complete：两轮128步、奖励重建max2.98023e−8，alpha分组和更新前采样模型归属正确、无评估。工程总384训练步和33600预热计算步；正式200轮预算独立。源码复核无阻断，准备正式启动并核实进度。

## 2026-09-18 正式训练启动与远端整合

已启动runs/geometric_random_recovery_20260918/training，trainer303368、monitor303369；声明cuda:0/4096环境/200更新/三层128/新几何奖励及当前XML正确。启动源码冻结37d219f，不在运行中切换任务、优化器或选模规则；当前仍按训练批次奖励选best，0自动开发评估。监视器已核实monitoring、无错误。

普通push发现远端新增bc39069与4afaf81；在独立worktree整合，保留远端完整曲线几何模式、PPO约束和完整回合选模工具，同时保留本轮因果前缀几何模式。合并代码通过377项CPU测试、4项跳过；奖励两父版本各500步状态/分项逐值一致。此次整合不代表已运行新合并优化器版本的GPU验证，正式实验仍对应37d219f。主工作区保留训练启动源码，远端合并版本通过普通快进推送交付；训练不重启。


2026-09-18 正式启动核实：runs/geometric_random_recovery_20260918/training 已完成前3/200更新、294912训练转移；第3轮采样9.575s，优化.056s，状态training，无自动开发评估。训练源码37d219f，trainer303368/monitor303369；监视器running且0错误。远端更新在隔离worktree合并，6a9813b已普通推送；主工作区源代码保留当前运行版本，避免中途切换，合并代码CPU377 passed/4 skipped，不冒称其GPU训练已执行。

## 2026-09-18 用户授权追加200轮几何训练
原阶段已complete，200轮/19,660,800转移。从update_0200完整RSL快照恢复策略、优化器与随机状态；仿真物理/ESO/历史重新预热。保持原冻结任务、奖励、128×128×128网络、物理与PPO设置，追加201–400轮，共19,660,800训练转移，预热89,600计算步；自动评估0，400轮或错误停止。新目录runs/geometric_random_recovery_20260918_continue200，best仅在本续训阶段已采样策略200–399中比较，400未采样计分；旧产物保留。

续训实际核实：已完成update201，累计19,759,104转移，本阶段98,304转移；首轮采样归属原policy200，优化器完成20个minibatch，loss有限。TensorBoard6006已加载initial200及continued200标量；trainer455172、monitor455173、TensorBoard456132，监视器running且0错误。实际初始化约248秒，首轮采样33.62秒；GPU存在并发工作，暂不估算独占吞吐或任务能力。

## 2026-09-18 用户追加合成单弯场景实车模型复现
读取用户附件主例：2→2.7m/s、约30度弯、最大曲率1/m、6秒观察、指令后3秒连续保持0.5秒。缺少原始曲率函数/候选生成器，明确采用sin²平滑单弯解释，非逐值复现合成轨迹。配置learning/configs/synthetic_turn_reproduction.json；先3秒预热再6秒观察，控制dt仍.005，3alpha＋1基线（跨alpha重建奖励）共4回合/最多7200物理步。参考速度/yaw slew显式覆盖以产生该急弯，约2.7rad/s峰值超过训练±1.8范围；不修改真实执行器、XML、网络或训练奖励。原文加减速度/侧倾代理仅作对照诊断，另算原文3秒严格门槛，不冒充当前训练回归时钟。

用户随后提供STTW_synthetic_reward_fit.zip；已安全读取原始代码与三候选数据。原曲线为路程sin²，长度2theta/kpeak=1.04719755m，候选生成侧倾设计帽.28rad。原三候选独立重算特征与附件最大差2.78e−17。先行4回合使用左端曲率采样，与原曲线最大3.413mm误差，保留为近似试测；新增源定义对齐4回合/7200步，以区间航向差产生参考命令，保持物理控制周期，单独目录synthetic_source_best252。合成24/24仅引用原文件，未重跑全部拟合。

源对齐试测发现预热时启用不同alpha策略使指令前速度/位置不同（alpha0约2.32m/s），故不作为原例同初态比较。保留该诊断，正式对比新增synthetic_matched_best252：所有方法前三秒均零残差ECBC+ESO，指令时启用冻结策略；额外4回合7200步上限，检查指令时完整物理状态一致，报告实测速度而非强称精确2m/s。

## 2026-09-18 400轮结果与源文档单弯对照
400轮39,321,600训练转移完成；续训best252为update253采样−.0493251/step，policy400未计分。追加阶段46.07分钟，采样中位8.52s；54项相关CPU行为测试通过，日志400×20=8000次优化minibatch、无物理失败记录，不等于任务成功。最后50轮速度RMSE .5633对上一阶段末50轮.4886，横向.1571对.1772；描述性批次比较，不认定因果退化。
对best252单独标准12残差+4基线、每条10秒：全部无物理失败，残差仅gentle alpha1末段保持1/12，基线0/4。12/12横向RMSE优于基线但12/12速度RMSE更差；9/12同alpha回报提高不能替代共同任务门槛。完整阶段/分项数据、跨alpha轨迹和每步/累计奖励：runs/geometric_random_recovery_20260918_continue200/analysis/REPORT.md。
用户追加附件及ZIP主例：原始sin²空间曲率30度/1m峰值，候选侧倾设计帽.28，评分.30，3个原候选独立重算特征误差≤2.78e−17；未重跑12/24全库。源对齐同初态物理测试synthetic_matched_best252为3秒统一零残差预热+6秒、3alpha+1基线7200步；指令时qpos/qvel等逐值一致，实际初速1.9596，非精确2m/s。当前参考离散与源曲线最大差1.412mm，终航向30.000002度。真实alpha0/.5/1最低速度1.791/1.782/1.775，横偏峰值.534/.523/.512m，3秒和6秒共同保持均未通过，均无物理失败。原候选有全路径预瞄/加减速限制，本策略不等价；参考yaw2.699和slew22.074明显超训练范围，不能据此直接否定当前奖励或认定只需34.2。未再训练或修改物理。
主报告：runs/geometric_random_recovery_20260918_continue200/synthetic_matched_best252/analysis/source_comparison/REPORT.md。两组近似/不同预热试测独立保留，各4回合7200步，不混入主对照；本次全部新增物理预算标准32,000+单弯21,600=53,600步。
诊断修正：浮点近等距的相邻片段可能使NumPy复核选邻段；新增从原路径、零进度独立JAX算术回放作为严格复核，不放宽阈值、不改训练/环境。冻结1601帧60KB工程夹具修复前失败，修复后通过并拒绝0.01m篡改。原始完整物理轨迹保留，未因诊断报错重跑。

## 2026-09-18 用户固定五场景15回合并要求补图
目的：每个场景直接查看三个alpha与基线的同图行为和奖励，避免跨多个alpha页面拼接判断。沿用best252原四标准与synthetic_matched_best252已完成的真实轨迹，不重跑物理；新增共享绘图入口并接入阶段面板出图，导出五场景各一张XY/每步奖励/累计奖励/速度误差总览和四类单图，PNG/PDF/CSV/NPZ及统一索引。基线物理单条、奖励按三个alpha分别重评分；未来15残差+最多5条首次基线，不恢复训练内逐轮评估。

## 2026-09-18 五场景15条件同图补齐
用户固定未来gentle/tight_turn/straighten/reversal/synthetic_turn各alpha0/.5/1，共15残差回合；基线各场景首次一条，之后条件匹配复用。规则写入AGENTS.md，覆盖旧12回合约定，不恢复训练内每轮评估。共享tracking_diagnostics.write_cross_alpha_comparison接入generate_panel；本次直接读取best252原四场景和同初态单弯轨迹，没有新增仿真/训练。交付5张总览＋20张单图，各PNG/PDF，以及5份CSV/NPZ；每张总览含XY、每步总奖励、累计奖励、速度误差。每步奖励symlog保留大惩罚与小变化；奖励基线按各alpha重评分，其余图基线共享一条。
核实5场景/15残差条件/同一checkpoint与seed，CSV和NPZ奖励、累计量及速度误差与30组已评分轨迹逐值一致，基线物理跨alpha相同。12项相关测试通过，检查代表性原场景/新增单弯总览并修正奖励轴可读性；所有索引链接有效。统一入口runs/geometric_random_recovery_20260918_continue200/analysis/five_scene_comparison/INDEX.md，原报告和标准图索引已添加入口。

## 当前续训安排（2026-09-18，覆盖上一评估安排）
用户要求暂停评估，训练到累计250轮后再考虑。主组rho34.2从update48的完整RSL snapshot继续202轮；原update1固定开发best保留。rho10原运行尚无snapshot，暂停旧流水线后排在主组之后从头训练250轮。每组累计32,768,000控制转移，主组新增26,476,544；准备计算另记。奖励/物理/模型/网络/PPO不变，恢复Actor/Critic/优化器/RNG但重新准备仿真初态。续训启用training_reward_selection以禁用全部初始化/中间/末轮开发评估；采样best仅是训练日志指标，不与旧固定开发best混比。运行入口支持--resume-run、--target-updates、--no-evaluation，使用新目录保留原始结果。原计划的best评估脚本延期，250轮结束不自动评估。

## 2026-09-18 阶段同步修复计划（用户批准实现并重启主组）
对照组在update36暂停；不再排队。新主组从头250updates，1024×128共32,768,000正式控制转移。任务/物理/奖励/0.8基础输出/网络/PPO不变，只修正采样与监控：准备后用当前策略真实闭环推进分层均匀的[0,2000)步相位，保留全部闭环状态；预推进计算和活跃转移另记。禁用采样reward best，训练期间仍不做开发rollout；所有checkpoint保存，固定完整场景选模留待后续。新增完整训练回合回报/长度/失败率（排除预推进后的不完整首回合，跨update策略变化只标训练统计）及分阶段奖励/速度/路径RMSE。先做相位状态/失败重置/跨batch回合统计测试、GPU短跑确认各阶段覆盖，再启动唯一主组和监视器/TensorBoard。

## 2026-09-18 用户确认新增直接RL控制训练
新任务配置asymmetric_direct_rl.json：正式任务base_output_scale=0，两路Actor分别直接映射±3rad/s前轮/±60rad/s后轮，执行器机械约束不变；共同3.5s、scale1稳定准备保持。ECBC/ESO及参考侧倾仍可计算并作为现有观测/奖励信息，正式任务基础命令贡献为零，不宣称移除了所有传统控制知识。沿用ppo_asymmetric_phase_spread.json从头250updates（32,768,000正式转移），rho34.2、奖励、网络、优化器、种子等相同，不加载旧checkpoint。保留在训0.8主组；rho10不重启。基于真实闭环分散相位，直接策略若提前失败须如实记录实际覆盖，不通过伪造时钟维持均匀。阶段末固定场景选模/比较另行执行，不用短片段reward选best。此为完整动作范围直接控制与有界残差方案比较，不是仅移除baseline的单变量消融。

## 2026-09-19 当前结果
phase_spread残差与direct_rl两组250轮均完成。训练日志诊断显示残差保持10秒生存但路径回归不佳；直接RL后期平均0.357秒失败、100%物理失败且KL回退180/250，不能因短回合误差小/回报较高判优。详见METHODS_AND_RESULTS最新节及runs/asymmetric_direct_rl_20260918/analysis/training_diagnosis/REPORT.md。本轮只作现有日志分析，未新增训练或固定场景仿真，最终checkpoint/alpha任务表现尚需配对验证。

## 2026-09-19 离散alpha三组重训（用户授权）
alpha每回合等概率抽取{0,0.1,1}并回合内保持；按三个精确值记录训练统计，不把0与0.1混入同一连续区间。主reward继续rho34.2，不修改失败罚/回归罚或KL机制。新鲜初始化三组：ecbc1（base1、残差1.5/10），ecbc08（base0.8、残差1.5/10），direct（正式base0、动作3/60）；共同3.5s scale1准备、相位分散、128×3 ELU和原PPO。
每组1024×128×200＝26,214,400正式转移，三组合计78,643,200；准备与相位分散另计。已有记录不覆盖、不加载旧checkpoint，不恢复采样best或训练内开发评测。按ecbc1/ecbc08/direct顺序自动执行；发生程序错误停止，不无上限续训。阶段结束弹窗；全部完成弹窗。入口支持重复--arm NAME=TASK.json；配置discrete_alpha_{ecbc1,ecbc08,direct}.json与ppo_discrete_alpha.json。现有直接RL失败风险未被本次采样变更自动修复，结果待核验。

## 2026-09-20 当前训练结果
离散alpha三组200轮已全部完成。训练证据优先支持进一步评测ECBC1+残差，尚不满足路径回归结论；0.8组175/200轮KL回退，直接RL末25完整回合100%物理失败（平均0.769s）。详细分alpha池化误差/窗口统计/复现脚本位于runs/discrete_alpha_three_control_20260919/analysis。未新增训练或固定场景仿真，未选择best；下一步建议先修KL缩步机制并做完整场景检查。

## 2026-09-20 当前交付：完整 ECBC＋残差最终轮可视化

用户确认只看离散α实验ecbc1的最终update_0200；候选比较已停止，没有新增训练，也不称best。完成五历史场景、α0/0.1/1、seed49001的15条策略＋5条配对原始基线；图包入口：`runs/discrete_alpha_three_control_20260919/ecbc1/final_review/panel/analysis/INDEX.md`。15条均存活到窗口末，但共同最终保持0/15；路径在4/5场景改善、速度在5/5场景差于基线。新增可复用入口`learning/cli/five_scene_review.py`，同图总览含路径误差；细节见当前VALIDATION及METHODS_AND_RESULTS末节。

## 2026-09-20 当前运行：速度精度奖励重训（已确认实际更新）

用户批准依据附件d884f02c修改奖励并启动新训练，并纠正alpha为0/0.5/1。范围为当前完整ECBC＋残差主组，新初始化200更新、1024×128×200=26,214,400正式转移，准备及相位分散另计；不自动扩到0.8/直接RL，不加载奖励身份不兼容checkpoint。控制器/物理/权限/网络/PPO不变，无训练内开发评估。

- [x] 新奖励开关默认关闭：欠速系数0.12→8、尺度0.1→0.05；超速系数8/尺度0.05固定；路径系数4→0.1。过程欠速带0.5→0.05，共同最终速度±0.05；路径过程带0.1→0.4，回归收紧至0.1。
- [x] 复用已发布参考触发的冲突回归时钟与外扰可观测离带时钟；不重置旧债务。核对Actor已有pending/elapsed上下文和无未来指令泄漏。
- [x] 测试附件候选排序、CPU/JAX、奖励重建、旧配置及身份兼容、保持及收紧。评估可选成本率cap100/普通奖励scale0.1、失败整步-200、一次deadline-5的有限回合失败激励界；通过后才启用，明确极端区间梯度截断。
- [x] 短GPU工程训练通过后冻结正式配置，启动200轮，注册TensorBoard/监视通知并确认实际更新。完整配对验收α1要求指令后速度RMSE≤0.8×基线，另保留失败/最终保持/超速/最大偏离；不宣称已达到。

用户最新确认：只训练完整ECBC＋残差，α=0/0.5/1，按TensorBoard `train/mean_step_reward`最大值保存`best_model`及`best_model.json`。使用`ppo_precision_speed.json`启用`training_reward_best_enabled`，奖励归属采样模型（更新前），末次未再次采样的update200不借用update199奖励；不增加开发rollout。新运行目录`runs/precision_speed_ecbc1_20260920`。

正式PID4155187；首4轮更新已接受，TensorBoard6006已加载真实训练标量，best记录已生成并核对为训练采样奖励最大值。工程与启动核验见`runs/precision_speed_ecbc1_20260920/startup_verification.json`，实时进度以该运行training/status.json为准。训练仍进行中，尚无本轮最终能力结论。

## 2026-09-20 单弯best复查完成

precision_speed_ecbc1已完成200更新，按训练奖励选择update0196（第197轮采样）。用户限定仅单弯，完成alpha0/.5/1＋共享基线；入口`runs/precision_speed_ecbc1_20260920/single_turn_review/panel/analysis/INDEX.md`。3～9秒速度RMSE基线0.09176、策略0.07412/0.07466/0.07528；路径RMSE基线0.40392、策略约0.102。三个alpha均最终保持、无超时/物理失败，但轨迹几乎重合，alpha1速度RMSE/基线0.820，未达0.8目标；超速峰值约0.155m/s且超0.05持续0.175s。不是全任务合格或泛化结论。

## 2026-09-20 三模式软预算奖励V1已启动250轮训练

用户提供下载目录sttw_reward_v1并批准实现与250轮训练。仅完整ECBC＋ESO＋残差，alpha0/.5/1、新初始化，1024×128×250=32,768,000正式转移；准备和短测试另计；按训练mean_step_reward最大值保存采样best。文档建议不加轮数被本次250轮明确要求覆盖。保留物理、200Hz、动作权限、网络及PPO参数。

- [x] 独立objective=soft_budget_v1：8alpha欠速/8超速/8(1-alpha)路径主成本、四项软带；1s缓冲后1.5s收紧，3s期限不延长；平滑G(C)=100C/(100+C)、普通scale.1、失败-200、一次deadline-5、超期成本率2。
- [x] 共享状态接口/秒观测不变，时钟仍按整数tick推导；旧奖励和身份不变。原始成本与有效奖励分别记录，不叠加旧precision分支。
- [x] 用附件参考实现核对候选、连续状态、CPU/JAX/分项/旧轨迹兼容；审查无效状态、真实失败与时间截断边界。
- [x] CPU闭环与有界GPU短测，通过后冻结250轮配置、启动单组训练/TensorBoard/通知，并核对首轮best归属。

正式训练PID1291444，源码fae5aaf已推送。首轮131,072转移完成，更新接受，KL0.002142，无非有限值；TensorBoard6006已加载实际奖励，best与采样模型归属核验通过。运行入口runs/soft_budget_ecbc1_20260920/INDEX.md；实时进度以training/status.json为准，尚无训练后能力结论。

## 2026-09-20 软预算V1 best单弯复查

250更新完成，训练奖励best为update0244（第245轮采样，mean_step_reward=-0.0037052934）。按用户要求仅复查synthetic_turn，alpha0/.5/1、seed49001，三策略＋共享ECBC基线。入口runs/soft_budget_ecbc1_20260920/single_turn_review/panel/analysis/INDEX.md。指令后3～9秒速度RMSE为0.07336/0.07369/0.07409m/s（基线0.09176），路径RMSE0.09964/0.09904/0.09852m（基线0.40392）。最终保持3/3、无超时/物理失败；三个alpha仍近乎重合，alpha1速度RMSE比基线0.807，未达≤0.8。超速峰值0.203/0.209/0.215m/s，比上轮约0.155更差；不能仅凭总体RMSE称全面改进。单场景外推开发诊断，不是泛化或硬约束保证。

## 2026-09-20 V1附件诊断报告与同状态alpha核查

报告runs/soft_budget_ecbc1_20260920/single_turn_review/diagnosis/REPORT.md，附逐步CSV、原始/有效成本、同状态mu/action/base/final_command NPZ和evidence.zip。复用既有best0244三轨迹，无新物理rollout或训练。真实roll峰值0.473～0.475rad、超0.30持续0.325s；同状态全十帧alpha替换的动作0→1 RMS差约前轮0.0101～0.0103、后轮0.00619～0.00636，最终命令差基本保留，不能把主要原因归结为限幅消差。3～3.39s参考窗口欠速积分略降/路径积分略升，为极弱预期方向，与全窗口RMSE反向须分开。转弯最低后步速度1.97976高于指令前1.95962，未展示主动减速。候选搜索、训练内core左右诊断、独立端点/三头仅为条件建议，未执行，不宣称全局无可行取舍。

## 2026-09-28 两个固定偏好几何网络（实现与预算）
用户批准两个独立从零网络，各100个PPO更新；不是各100个环境episode。隔离目录STTW_CONTROL_fixed_endpoints，基于4cafb53复用现有几何投影/日志/训练修复；主目录与历史证据不覆盖。旧85cacdc是时间目标，最新批准方案是几何目标，不能把本实验标成旧版精确复现。
- [x] 新配置复用asymmetric_geometric_huber，有限10:1权重保证alpha0/1都有位置成本；欠速alpha0轻、alpha1重，超速共同强约束；关闭会强迫弯道恢复原速度的期限奖励/罚，保持真实位置惩罚，不奖励停车或时间落后。
- [x] 固定alpha仅在奖励配置中，无Actor alpha输入；拒绝隐藏随机alpha；保持三层128、10帧物理历史。
- [x] 固定侧倾目标支持控制器、Actor误差、奖励和最终诊断一致；保持真实侧倾动力学及物理安全阈值。用户确认采用6.84°固定目标（不是锁死真实侧倾），prepare=0工程初态。
- [x] 连续限变化率随机指令与同方向曲率范围工程筛查；路径独立积分，不以实际车辆重定位，不奖励时间沿程/yaw-rate误差。
- [x] CPU行为测试、两组GPU各2轮小批验证通过，正式100轮/组队列已启动，TensorBoard6006及监视器同步启动。不自动续训/训练内评估。
正式预算每组1024×128×100=13,107,200训练转移，两组26,214,400；相位初始化每组最多2,048,000计算步另列。10s任务，不更改dt/XML/执行器。准备状态预算随最终固定侧倾筛查声明。最终配置与停止条件冻结后才启动。

实现检查：435 passed、5 skipped；固定侧倾ECBC须同步采用ratio(speed)×roll_target平衡转角，否则破坏平衡（错误候选4/4约1.2s失败，修复后4/4完整10s）。新基线不直接响应外部几何转向请求，该请求仍进入Actor，位置修正由残差承担；不能与历史动态路径ECBC混称。当前GPU短测及启动状态以runs/fixed_endpoints100_20260928/status.json为准。

启动记录：实现c91f19b已推送，正式队列PID252257，监视器252258，TensorBoard239572；alpha0先运行、alpha1排队。两组GPU各128转移均完成，无非有限候选，奖励分项重建最大误差3.73e-9。固定目标6.84°及同方向弯道已获用户明确确认。启动记录是时间快照，实时状态以运行目录为准。

## 2026-09-28 双端点训练后分析（完成）
- [x] 核验100×2更新、采样best归属、分窗口奖励/误差/loss/KL趋势。
- [x] 冻结alpha0 best0059、alpha1 best0072，seed49001；历史五场景10策略+5共享固定侧倾ECBC基线，CPU评估；无新训练。
- [x] 复用奖励重建及绘图模块；增加显式独立端点模型组合支持，禁止默认混checkpoint；同场景XY/每步/累计/速度及位置误差PNG/PDF/CSV。
- [x] 按工况与时段报告失败、共同最终门槛、速度/路径取舍，区分训练范围内和外推；更新方法台账及验证，提交推送。

完成结果：两组100更新已结束；best0059/0072五场景10策略+5基线已完成，10/10最终保持不合格。存在相对速度/路径偏好方向，但仍绕圈、路径米级偏离，不能称已学会正确过弯。报告runs/fixed_endpoints100_20260928/review/analysis/REPORT.md；不自动续训。工具验证438 passed、5 skipped。

## 2026-09-28 附件方向性奖励下一轮（进行中）
用户批准按附件修改后再训练，继续奖励选best并新增欠速/超速诊断。现有asymmetric_geometric_huber已具备拆分逻辑；本次复用实现，配置改为tracking_rate4、speed_scale0.1、欠速带0.5→0.2随alpha，固定超速带0.05、最终欠速0.2、启用离带回归时钟收紧（路径带也按已有共享收紧收回共同0.1m）。不重复创建reward模块，不改基础控制器、物理、PPO、网络或残差权限。
- [x] 数值对照、方向日志池化峰值/时长/积分测试；保留旧配置身份。
- [x] CPU逐步奖励复核及两组各2更新×8环境×8步GPU短测；失败终止罚尺度审计。
- [x] 新目录冷启动alpha0/1各100 PPO更新，1024×128/轮，总26,214,400正式转移；初始化每组≤2,048,000计算步另计。准备0、任务10秒；异常停止队列、不自动续训。
- [x] TensorBoard6006同时显示旧/新运行，已核验首批标量和方向日志，监视器正常，提交推送。

正式训练队列732007已启动，监视器732008正常，alpha0处于相位初始化、alpha1排队；源码91bdb8d已推送。最终全套442 passed、5 skipped。TensorBoard6006当前服务PID721652，previous/directional两组标签共用入口；新标量等待正式首批采样。

正式首轮131,072转移已完成，16批优化接受、无非有限值；TensorBoard已加载directional_alpha0及超速峰值等新标量。alpha1仍按队列待alpha0完成后运行。

## 2026-09-28 Teleop preference governor V1 (isolated execution)
User-approved specification: docs/teleop/SPECIFICATION.md; original JSON copied unchanged.
Base b446abae131f7a3698c884f1872f8049e9ba3c6e, branch experiment/teleop-pref-governor-v1.
No training, no policy loading, no push. Existing workspace and runs untouched.
Plan/progress kept here per repository rules (writing-plans / planning-with-files).
- [x] Audit base, instructions, dependencies and frozen actuator/controller/XML limits.
- [ ] Add strict independent schema, causal v/delta command stream, exact reference integration.
- [ ] Test and implement complete-state kernel: shared old ESO state, one commit, unchanged actuator.
- [ ] Contract tests, one-step/240-step replay, CPU/MJX audit and original compatibility suite.
- [ ] Shared 700-tick preparation; B0 N0--N5 gate before governor functional panels.
- [ ] Predictor summary via vmap/scan, 144+80 candidates, lexicographic selection and recovery.
- [ ] Gate A G0/G1; Gate B C25 pair then C30 pair; Gate C four frozen PCG64 streams.
- [ ] Recovery ablation only after gates and if budget remains; reports, local commit.
Budget: 80M predictor ticks including padding/benchmarks; <=128 contract rollouts;
<=42 main and <=4 ablation episodes (16s each); shared preparation 3.5s.
Any failed gate stops later large panels. No automatic tuning or budget extension.
Source audit: dt=.005, physics dt=.0002 (25 substeps), final 3/60, residual 1.5/10,
mechanical steer .8, ESO start 3.0. Only fixed roll references removed in new task.

Implementation progress: independent schema/reference/kernel, full-state MJX predictor,
144+80 fixed shape candidate search, lexicographic endpoints, recovery state machine,
persistent budget, staged CLI and static evidence writer now implemented. 14 new pure/
CPU tests passed; original suite including first five additions: 447 passed, 5 skipped.
Shared 700-tick MJX preparation passed (v=2.25384784, roll=-4.89e-6 rad).
First engineering replay attempt hit a readonly NumPy-view masking bug after 35,520
reserved predictor ticks; logs preserved. Fixed with non-mutating masks and regression
check. Second attempt is running in contracts/attempt_0002; left bypass replay differences
are exactly zero through 240 ticks including qacc_warmstart and ESO. GPU is shared with
unrelated jobs; 170.4s first scan compile+execution is not a realtime result. No functional
gate passed yet. Code review identified and repaired phase continuation, early-budget
reporting, Gate A bypass checks, failed-state freeze, and non-bypass replay coverage.

### User pause and desktop detector (2026-09-28)
User asked to stop waiting, provide a detector, and show a completion popup. No new
functional experiments will start until the user resumes. Both previously dispatched
short checks have finished. Left/right bypass replay max errors are zero; the 144-copy
candidate consistency check FAILED (max duplicate difference 9.714823681861162e-05).
Its cause is not yet diagnosed; no gate acceptance is inferred. CPU-device MJX governed
short scan completed: compile 26.9698s, execute .39085s, no physical failure; no matching
step replay yet. Later implementation/CLI code remains unvalidated end-to-end.
Evidence: runs/teleop_pref_governor_v1/20260928T055716Z/HANDOFF.md.
Budget: predictor 71,760/80,000,000 ticks, plant preparation 700 ticks, 9 contract
rollouts, 0 main episodes, 0 training. Original attempts and budget charges retained.
Read-only detector: `python3 learning/cli/teleop_governor_status.py --watch --notify`.
Uses run-local checks_watch.json, persists notification receipt to avoid duplicate popup;
no automatic continuation. Popup title explicitly requested by user: 偏好控制器仿真结束;
body distinguishes short-check completion, failed check, and unrun later panels.

### 2026-09-28 resumed contract diagnosis (user: continue)
Existing ledger retained. CPU float32 144-copy diagnostic confirms original snapshot
unchanged and duplicates exactly equal, yet batch vs scalar replay terminal qvel
max error3.7145146052353084e-5 exceeds unchanged1e-5 tolerance. Original check conflated
state mutation with duplicate arithmetic equality. Next bounded diagnostic: same full
MJX/JAX equations/model/solver and complete initial snapshot, both plant and predictor
float64, to test batched rounding sensitivity. No XML/threshold/window/grid/controller
parameter change. This arithmetic precision change is explicit; old float32 evidence
remains unaccepted. Reserve34560 batch +240 scalar ticks per raw/governed diagnostic;
ledger includes all work. No main episode until actual batch/step consistency passes.
Float64 same-snapshot diagnostic completed: raw terminal qvel difference7.0980e-11,
governed1.3804e-14; qpos6.4574e-12/2.0539e-15. Real snapshot preserved, duplicate
results equal. Initial migration compile error (contact geom int32 vs JAX64 int64)
is retained; the fix casts only those indices without changing values. Added verified
complete-state promotion, stable governor scalar dtype, precision/device provenance,
and production shape1/144/80 vs independently rebuilt executed summaries contract.
Current 25 targeted tests passed. Next production contract attempt remains in the same
run and budget; no main panel until it passes. Notification keys separate this resumed
stage from the already-delivered completion popup.
Production contract resume_validation passed: four explicit/scan240-tick replays,
plus real predict() shapes1/144/80 versus independently reconstructed physical/cost/
constraint summaries. Maximum error7.09802e-11; Boolean constraints agree; complete
snapshot unchanged. 467 passed/5 skipped in full compatibility suite. XML, control
parameters, candidate grids, thresholds and permissions unchanged; execution now
explicitly uses MJX/JAX CPU float64 with exact-value snapshot migration. Old float32
failures remain failed. Warm144 prediction ~69.25s: not real-time20Hz. Proceed offline
within original80M ledger; gate order unchanged; detector will popup at completion or
failure stop. At launch preparation700 ticks, predictor301440 ticks, contracts21,
main episodes0. Numerical evidence: numerical_diagnosis/REPORT.md under original run.


## 2026-09-28 Direct command V3 — saved and stopped

Independent branch `experiment/direct-command-policy-v3`, teleop parent `92afde6`,
training execution snapshot `8211d89`. One shared alpha0/1 Actor345 / Critic346 /
two Gaussian latents changes raw-relative speed and steering references, retaining
original physics, ECBC/ESO, 200Hz/50Hz timing and actuator authority. No analytic
allocation, parameter coordinator, search or old checkpoint enters the V3 path.

Fresh 8x16x2 engineering and separate fresh512x128x20 PPO completed. Last checkpoint
update0020. Main review reached the60s per-episode wall cutoff; three1000-tick/5s
prefixes saved, random88001 unrun. No restart or continuation. Main[2.5,4.5)s speed
RMSE B0/alpha0/alpha1=.178115/.117384/.114678 m/s, steer=.037576/.049995/.050479 rad;
peakroll=.384821/.351292/.351265 rad. Both policies sacrifice steering/heading to
retain speed; alpha separation is too small and work range fails. Direction recovery
is unverified because the raw steer has not yet settled by the saved endpoint.

Evidence `runs/direct_command_v3_20260928/INDEX.md`; live historical metrics at
http://localhost:6016 (20 reward scalars loaded, physics worker stopped). Compile185.39s,
preparation/interfaces/smoke78.78s, pilot526.67s, review60.26s including~.26s exit latency;
all checks/plots/overhead retained in the same1800s ledger, total about15min. No push.
Implementation/schema/run instructions: `docs/direct_command/README.md`.

Post-run corrections: raw!=bounded float32 diagnostics falsely reported residual
clips; source flag fixed and saved review clips reconstructed without editing NPZ.
Full training motor/final clip aggregate cannot be corrected from two diagnostic
environments, so it is not saturation evidence. Added optimizer-nonfinite stop gate
before review (not triggered in this finite run), independent saved-trace audit,
matched-alpha baseline re-scoring, physical and reward plots. Control mathematics
and this run's trained checkpoint were not changed. Final related suite48passed.


## 2026-09-28 User-authorized V3 500-update trial — implementation plan

User explicitly replaced the20-update cap with500 updates. Keep the V3 method and
its fresh-initialization requirement: separate fresh shared Actor/Critic/optimizer,
seed73,512x128x500=32,768,000 policy transitions,131,072,000 control ticks maximum.
Config `learning/configs/direct_command_500.json` changes only update/sample and
training/total compute budgets. Training15000s, total16200s; compile300s,
engineering120s and existing restricted two-case review budget remain unchanged.
No old experiment overwrite, no policy import, no automatic beyond500 continuation.

Plan: (1) make existing CLI/status/budget honor explicitly frozen run counts while
validating all method fields against V3; (2) tests for500 acceptance, immutable method
rejection and sample-count consistency; (3) launch new bounded campaign and isolated
TensorBoard, verify a current pilot reward scalar and checkpoint; (4) record source,
run identity, live progress and stop conditions. Existing isolated branch reused.


Launch verified: run `runs/direct_command_v3_fresh500_20260928`, source`cc03aa6`.
At verification2/500 formal updates,131072 transitions, fresh initialization;
checkpoint update0002 exists and TensorBoard http://localhost:6018 loaded both
formal pilot reward scalars. Run remains active; status.json is authoritative.
Required interface checks and8x16x2 smoke passed. Existing20-update run and logs
remain unchanged. No remote push, no other process stopped. Source regression
checks for config isolation, sample budgets, PPO and evidence modules passed.


## 2026-09-29 Frozen residual lower controller / fresh upper500
User requested key evidence upload and a new500 trial using path_rsl_4096 evaluation
alpha_2/seed_49001/force_right/residual. Source update0200 is pinned by sidecar/payload
hash; lower alpha=1,280 inputs,200Hz history, frozen ELU256/128 tanh actor. Existing
V3 upper Actor345/Critic346,50Hz,PPO512x128,alpha0/1 and reward remain unchanged.
Plan: verify pinned lower input/output and geometric history adapter; combine raw-to-
governed ECBC change plus frozen residual under one shared±1.5/10 authority; run
interfaces8x16x2 then fresh upper500, separate output, checked TensorBoard. No old
upper/optimizer import. Keep original V3 physics (old lower source XML differs).
Lower world-fixed path is integrated causally from governed commands, historical
nearest-segment curvature, with endpoint tangent extrapolation; not old bend-task
reproduction. Original raw-command reference/reward remain independent. Lower
history/return/path state reset per episode; ESO committed once. Preparation uses
original ECBC bank, lower activates at task reset. This transfer is experimentally
unverified until new checks/evaluation. Budget:compile600s,engineering300s,training
15000s,review600s,per-case240s,total18000s. Stage/time/numerical stops remain.
Existing456 result is not coordinated tracking success; concise shareable evidence
in docs/evidence/direct_command_fresh500_20260929. No new panel expansion.


Launch verification: frozen-lower500 run passed both4s interface replays and
8x16x2 engineering updates. Formal fresh upper reached2/500,131072 transitions,
checkpointupdate0002 saved, ~27.4s/update. TensorBoard http://localhost:6021 loaded
both formal reward scalars; watchdog monitoring with zero errors. Run remains
active; live status.json is authoritative. No claim of new control improvement.


## 2026-10-08 冻结残差底层500轮结果核查

run direct_command_frozen_lower500_20260929已完成500/500（32,768,000转移），
update0500完成主场景/随机各三方法全部16秒，共6回合。未达到协调或恢复要求。
主场景[2.5,4.5)s B0/alpha0/alpha1速度RMSE=.00937/.13613/.13750m/s，转角
RMSE=.12914/.09146/.09149rad；峰值侧倾=.3753/.4373/.4377rad。上层减小转角
误差但牺牲速度，两alpha几乎无区别；工作界及回正保持不合格。随机同样未恢复。
训练无硬KL/非有限停止、0/40960结束回合物理失败，但成本触顶全程约37.6%、
末20轮约48%，工作界违反约4.6%。这些是跟踪问题，不以存活/奖励替代成功。
冻结残差底层在因果governed路径迁移中也未可靠回正，不能据此单因果归咎ECBC。
六条保存轨迹独立重建及导出一致性核对通过。仅后处理，无追加训练/物理评估。
关键证据：docs/evidence/direct_command_frozen_lower500_20261008/README.md；
本地完整索引runs/direct_command_frozen_lower500_20260929/INDEX.md。

## Lower random-command residual implementation plan (2026-10-08)
User delegates reward/hyperparameters and authorizes fresh1024envs x128steps x200updates.
Design: reuse TeleopEnv physics, closed_loop_kernel.controls, command slew, RSL rollout/GAE and guarded optimizer. New lower task has no path or alpha inputs; Actor210 (10x20frames+10mask), Critic211 including remaining duration, ELU256/128, Gaussian2 -> tanh -> bounded front/rear residuals. All200Hz. Original observation fields/scales reused, actual body forward velocity is explicitly simulation-assisted.
Reward rate:2exp(-(ev/.15)^2)+2exp(-(ed/.05)^2)-.2Huber(ev/.2)-.2Huber(ed/.1)-4Huber(max(abs(roll)-.26,0)/.05)-.02sum(action^2)-.05sum((action-prev)^2). Multiply dt=.005; physical failure -10 additional. No total cost cap; no heading/path objective. Finite10s task terminal, rollout boundary bootstraps, terminal never bootstraps. Tracking limitations of large speed/steer commands remain visible.
Commands:1.5..3m/s,70% narrow±.12rad,20%±.30rad,10%straight per target; targets hold.8..1.8s,final2s straight at sampled speed; speed slew.3.. .8,steer.15.. .45. Stateless seed/env/episode sampling; no future rows in observation.
PPO:1024x128,4epochs,8minibatches,Actor lr1e-4/Critic3e-4,gamma.9975,lambda.95,clip.2,entropy.001,grad1,std.15 bounded.05.. .5,soft KL.01/hard.05 rollback-stop. One seed830081. Max9000s wall including compilation/smoke/formal. Engineering8x16x2 separate initialization; formal200 maximum, no auto continuation/development panels. Save every10 and last, training-reward best belongs to pre-update model only.
Plan:
- [x] Tests for reward ordering, random schedules/causality, reset history, zero residual ECBC parity and PPO global KL.
- [x] Implement lower task and small training entrypoint; reuse existing physical and optimizer modules.
- [x] Run CPU tests and8x16x2 GPU engineering smoke; verify finite gradients and weights actually change.
- [x] Register run under shared lock; start fresh1024x200, verify reward scalar HTTP TensorBoard and checkpoint/progress.
- [x] Commit isolated implementation after launch; no automatic push. No claims of control improvement until paired trajectories evaluated.

## 2026-10-08 Frozen registered lowers / independent upper endpoints
User authorizes R196 and R244 ALPHA1 lower controllers, upper alpha0/1 retrials250 updates. Interpret as2x2 independent upper policies (4 groups), rather than confounding lower identity with upper preference. No old upper or optimizer loaded. Same seed73,512envs x128policy steps x250=16,384,000 policy transitions/group;65,536,000 physical ticks/group. Upper50Hz, lower200Hz; original16s upper task/reward/command/action mapping preserved. Frozen lower history capacity extends2001->3201 solely to support16s; contract fields/scales/normalization/return rules preserved. Composition follows frozen lower: base ECBC at governed reference + original±1.5/10 residual, final original limits. This differs from former raw-centered shared correction budget; do not label single-variable comparison with old500 results.

Budget/group: compile900s, engineering600s, training14400s, restricted final review1200s (per-case480s), total18000s. Four groups max72000s wall; serial workers, stop queue on group numerical/interface/budget error, no retries or extra updates. Final comparisons: each lower's zero-upper baseline and own alpha0/1 finals on original main+random16s schedules; no broad panel. Save partial/failure evidence. TensorBoard all group directories separated; live verification receipts and progress are recorded in the run directory and shared research ledger.

Plan (brainstorming / writing-plans / TDD; user delegates implementation and authorizes launch):
- [x] Registry adapter preserves310-field Actor, mask/history, fixed lower_alpha1 and independent state; parity with validated Transfer.
- [x] Parameterize upper training single-alpha grouping without changing historical defaults; keep separate model/checkpoint/optimizer identities.
- [x] Short8x16x2 fresh engineering checks; validate lower reference-centered control, ESO once, raw reward/reference, registry hashes.
- [x] Register/launch bounded4-group queue and completion review, verify current reward in TensorBoard, commit after launch.
