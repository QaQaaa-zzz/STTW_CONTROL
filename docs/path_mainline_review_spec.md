# STTW Path Feedback：主线推进与训练运行修复指令

## 0. 本轮决定

保持当前主线：固定几何路径 → Pure Pursuit 实际位姿反馈 → alpha 上层直接输出 Δv/Δδ → 既有一阶修正滤波与限速 → 冻结 local300 + ECBC/ESO → 车辆。

本轮不重训底层、不切400、不调Pure Pursuit到零上层完全合格、不加规划神经网络/世界模型/搜索器、不蒸馏、不增加下层alpha/path/return任务。当前两个端点分别训练的事实保持，合并留到端点可用以后。

继续用户已批准的每端200次更新。代码层面只修影响训练可运行性与模型选择的明确问题，补齐主线指标。不要在尚无新上层结果时又换奖励函数。

审计基准：
- 报告分支 experiment/path-feedback-v1 @ 61f612aa82f695bf5b57fb354cb5831a5061f9ac。
- 报告声明实际启动代码 8e16cd105c7cdca88ec151b73695cc327a2096ba。
- run: /home/qy/STTW_CONTROL/runs/worktrees/path-feedback-v1/runs/path_phase_C_200_20261010。
- 远端状态是2026-10-10 17:16:06 +08:00快照：尚在zero_upper验证，上层训练转移0。不得把这个快照当作现在的主机实时进度。

## 1. 先读实际运行状态，禁止重复启动

读取原run的 manifest.json、status.json、alpha0/alpha1/last_completed.json、metrics.jsonl、最近launch日志及进程cwd/命令。一次性汇总：每端采样批数、已接受更新数、实际PPO reward条数、最新完整checkpoint、当前阶段、是否报domain/NaN/KL错误。不要仅以PID存活或TensorBoard HTTP200声称正在训练。

- 若原进程正常训练：不kill、不重启、不热改内存行为，不并行启动同一任务副本。在独立工作树准备下面的补丁，于可解释的保存边界使用。
- 若已经完成200/端：直接分析已有晚期验证和best，不重新训练一遍。
- 若因有限状态的PP domain_exit停机：保留故障回合与旧checkpoint，按第3节修复后从最后完整学习边界继续，不全部从零开始。
- 若还在zero_upper：只完成缺失的同协议基线，不因为旧B的末段航向失败再停止上层训练。不重复已完成的六条完整基线。

快照和schema中的历史“not_trained”“automatic_training_authorized=false”文字不能覆盖后续明确的执行授权；依据run manifest、实际命令和日志确认，不偷偷改写历史。

## 2. 必须冻结的实现参数

| 部分 | 固定值 |
|---|---|
| 物理子步 | 0.0002 s |
| 底层 | 25子步一次，0.005 s / 200 Hz |
| 路径发生器/上层 | 0.02 s / 50 Hz |
| 下层 | 固定local300，原210维局部观测，无alpha/path/global debt |
| 上层 | 两个独立Actor，分别固定alpha=0/1；351→128→128→64→2 ELU |
| Critic | 352维，附剩余任务时间 |
| 动作 | 当前名义参考的Δv约[-1,+.25] m/s、Δδ约±.20 rad；沿用tanh映射 |
| 修正滤波 | tau_v=.10 s，tau_delta=.08 s；每5ms更新；状态入网 |
| 参考与执行限制 | 完全沿用当前配置，不借修复扩大权限 |
| PP | 实际弦长Pure Pursuit；lookahead=clip(.7*max(v,1.5),.8,1.8) m；不按曲率调速 |
| 路径 | reset生成后固定，连续局部投影，不重锚、不按墙钟提前换曲率 |
| 任务 | 沿路径前进、穿目标截面并保持出口路径，不包含停车 |
| PPO | 512环境×128策略步；Actor lr1e-4/Critic lr3e-4；gamma=.997；GAE=.99；epochs4；minibatches4；clip=.2；std初值[.25,.10]；范围[.05,.5]；entropy=.001 |
| KL | 原soft .01/hard .03及已实现回滚机制，不新加多套更新器 |
| 数据 | 30%普通/缓弯，50%左右60–100°弯，20%S；v1.8–2.6，R1.8–4；20%一次性路径锚定偏差 |
| 回合 | 20 s；1000上层步，4000底层步 |

这些是沿用本轮设置，不是宣称它们已经最优。不要为了“再优化一次”同时改上述几类变量。

## 3. 首要运行问题：区分任务域退出与工程故障

### 3.1 当前源码问题

path_reference.pursuit把以下三类情况并入invalid：前瞻点在后方/太近、路径航向误差达到pi/2、非有限计算。path_command_training.termination_flags把domain_exit和工程故障合并。PathBatch不重置该环境，训练器遇任一个环境就抛异常丢弃整批。

前两类在路径数据有效、状态有限、reset合法的情况下，可能只是策略没有跟住路径。它们不应和文件损坏/NaN一起结束512环境的全部学习。

### 3.2 新明确语义（terminal_contract_v2）

A. engineering_fault：NaN/Inf、权重/观测形状错误、模型或冻结文件身份错误、索引/投影代码错误、非法初始路径等。保存快照并停止训练；不能sanitize成成功。

B. tracking_domain_failure：路径和状态均有限、reset有效，但执行动作后超出当前前向Pure Pursuit的适用域（原body_x<=.05或|heading_error|>=pi/2条件保持）。这是声明任务失败，不等于物理摔倒。

C. physical_failure：原接触/倾倒等物理失败，保持定义。

D. finite_task_end：真实20秒任务终点。

B/C/D均只终止对应环境并reset，其他环境照常采样。A才终止整个训练。发生B不扩大适用域、不旋转原地、不重置参考到实际轨迹、不继续运行已经失效的PP。

B使用与C相同的保守失败吸收计分一次：
R_fail = -5 - .1*.02*5080.2*(1-gamma**N)/(1-gamma)，N包含当前剩余策略区间。

当前区间如已有部分5ms有效步，失败计分替代该区间正常总奖励一次，不重复累计、不给后续padding计奖。20秒正常完成无额外失败罚。B/C均不bootstrap，普通128步rollout切段正常bootstrap；bootstrap不能读auto-reset后的新回合来替代失败末态。

统计分别保存physical_failure_count、tracking_domain_failure_count、engineering_fault_count。若同一步B/C同时发生，终止和罚只执行一次，物理失败作为主要end_code，附加域状态另记录。

reset即域外：当前路径锚定范围通常很小，必须先归因生成器/坐标错误；不能用不断reset掩盖坏路径。

### 3.3 生效方式

这是终止合同修复，不改变正常轨迹动作和逐步奖励。不能热改正在执行的图。若已触发停机则立即修；若运行正常，可在保存边界或本阶段完成后合入下一阶段。

继续训练继承同架构Actor/Critic/log_std/Adam，不机械清零。旧终止版本和新版本分开记载，父学习步数不丢失，废弃未完成rollout不作为已优化数据。恢复路径计数沿用next-unused key逻辑；物理从原完整准备状态重置，并明示不是逐帧连续恢复。

仅需一个两环境小测试：一个有限域失败只reset自己且罚一次，另一个不受影响；另一个NaN测试必须仍停止。无须重跑底层全套检查。

## 4. best model：新增任务一致选择，保留旧best记录

### 4.1 当前问题

path_command_selection.score_key是：
(physical_failures, working_failures, primary_score, goal_hold_failures, worst_error)。

primary_score是连续浮点值且没有通过容差。主误差改善1e-6就能压过完成目标/末段保持的改善，不能据此把该模型直接称为最适合整个任务。

### 4.2 不改旧排名，增加并行best_task_model

旧best_model.json、旧score和轨迹保留。只对已存在的同一固定DEV轨迹离线计算selection_schema=path_task_best_v2，输出best_task_model.json。使用独立原子指针与SHA引用，不覆写旧teleop/local/R196任何best。

逐场景定义：
- conflict窗口仍使用原chi>0，保留旧口径；chi>=.5另作描述性强弯窗口，不偷换旧窗口。
- 有conflict样本：alpha0主误差为RMSE(ey)/.10，alpha1为RMSE(ev)/.05。
- 无conflict样本：普通目标使用max(RMSE(ev)/.10, RMSE(ey)/.10)，两端相同。
- 主误差>1计一次primary_band_failure；这是本轮明确的工程目标，不把旧结果重写成旧协议已失败。
- 目标截面/路径进度/末1秒保持使用原定义：|ey|<=.10、|eh|<=.05、|ev|<=.10；全程peak_roll<=.302。
- finite tracking_domain_failure不是工程无效数据：该候选计任务失败，不能因轨迹短而省略失败场景；普通非终止截断、丢帧、NaN仍不允许选best。

推荐排序：
(N_physical + N_domain,
 N_working_failure,
 N_primary_band_failure,
 N_goal_hold_failure,
 max_case_primary_ratio,
 worst_case_normalized_error)。

qualified_task表示上面四类计数全零。qualified_preference是第5节的配对结论，不能由单模型qualified_task自动推出。

所有候选保持相同DEV路径、初态、v_user、时长、下层和投影协议；PP输出是反馈量，两种策略的nominal可能不同，不要求它们逐值相同。

最终从best_task_model指定路径加载Actor核验SHA，报告该来源更新及其与last/旧best的差别。候选范围仅100/150/200等已评点，不声称每次更新都找到全局物理最优。已有轨迹直接重评分，不为改指针重跑全套仿真。

## 5. 主线必须交付“配对路径/速度取舍”，不能只有两份reward

固定DEV先保持已有6条：straight/left90_R2/right90_R2 × 2.0/2.6m/s，全部20s。

每组比较零上层B0与两个端点best。对左右R2@2.6，至少报告：
- 全程及弯道段实际速度RMSE、实际横向RMSE、路径切线航向RMSE；
- 平均欠速、最低实际速度、相对入弯前实际速度变化；
- 上层保护前请求、filtered、实际offsets、governed、actual；
- peak_roll、越工作界时长、物理失败/域失败、目标截面与进度、末1秒保持；
- 原始误差分解actual-governed与governed-nominal，仅用于局部控制诊断。

配对差异：
D_v = E_v(alpha0)-E_v(alpha1)，
D_y = E_y(alpha1)-E_y(alpha0)。

两者为正才支持方向正确。还要报告幅度，不能把小到数值噪声的区别当成果。开发阶段可标注D_v>=.03m/s、D_y>=.02m为“有可辨识差异”的参考标准，不加到reward、不强迫本来不冲突的场景人为分化。

如果两个端点已经同时满足速度/路径/共同安全，不必为了图分开强迫牺牲。反之，如果alpha1只是更接近目标速度但两端都越工作界，不称任务成功。

弯道是空间段，不按同一墙钟窗口假定车辆位于同一位置。增加按固定弧长bin的描述表以检查速度慢的策略是否只因时间权重不同得分；不平移曲线、不重锚、不追赶时间位置。

名义转角是PP的内部建议，不是几何任务主目标。不要重新加入很大的actual_delta-nominal_delta或“不准修改转角”项，那会惩罚为了路径恢复做出的必要调整。

## 6. 奖励与训练分布：本轮先冻结，不生成另一套大权重

实际沿用：
C_v=(10-9chi(1-alpha))*H(ev/.1)
C_path=(10-9chi alpha)*(H(ey/.1)+.5H(eh/.1))
C_primary=20chi*((1-alpha)H((|ey|-.1)+/.1)+alpha H((|ev|-.05)+/.1))
加已有共同roll/working_roll/roll_rate/overspeed/low_speed/offset_magnitude/offset_rate，独立分项cap，总界5080.2，无总裁剪。

仅终止合同见第3节。网络结构、滤波tau、主项系数、动力学不变。

记录每25更新：
- 完整回合总回报、失败和任务域失败率；
- 普通段、弯道chi>0、强需求chi>=.5、出弯直线各自的样本数与速度/路径/航向RMSE；
- 三类route占比，左右符号、v/R分布；
- 请求/执行的有符号Δv和Δδ统计，参考slew与幅值裁剪分别报告；
- 每个cost原始/有效均值与cap比例；
- PPO accepted_epochs、KL、log_std、实际学习率；
- physical_steps、policy_steps、完整回合数、训练/编译/DEV墙钟分别统计。

现在训练有大量直线尾段，速度/半径独立抽样；“一共很多环境”不等于强需求段样本充足。但未取得真实统计前不得立即改课程。若200后两端在强需求段无差异且其样本极少，才提出后续定向增加高速度小半径组合比例；保留普通与恢复，不让每个回合都极端。需另建sampler版本，不热改本轮。

## 7. 运行效率：不为日志或重复基线阻塞主线

1. 零上层6条同协议基线只需一组物理轨迹，按两个alpha分别离线计分。resume父目录已有完整结果时校验路径/初态/模型/执行时序/SHA后复用；只补缺失案例，不重复全部。
2. TensorBoard HTTP失败只写警告。SummaryWriter本地事件、metrics和checkpoint有效时继续训练；不能因为_verify_reward_http超时杀死训练。日志盘写入失败仍是重要故障，不能忽略。
3. 不为每个20ms区间更新status和生成PNG；训练状态每完成更新写一次，DEV每0.5–1s写一次即可。
4. 先不要另造性能框架或迁移引擎。需要优化时复用已有1s lax.scan批量验证，等价数据检查只限同一短样本，不顺便改物理、精度或控制周期。
5. 不新增20/25更新的场景评测。已有100/150/200按数值评测，最终best生成完整图。

## 8. 完成200以后怎样决定是否延长

本轮当前已授权每端200；继续完成这一预算，不提前因零上层末1秒航向失败停止。

在100/150/200固定验证之间，用第4节新任务指标和原指标同时观察。训练每25更新比较完整回合与分工况统计，不能仅凭某个rollout总reward。

- 已有明确安全的取舍且最终任务合格：进入第9节，不为了凑轮数续训。
- 末两个晚期点主要计数减少或同类物理质量改善>=2%，训练对应工况也改善：建议再追加100，到300；仍有证据再到400，不能当作已自动获批。
- 只有训练reward变好而固定DEV不改善：先检查强需求覆盖、cap与过拟合，不自动加轮数。
- 连续三个晚期点无计数改善、质量改善<1%，训练对应任务变化<2%且仍不合格：报告未达标平台期；只选一个明确问题修改，不再同时改reward/底层/规划器。
- 单个checkpoint更差不等于整个训练倒退；best永久保留。

现load_plan硬性限制updates<=updates_initial(200)。实际批准扩展后，才能把运行预算检查改为显式授权target<=updates_max(400)；不通过编辑“updates=300”后忽略校验绕过合同。执行预算与学习任务身份分开记录，不能忽略所有config SHA。源reward/网络/滤波/下层保持一致时沿用学习器状态，无需重训；路径计数使用next-unused key，物理重启不是逐帧续跑。

## 9. 端点可用后，只补最少量独立证据

原6条是开发集，用于选best，不称独立泛化成功率。

两端配对有效且有共同安全之后，冻结模型再新增最多3条未用于调参的完整路径：一个不同半径/角度左弯、一个镜像右弯、一个S弯；每条同初态/速度比较零上层与两个端点。事先固定路径和成功标准，失败不能删掉。

最终再做两个教师到一个alpha条件学生的合并方案，本轮不蒸馏。不要为达到网络数量目标提前压缩尚未可靠的行为。

## 10. 哪些情况才返回底层问题

在下列情况前，不重训local300，不要求其在每个5ms时刻理想跟踪：
- 在已经存在的上层轨迹中，governed温和且持续稳定至少0.5s，仍有速度误差>.15m/s或转角误差>.04rad持续>.5s，且任务失败与此对应；
- 最终执行器命令合理但实际运动受明确力矩/接触限制；
- 主线上层给出合理动作后，已达到原权限边界且仍无法满足指定工作域，需要重新讨论可实现域。

即使满足，先用已有日志定位，不立刻新建400轮底层任务。下层单独RMSE不完美不是阻断理由。上层频繁改变指令时没有稳定样本，应报告N/A而不是要求底层零延迟。

## 11. 实施文件与最少检查

文件：
- path_reference.py：拆开finite domain与numerical fault。
- path_command_env.py：域终止标记、失败计分一次、保留后状态和有效步数。
- path_command_training.py：对应环境reset、日志非关键服务失败不抛停、续训不重复zero基线、统计完整回合/工况。
- path_command_selection.py：保留legacy，新增task_best_v2指针及独立域失败类别。
- path_command_reporting.py：配对指标、不同nominal来源、convergence和best/last身份。
- 配置仅新增明确的运行/终止/选模版本，不改已有网络/reward系数/滤波/物理。

新增检查上限：终止与只reset一个环境、best容差反例、缓存身份、同alpha重评分4类；已有测试直接复用。不全仓回归、不重新查ESO mask、不重跑A/B。

验收反例：
- 两个安全候选主误差均在容差内，一个微小更准但未完成目标，另一个完成：新task_best应选后者。
- 一个有限偏航域失败环境不停止另511个环境；NaN仍停。
- 旧best/模型SHA不变，新终止版本和废弃rollout有明确记录。

## 12. 最终交付

首先交付主线结论，而不是测试数量：
1. 实际训练到哪一步，是否正常推进，来自本机哪次时间快照。
2. 是否已有同路径下alpha0更多保路径、alpha1更多保速度的实际证据。
3. best与last及legacy-best的来源差异，目标/安全/保持是否满足。
4. 底层缺陷是否确实阻断任务；若没有，不再延长底层研究。
5. 下一步是继续100次、补3条独立路径，还是针对一个具体错误改动。

保存配置、源码commit/patch、原始NPZ/指标、模型身份和有限故障证据。未经明确请求不自动push，不覆盖其他worktree，不操作JIT/DVGC。读取状态不是后台监控任务，不做无限轮询。

## 审计来源（固定版本）

- https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/61f612aa82f695bf5b57fb354cb5831a5061f9ac/docs/evidence/path_feedback_v1_20261010/INDEX.md
- https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/61f612aa82f695bf5b57fb354cb5831a5061f9ac/learning/src/sttw_control/path_command_training.py
- https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/61f612aa82f695bf5b57fb354cb5831a5061f9ac/learning/src/sttw_control/path_command_selection.py
- https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/61f612aa82f695bf5b57fb354cb5831a5061f9ac/learning/src/sttw_control/path_command_reward.py
- https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/61f612aa82f695bf5b57fb354cb5831a5061f9ac/learning/src/sttw_control/path_command_env.py
- https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/61f612aa82f695bf5b57fb354cb5831a5061f9ac/learning/src/sttw_control/path_reference.py

这份文件是基于已推送源码的实施意见，不是新修复已运行成功的证据。未读取到实时训练reward时，不宣布已收敛或已经失败。
