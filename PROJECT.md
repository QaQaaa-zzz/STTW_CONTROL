## dt005训练53轮与后续候选比较准备

训练pid4094392已核实live，53/150更新、最近采样5.852秒/优化.186秒。新select_phase_u_candidates.py只生成候选清单：按每个已存周期模型自己的下一次rollout奖励排序，取最多三个不同更新数，阶段完成后追加最后模型；不把model_25错误归属update25（原runner保存后实际26）。当前sample best仍update0，短启动批次与完整episode分布不同，不能据此判后续模型无用。清单是DEV筛选，最终还需同工况物理比较，未自动采用或声称训练收敛。

接续进程4081935 live elapsed9:52，真实source_exit_snapshot与264个lower步已保存，出口真速度2.353m/s、近零横向误差、roll峰值约0.102度；这是本来源退出状态，不能宣称强残余扰动或左/右候选差异。B仍运行。GPU错误参考NVlabs/RoboLab docs/env_vram_size_guide.md列出narrowphase kernel launch作为显存故障特征，只支持资源问题候选解释，不证明本机确切因果；本轮未额外启动GPU任务。

## 同车H75/Local300到跳跃入口已接通，首跑引擎错误不采用

跳跃评估复用IsaacPathRollout作为可选ground_bundle，在同一个commonasset与.005物理实例中锁摆运行H75+Local300，再原位释放摆给PhaseU/RB，控制周期.005/上层.02，保存ground身份。ground_alpha=.5首跑输出runs/pendulum_sttw_jump_dt005，日志出现GPU narrowphase/friction kernel launch errors，确切原因未定；虽写出results.json，已INVALID.json标明不可用于兼容或能力结论。保留runtime.log，不将其曲线当有效实验证据；后续需补无引擎错误运行。没有修改其他任务或新训练预算。

ContinuousTask新增纯任务接口：已知地图允许路线→经验能力查询→相近完成概率下选成功时间，未知/不可通行返回剩余任务修订请求，保留当前有效参考；实际到达、直立、区域条件才推进完成序列。三个单元用例验证无真实运行数据时不伪装完成/不选已阻塞路线；不称LLM已部署或在线仿真闭环已完成。

## 跳跃训练已切换0.005秒；同车STTW调度同步适配

用户要求同车STTW与跳跃兼容并中止旧训练。旧pid4004504已停止，23轮记录保留，旧采样reward best实际update0用于新阶段weights-only初始化，fresh optimizer，不冒充23轮精确续训。新配置isaac_phase_u_training_dt005.json：4096×24×150，physics.005/sensor.005/policy.02，保留原模型网络、Go2 PPO与随机化。初始化首次因并发显存OOM失败（0更新）目录保留；仅降低预分配接触容量到2^22/2^22/2^24后运行成功。

真实新pid4094392、runs/phase_u_isaac150_dt005链接数据盘，4轮时采样6.028s、优化.156s；旧阶段最近采样78.733s，属于不同运行/并发负载实测，不称严格单因素加速。TensorBoard沿6006保留所有原入口并新增phase_u_dt005，HTTP200；新通知器延迟600s启动。未宣称训练效果或完整兼容已通过。

IsaacPathRollout已按plant.physics_dt推导每个.005s地面控制周期的物理步数：原.0002为25，新.005为1，H75仍每4个地面控制步执行。原STTW在跑任务未热更。4项状态/任务管理单元测试通过；同车锁摆STTW→释放摆跳跃仍需完整实际轨迹验证，不能只凭两端可调用就称成功。

## 0.005秒短测完成（非等价替换）

runs/pendulum_jump_dt005_small_buffers/dt_comparison.png为同冻结source/RB与旧.0002实际XY、根高度及侧倾对比。仅小批量缓冲各2^20，日志无PhysX错误；采样11.6658秒/764lane physics，0optimizer。两者.78秒接管，source1.65秒、RB1.91秒45度侧倾终止；最高根高度.40634/.39195m。旧source2.09秒/RB1.865秒、采样40.8938秒。窗口及共享负载不同，不将总耗时比当训练加速倍率。地面接近与起跳形状基本保持，落下后响应明显变化，不能直接认为动力学等价或跨平台跳跃成功。旧150轮训练没有改步长。先前OOM结果仍无效且保留。

## 同出口六候选实际采集入口与固定场景准备

prepare_connection_candidates.py复用geometric_path.build_path，准备固定直行A与左右90度B路线，不按车辆偏移移动原参考；两速度1.5/2.3m/s×alpha0/.5/1，各B观察10秒、2m原路径走廊、弯后5m目标/2m到达半径，全候选均启用已有Dubins重规划。source几何进度3.1m触发保存，不能称所有轮接触已离开地形；源斜向凸起支持最远3m，真实接触状态仍需轨迹解释。terrain复用现有place_isaac_terrain.py/Isaac Gym转换，3cm凸起、0.1m网格、0.005m高度量化。准备产物runs/connection_candidates_nominal，未启动采集。

collect_connection_candidates.py接既有状态保存/恢复，保留历史和ESO等，分别执行两任务六候选，保存原始XY/固定路径误差/失败/到达及JSONL。先在GPU累计轨迹，结束后导出；包含全窗口存活与走廊约束，不仅末端。当前只是单来源DEV准备，无法单凭12个候选拟合或评价泛化；后续需多个真实来源运行。物理预算上限780000 lane steps、0optimizer，待GPU资源可用后执行。完整PhysX内部缓存仍不在快照范围，限制保留。无新训练或实机。

## 可配置物理步长与0.005秒短测状态

IsaacLabVectorPlant新增physics_dt（默认保持0.0002），状态更新时间与记录随实际步长；PhaseU策略周期0.02、传感周期0.005保持，物理子步自动推导，动作DelayBuffer原有逻辑按实际物理dt换算。跳跃评估同样使用实际子步并记录。新配置isaac_pendulum_jump_dt005.json为两策略最多4秒、1600 lane物理步、0优化预算。

首试runs/pendulum_jump_dt005发生PhysX GPU分配64MiB失败，尽管进程返回0，标记INVALID.json，runtime.log保存；产出的曲线/数值不得用于步长结论。并发时原训练占9607MiB、另一任务6765MiB、新实例6699MiB；未停止其他任务。尚未证明0.005秒可用或提速，等待资源后再次执行；原150轮进程未热更。后续需另存重试结果，旧错误记录保留。

## 同出口连接候选的控制状态保存/恢复接口已实现

IsaacPathRollout.capture_connection_state在50Hz边界导出单lane车体/关节位置速度、ECBC ESO/扰动/gains、Local300历史/前动作/偏航率、H75历史/mask、filtered/offsets/执行模式、原参考/命令、进度/跟随器及物理控制时钟。restore_connection_state把该出口广播给匹配原模型/网络/控制器的名义批次，不用重新预热替代真实出口。begin_connection只绑定下一路线/速度/alpha与新任务计时，保留历史、ESO、滤波和原参考slew。

当前范围仅原STTW、名义条件、无偏好自动切换的接续采集；训练随机化delay/RNG和带摆制动状态不在此接口范围，不伪装通用快照。PhysX内部solver/contact缓存尚未序列化，故称“可访问物理＋控制状态”，不称完全同引擎内部状态重放。真正物理接续尚未执行。新增状态广播测试覆盖坐标原点、独立保存、ESO/历史/滤波和时钟；连同能力评价共5项通过，单元fixture不是实验结果。后续真实出口采集仍须检查实际配对轨迹与scene身份。

## 第三章接续评价补充按原始运行整体留出的DEV评估

在训练等待期间复用ConnectionCapability，新增evaluate_connection_groups和薄CLI evaluate_connection_capability.py。输入是实际执行的同snapshot/task/skill全候选配对结果；origin_id将一次上游运行产生的所有出口、左右支路和候选整体留出，避免同源状态进入查询库。每组输出预测、支持量、所选候选真实完成标签与成功条件代价、实际候选中最优参考，以及有支持预测的Brier；未知保持None并单独计数，不能当失败/成功或凭空给备选路线。

候选缺项/重复或同snapshot候选状态不一致在输入处报错，避免把未执行候选当负样本。四项ConnectionCapability测试通过，新增单来源多出口左右支路全留出后应全部unknown，以及缺候选拒绝；均为明确单元fixture，不是机器人实验数据。真实6候选同出口bank尚未采集，现有非配对H75曲线不回填此bank；因此未声称能力评价准确率/第三章实验完成。工具角色为leave-one-origin-run-out DEV，不是独立TEST、规划器闭环成功率或实机证据。本轮没有新物理/训练查询，既有PhaseU进程按UTC07:55检查安排保留。

## 训练后冻结导出与多候选接管比较入口准备完成

`export_trained_phase_u.py`显式指定checkpoint和source-bundle，严格加载原网络state_dict后导出Actor/critic/normalizer，继承动作映射并记录当前checkpoint路径/SHA及selection infos；不默认用last，不改权重。`evaluate_isaac_pendulum_jump.py`按skills数量创建lane，可同场比较source/RB/训练best或last。新增plot_isaac_pendulum_jump.py直接复用已保存NPZ，已用aligned旧轨迹出图并查看，无重跑物理。训练后候选尚未执行。

原Go2 runner.save中途iter字段可能仍为0，因为current_learning_iteration到整段结束才更新；当前运行的best_step_model.json及checkpoint infos.update已正确记录采样模型更新号。导出分别标stored_checkpoint_iter与scored_policy_update，后者用于best语义；源码后续保存best已改直接写iter=实际update。此修改不热更当前训练进程，未重启/未改其权重；避免把metadata0误称最佳一定在update0。普通定期checkpoint仍保留上游语义，以日志/文件轮次另行识别。

训练pid4004504在本轮首查运行71秒且initializing，TensorBoard服务HTTP200，尚无本运行标量；随后只做离线评估入口，不轮询。下一次训练状态检查不早于启动约10分钟（UTC07:55附近），再以实际首轮耗时安排后续。目录runs/phase_u_isaac150链接数据盘，预算仍150updates。

## 原Phase U完整环境接线通过，启动4096环境有界适配

`PhaseUEnv`已接真实GPU物理、原92/8 reset、原事件/奖励/106critic、startup摩擦/质量惯量/COM、四动作增益/延迟/传感噪声及4s推扰。接口检查runs/phase_u_env_integration：4lane×8policy=3200physics lane steps、0optimizer，采样4.059s，初始化2.558s（不含App加载），奖励/观测有限，partial reset通过；不是恢复能力验证。共同车natural rootz=.254与原donor .15不同，明确以模型几何适配避免穿地；8%空中RSI范围原样保留。训练为原PhaseU全程四动作控制，不是锁摆地面交接实验。

正式入口train_isaac_phase_u.py导入原Go2 OnPolicyRunner/PPO，仅替换模型任务adapter与采样前best记录；4096env×24步×150updates=14745600policy transitions/1474560000lane physics steps，50Hz策略/5000Hz物理，原网络/critic/归一化权重，fresh optimizer。当前Go2源码PPO为lr1e-5/adaptiveKL.01、5epochs/4minibatches、gamma.99/lambda.95/clip.2/entropy.01。训练奖励归属更新前模型，best不是固定任务合格，最后模型另存。产物计划数据盘/media/qy/6A7E3DB07E3D75C3/STTW_runs/phase_u_isaac150，根盘约30GB保留。训练尚待实际进程记录，其他线程地形任务不停止。

## 四动作Go2式随机化适配完成，实际GPU DelayBuffer检查通过

`PendulumControlRandomization`复用Isaac Lab DelayBuffer，动作按物理0.2ms时钟延迟25–75tick（5–15ms），IMU/关节/速度按源Phase U 5ms采样时钟延迟1–3tick；50Hz策略历史与传感器采样拆开，不将延迟取整成0或20ms。转向位置、后轮速度及hip/knee位置servo均按原增益范围随机化，原力矩/动作范围不变。角度偏置仅进观测，物理状态/奖励终止不改。25维frame新通道噪声在isaac_phase_u_task.json明确：加速度0.2m/s²为显式任务映射，已知地图障碍距离/根高度不加噪声；不能称与Go2腿部观测逐字段相同。现有startup摩擦/质量惯量/质心函数继续复用，周期推扰仍待环境接线。

实际env_isaaclab cuda:0调用原DelayBuffer测试：25tick精确延迟、原始frame未改、last_action和地图字段保留、批量gain可用且力矩仍≤30Nm、局部reset清空对应历史均通过，runs/phase_u_randomization_port/result.json。首次torch.long与原buffer int32索引赋值不兼容，已按既有模块改torch.int，错误日志保留。0物理/0优化，不是随机化训练已完成。两项已有观测行为测试通过；下一步环境闭环接线与正式runner调度，不再重复已验证纯数值模块。

## 原Phase U事件/终止与106维critic接口完成

`phase_u_events.py`批量实现原一次性jump_signal、consumed锁存、ascending/height/apex、速度卡住与失败优先级，保留原step计数和timeout；原env需要先初判terminal→advance_events→同旧step终判，不能把apex当终止成功。runs/phase_u_event_port对只读原semantics.py比较32lane×12步，共384事件转移＋384终止组合、252字段比较一致；覆盖本配置velocity-stuck分支，不声称旧window-stuck分支已验证。没有物理/优化更新。

TorchPendulumObservation.privileged保留原106字段：Actor76、原顺序qpos12/qvel11、Euler3、world线速3、contact1；root姿态使用声明mount虚拟直立框架，freejoint角速度用本体坐标，轮位置与轮速度一致反号。共同车额外被动后悬架仍有真实动力学，但不加入原critic输入以保持网络维度；这是明确状态投影，非模型等同。两项观测测试和一项回报修正测试通过。训练环境/四动作域随机化还未接完，未新训、未重跑物理。

## 原Phase U Actor/critic已接通原Go2 PPO接口

`phase_u_actor_critic.py`保留Actor76→256→256→256→8、critic106→256→256→256→1与SiLU，原checkpoint两网络/归一化均从只读来源导出至runs/phase_u_training_bundle/source。原四均值＋四状态相关scale经softplus+.001，不换成Go2独立std头。rollout存latent动作，环境必须tanh一次再原映射；log probability和采样entropy保留tanh Jacobian，KL比较latent Gaussian。归一化暂冻结原值，续训为权重/critic初始化＋fresh optimizer，不是exact resume。

`runs/phase_u_training_bundle/adapter_check.json`：16输入确定性动作与原JAX max差2.68e-7；latent范围[-4,4]的logprob差1.22e-4（按相对3e-6/绝对3e-5通过，大幅负logprob浮点尺度），不能写绝对误差<3e-5。直接导入原Go2 algorithms/ppo.py执行16个合成转移、1次更新，loss有限且原Actor权重确实更新；不保存更新后checkpoint、0物理、不是训练结果。原source与RB推理bundle不覆盖。导出CLI新增skills筛选并附critic数组，避免无须重导RB。

训练环境的原事件/终止、106维critic观测、四动作随机化仍待接入；未运行4096物理训练，未宣称性能。第三章完整能力评价/任务管理/跨技能仍未完成。

## 原Phase U奖励移植完成，训练环境仍待接线

`learning/src/sttw_control/phase_u_reward.py`实现GPU可批量的原21分项奖励，保留分段侧倾/俯仰/偏航、速度、起跳高度、动作、实际摆功率与终止罚。`runs/phase_u_reward_port/compare.py`直接调用只读DVGC/JIT原函数，128组含分段边界与终止组合逐项对比：总奖励最大差0，唯一非零分项speed最大5.96e-8，0physics/0optimizer。原env.py还有stuck/yaw导致整回合回报固定-100的修正；已保留单独phase_u_episode_reward，修正不能再次裁剪，也不把它漏作原奖励。专项测试覆盖历史累计700时末步-800、累计-300时末步200，均最终-100。

`learning/configs/isaac_phase_u_task.json`保存原resolved合同与来源SHA，明确不是可执行训练配置。原Phase U训练propulsion-ascent，事件apex高度.5m、起跳区x2.5–4、初速度2m/s、8%airborne RSI、400×.02s回合、侧倾35deg/俯仰75deg等保留；前两轮45deg停止仅接口DEV，不冒充原任务验证。首次apex事件奖励不是完整过台/落地成功。原8logits（均值与状态相关噪声）不能直接换成Go2默认4输出+独立std，否则违反保留网络要求；需兼容adapter再复用runner。随机化现有模块针对2动作/7测量，未接4动作/76观测前不能宣称Go2随机化生效。无新训练。

## 对齐跳跃接近航向后仍失稳；诊断循环减负已实跑

`runs/pendulum_jump_aligned/xy_jump.png`（实际查看）显示接管前沿直线接近，0.78s释放时y约0.00083m；Phase U2.09s、RB1.865s达到45°终止，根最高.3996/.3906m，未完成过台。相比首轮约.204m交接偏移，朝向对齐排除了该明显接近误差，但不能排除其余模型/交接差异；目前证据支持准备基础跳跃适配，不支持直接采用RB或继续追求地面微小误差。

实际20900physics lane steps/40.894s、0optimizer，原首轮19900/62.698s：约511对317lane-physics steps/s（1.61倍观测比）。初态不同，非严格单因素性能基准；不外推4096训练吞吐。实现仅诊断可选减负：旧plant.step默认完整刷新；本跳跃任务在5000Hz伺服内只刷新关节，控制边界取body/root，接触物理保持。该入口没有接触奖励/终止依赖，未来训练必须按原PhaseU任务接回所需接触和事件。此前GPU集中导出也生效。下一阶段移植原PhaseU任务与奖励到现有Go2训练runner，沿用原76-256-256-256-8网络和共同车辆，接通适用域随机化后声明训练预算；尚未开始训练。

## Phase U/RB首轮接管完成；后续训练禁止照搬慢速诊断循环

`runs/pendulum_jump_pair/xy_jump.png`与原NPZ记录两策略完整实际轨迹，0.78s在近似相同状态释放；共同快照重置相同，但并行物理有微小差异，并非逐位相同。Phase U1.99s、RB1.75s达到45°侧倾停止线，根最高0.401/0.390m，尚未过台，不能判跳跃完成或RB优越。初态保存有非零航向，因此世界y=0直线为期望接近参考，不冒充初始航向指令积分线。实际19900physics lane steps/0optimizer、采样62.70s；单初态DEV，不是成功率或策略能力定论。保留车体与donor差异，暂不盲目续训。

用户要求后续训练速度：本次5000Hz逐Python物理步、强制传感器/全状态刷新、两lane及逐控制步CPU导出为诊断流程，不能直接作为4096训练入口。已将该评估轨迹缓存保留GPU并在结束集中传CPU（未重复物理，不宣称加速倍数）；正式训练复用现有Go2式批量GPU采样/PPO/随机化，实现必要接触终止与奖励而非删除相关物理语义，保持自有模型/网络/频率/难度。分阶段测采样与优化耗时后再报告速度。三个接口/伺服行为测试通过，接管真实试跑结束，无训练、无push。

## 原Phase U与RB先配对试跑，续训条件保持用户模型与网络

最新用户允许同时接原Phase U底层跳跃和RB候选先比较，效果不足再训练。`source`的binding receipt追溯原Phase U transition4988928；rebind目录transition0不代表未训练。后续如需Isaac续训，复用Go2 PPO/采样及域随机化实现，保留自己的车辆、网络结构、动作/观测和任务合同，不直接替换Go2策略。

`learning/configs/isaac_pendulum_jump_pair.json`声明同锁摆地面快照平移到x1/y0、两策略各一lane、原3.6m前缘/0.16m高台、进入原x2.5–4起跳区释放，4秒/40000physics lane steps/0optimizer。地面200Hz，跳跃50Hz，物理5000Hz。修正障碍观测为实际碰撞几何最前端距离，保留真实物理FIFO；地面last_action=0显式表示尚未执行跳跃策略。输出runs/pendulum_jump_pair，当前已启动，未宣称跳跃成功。共同车质量8.92421kg、轮半径0.1m、转向1Nm/后轮3Nm，与原donor不同，结果不是只切换引擎的单因素实验。

## 锁摆左右弯完成，原跳跃策略与RB候选GPU推理就绪

`runs/pendulum_h75_locked/RESULTS.md`包含两个弯向的同图五alpha/Zero、全程路径速度指标。H75 10/10回合完整10s并goal；右弯pathRMSE .293/.164/.134/.102/.087m，尾段摆动和严格hold失败保留，按用户决定推进不重训。Zero两弯提前域退出，不称摔倒。左弯旧出图错误已修，不重复物理。

`export_pendulum_skill.py`通过原PendulumSkill/checkpoint身份加载，原样导出原始source与rb_run1_candidate的76→256→256→256→8 SiLU权重、mean/std、动作映射到runs/pendulum_torch_skills。Torch确定性tanh前四logits推理保留原网络/权重，无蒸馏。各16个声明数值输入与原JAX比对；Isaac Lab环境cuda:0两策略动作max误差2.384e-7、目标max误差≤2.861e-6。这里是推理实现一致，不是物理sim2sim或跳跃成功。候选Run1仍非best选择结论。JAX依赖已移入原适配类局部，Torch观测/推理无需在Isaac环境装JAX。

下一步真实共同模型上地面锁摆→释放→原策略/RB接管，保持真实FIFO与物理状态；观测实际加速度及地面动作历史约定尚需显式接入。未运行完整跨技能任务，不因GPU推理通过称章节完成。

## 锁摆H75左弯完成；RB观测组装实现待物理接入

`runs/pendulum_h75_locked/RESULTS.md`先给实际XY。五alpha皆完整10秒/goal，路径RMSE .232/.158/.110/.064/.083m，速度RMSE .149/.105/.076/.055/.039m/s；严格final_hold按各行记录，不因此重训或要求单调排序。Zero9.18s domain_exit，末roll-.02491rad，非摔倒。平面绘图错误plane/identity.json已原位修复并复用轨迹补全，原批处理退出后仅继续右弯session19125，未重跑预热/左弯。用户认可当前效果并要求推进。

新增TorchPendulumObservation组装RB3×25+signal，复用原rotate_inverse及原JIT字段顺序：按mount将chassis转到虚拟RB直立坐标，轮速符号转换一次；保留实际世界x速度/高度/障碍相对x，不伪造donor质量/半径/高度。50Hz连续FIFO，last_action需调用方明确来源，地面模式不能把虚构RB动作称实际已执行。当前只是数值接口，未接真实IMU加速度/策略滚动；共同chassis root COM为0的加速度假设已注明。三个观测/伺服测试通过，验证姿态坐标、静止specific force、轮速符号和三帧顺序，不称RB接管成功。

## RB交接动作接口：复用原车备用转向位置伺服

源码核对：原RB的四动作映射为转向角/后轮轴转速/hip角/knee角，而地面STTW为转向角速度/后轮轴转速，不能直接串接。共同STTW模型已保留未启用的cmd_steering_p（kp10、kd0.1、角度±0.8rad、力矩±1Nm），因此复用它作为显式跳跃位置接口，不自行拼角度到角速度转换。旧RB donor的转向力矩±5Nm、后轮±6Nm与共同车不同；共同车仍保留转向±1Nm、后轮±3Nm，不能假设原RB直接成功。

IsaacLabVectorPlant.step新增可选steering_position_target（batch×1），显式选择时关闭同关节速度servo扭矩，只使用原位置servo；hip/knee目标沿用已有接口。TorchPositionServos的allow_inactive仅用于显式选择已有备用执行器，缺省保持旧检查。两个servo行为测试通过（包括原力矩限制与阻尼），尚未执行RB闭环或跳跃接管。进入跳跃需要先释放摆，持续保存真实历史；原RB 3×25+signal观测尚需接入真实共同车坐标/IMU/轮速，未用零历史替代。

锁摆H75批次session25963已完成预热进入left90_R2，沿原预算运行，右弯稍后；不重启、不训练。当前新增为复用原执行器的接管接口，不是跳跃成功证据。

## 地面锁摆接口已修复，进入H75左右弯执行

`runs/pendulum_isaac_locked_jointonly`：3秒转弯/15000physics/0optimizer，髋摆角最大漂移0.000100017rad（0.00573°）、膝4.768e-7rad；约束瞬时角速峰值仍0.09054/0.00223rad/s，不宣称完全零微振。无PhysX限位报错，peakroll5.59°、未终止。释放接口恢复原limits缓存且位置未重置，释放后RB动态控制尚未验证。旧等限位/窄限位tensorAPI均会全关节回写，两个带报错记录保留；最终用独立USD摆关节更新避免碰车轮。

用户认可当前行驶效果，继续推进，不再重复七工况或训练。已启动session25963，runs/pendulum_h75_locked：同锁摆模型3.5秒ECBC预热生成初态，继而原H75/Local300左右弯五alpha+Zero各10秒；最多617525physics、0optimizer，结果pending。理想极小游隙制动为显式仿真假设，未声明实机机械锁能力。下一步完成路径任务后再接释放与RB，维持跨技能统一模型目标。

## 地面固定摆、跳跃释放及继续推进

用户认可当前行驶表现，要求继续推进，不为缩小误差重复调参/训练；但地面摆应固定，跳跃才由RB接管。新增显式pendulum_locked选项，地面以当前hip/knee角±1e-4rad极小游隙制动（裁到native limits），不额外添加摆位置servo扭矩；set_pendulum_locked(False)恢复原范围、目标对齐当前角、不reset车体/关节。此为仿真制动模型，不是已验证实机机械锁或原30Nm电机能力。

首版等上下限受到PhysX报错，runs/pendulum_isaac_locked保留不作为接口通过证据；现有极小游隙修复正运行同3秒/15000physics/0optimizer短测，session52089，runs/pendulum_isaac_locked_clearance，记录实际角度/角速及释放前后状态。默认旧plant行为不变。旧H75 PD保持straight已完成并出图，左右弯调度因用户固定摆要求停止，避免继续错误合同。修好接口后直接往H75/跨技能推进，不重跑全七工况；完整RB交接尚未验证。

## 带摆原七工况结果与H75路径闭环接续

标准评测已正常退出，14lane均完成10s，无物理终止；包括预热总945000physics、0optimizer、416.604s。`runs/pendulum_local300_standard/RESULTS.md`先展示实际/原参考XY、逐时刻最近路径距离及速度误差，保留原轨迹、所有方法指标及原奖励重建。带摆Local300回正/回正后小转向峰值偏离1.08–1.47m，超过0.5m持续2.05–4.30秒，不能因存活判跟踪合格；持续正负转弯优于纯ECBC。原车历史对照初态不同，图按首记录位姿对齐，不能宣称严格单因素效果。

原H75路径闭环评测已启动session80548，目录runs/pendulum_h75_standard：原生straight/left90_R2/right90_R2，每条五alpha+Zero、10s、最多900000physics、0optimizer；使用实际带摆预热初态，原网络参数不改，必要重规划保留。先看路径反馈能否纠正底层开环指令累计偏差，再决定适配。未证明带摆地面技能通过，更未证明跳跃交接。plot_isaac_gpu_evaluation原位支持无曲率直线路径，避免空弯道索引；语法/diff通过。

## 全过程路径分析已接入，标准物理评测继续

`compare_local300_sim2sim.py`新增显式`--start-relative`和`--historical-lab`，用于不同模型预热状态下对照完整轨迹。对每条实际轨迹及对应原指令参考按首个记录位姿平移/旋转，保留原始数据；同图可包含原MuJoCo、原Isaac Local300、带摆Local300与带摆ECBC。新增逐时刻到原参考折线的最近距离、全观察窗口RMSE/最大值、超过0.5m累计时间、速度误差及NPZ；0.5m只是描述统计，未新增任务合格门槛。最近几何距离不能代替沿程进度或最终任务完成。

分析器以现有原车轨迹在`/tmp/sttw_standard_plot_check`执行成功并查看图，属于出图集成检查，不是带摆实验结果。session93823当前已完成3.5s预热，14lane仍active，继续原10s评测；无重启、无额外物理。

## 带摆模型按原STTW标准评测（运行中）

用户要求以全过程XY路径跟踪判断底层是否够用，不能以末端/存活替代。`isaac_pendulum_local300_standard.json`复用原七个Local300指令流和10秒窗口，14lane配对原Local300与ECBC，另含3.5秒共同带摆车ECBC预热；最多945000 lane physics steps、0optimizer。原位复用`evaluate_isaaclab_local300.py`，支持显式asset、同模型fresh warmup与paired_original；旧保存状态模式继续可用。预热及真实终止保留，生成同模型initial_state用于后续H75接口。相较历史MuJoCo/无摆车有初态及质量差异，不能称完全配对sim2sim。

实际输出`runs/pendulum_local300_standard`，session93823，日志`/tmp/sttw_pendulum_local300_standard.log`。结果待返回；下一步首先检查七工况实际/参考XY和回正后偏离时序，再做H75五alpha，不能由上一8秒小转弯诊断推断底层通过。此次无权重/ECBC参数调整，无训练。语法检查通过，运行中直接检验集成。

## 带摆四组转弯诊断结果

`runs/pendulum_isaac_turns/RESULTS.md`、`xy_response.png`及`tracking_metrics.json`保留四条完整轨迹。实际完成8s/160000physics/0optimizer，4lane未触发45deg终止；初始速度均2.0，reset修复在本run得到确认。3–8s正/负方向速度RMSE：ECBC .05524/.05575m/s，Local300 .03440/.02848m/s；峰值roll ECBC5.85/6.04deg、Local3007.80/7.75deg。8s末端yawrate ECBC-.00424/+.04225rad/s，Local300-.15020/+.13044rad/s。Local300改善速度但放大roll瞬态及尾段偏航，不能认定整体更优或共同任务已合格；回正窗口仅1s，保留其限制。

下一步接H75任务级参考检查实际路径行为，仍保留ECBC对照，之后据行为决定底层/中层适配。未启动训练、未证明跳跃交接。采样255.50s，进程退出清理与物理完成分开记录。

## 第三章经验接续查询实现（真实能力库待采集）

`empirical_capability.ConnectionCapability` 接收已执行的闭环结果记录，字段state/task_id/skill_id/candidate_id/complete/success_cost。task_id应标识下一任务弯向、道路和连接条件；skill_id必须绑定实际模型/策略/执行版本，不使用含混的“best”。state特征尺度、Gaussian bandwidth和minimum_support由冻结开发协议显式传入，不在查询时拟合；正式建库需另保留完整快照与轨迹来源。

按每个候选返回加权完整任务完成率、仅成功样本的条件代价、sum(weights)支持度和样本数；低支持返回known=False、概率/代价None，全部未知返回best_candidate=None。无成功样本时成功代价为None；词典序先完成率后成功代价，失败记录上的cost不参与。执行管理器仍需使用声明的完成门槛/备用路线；支持度不是置信区间，经验比例不是已校准概率，也不是安全证明。

13项查询/旧force-grid测试通过，验证任务/策略隔离、远距离未知、完成优先及失败代价排除。测试为人工小样本单元测试，不是物理实验；当前未建真实6候选库、未连接LLM，不能称能力评价已验证有效。

## 带摆左右转弯配对诊断已启动

`learning/configs/isaac_pendulum_turns.json`：4lane正/负0.12rad转向，各比较原ECBC与原Local300＋ECBC；2m/s、3秒零残差预热并填充history、3–7秒slew转弯、7–8秒回正。共同带摆模型/固定摆姿态不变，Local300原权重与1.5/10缩放。最多160000physics steps、0optimizer，输出 `runs/pendulum_isaac_turns`；此处结果尚待采集，不是H75或跳跃任务完成。

源码核实reset首帧问题：官方write_root_velocity_to_sim实际写COM速度，不刷新已用root_link_state_w；更正调用write_root_link_velocity_to_sim，与本plant观测/物理初态bank的root frame契约一致。原chassis COM为0，名义物理速度相同。旧三秒诊断原始首帧保留，不重跑。

前一次三秒诊断结果已完整保存，旧进程3760642退出清理停留超过4分钟，已发送SIGTERM回收本任务进程；未干预其他仿真。新入口直接使用官方app.close(wait_for_replicator=False)。实际session78985/PID3779253已记录200控制步、4lane active，正在继续声明预算。分析器支持批量原轨迹/参考/真实终点，不以存活取代转弯完成。

## Isaac Lab 共同带摆车首次物理诊断

`runs/pendulum_isaac_ground/limit_rounding/RESULTS.md` 与 `xy_response.png`：一lane平地、原ECBC（无Local/H75）、固定摆姿态，完成3秒/15000physics steps，未触发45deg侧倾终止；峰值侧倾0.20002deg、末端速度1.94705m/s。这是单名义直行接口证据，不是弯道/地形/RB交接验证，不据此声明泛化。实际Isaac总质量8.92421009kg，自碰撞开启，25对body按源mask/exclude/焊接及相邻关系过滤。首次采样速度2.3与reset请求2.0不一致，可能为constructor/reset缓存，原数据保留并标图；不是新初始化加速实验。采样墙钟68.46s。

实现：原位增加hip/knee显式affine position servo（原kp100/kd6/30Nm和目标范围），0.2ms物理，5ms ECBC；GPU张量/物理。关节初态用逐名称配置，避免Isaac regex重叠；knee初态2.499999rad避免USD弧度/角度边界舍入，原关节范围未改。前两次启动失败无有效轨迹，原文件/日志保留；最终结果已保存，物理不重跑。新增servo限幅/阻尼测试、带摆导出2测试及原URDF回归通过。

后续先修正reset首次读取并接转弯参考/冻结网络，不因资产变化自动重训；按真实任务行为决定ECBC与网络适配。原STTW评测不改。

## 带摆 URDF 导出与运动学检查

现有 `isaac_urdf.py` 原位支持非原点单hinge和圆柱碰撞；URDF link改以hinge为原点，COM/geom/子节点origin同步平移，`link_frame_offsets`记录变换。`export_isaac_asset.py --preserve-self-collisions`记录源contype/conaffinity及显式exclude，不再用原无自碰撞假设拒绝候选；默认原STTW路径不变。实际候选在 `runs/pendulum_common_asset/isaac_urdf/{vehicle.urdf,contract.json}`。

2项带摆测试＋1项原URDF回归通过。独立MuJoCo URDF读取在(hip,knee)=(0,0),(-1.2,2.5),(.3,-.7)三姿态核对所有link质量与世界质心，确认膝轴偏置映射未改变运动学。这是导出验证，无物理步/训练，不是Isaac仿真通过。现有Isaac地面plant仍关闭self_collision，针对需要pair filtering的资产明确拒绝导入，防止静默丢掉摆碰撞；下一步原位实现源body/geom碰撞规则和摆位置执行器后运行短闭环。

## 第三章共同车体候选资产

`pendulum_asset.py` 与 `learning/configs/pendulum_common_asset.json` 构造 STTW 原车体＋RB 双节摆及髋电机的仿真候选，不修改两份原XML。完整候选在 `runs/pendulum_common_asset/complete_mechanism/{vehicle.xml,contract.json}`；最初仅双臂的中间产物保留在父目录，不作为完整机构。安装旋转Rx(-90deg)、平移0是显式仿真设计，不是已测实机安装。

实际编译总质量8.92421kg，原车5.43421kg，新增双臂3.03kg＋髋电机0.46kg；nq13/nv12/nu6。针对性测试确认原车body质量/惯量/局部位姿、关节轴心/范围、执行器gain/bias/限幅保持，膝关节原始偏置[0,0,.11]保留。无physics、无训练、未导入Isaac；不能认为旧RB策略可直接复用。原URDF导出器不支持该膝轴偏置、圆柱碰撞与新增自碰撞mask；下一步原位扩展导出与Isaac执行器，而非删除碰撞或改关节消除报错。

## 第三章最新执行约束：Isaac Lab 与带摆策略适配

用户最新要求：带摆平台采用 Isaac Lab，以除摆机构外与 STTW 一致的共同车体为目标核对与适配。允许调整 ECBC 参数、蒸馏或重训带摆底层，以及蒸馏并续训 alpha 网络；覆盖此前第三章不新增 RL 训练要求。原 STTW 模型及已选权重独立保留。先核对资产和接口，再评估原控制器，随后按需要适配网络；后续训练继续 Go2 随机化/GPU流程、原任务难度和五档 alpha，具体预算逐阶段记录。旧 RB 模型/策略来源保留，不能把共同车体适配后的结果冒充原模型复现。

原生 MuJoCo 的 `runs/pendulum_ground_compat` 仅为已完成接口排查，不作为正式跨技能环境。后续转向 Isaac Lab；文本 LLM/能力工具/受扰任务修订的修订版总体目标继续有效。

## Pendulum-bicycle skill lineage located and model pair identified

`learning/configs/pendulum_bike_skill_sources.json` binds the reviewed source and first Full/Run1 student record from the existing10k evaluation request. Actual payload hashes and checkpoint XML identities match: source actor`a06acbc8…`, Run1 actor`c3eae39c…`; payloads are accessible. Run1 is an interface-preparation candidate, not a new best-model selection or a claim that all historical Full labels share one recipe. Preserve the existing training-record audit differences.

The checkpoint-bound model is `/home/qy/DVGC/assets/orange_bike_4kg_horizontal.xml`, SHA`0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`; the previously found quadruped_jump_jit/framework asset is byte-identical. This selects the correct model family by checkpoint evidence rather than the2kg/4kg filename. No external file was modified.

Direct MuJoCo model compilation (no stepping) gives current STTW mass5.43421kg/nq11/nv10 versus pendulum5.091kg/nq12/nv11. Both XMLs expose four actuator slots, but STTW has alternative steering interfaces while the skill controls steer/rear/hip/knee. `cmd_steering_v` in the pendulum XML is a POSITION actuator despite its name. Pendulum wheel joint addresses also differ; raw qpos/qvel copying is invalid. Compilation details: `runs/pendulum_bike_handoff/compiled_models.json`.

Skill contract:76-D = three25-D history frames + jump_signal; action order steer/rear-wheel-drive/hip/knee. Original skill runtime uses5ms physics and20ms control (constants.py/env.py); XML default is1ms and is overridden by runtime. Current STTW uses0.2ms physics,5ms lower,20ms H75. A shared physical model cannot change integration timestep merely at policy handoff: unified timing and ground behavior on the pendulum model remain to be evaluated, without claiming existing checkpoints retain performance. Preserve actual vehicle/actuator/history/event states through handoff; do not use Go2 mapping or pad observations.

Reusable upstream entries are `build_unified_formal_environment`, `load_checkpoint`, `make_checkpoint_policy`, and `unified_envelope_snapshot` for its own model. No new runtime wrapper, training or physics evaluation was launched in this identity pass. Next: read exact action mapping and state-history lifecycle, then implement a bounded interface on the pendulum model while leaving the completed STTW bicycle chapter intact.

## 当前执行顺序：STTW 走通并形成论文后再启动新章节

用户最新明确要求：先把 STTW 走通形成论文，再开始新章节。当前工作集中于原双轮车、已选橙色 Local300 与原 H75 的越野偏好控制：整理全部已完成轨迹与机制证据，补齐影响论证的缺口，形成与实际结果一致的论文主稿。不得以仿真主体完成或接口加载成功替代论文交付与实车证据。

带摆双轮车、RB 接续、6/36 能力库及 VLM 新章节工作暂不推进；保留已完成的来源绑定和推理适配代码，不删除、不继续扩展。整体目标仍保持，只有阶段顺序调整。STTW 实车尚无执行证据，需要实际部署条件与执行授权；先完成能够开展的仿真证据整理和写作，不虚构实车结果。当前工作树实际论文源为 `docs/paper/MANUSCRIPT.tex`，旧规则提及的 `MANUSCRIPT.md` 在此工作树不存在；后续在实际主稿原位维护，不另建并行主稿。

## Pendulum skill inference adapter available

`learning/src/sttw_control/pendulum_skill.py` loads either `source` or `rb_run1_candidate` from the bound registry, reusing JIT checkpoint loading, original PPO inference/normalization, host model and action mapping. Reference dependency: add `/home/qy/DVGC/JIT/src` and this checkout's `learning/src` to PYTHONPATH; use `/home/qy/mujoco_playground/.venv/bin/python`. No external source edits. `PendulumSkill(registry_path, skill_id).infer(observation76, knee_position)` returns normalized action and native targets `[steer rad, rear rad/s, hip rad, knee rad]`. Caller owns observation FIFO, last-action memory and physical state; inference never resets or steps them.

Actual inference evidence: `runs/pendulum_bike_handoff/inference_interface.json`, both frozen actors loaded and returned finite four-channel actions on a declared synthetic zero observation. Source output exactly matched the original full-environment inference observed this turn. This is interface verification, not a reachable-state, jump-success or sim2sim result; physics/optimizer steps are zero.

The original full training-environment builder requires a DVGC working directory for historical relative paths and rejects the RB training record under its current aligned-PPO-budget check. The inference adapter therefore uses the identity-bound phase config and host model without constructing training/reset pools. It does not relax checkpoint identity checks or modify the historical budget. Full simulation reset/event/landing lifecycle still needs integration.

Native mapping is steer=0.8*a0 rad, rear=clip(12+12*a1) rad/s; hip/knee use piecewise absolute targets centered at -1.2/2.5 rad with original actuator bounds. Knee center equals its upper bound, so positive normalized knee actions remain at 2.5 rad; this is original semantics, not a conversion bug. Do not feed STTW steering rad/s directly into the position actuator. Current STTW selection remains unchanged. Next: apply ground control on this same pendulum model and preserve its FIFO/events through the skill transition; no raw state-copy between different MJCFs.

## User clarification: STTW remains bicycle; cross-skill platform is pendulum bicycle

User explicitly clarified the platform: STTW continues on the current bicycle, while cross-skill tasks use a bicycle with a pendulum. Go2 is a training/transfer reference, NOT the cross-skill robot or policy to attach. This corrects the preceding exploratory Go2 compatibility assumption; no Go2 adapter, training or model change was performed. Bark clarification notification was delivered and answered.

Read-only XML inspection located two relevant candidates: `/home/qy/STTWproject/dvgc_mjx_v21_port/orange_bike_2kg_horizontal.xml` and `/home/qy/quadruped_jump_jit/framework/assets/orange_bike_4kg_horizontal.xml`. Both have `hip_joint` and `knee_joint` in addition to steering/front/rear wheel joints, with steering position, rear-wheel velocity, hip position and knee position actuators. The selected current STTW model has no hip/knee joints. Source hashes and exact actuator attributes are retained in `runs/pendulum_bike_handoff/model_inspection.json`; filenames do not establish the correct trained policy/model pair. Neither candidate is adopted yet.

Next required compatibility work is to identify the pendulum-bike source/RecoveryBridge checkpoint and its exact model/control/history contract, then define real physical-state-preserving ground-to-skill handoff. Do not pad observations, map a Go2 twelve-joint action, assume a fixed pendulum is dynamically equivalent to the current bicycle, or silently transplant masses. All exploration remains read-only outside STTW_CONTROL. Current frozen Local300/H75 and completed terrain results remain unchanged. Hardware evidence and continuous-task completion are still missing; this platform clarification does not mark them complete.

## Frozen offroad runtime entry prepared

Current simulation selection is now explicit as `current_offroad_simulation_id` in `learning/configs/frozen_middle_registry.json`, with `isaaclab_h75_offroad_selected.json` as the six-lane all-replanning nominal entry. Existing U170/historical aliases remain unchanged. No weights or controller behavior changed; necessary actor/config/asset/warm-state identities recorded once. This new convenience configuration has not itself been executed and is not a new evaluation result.

`docs/FROZEN_MIDDLE_NETWORKS.md` documents actual scheduling: Local30010-frame history at5ms, H7516-frame history at20ms, H75 correction processing at5ms. Do not apply the task brief's generic5ms history wording to both networks. Runtime code and immutable checkpoint configuration are the source. World pose/true speed dependencies and measured hardware startup/interface gaps are explicit; no hardware command was sent.

## Contact-force semantics resolved and all-alpha phase alignment complete

Existing installed IsaacLab commit090aed181 documents `net_forces_w` and `force_matrix_w` as world-frame NORMAL contact-force vectors (not friction-inclusive totals). `isaaclab_vector_env.refresh` stores the former; `isaac_path_rollout` stores the latter per wheel/filter at0.2ms. Documentation for this generation: https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.sensors.html . Do not substitute newer main API semantics.

`learning/cli/plot_isaac_contact_alignment.py --run runs/h75_common_state_channels` now produces `analysis/contact_alignment/{RESULTS.md,front_aligned.png,rear_aligned.png,summary.json,alpha*.npz}` from the five natural branches. Quantity is norm(sum of per-filter normal vectors), NOT sum of per-contact scalar loads. Existing world-Fz graphs remain valid vertical components; earlier loose wording as total contact force is corrected here. No physical rerun, new filtering or estimated terrain-normal projection.

Front first raised contact1.5506–1.6002s; rear first1.7448–1.8170s; rear last2.0106–2.1156s across alpha. Thus the natural histories differ before alignment. Full-window raw peak front resultants191.7–257.5N, rear227.7–369.1N are instantaneous simulation samples, not measured hardware loads or a robustness ranking. Phase charts show raw0.2ms resultant pulses,5ms executed speed/steer correction, roll rate and signed path error, with first/last front/rear markers. Local path error need not follow the whole-window preference ranking. Interpret alongside the common-exported-state hold/zero experiments; temporal coincidence alone is not causal mechanism proof.

This resolves the log-semantics/phase-figure gap in the task coverage map below. STTW hardware evidence remains missing; later capability/continuous-task work remains incomplete. Continue to use frozen original H75/Local300 and independent slew, with adopted tangent-heading replanning.

## Tangent-heading closed-loop confirmation complete; STTW evidence figures supplemented

`runs/h75_shared_switch_tangent/DECISION.md`: declared5-branch10s evaluation complete (250000physics,0optimizer). All5 arrive and remain active10s;11 plans contain0 LRL/RLR loops and span2.178–4.506m. Reward replay discrepancy5.93e-08. Adopt the existing PythonRobotics adapter with analytic tangent-heading candidates; retain original H75/Local300 and independent slew. The earlier10.480m loop is preserved in `h75_shared_switch`; no universal no-loop claim. No further planner scan or training.

Saved-data chapter figures now include `runs/h75_terrain_speed_sweep/evaluation/PROGRESS.md`: ten terrain/speed conditions, all fivealpha plus matched Zero, progress-indexed true-speed/cross-track errors and `tradeoff_common_window.png`. Scatter uses the per-condition common active window (early Zero failure is not padded); full actual XY, rewards and failures remain in original RESULTS/METRICS. This supplements the existing fixed-route DEV data and does not relabel it as new-goal/replanning evaluation.

STTW simulation evidence is substantially assembled; real-robot evidence and the later unified-model/capability/VLM chapter are still incomplete. See the current requirement map in METHODS_AND_RESULTS. No hardware run or later-chapter completion is claimed.

## Shared-state switch complete; small terminal-heading repair pending closed-loop confirmation

`runs/h75_shared_switch/analysis/DECISION.md` contains actual XY, reward diagnostics and branch metrics. All5 branches reached the2m goal disk and completed10s. In the first0.5s after alpha1→0, independent/common/steer-priority differences are comparable to natural/restored submillimetre differences; steering-priority did not activate. Keep original independent slew. Keeping alpha1 reduces speed error and increases path error in this window; no universal ordering claim.

A new saved plan at4.0s was10.480m LRL despite goalturn_end+5m/radius2m. Previous final_goal success does not generalize to all nearby states. Adding analytic circle-tangent terminal headings to the existing PythonRobotics adapter gives2.604835m at identical pose/goal/radius; closed-loop confirmation pending. No replacement planner, training, or goal change.

Executed-zero analysis also complete:10cases/50branches,960500physics,0optimizer; all survived their2s postfork windows. Entry speed-zero reduces path RMSE in5/5alpha conditions but increases peak roll in5/5. Exit speed-zero reduces speed RMSE in5/5, path in3/5. Short-window mechanism evidence, not full-task success or universal benefit. `runs/h75_zero_correction_channels/analysis/{RESULTS.md,HOLD_ZERO.md,hold_zero_comparison.json}` preserves all conditions and prior hold comparison.

## First executed-zero case completed; remaining declared cases running

Entry-projection alpha0 finished at3.6s (fork1.6s plus2s),90000physics/0optimizer. All five branches remain active through the observation window. Saved selected correction components are exactly0 for the declared40 lower ticks; first natural/restored policy observation difference0, reward replay discrepancy1.91e-8. In the0.5s after release, natural/restored XY max difference1.34mm, speed-zero13.18cm, steer-zero1.18cm and both-zero13.20cm. These are descriptive local effects in one case, not a better-controller or generalization result. Full two-second response and negative outcomes remain in the report.

Partial report: `runs/h75_zero_correction_channels/analysis/RESULTS.md`; first actualXY at `analysis/entry_projection/alpha0/xy.png`. Pipeline PID3476615 continues the declared five-alpha entry/exit set. Analyze the final collection before comparing zero with prior hold. No training, new planner, or extra scenario scan.

## Executed-zero common-state intervention registered

Previous goal turn completed replanning integration and changed authoritative evidence (progress). Continue frozen H75u77/selected Local300; no training. `runs/h75_zero_correction_channels` adds the requested executed-correction-zero variant to existing hold experiments: all five alphas, geometric entry/exit forks, five branches (natural, restored control, zero speed, zero steer, zero both). The selected executed H75 correction is forced to0 for.2s before absolute reference bounds; filter/history/ESO persist, release uses existing slew. Onset deliberately need not satisfy the normal correction-slew bound, so this is a diagnostic intervention, not a deployable limiter.

Reuse existing fork/evaluation/analysis code, no copied controller or planner. Observe2s after each fork; preserve original10s reward horizon and task clock. Maximum10cases×5lanes×10s=2500000physics, expected~1000000 with early collection end;0optimizer. Contact-force logging uses existing body-world forces at5ms. Avoid expensive repeated0.2ms contact-point collection; historical contact phases remain separate and newFz is not called slope-normal load. This local mechanism window fixes reference geometry to interpret channel effects; it is not a complete goal-task comparison or evidence to remove the required execution replanning stage.

Natural versus restored controls retain the PhysX-cache-copy limitation. Pair intervention effects against restored control. Prior hold full traces are preserved and can be restricted to the same postfork windows for comparison; no need to rerun them. Script `runs/h75_common_state_channels/execute.py --output runs/h75_zero_correction_channels` now accepts an output root. One declared pass, no retries or additional training. Direct executed-zero/bounded-release behavior check passed; actual physics results pending.

## Goal-replanning evaluation completed: keep original H75

Final evidence: `runs/h75_lab_goal_replanning_final_goal/{DECISION.md,RESULTS.md,metrics.json,xy_models.png,xy_models_bend.png,xy_switch.png}`. Originalu77 and reward-peak update33 each complete all five alpha plus1→0 cases:6/6 enter the new goal disk and remain active for10s. Both Zero baselines enter the goal but later terminate on follower-domain exit, not a documented fall. Final goal rule: turn_end+5m, arrival radius2m; old/interim contracts remain separate. Actual switching is2.3s in both fixed/replanned lanes of both models.

Final plans contain no LRL/RLR loops; lengths original2.159–4.476m and best33 2.208–5.771m, all short relative to endpoint distance. Actual full/bend/switch plots and per-step/cumulative reward/component/speed data retained. Reward replay max discrepancy3.58e-7. Full-window original-path errors include deliberate forward-exit tail after goal; not a fair old-path recovery score. Original remains adopted; best33 retained without adoption, no more training/planner sweep. Six bounded evaluations total4200000 physics lane steps/0 optimizer updates; final pair1400000. The previous active training stopped by user at59/150 with artifacts preserved.

Tests: existing planner12 tests passed; actual GPU rollouts complete; no new package installation. New Lab adapter reuses the same NumPy planner geometry without importing JAX, changes only route projection/follower and preserves physical/controller/policy states. Latest STTW experiment variants and chapter architecture remain as recorded above; replanning is a required execution component, while the fixed-route copy here is integration evidence only.

## Current goal geometry: finish beyond the bend, accept a region

User asked for simple usable replanning, not a planner search. Keep the existing pinned PythonRobotics implementation. All new declared bend tasks place the goal at `turn_end + 5m` or farther; current evaluation sets exactly5m. Arrival radius is2m. The earlier1.5m trial still generated an11.026m LRL path at1.586m distance, so it is retained as a negative result rather than called fixed. The final paired run is `runs/h75_lab_goal_replanning_final_goal` (original u77 and reward-peak u33,14 cases each,10s,1400000 physics lane steps,0 optimizer). Earlier goal contracts and artifacts remain immutable; changing the goal prevents direct old/new success-rate comparisons.

Upon measured arrival, an already-replanned route hands off to a straight forward extension from the measured pose, keeping all policy/controller state. This maintains motion for the remaining observation window; it is not an obstacle-validated bypass or a completed next task. Per-step execution-path rewards and original-route diagnostics remain separate. Dubins is a simple geometric connector under curvature limits, not a model of roll dynamics or hard obstacle feasibility. No further training or alternative planner implementation is queued.

## User clarification: replanning is part of the execution system

Replanning is a required component of the intended task execution loop. The current fixed/replanned paired run is an integration diagnostic, not a proposed main-method ablation or a reason to omit planning. Preserve both original-task geometry and active execution geometry. Both earlier and latest experiment variants may be pursued; choose representative main figures based on explanatory value, retain negative results and state the selection scope. The latest attachment guides the work rather than overriding the user's direct choices or making every suggested parameter compulsory.

## Latest task scope and original-H75 replanning comparison

User stopped continuation at59/150 updates; this is a user-directed stop, not convergence. Preserve checkpoints/logs and cancel automatic stage-end review. Keep original H75 u77 and orange Local300 adopted. Also evaluate the stopped run's Episode/reward peak at update33, not a new training run.

Latest task brief: `/home/qy/.codex/attachments/da992a6d-def8-4674-a0f1-0a661cee6df3/貼上的文字.txt`. STTW remains frozen H75, fixed-original-path terrain preference, contact-aligned channel intervention, and spatial alpha switch. Five alpha levels remain; no strict per-case monotonicity or tiny-error gate. Existing channel-hold experiments do not fulfill the newly requested correction-zero intervention; mode comparisons also need an identical full snapshot. Existing2–6cm bumps exceed brief h/R=.05/.10/.15 with .1m wheels (5/10/15mm), so retain them as stronger DEV and add shallow standard conditions when completing the chapter. Common progress is a comparison with honest negative outcomes, not a required improvement. Hardware collection remains unexecuted and separately authorized.

Replanning is an auxiliary task-completion comparison, not a replacement for STTW original-path preference evidence. Reuse existing PythonRobotics Dubins, from measured pose to unchanged original goal; keep policy/history/ESO/filters. Config `learning/configs/isaac_h75_goal_replanning.json`: inspect every.5s, departure.3m or invalid follower, radius1.8m, stop planning after entering original1m goal disk. Nominal4cm bend,10s,14 lanes per model = five fixed alphas,1→0,Zero each with/without planning. Two models total28 episodes/1400000 physics lane steps,0 optimizer updates; output `runs/h75_lab_goal_replanning`. Original-path diagnostics and execution-path reward are separate; no obstacle-awareness or safety certificate attributed to Dubins.

New chapter follows the latest four-module architecture: asynchronous VLM task planner, geometry/empirical capability tools Q(z,g,u,skill), deterministic execution manager, frozen STTW/RB. VLM outputs task stages/approach/corridor/priority/fallback identifiers; Q selects speed/alpha with support and unknown handling; manager monitors entry and switches to validated forward bypass without awaiting VLM. Six single-stage and36 two-stage candidates and the four comparison groups follow the brief. Ground and jump capabilities remain separate. This is an adopted plan, not implemented functionality; retain STTW-first order, do not start cross-project changes or hardware from the attachment alone.

## User decision: retain current simulation frequency

User confirmed keeping the current setup. Retain5000Hz physics and200/50Hz control; close the pending1000Hz proposal without execution. Continue the already-running150-update H75 horizon continuation and declared stage-end candidate review. No additional acceleration experiments queued.

## Original-horizon H75 continuation registered

Configuration `learning/configs/go2_h75_offroad_horizon.json` restores original50Hz gamma .997 and GAE lambda .99, matching the unchanged terminal-reward discount. All other repaired Go2 continuation settings remain:4096×24×150, original320-prefix TRAIN task and signed pulses, shared Go2 DR, switches and replicated terrain, fixed1e-5 learning rate, unchanged network and selected Local300. Starts from originalu77 Actor/Critic/std with fresh optimizer, not rejected continuation candidates. Output `runs/go2_h75_offroad_horizon150`; budget14745600 policy transitions/1474560000 physics lane steps,150 iterations, estimated~4hours. No claim that discount mismatch is the sole cause of previous regression.

Stage-end review uses `learning/configs/go2_h75_horizon_review.json`, no earlier segments, at most reward peak/periodic/final candidates. Same flat/bump/switch DEV, cached original/Zero reused, at most2400000 additional physics lane steps. No automatic adoption or unlimited continuation. TensorBoard6006 and stage notification retained; active process identity is in the research ledger.

## Matched low-speed flat controls completed

`runs/h75_low_speed_flat/comparison/RESULTS.md` and `xy_matrix_full.png` compare all24 saved flat/ramp lanes at2.0/1.7m/s. Two new flat cases completed600000 physics lane steps/0 optimizer updates in279s, old ramp trajectories reused. At2.0m/s all flat lanes pass original hold; alpha0 flat path RMSE .0506m versus ramp .435m, indicating terrain-triggered subsequent oscillation in this matched case. At1.7m/s alpha1 already oscillates on flat (path RMSE .179m versus ramp .220m); final1s flat path/heading peaks .248m/.349rad are substantive. Thus low-speed alpha1 behavior is not solely a hill effect. Flat Zero's hold miss is marginal heading .051365rad versus .05, with path .01284m and speed .02935m/s; do not group it with alpha1 instability.

Retain original selected models and all negative outcomes. Stop this bounded terrain/speed screen: no inability to climb was demonstrated, and alpha1 is not universally better. Next return to the declared preference-switch recovery work, taking the already identified training temporal-discount mismatch into account before any further continuation. No additional training or physics queued here.

## Active matched low-speed flat comparison

Configuration `learning/configs/isaac_h75_low_speed_flat.json`: flat mesh at2.0/1.7m/s, original H75/Local300, same complete2.3m/s warm state as completed24cm ramp cases. Budget600000 physics lane steps/0 optimizer updates; output `runs/h75_low_speed_flat/evaluation`. Reuse all ramp trajectories and existing evaluator/plotting. Separates low-speed policy behavior from added terrain response; no new training or changed physical limits.

## Lower-speed ramp results: no climb failure, recovery is non-monotonic

`runs/h75_uphill_low_speed/evaluation/UPHILL.md` and `uphill_comparison.png`:24cm ramp at2.0/1.7m/s, all five alphas and matched Zero,12/12 survive10s and cross crest and goal. Completed600000 physics lane steps/0 optimizer updates in287s. No alpha0 climb failure occurred. At2.0m/s, alpha0 has substantial post-ramp lateral oscillation (path RMSE .435m vs alpha1 .185m), and fails final hold; other five lanes pass. At1.7m/s, alpha0/.25/.5 and Zero pass final hold, while alpha.75/1 retain lateral oscillations and fail it. Alpha1 path RMSE .220m versus alpha0 .087m. Thus reducing speed does not monotonically favor alpha1. Actual ascent minimum speeds at1.7m/s are1.298/1.294/1.297/1.300/1.290m/s: entry-speed ordering does not persist as a minimum-speed advantage. Full actual XY/reward panels inspected; no early physical failure hidden.

Both lower-speed conditions start from the same saved2.3m/s warm state and include the request-change transient. These requests lie below the exported original H75 TRAIN speed range2.409–2.799m/s; performance here cannot establish in-distribution robustness. Next use matched low-speed flat controls to distinguish pre-existing low-speed behavior from terrain-induced recovery, rather than assuming motor power/kinetic energy causality or blindly extending hill scans. Preserve all results and selected original H75/Local300; no new training started.

## User-directed lower-speed uphill screen

The user directed lowering standard speed. Replace the unexecuted32/40cm height extension with existing24cm terrain at2.0 and1.7m/s requests, all five alphas plus Zero,600000 physics lane steps/0 optimizer updates. Configuration `learning/configs/isaac_h75_uphill_low_speed.json`; outputs `runs/h75_uphill_low_speed/evaluation`. Original H75/Local300 and physical/actuator limits unchanged. Both conditions reuse the complete2.3m/s warm initial state, then execute the lower request; actual entry speed will be measured rather than claiming equilibrated low-speed initialization. Reuse the existing runner/plotting and ramp asset. This is nominal single-state DEV, all outcomes retained. No higher ramp was generated or executed.

## Short-uphill screen completed: traversal hypothesis not yet observed

`runs/h75_uphill/evaluation/UPHILL.md` and `uphill_comparison.png` summarize all three8/16/24cm ramps, five alphas plus Zero.18/18 lanes survive10s and cross the original goal/crest projection;900000 physics lane steps,0 optimizer updates,478s wall time. No alpha0 inability to climb was observed. At24cm alpha0→1 minimum ascent speeds are1.612/1.640/1.688/1.749/1.782m/s; Zero1.802m/s. All outcomes retained, including successful Zero. Actual wheel-height changes are stored in uphill_metrics.json.

At24cm alpha0 has visible post-ramp path/speed oscillations; path RMSE .1665m and speed RMSE .1464m/s versus alpha1 .1892m/.0900m/s. Its original final hold flag is false solely because last1s heading peaks .050030rad versus .05rad; path .02744m and speed .06085m/s remain within limits. This tiny threshold crossing is not treated as practical traversal failure. Other17 lanes pass original hold. Zero at24cm has lower path/speed RMSE .0992m/.0759m/s but higher peak roll .369rad versus alpha1 .281rad; no universal H75 superiority claim. No measured motor work was recorded.

Original H75/Local300 remain adopted. This completed screen supports speed/attitude tradeoffs but not the proposed pass/fail separation. Next assess whether a bounded steeper-ramp extension addresses the hypothesis before changing training; preserve limits and all outcomes. Preference-switch1→0 recovery remains unfinished, and subsequent task-connection work has not begun.

## Completed terrain/speed matrix and short-uphill follow-up

Original H75 u77 + selected orange Local300 completed 10 terrain/requested-speed conditions (50 alpha lanes and 10 matched Zero lanes). All50 H75 lanes survive10s, cross the original goal and satisfy original final hold. Speed RMSE decreases across all five alpha levels in every condition; path error is not strictly monotonic. Zero fails physically in6/10 conditions; the remaining4 do not satisfy final hold. Preserve actual early-failure windows. Evidence: `runs/h75_terrain_speed_sweep/evaluation/RESULTS.md`, `METRICS.md`, `xy_matrix_full.png`, `xy_matrix_bend.png`. Eight new conditions used2400000 physics lane steps,0 optimizer updates; two matched historical conditions were reused. Single initial-state nominal DEV only;2.6m/s starts from the same2.3m/s warm state.

User clarification: mild working-roll exceedance alone is acceptable when traversal and subsequent recovery remain satisfactory. The6cm/2.6m/s alpha1 peak .316699rad exceeds the historical .302rad working threshold, but completes goal/hold. Retain raw flags and thresholds; do not reject this model solely for that overshoot. This does not establish tolerance of physical failure or sustained instability.

Next registered experiment uses `learning/configs/isaac_h75_uphill.json`:8/16/24cm half-cosine ascents over s3.8–5.0m, persistent plateau, same2.3m/s request/initial state/actuator limits for five alphas and Zero. Budget900000 physics lane steps,0 optimizer updates; reuse native evaluation and reward plotting. Outputs `runs/h75_uphill/evaluation`. Tests the user's entry-speed/traversal hypothesis; no outcome is presumed. Existing switch1→0 recovery issue remains unresolved; the later connection chapter has not started.


## H75 offroad continuation and channel evidence — current stage

The selected lower remains the user's orange Isaac Lab original Local300, frozen at `runs/local300_torch_portable_parity/actor.npz`. The adopted middle remains original H75 u77; continuation candidates are not yet adopted. Small sim2sim errors are not a gate for project progress. Complete STTW offroad work before the subsequent task-connection chapter.

The Go2 runner/storage/PPO adapter and faithful original H75 task port are implemented. The registered `runs/go2_h75_offroad150` continuation uses GPU physics/tensors/network,4096 environments×24 steps and150 total updates, original128/128/64 alpha-modulated H75 network,320 original TRAIN prefixes/full routes, original command speeds/deadlines and signed force events, shared Go2 domain randomization, plus terrain and eligible within-episode alpha switches. Configuration: `learning/configs/go2_h75_offroad.json`. The continuation completed all150 updates; the last24 updates with native terrain/vehicle replication took2119 s (35.3 min), with final sampling87.49 s/update. Fixed-case review of updates70,51 and150 completed48 cases; none is adopted because full trajectories retain substantial lateral displacement. Original H75 u77 remains selected. The next declared terrain/speed matrix reuses two original cases and collects eight new six-lane cases; old external-terrain evidence is preserved in `runs/go2_h75_offroad150_external_groups`. First repaired update sampled in93.10 s versus234.36 s before (2.52× observed). Live completion is recorded in the run's `status.json`, not this document. Do not restart a live process because an observation times out. Training reward nominates at most three candidates for the already-registered fixed-case review after150 updates; inspect full trajectories before adoption. TensorBoard: http://127.0.0.1:6006 .

Original H75 with the selected lower has completed all ten entry/exit-projection channel conditions across five alphas:50 branches×10 s without physical failure, exactly2500000 allocated physics lane steps and0 optimizer updates. The collector has exited normally; no further mechanism physics run is queued. Same-state five-alpha forward queries and all saved physical branch trajectories, signed/cumulative rewards, errors and actual wheel contact phases are indexed at `runs/h75_common_state_channels/analysis/RESULTS.md`; this is nominal single-initial-state DEV, not independent generalization evidence. Full-tail reload effects and non-monotonic results are retained. No additional precision experiment is required before advancing the declared work.

The portable bank remains `runs/h75_original_training_portable`:65 ordinary/126 single/129 S prefixes, original2.409–2.799 m/s requests and full physical/controller/history/filter clocks. The earlier320-state forward parity check is complete and is not repeated. Detailed methods/results remain in `docs/METHODS_AND_RESULTS.md`; current decisions and experiment state are authoritative only in `/home/qy/STTW_CONTROL/research-hub/STTW_PROJECT_STATE.md`. Later sections below retain historical stage snapshots.


## Spatial alpha switch on the 4 cm bend bump

Original H75/selected Local300 frozen. Six10s nominal lanes compare0→1/1→0 with independent/common/steer-priority execution after s≥5.3m;0.0002s official mean wheel contact points/force matrices captured. All switches occur after rear raised-contact clearance and before turn end.0→1 meets original goal/hold/work checks in3/3;1→0 crosses goal but all3 fail final hold with persistent path oscillation. No optimizer updates.

Steer-priority has zero post-switch activation opportunities (speed dominates limiting); no candidate adopted as an improvement. Recorded prefixes differ by up to~8–9mm/rad component values, so small mode differences cannot establish causality. Frozen original H75 declares constant alpha per episode; history/state distribution shift under switching is a supported hypothesis, not a proven sole cause. Next same-state channel intervention and faithful original-task switch curriculum with shared Go2 DR. Full XY/reward/contact evidence: runs/h75_bend_switch_execution/analysis/RESULTS.md.


## H75 shares Go2 domain randomization

IsaacPathRollout now connects the same IsaacLab randomize_vehicle_properties/VehicleControlRandomization and push event as Go2LocalEnv. Training defaults to the shared Go2 DR settings and selected repaired contact options; nominal evaluations remain unchanged by default. One declared6-lane5s engineering run completed150000physics/0optimizer, actual randomized material/mass/COM/25–75physical-tick delays saved, six push events at4s and finite observations. ECBC/reward/termination use clean state; policy history uses augmented measurements. No new PPO training or DR robustness claim. Evidence: runs/h75_go2_dr_integration/RESULTS.md.


## Go2 randomization requirement and first 4 cm terrain DEV

All subsequent lower/middle/student training must reuse the Go2 randomization contract and applicable training processing, recorded in AGENTS.md. Current lower implementation includes friction/mass/inertia/COM, gains/offset, physical-time action/motor/IMU delays, observation noise and periodic pushes. The older middle terrain training path has not yet connected the shared randomization; it must be updated before any new formal training. No new training launched here.

Saved results: runs/h75_selected_local300_bend/RESULTS.md. Original H75 and selected orange Local300 unchanged; same initial state, topology-matched flat mesh vs 4 cm oblique bump, six lanes each, 600000 total allocated physics steps, zero optimizer. All five H75 alpha levels reach full10s; bump Zero fails at6.72s. H75 bump path RMSE alpha0→1 .052→.119m and speed RMSE .127→.063m/s; no independent generalization claim. Figures overlay five fixed alpha colors per model with Zero, original path, bend detail and true endpoints. Goal/hold terrain qualification remains separate from survival. Contact timing currently only support/net-force proxies; exact contact-stage mechanism remains unverified. Same-state alpha queries retained for flat and bump. Next contact-stage/switching work, then any necessary training with shared Go2 DR enabled.


## Original H75 on the user-selected orange Local300 — matched flat DEV

Completed `runs/h75_selected_local300_flat`: unchanged H75 u77 plus original Local300, alpha=0/.25/.5/.75/1 and Zero, same full warm initial state, fixed left90_R2 route and user speed2.3m/s, 10s, friction0.8 on both engines. Native MuJoCo188.94s and Lab106.95s; total600000 allocated physics lane steps, zero optimizer updates. All five H75 conditions in both engines cross the goal section and meet the original final1s speed/path/heading hold and working-roll checks. This is one route/initial state, not a full standard panel or offroad qualification. Zero survives both, but MuJoCo fails final hold and Lab exceeds working-roll0.302rad (peak0.3178rad).

Paired H75 XY RMSE0.0955–0.1163m; endpoint separation0.1913–0.2152m. Lab path RMSE alpha0→1:0.0592→0.1122m; speed RMSE0.1323→0.0478m/s. This preserves the expected preference direction in this case. Per-lower-step Lab rewards reconstructed against saved policy rewards, max difference5.96e-8. Full/cumulative reward, signed components, targets/filter/executed offsets and matched-alpha Zero rewards retained.

User requested grouping by model: `comparison/xy_mujoco_all_alpha.png` and `xy_isaac_lab_all_alpha.png`, each full XY plus equal-aspect bend detail with fixed five-alpha colors, Zero and original path. Raw NPZ retained; no extra physics for plotting. Report: `runs/h75_selected_local300_flat/comparison/RESULTS.md`. Historical MuJoCo/selected Lab comparison included front friction10→0.8, so old differences cannot all be attributed to engine; original evidence preserved with correction. Next keep original H75/selected lower and start the STTW terrain stage; no lower/H75 retraining presently justified by this case.

# Current adopted lower: orange Isaac Lab original Local300


## User-selected orange-curve Local300 lower

User explicitly selected the orange “Isaac Lab original Local300” trace in runs/local300_contact_repair/mujoco_lab_xy.png. Adopted actor sha517f41fe825282ac4d4260a9757802e2258c644a1b235d0c328cd12e7b45f125, original256/128, ECBC5/40/filter8 and repaired Lab contact options frozen together in learning/configs/isaaclab_selected_lower.json. Actor hash and controller/plant settings checked against the actual evaluation contract. Known return XY differences remain documented; adoption is for subsequent simulation work, not exact sim2sim equivalence.

Warm1000 training stopped on user direction at787updates; checkpoints/logs retained, queued review and stage notifier canceled before extra evaluation. No new training queued. Next use unchanged original H75 on standard plane for matched MuJoCo comparison, then terrain after evaluation.

Older active-run snapshots below are historical.

# Active: original Local300 weights adapting in native Go2 framework

`runs/go2_local300_warm1000`, PID737669/session4168,4096×24×1000weights-only adaptation. Original256/128 Actor weights copied exactly; fresh critic/optimizer/std. Original command difficulty/reward and Go2 DR retained. Estimated45min. TensorBoard http://127.0.0.1:6006 includes local300_warm. Post-review session92097 will select≤4candidates and evaluate original7cases; no automatic model adoption/H75/terrain.

Prior adapted500 has completed and all4candidates failed tracking0/7 despite survival7/7. Full XY reviewed; no new candidate adopted. Original transferred Local300 remains the frozen working reference.

# Active training and queued fixed-case review

Training649378/session57316 continues its500update budget. One-shot follow-up session64261 will evaluate at most4reward-screened snapshots plus final (4total), using original seven cases and saved baselines; output runs/go2_local_adapted500/final_review. No new training or H75/terrain stage is queued. Review requires actual full trajectories, not just artifact status.

# Active: residual-task adaptation500



## Residual-task adaptation run launched

`runs/go2_local_adapted500`, PID649378, tool session57316: fresh4096×24×500 GPU budget. Native Go2 network512/256/128ELU retained because no width-causality evidence exists. Task-specific initialization restores std.15 and a zero actor output head; PPO learning rate fixed1e-5. Physical-time discount mapping uses gamma=.99**(.005/.02), lambda=.95**(.005/.02). Original Local300 command families/difficulty/reward, all8initial states,10s and original control/physical timing remain. This is a repair bundle, not a single-factor ablation.

Source audit verified an additional wiring mismatch: native Go2 policy observations are delayed/noised while PD uses current dof state; old STTW augmentation also drove ECBC/ESO. New explicit controller_clean_feedback option supplies current plant measurements to ECBC and its steering-limit feedback while retaining prior policy noise/delay, command delay and physical randomization. This is a training-model assumption, not proof of deployable noiseless state feedback. Historical default remains false.

Startup actual: first4updates complete, ~2.96s/update (~25min estimate). TensorBoard first13 logged updates keep actual LR1e-5 and std.14997–.15054; no claim of convergence.6006 HTTP200 and adaptive/adapted run groups loaded, browser opened when desktop available. Existing stage notifier bound toPID649378 with delayed check. Primary best waits for completed episodes, selected at Episode/reward peaks; short-block best separate. Expected output<.3GiB versus33GiB free at launch, preserving30GiB budget. No H75/terrain run yet.

Earlier launch and stop records below are historical snapshots.

# Current: reward selection corrected; no new lower candidate adopted

Episode/reward mid-run peaks are valid completed-episode reward. Old model_best.pt was the initialization winner of short24-step blocks and is not a learned best. Four episode-reward-screened snapshots (completed276/201/301/latest451) now evaluated: survival7/2/7/5 of7, finaltracking0/7 each; full XY in runs/go2_local_original500_episode_candidates/RESULTS.md. No H75/terrain progression on these candidates. New primary selector now follows the TensorBoard completed-episode reward; short-block best diagnostic separate.

Original Actor/Critic256/128 hidden widths; new512/256/128 is3.13x Actor parameters, with no causal evidence of over-sizing. Native Go2 transfer requires accounting for action/exploration and50Hz versus200Hz timing. Native source/log comparison report: runs/go2_local_original500/TRAINING_DIFFERENCES.md. Fixed-learning-rate rerun remains prepared, not launched. Next restore task-appropriate near-zero residual initialization and fixed LR, preserve original command difficulty/reward, then run the declared bounded stage. Original Local300 remains frozen reference.

# Current: adaptive-LR run stopped; fixed-LR rerun prepared



## Adaptive learning rate failure — interrupted at463 completed updates

Actual TensorBoard LR rose from initial1e-5 to5.77e-4 after update1,6.57e-3 after update2 and1e-2 after update3. The native Go2 implementation applies factor1.5 per minibatch and caps at1e-2. Near update440, exploration std5.89, finished-episode failure approximately100%, duration around1.1s, value loss thousands. This is sustained degradation, not harmless early fluctuation. Copying the initial LR without controlling its actual adaptive range was insufficient. Shared Go2 implementation does not establish suitability for this two-wheel residual task.

Run stopped at463 completed updates; all original logs/checkpoints retained, status_before_stop.json saved. Latest periodic file model_450.pt represents451 updates; no500 completion claim. Next configuration prepared with learning_rate1e-5 and schedule fixed, all other training/task settings unchanged. No new run launched yet. Learning-rate escalation is observed; its sole causal responsibility is not established.

The following launch snapshots are historical.

# Active lower adaptation training

`runs/go2_local_original500` is running: GPU4096 environments,24steps/update,500updates, native Go2512/256/128ELU Actor/Critic and PPO. Original Local300 eight-state reset/command difficulty/reward restored; repaired contacts, Go2 randomization and physical-time delays. Fresh initialization, not Local300 weight continuation. Initial measured iterations2.9–3.1s (~25min estimate, not completion). Actual randomized friction0.308–0.999 and compliant stiffness entries−7833.38/−5324.41 read back across4096 environments. GPU reset startup succeeded. TensorBoard http://127.0.0.1:6006 HTTP200 and current scalar tags loaded.

Training PID543737, tool session30453. Training rewards belong to sampled pre-update models; `training/model_best.pt` saves the best sampled candidate. Fixed-case comparison will include reward-selected candidates and final, with full XY against original MuJoCo and matched ECBC; no reward-only adoption. H75/terrain remain downstream. Next ordinary training check ~20minutes after startup, unless exit/error is delivered.

# Current model-response and Local300 comparison

Original Local300 with repaired Lab contacts now completes seven standard cases and meets final speed/steer hold in 7/7, but return-case full XY difference RMSE remains 0.61–0.69 m and endpoint separation 1.50–1.71 m versus original MuJoCo. Full XY plus matching Lab ECBC: runs/local300_contact_repair/RESULTS.md. No new training or H75 run. Four ECBC candidates gave mixed effects; retain original wc5/wo40/filter8. Saved yaw analysis locates most return-heading mismatch by6s (6.2–7.4degrees); remaining migration mismatch is explicit, not hidden by final hold.

Future lower uses native Go2 Actor/Critic512/256/128 ELU and runner/PPO, original Local300 commands/difficulty/reward. Eight-state reset/conditional command export is now implemented and checked; compliant stiffness is preserved in the friction-randomization code. Live GPU reset/material readback remains pending before training. H75 unchanged first after lower qualification, then STTW terrain with five alpha levels.

# Latest priority correction: plant response first

Before selecting or tuning ECBC, compare original MuJoCo and Isaac Lab plant responses from the same physical initial state and identical input commands. Examine full XY and actuator/wheel/body/attitude time histories, not just final tracking hold or survival. The27-parameter ECBC screen has completed but is not adopted and further tuning is paused. Current short paired open-loop response diagnostic: runs/plant_response_pair, four1s cases,40000 total physics lane steps,0optimizer. Then fix identified differences, tune ECBC, restore original Local300 training/reward, test unchanged H75 first, and only after those pass proceed to terrain.

# Latest execution order — 2026-11-05

Finish the four saved candidate comparison, tune zero-residual ECBC+ESO until performance approaches the original baseline, then train the lower residual on original Local300 command families/difficulty/reward. Compare original training logs using the same per-step and complete-episode reward definitions, plus fixed-case tracking; no reward-only adoption. Preserve Go2 runner/PPO/domain randomization/friction/physical-time latency. First test the unchanged original H75 on the qualified lower controller; only finetune H75 if needed. After H75 passes, resume STTW terrain tasks with five alpha levels. No residual training while ECBC remains unresolved. This supersedes earlier immediate middle distillation and single-candidate selection statements below.

Four-candidate comparison is complete; update500 survives7/7 but final tracking0/7, hence not adopted. ECBC27-configuration screen is now running, 189 GPU lanes, original seven cases,10s,9,450,000 physics lane steps/0 optimizer. See runs/ecbc_parameter_screen and research ledger. Original Local300 update300 mean_step_reward=-0.001390266875, last20 updates mean=-0.001749556545; rollout phase matters. Historical positive reward ~20 is incomparable.

# Latest selection and monitoring instruction — 2026-11-05

Newest task-fidelity correction: preserve original MuJoCo command distribution and task difficulty (ordinary/turn-return/post-return-small/reversal, edge ranges and fast slews), reward and task details. Go2 remains the framework/PPO/randomization reference; friction and latency follow Go2, with physical-time conversion. Do not simplify the task to get a running migration. `go2_local500` is explicitly a simplified-distribution historical candidate; it cannot establish original-task transfer even if its training reward is high. Restore the original source specification before the next task-aligned training stage.

Original `sample_local_v2` functions have been reused to export32768 command schedules to `runs/local300_command_bank` with counts8219/11328/8323/4898 for the four families. This is a finite training bank sampled from the original specification, not a newly invented sampler or fixed evaluation set. Export required no physics/optimizer and no GPU work. Task integration, original reward restoration and full Go2 randomization alignment are still pending.

Select the saved candidate using native Go2 training reward first, then evaluate only that candidate on the fixed seven cases (plus matching zero ECBC baseline). Fixed cases do not rank or select checkpoints. `select_go2_local.py --training-run runs/go2_local500` writes `best_training.json` from TensorBoard `Train/mean_reward`, using the first subsequent rollout log for each saved post-update checkpoint. This native statistic averages the latest100 completed episodes and spans previous policies: report it as a heuristic training-reward selection among saved candidates, not exact frozen-policy/all-update best. The final checkpoint without subsequent score is unscored, not a default winner. Earlier fixed-case-ranking plans below are superseded.

User requests no frequent polling. `runs/go2_local500/next_check.json` records estimated next check around00:31:48 PST; avoid checking training before that time unless new input or an error requires it. The stage notifier was restarted alone with an initial delay until that estimate, then60-second checks if still running; training PID140402 remains untouched.

# Current direction — native Go2 framework for STTW (2026-11-05)

Current implementation: `train_go2_local.py` calls the upstream `OnPolicyRunner.learn` with `Go2LocalEnv`; only the status logger subclasses the runner. `go2_local_training.json` declares fresh 512/256/128 networks, 4096×24×500 = 49,152,000 lower transitions (1,228,800,000 physics lane steps). This is a new lower candidate, not Local300 weight continuation. The original Local300 file supplies the preserved observation/history helper; its inference is not called during this training.

New task uses Go2-style random command resampling, individual reset, positive exponential speed/steer tracking with posture/action costs, failure penalty 5, and upstream timeout bootstrapping at 10 s. It is a continuing tracking objective; there is no remaining-time critic input or old finite-horizon failure-tail penalty. Reward values are not comparable to old Local300 returns. Nominal frozen seven-case comparison remains the selection evidence after training. Configuration records vehicle-scaled randomization and the unchanged actuator limits: friction .3–1, body density ×.9–1.1, base COM ±1 cm, servo gains ×.9–1.1, steer sensor zero offset ±.01 rad, 5–15 ms delay, noise, and four-second velocity pushes. This is the declared new training distribution, not a claim that Go2 uses these exact vehicle-scaled values.

Native GPU training is now running in `runs/go2_local500` (PID140402 at ledger revision3144); at least23/500 updates observed, approximately2.37s sampling and0.10s optimization per4096×24 batch. TensorBoard: http://127.0.0.1:6019. Three startup errors were corrected and retained in separate failure directories. No sim2sim qualification or controller adoption is claimed. Latest user instruction lowers STTW system free-space reserve to30GiB; pip download-cache reclamation restored51.71GiB before launch. Next: monitor this run, compare saved candidates on the nominal seven commands, then decide lower adoption before middle distillation.

Latest user instruction supersedes the earlier network-size restriction: use the Go2 framework as the training implementation, including its network architecture if useful, then distill the original middle policy. First complete the lower task and MuJoCo–Isaac Lab comparison; only then middle distillation and offroad work. Previous dated entries below are historical, not live training status.

Source: `/home/qy/projects/My_unitree_go2_gym`, registered `go2_spring_jump`, `Go2_Spring_Jump_Cfg_Yu` / `Go2_Spring_Jump_PPO_Yu`. Direct reuse targets: `OnPolicyRunner`, `ActorCritic`, `PPO`, `RolloutStorage`, GAE, optimizer, adaptive KL scheduling and checkpoint logging. `go2_training_config.spring_jump_recipe` reads the actual source configuration without importing incompatible Isaac Gym bindings into Isaac Lab. Native actor/critic widths are 512/256/128 ELU; 4096 environments, 24 steps/update, five epochs, four minibatches, lr 1e-5 adaptive, KL .01, gamma .99, lambda .95, entropy .01. Lower stage budget remains 500 updates; Go2's 50000-update default is not the authorized budget.

Required environment rewrite: expose the upstream VecEnv interface, reset individual terminated environments immediately, distinguish timeout bootstrapping from physical termination, clear every controller/history/delay state at reset, and resample commands and training randomization. Replace the old inactive-padding and hand-written sampling loop for the successor task. Keep Isaac Lab as the simulator backend and implement vehicle-specific model, observation/action mapping, ECBC/ESO, actuator constraints, reward and termination there. The existing H75 continuation entry remains historical and is not the successor runner.

Go2 training randomization includes friction .3–1, added base mass ±1 kg, link mass ×.9–1.1, COM ±.03 m, PD gains ×.9–1.1, motor-angle offsets ±.035 rad, sensor noise, command/motor/IMU latency 1–3 physics steps, and pushes every 4 s (xy .4 m/s, angular .6 rad/s). These are source values, not yet adopted STTW ranges. Go2 updates delay buffers at physics dt=.005 s: 5–15 ms corresponds to 25–75 STTW physics steps at .0002 s. Randomization uses Isaac Lab's existing event/delay utilities where applicable; actuator gain randomization must act on STTW's external velocity servos, not unused implicit PD gains. Absolute mass/COM/push magnitudes need explicit vehicle-scale configuration. Nominal evaluation preserves the original physical vehicle and disables training randomization. No quadruped joint permutation or jump-specific reward is applicable.

Middle transfer: preserve frozen H75 as teacher, collect teacher outputs on student-visited states (including matching teacher history/filter state), distill both correction channels conditioned on alpha=0/.25/.5/.75/1, and then finetune the student in the same Go2 framework. New network weights are a new student identity, not an exact Local300/H75 resume. Record supervised and PPO stages separately; compare original requested references, not student-modified targets.

Implemented so far: source recipe reader, configurable ECBC baseline, selected-lane plant reset (including exported initial state), and separated Local300 history observation construction for a new upstream actor. Randomization adapters now call Lab's existing material/mass/COM events and DelayBuffer; external velocity-servo gains, steering sensor zero offset and measurement noise are supported without changing nominal torque limits. Property events run once at startup because their PhysX setters use CPU tensors; control buffers remain on the GPU. Randomization is not yet wired into a running training task. Pending: native VecEnv task with full controller/history/delay reset, explicit randomization configuration and pushes, successor training launch, and distillation. No new training result is claimed. Original Local300 standard comparison and weights remain under `runs/local300_sim2sim` and their original source paths. Authoritative decisions and execution state: research-hub/STTW_PROJECT_STATE.md.

# Latest task specification — bend-local terrain (2026-11-05)

The e2ea098b task attachment retains three STTW studies: nonflat preference primitives, contact-phase channel interventions, and externally scheduled alpha switching. STTW remains first; later connection/RB studies wait. Five alpha levels remain authoritative.

`learning/configs/isaac_bend_bump.json` and the existing `place_isaac_terrain.py` now generate a compact oblique 4 cm bump on the original fixed H75 curve. Nominal support ends at route s=4.8 m, switch crossing is s=5.3 m, and turn ends at s=6.3274 m. These are spatial scene definitions, not observed wheel-contact events. Artifact: `runs/isaac_bend_bump`; fixed-preference trajectories, contact phase extraction and same-state switching interventions remain pending. Current GPU150 uses its frozen original flat/slope/rough-slope assets without modification.

# STTW offroad — current implementation direction (2026-11-04)

User priority: finish STTW offroad before continuous-task/RB work. Stop prolonged mesh parameter scans; use standard Isaac terrain generators and supplement slope/rough-terrain training. Full alpha coverage is 0/.25/.5/.75/1. Current candidate uses 0.1 m horizontal terrain sampling, contact/rest offsets .01/0 m, physics 5000 Hz, lower 200 Hz and middle 50 Hz. H75 Torch inference/history/control are integrated; LOCAL300+ECBC/ESO stays frozen for the first middle finetuning stage. GPU PhysX/tensor rollout and PPO are implemented; 4096 environments ×24 steps ×150 iteration training is active. Actor architecture remains unchanged. Current evidence and limitations: docs/VALIDATION.md; authoritative live decisions: /home/qy/STTW_CONTROL/research-hub/STTW_PROJECT_STATE.md.

---

# Path Coverage V2 — 2026-10-11

Reset-distribution-only fork from alpha0 best100 / alpha1 best150. [Execution contract and verification](docs/path_coverage_v2.md). Default +100 each; conditional +50 blocks capped +200. Old results preserved; no auto push. Live facts remain in the shared research hub.

---

# Phase C training wiring — 2026-10-10

[Implementation, proposed budget and validation limits](docs/path_feedback_phase_C.md). New351/352 geometric sampler/PPO lifecycle/checkpoint/best/report wiring is prepared. No new vehicle physics or alpha training; Stage B failures remain unchanged. Shared research hub is authoritative.

---

# R196 lower interface repair — 2026-10-09

Authorized bounded diagnosis and local candidate preparation. Live V5.2 remains untouched. Plan: immutable declaration/interface audit; one full-state B0 fast_turn12s diagnostic replay (optional alpha0 once); control-chain evidence; opt-in210D local candidate with screened commands, component-bounded finite reward, late physical best and V5.2 KL handling; necessary CPU checks; report and shared ledger. No formal lower training or deployment. Current facts are in research-hub/PROJECT_STATE.md. Candidate implementation committed63c9d68; six new CPU checks plus five directly related legacy checks pass. Formal local training remains not_run. Offline original-trace ECBC/ESO/return reconstruction and three control-chain plots completed; the sole B0 diagnostic replay completed12s with no failure:36/60000 steer-force substeps touched limits (22 in first mismatch,0 in second). Native path jumps and tiny original/replay differences recorded; no alpha0 replay. Candidate horizon corrected to10s in6e47341.

---

# V5.2 launch resumed — 2026-10-09

User explicitly requested 开启训练, revoking the earlier pause. Lower-tracking limitation remains a known result. One engineering8×16 check completed two total PPO batches (one per endpoint), finite and no stop. Formal run planned at runs/preference_v52_tracking_best_start_20261009: parent100 Actor/log_std, fresh Critic/Adam, four value-only rollouts each, initial200 additional PPO each, original evidence-gated100 extensions capped400. No20/25 scene validation or fixed wall cutoff. TensorBoard6016 separates engineering/formal. Historical paused run and audit are preserved. Current live facts are in the shared ledger.

---

# V5.2 training paused by user — 2026-10-09

LOWER_TRACKING_LIMITED: saved fast_turn B0 and alpha0 show persistent steering tracking errors after governed quiet intervals. User explicitly chose “按附件暂停训练”. New physics0, new PPO0; engineering check not launched. V5.2 code prepared and7 focused CPU checks passed; GPU integration is unverified. Parent best/last distinction, physical qualification, zero-shift decomposition and all intervals: docs/V52_VALIDATION.md. No formal launch or automatic extension while paused.

---

# V5.2 implementation — 2026-10-09

Approved design: attachment STTW_V52_Codex.md and endpoint JSONs. Base26c3fed merged with performance7ab2cd8 before algorithm edits (405cda1). No physical/observation/action/lower changes. Read-only parent18-trace NPZ audit first; then three reward changes and5080 independent cap/failure bound; Actor+std only hot-start, Critic/Adam reset; four value-only batches; safety-first fixed-protocol best and bounded convergence extension. Numeric validation100/150/200 then every50, plots only final reloaded best. Parent counts and new counts separate.

Implementation plan: (1) finish saved-data audit and severe tracking intervals; (2) reward/config with independent reference tests; (3) selection metrics, atomic identity and convergence; (4) integrate existing optimized campaign, complete-episode summaries and CLI; (5) one8x16x2 engineering check then authorized200/max400 training with checked TensorBoard. No repeated lower interface or scale tests. Shared research ledger is authoritative current state.

---

# RTX 4090D performance implementation (2026-10-09)

Authorized specification: user request, baseline 071fc6f; implementation base 77f715f retains evaluation-budget repair. Isolated branch performance/r196-v51-4090d. No formal training, no changes to physics/reward/control/PPO contract, no DVGC/JIT writes or process termination.

Plan (sequential A/B commits): (1) reusable five-mode benchmark and current baseline; (2) resolved dimensions; (3) device statistics and exact reset events; (4) batch reset guard; (5) PPO synchronization; (6) batched evaluation plus numerical audit; (7) profile-gated geometry/cache/scale decisions. JAX rollout only if remaining framework handoff cost warrants it. Each benchmark uses two warmups and three restored-state measurements; compile/preparation separate. Existing tests and complete-state comparisons gate adoption. Raw artifacts stay under ignored runs/. Current research status remains in the shared ledger.

Progress: completed evaluation A/B: median752.835059s to239.373127s (3.15x), all18 reward audits pass. V5.1 batch evaluation is now the default with explicit sequential fallback. Small trajectory differences remain (maximum compared XY0.0211668m); strict numerical equivalence is not claimed. Dimensions, PPO and shared cache enabled; summary/reset remain opt-in. User requested fewer tests: optional remaining scale queue cancelled, raw evidence preserved, no formal training. Report and all six XY links: docs/PERFORMANCE.md. Existing100/100 experiment and DVGC/JIT untouched. No push per worktree override.

---

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

## R196 Preference V5.1 strong-primary experiment (2026-10-09)

User-authorized successor to V5, based on published implementation `6a264f7`. The
frozen R196 lower_alpha1 controller, ECBC/ESO, governed-centered lower interface,
physics, authority, upper345/critic346 network and action mapping remain fixed. V5.1
adds the endpoint-selected actual primary-excess cost (weight40, cap400), replaces
`yaw_damping` with reward-only signed `yaw_recovery` (cap20), and trains from the
first batch on one16s distribution containing20% virtual initial-heading-error resets.
Physical reset state is unchanged and the lower initializes from actual pose; evaluation
uses zero initial heading offset. Failure bound is the2080 sum of component caps and
there is no total-cost clip.

The attachment math and focused production checks pass; old V5 reward/task/PPO checks
remain passing. A separate8env x16step x2update engineering run completed with fresh
Actor/Critic/Adam/std and finite optimizer metrics. Formal endpoints are to be newly
initialized, seed87,512x128, max100 updates, checks at25 and100, with no automatic
extension. Control benefit remains unverified until the declared physical evaluations.

Formal result: both independent endpoints completed100/100 updates. The first update100
evaluation stopped after `straight_hold` because evaluation ticks were incorrectly charged
against the exact training-transition ceiling. Evaluation was separated from that ceiling
and the five missing cases were run from the saved policies; no PPO batch or checkpoint
write occurred. On `fast_turn` [2,4)s, alpha0 steer RMSE=.01862rad passes .05 and alpha1
speed RMSE=.04997m/s passes .08. Both fail the common roll requirement: peak
.31540/.36761rad and2.160/2.445s above.302rad. Final joint speed/steer/heading hold
passes alpha0 3/6 and alpha1 1/6. Fast-turn heading fails at10s and16s, with16s RMSE
.07749/.24961rad. Straight hold retains unhelpful speed offsets .0753/.0442m/s and
worsens B0 speed error. Reward reconstruction passes, but the result is unqualified and
not adopted; no250/500 continuation.

Remote-review evidence bundle: `docs/evidence/r196_preference_v51_20261009/README.md`.
It includes XY-first plots for every fixed case, paired update25/update100 metrics and
5ms trajectories, frozen configurations, failure/recovery receipts, and TensorBoard
image plus numeric scalars. The local full run remains under `runs/`.

## 2026-10-10 Lower first no-alpha v2 — implementation plan

User-authorized attachment specifies fresh210/211 ELU256→128 local two-residual lower, unchanged ECBC/ESO/plant/limits. Isolated from4e956f0; upper2eeef42 best0 added200/best1 added250 frozen. Five fixes in lower_command/training/validation: consume configured four-family/range/slew sampler, independent small direction at7.5 (≤7.6)s, fine.01rad small acceptance, common immutable issued stream, complete rollout-boundary resume. No deployment screening or old310 weight surgery.

Seven focused new CPU checks pass; prior11 checks and B0 replay reused. Sole engineering8×16×2 completed. Formal fresh512×256×200 started; corrected accepted-count metadata after six updates via full-boundary exact resume (no sample repetition). Status and PID in runs/noalpha_v2_formal_20261010. Fixed numeric validation100/150/200; evidence-based100-block extension≤400. Local pass then one old mismatch full-state comparison; both gates then frozen upper straight/fast_turn combination. Final physical evidence must precede adoption; no push.

Source paths: learning/configs/lower_local_tracking_no_alpha_v2.json; lower_command.py; lower_command_training.py; lower_command_validation.py. Resume: --formal-only --resume-checkpoint <run>/training/checkpoints/resume_boundary.pt --target-updates <total≤400>. Old boundaries/future-artifact overwrite rejected before writing.

## 2026-10-10 Path Feedback V1 工程交付
隔离自ab195e7，新增几何模式与351/352接口，A复用旧四条，B两条共6000步已完成，C未训练。当前事实及用户“快速验证主线、不追求基线完美”决定见共享台账STTW-PATH-FEEDBACK-V1；工程证据[INDEX](docs/evidence/path_feedback_v1_20261010/INDEX.md)。不替换旧teleop、best或物理，不自动push。

## 2026-11-05 Isaac坡地补训工程阶段
官方slope/rough_slope已接入平地入口与连续出口；H75完成2次有界PPO更新，权重真实变化，五档alpha共10回合。证据`runs/isaac_h75_terrain_training`，验证见docs/VALIDATION.md；最终模型未评估，下一步同场景新旧策略与冻结底层配对。当前唯一事实仍以research-hub/STTW_PROJECT_STATE.md为准。

32轮扩大续训已启动：承接两轮工程阶段，三地形×五alpha每轮混合，原奖励不变。输出runs/isaac_h75_terrain_continue32，TensorBoard6018；不因两轮阴性结果否定训练方向，阶段末再固定配对评估。

## 新版规划主线（2026-11-05）
以用户新附件491cddf6-4b37-40f8-99ff-2e375d2c32c8/pasted-text-1.txt为当前规划来源。STTW依次完成：真实非平坦地形上的双通道偏好、接触阶段对齐与共同状态通道干预、弯内给定alpha切换及转向优先执行。五档alpha保持；独立执行为主基线，双向共同gamma为机制对照，转向优先候选保留原转向修正进度，仅必要时放慢速度修正。原低通顺序、非对称修正量限速与最终参考边界不变，不预断候选优于Zero或独立执行。旧剩余回归距离输入/奖励不再作为本阶段必先实现项。
当前4096环境GPU物理+GPU张量PPO150轮已运行，网络规模不变；下一在新版弯内短横坡/斜向凸起场景保留接触阶段、目标/滤波/实际修正和限速激活，再固定偏好/共同状态查询/短干预/切换配对。后续章节C6/S36/S36-R及RB交叉执行仍等待STTW工作完成。


## GPU bend fixed-alpha collection (2026-11-05)
`evaluate_isaac_gpu.py` reuses the GPU training plant and controller for the original H75 five fixed alphas plus one Zero lane (frozen LOCAL300+ECBC/ESO with no H75 correction). Six-second DEV budget: 180000 physical lane-steps. Terrain/config: `isaac_bend_bump`; output `runs/isaac_bend_fixed_completed`. Collection records 50 Hz actual rewards/failures and 200 Hz all-lane poses, measurements, targets, offsets, filters, rate-limit flags, reward components, rigid states and net contact forces. Contact observations are lower-step samples, not exact 5000 Hz onset times. A first attempt failed after 150 physical lane-steps on a nested reward-dict logging error; retained in `runs/isaac_bend_fixed`, repaired in place. No model/physics changes. `plot_isaac_gpu_evaluation.py` produces actual/reference XY and per-step/cumulative reward and speed-error overviews from saved data; full mechanism/contact interventions and matched alpha-rescored baseline reward diagnostics remain pending. Training process continues its original frozen source independently.


## Bend-bump original H75 DEV result (2026-11-05)
Six-lane GPU evaluation completed, 180000 physical lane-steps. `runs/isaac_bend_fixed_completed/RESULTS.md` contains all-lane XY and reward/speed overview with original source NPZ. Alpha .25/.75 reach 6 s; alpha 0/.5/1 and Zero terminate at 3.820/.915/4.855/3.695 s. The .5 failure is before terrain. Neither full-window survivor reaches original goal s=10.3274 m; survival is not task success. Per-lane metrics use unequal observed windows. Same-state five-alpha query on .25's 300 decision states: 833/1500 rate-limited, no steering-dominant progress case; steering-priority increment is identical to independent execution for this corpus. Early-failed .5's 46-state query has two steering-dominant cases before terrain, not terrain mechanism evidence. No physical intervention or candidate closed-loop benefit yet. Full baseline rescoring/contact-phase diagnostics remain pending.


## GPU wheel-contact investigation and comparisons (2026-11-05)
Original H75 previously ran all 10 slope/rough-slope CPU PhysX cases for 5 s. The GPU change produced an immediate contact shock, not a training/terrain-coverage failure: first 5 ms front wheel rate 8.39 rad/s vs CPU 22.9464. Original first observation/reference matched, old CPU-interface entry with GPU physics reproduced it, and a 1 m airborne diagnostic recovered the CPU wheel rates. This localizes the discrepancy to contact handling; the internal cooking defect is not proven.

Rejected attempts: single VHACD hull worsened the response; equivalent mesh-unit rescaling did not fix it. User-requested 220→200→180→160→140→120→100→80→60 vertex descent first matched the initial interval at 60; 64/63 failed and 62 initially matched but failed all six full episodes within .08–.16 s. Short-step matching alone is insufficient. All candidate meshes and results are retained under `runs/isaac_hull_descent` and `runs/isaac_hull_refine`.

Complete same-original-H75 bend comparisons: original 226 vertices H75 2/5 survive 6 s, Zero fails; regular 38 vertices H75 5/5 survive, Zero fails; regular 50 vertices H75 5/5 and Zero survive. These are one-seed DEV window survival counts, not goal completion or transfer proof. The collision approximation changes while semiaxes/mass/inertia/frequency remain fixed. Actual/reference XY, source-OBJ images and common-window metrics: `runs/isaac_wheel_comparison`; rejected 62 has its own `rejected62` panel. These figures show source geometry, not the engine's internally cooked geometry.

NVIDIA PhysX documentation states GPU-compatible convexes constrain both vertex and convex polygon counts. Estimated distinct source supporting planes: 226 vertices→256 planes; 38→48; regular 50→64; irregular 62→110. This supports a representation hypothesis but does not prove an SDK defect or a universal empirically observed threshold. References: https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/_build/physx/latest/class_px_convex_mesh_desc.html and https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/GPURigidBodies.html .

GPU150 completed 150 attempts, 141 accepted updates on the original 226 representation. Preserve this stage separately; do not attribute collision corrections to learned improvement or adopt its checkpoint as physically matched evidence. No automatic further training was launched. Installed IsaacSim5.1/IsaacLab contact replay of original mesh is now being attempted. First import failed before physics due to decimal-point mesh basename used as invalid USD prim name; staging identical mesh bytes under valid names repairs the importer issue, not yet the physical discrepancy.


## Installed Isaac Lab original-mesh interval (2026-11-05)
`runs/isaaclab_original_interval_matched` completed 25 physical steps with the original 226-vertex wheel: front23.003996rad/s versus old CPU22.946377/old GPU8.392765 at5ms, root forward2.299083m/s and roll-1.037e-5rad. No old GPU start-up impulse in this interval. Readback confirms all six materials .8/.8/0 and collision contact/rest .01/0, matching masses. The importer needed legal USD mesh filenames (identical source bytes) and explicit deinstancing before applying collision settings. Preliminary unmatched run remains separate. Do not call a25-step replay full closed-loop H75 validation or infer exact imported inertia/cooked-shape equivalence. Next: integrate existing H75/lower/controller scheduling into the installed Lab plant and record full original-mesh trajectories. Figure/report entry: `runs/isaac_wheel_comparison/RESULTS.md`.


## GPU execution requirement and Isaac Lab batching (2026-11-05)
User requires GPU physics, GPU state/control/policy/PPO tensors and 4096 training environments, retaining the H75 network and 5000/200/50 Hz timing. Existing `isaac_h75_gpu150.json` already specifies 4096 environments, 24 rollout steps and 150 iterations; that historical Gym stage is complete. Six-lane evaluations are five-alpha plus Zero comparisons, not the training batch size.

The Lab plant now reuses installed Isaac Sim GridCloner/Cloner physics replication, creates the robot once and shares each terrain mesh through USD references. The 4096-environment initialization exposed PhysX allocation/capacity errors and is invalid despite a finite tensor receipt. The training entry now imports Go2 RSL-RL PPO and RolloutStorage directly from `/home/qy/projects/My_unitree_go2_gym/rsl_rl` (checkout 5c47fa3), with an H75 network interface and inactive-lane minibatch mask. Upstream GAE, losses, Adam steps and per-minibatch adaptive KL schedule are reused; the former custom hard-KL rollback and std clamp are removed explicitly. STTW timing, architecture, rewards and synchronous resets remain. Formal 4096×24×150 Lab execution is `runs/isaaclab_h75_go2ppo150_running`, TensorBoard http://localhost:6019/. Earlier startup attempts and engine-error updates are retained and not adopted. The installed PhysX requested found/lost and total aggregate capacities 2**24 and found/lost aggregate capacity at least 100663296 (configured 2**27); per-environment collision groups pair vehicles with their own terrain. Runtime completion remains pending.


## Standard flat-ground H75 comparison: migration discrepancy (2026-11-05)
User corrected the scope to the original standard plane only. Reused CPU `runs/isaac_h75_curve_probe` and collected Lab `runs/isaaclab_cpu_parent_comparison/plane`, same original H75 and original wheel source. Old CPU five alphas all survive 5 s; Lab alpha0/.25/.5/1 fail at approximately4.375/3.465/3.385/4.400s, alpha.75 survives5s. Zero CPU2.185s, Lab2.335s. This contradicts the prior broad claim that migration reproduction was established by bend-bump survival. No task-success claim.

Actual/reference XY: `runs/isaaclab_cpu_parent_comparison/plane/cpu_lab_xy.png`; speed/roll/steering: `cpu_lab_states.png`; common-window differences and source plotting script retained beside raw trajectories. The alpha.5 speed is already2.089 versus CPU2.306m/s at0.1s, before the bend, so diagnose plant/import/control differences first. Current4096 training PID28577 remains SIGSTOP with live state preserved pending this discrepancy; do not silently restart it or adopt its weights. Earlier slope collection completed but is outside this narrowed delivery; rough-slope expansion cancelled.


## Local300 first: MuJoCo to Isaac Lab (latest user direction)
Terminate the H75 continuation and finish lower-controller transfer before re-enabling H75. PID28577 is terminated; the first two failed formal training directories were deleted at user request (about6.7MiB). The previous Gym CPU comparisons remain diagnostics only, not the requested reference.

Original MuJoCo Local300 references and issued command streams were located in the lower-first-noalpha worktree, `noalpha_v2_formal_20261010/training/final_best_validation_0300` and `training/reference_streams`. The seven existing standard commands are reused; no new terrain matrix. The source starts from prepared bank index3, including controller/actuator state, so this needs mapping before a matched comparison. `learning/configs/isaaclab_local300_transfer.json` records exact locations and the lower-only scope.

Import inspection rejects the unverified inertia-toggle explanation: installed URDF importer defaults import_inertia_tensor=True; saved USD contains the expected link masses/inertia entries, joint damping/friction/armature0. This inspection does not establish complete simulator equivalence. If adaptation training is needed after concrete interface/physics fixes, user authorizes it; planned bounded stage is500 iterations,4096 environments×24 lower steps, original256/128 network,5000Hz physics/200Hz lower. No new lower training has started.


## Local300 direct MuJoCo–Lab standard comparison (2026-11-05)
Completed350000 physical lane-steps,0optimizer,92.57s. MuJoCo original Local300 survives all7×10s; Lab original Local300 survives only straight_speed_change. Lab failures: steady_positive5.340s, steady_negative3.760s, return_positive3.615s, return_negative4.885s, post_return_small_positive5.945s, post_return_small_negative3.640s. Straight speed difference RMSE.596m/s over10s; survival is not matching tracking. XY and speed/steer/roll figures plus raw comparison are in `runs/local300_sim2sim/lab_parent`.

The original bank index3 has physical_tick700 (ESO already enabled), controller and previous command/reference transferred. MuJoCo wheel centers are z=.09592/.09599m with nominal radius.1m. This is an engine contact-equilibrium difference, not proof of a controller error: Lab first5ms rear/front wheel rates34.83/11.49rad/s. MuJoCo contact warm starts/compliance are not transferred by joint/root state mapping. The prior inertia-import hypothesis was rejected. H75 is not loaded; original H75 continuation remains terminated. Further lower adaptation uses the authorized500-iteration stage only after the concrete initial-contact issue is addressed; no new training yet.
