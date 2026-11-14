# STTW → RB → STTW：当前交接诊断

## 锁定条件续训完成：仍5/9，未改善完整交接

![累计300模型全部前后弯道XY与真实终点](turn_lock300_xy.png)

|前段|后段|锁定条件续训300|
|---|---|---|
|直线|直线|通过|
|直线|左弯|超时|
|直线|右弯|通过|
|左弯回正|直线|通过|
|左弯回正|左弯|通过|
|左弯回正|右弯|越廊|
|右弯回正|直线|通过|
|右弯回正|左弯|超时|
|右弯回正|右弯|越廊|

训练追加150更新到累计300，7760.615秒；严格完成事件7/159117，失败159109、超时1。这是随机训练终止事件，不是独立成功率。共同300checkpoint九路线仍5/9；相对归位150新增路线2/3、丢失1/8，净改善为零，未采用为更优模型。Bark训练完成回执已sent/HTTP200。

![本阶段实际训练曲线](turn_lock_training.png)

交回记录显示路线3/8转向角速度约19.911/16.156rad/s；路线3最终通过、路线8失败，所以它是待处理动态风险，不能声称唯一失败原因。路线5约8.830rad/s且交回0.5秒后航向增0.375rad，最终越廊。原奖励惩罚姿态/横向位置，但未直接惩罚落地后的横向速度、偏航角速度和转向角速度。

下一有限阶段从300的Actor/critic/normalizer初始化、fresh Adam，新增双轮远岸接地期间每20ms惩罚 `0.02*(2*vy²+0.5*yaw_rate²+0.05*steering_rate²)`，分别日志记录三项批均值。原权重为零时保留旧奖励；不加硬门控、不改执行器权限、不改九路线和最终成功门槛。六入口、Go2随机化、4096×24、严格锁定条件不变；150轮后停止，无自动延长。原奖励300保留对照；新奖励与fresh optimizer共同变化，不能称纯单因素实验。训练历史重置的部署差异仍未解决，必须完整链复测，不能从奖励推断交接成功。

原始结果与合同：`rb_adaptation/approach_departure_turns/turn_lock300_evaluation`、`landing_dynamics_protocol.json`；新run `rb_adaptation/landing_dynamics150`。

## 前弯—跳跃—后弯实测：两模型均5/9，交接尚未完成

![新归位150模型九路线实际与参考XY](approach_departure_homing150_xy.png)

![父模型同九路线实际与参考XY](approach_departure_parent75_xy.png)

|前段|后段|归位150|父75|
|---|---|---|---|
|直线|直线|通过|通过|
|直线|左弯|通过|超时|
|直线|右弯|超时|通过|
|左弯回正|直线|越廊|通过|
|左弯回正|左弯|通过|越廊|
|左弯回正|右弯|超时|越廊|
|右弯回正|直线|通过|通过|
|右弯回正|左弯|越廊|超时|
|右弯回正|右弯|通过|通过|

同一候选覆盖九条完整链，新150耗时72.500秒，父75耗时70.278秒。新150九条均起跳、触远岸并锁后交回，完整目标仅5/9；两条超时、两条越廊，不能把回交等同任务成功。新150相对父75新增2条、丢失2条，未采用为更优模型。所有轨迹与失败在 `rb_adaptation/approach_departure_turns/{evaluation,parent_evaluation}`，DEV，不是独立成功率。

前弯实测RB入口有约−0.074/+0.067rad侧倾，旧直线约0.006rad；部分交回横向速度约−0.414m/s。归位150训练7684.585秒完成150更新，success事件1684、failure事件148410、done150091，事件标志有3次重叠，不能相加作为互斥类别或独立成功率。训练阶段Bark旧回执为URLError，未成功送达。

训练成功用关节姿态L2误差<0.15rad、速度L2<0.5rad/s，而实际锁定用每关节<0.01rad、<0.1rad/s。新有限续训用六个真实入口（两来源×三前段，去掉重复后段前缀），收紧L2到0.01rad/0.1rad/s，保证训练关节条件覆盖实际锁定条件；原Go2随机化、其他奖励、网络及4096×24不变，从150恢复权重/Adam，追加150到300。此为修改任务分布和完成条件的DEV阶段，不是单因素对照；不改变全链原最终验收。仅qpos/qvel初始化、训练历史清零，与部署继承历史仍有差异，必须复测全链。

## 当前顺序与新增覆盖（用户要求）

等待预计超过30分钟或每次恢复后累计推进达到3小时即暂停，等待用户继续；不再启动后续阶段，已授权后台任务保留原预算。

先完成 STTW→RB→STTW 交接，再按 `/home/qy/下载/Chapter3_Execution_Outline_CN.docx` 推进第三章规划；研究任务不替换，方法可调整。现有三路线跳跃前均为直线，不能作为前弯交接证据。

已准备九种组合：前段直线／左弯回正／右弯回正 × 后段直线／左弯／右弯。前弯位于x=2.5–8.3m，横向偏移峰值0.55m；沟前8.3–10.3m、沟后10.6–12.6m仍为原平地（平地不限制落地后的转弯）。地图、沟、最终目标及保持标准不变，连续运行、不重置状态。首轮每个共同checkpoint九个名义DEV回合、各最多10秒；真正初差验收另报，不与名义重复混算。

配置及路线编号：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/rb_adaptation/approach_departure_turns/{config,protocol}.json`。已核对路线前进性、曲率有限、前弯回正与地形配置不变；**九路线物理测试已完成，结果见上表；整体交接仍未通过**。复用现有GPU批量评测入口，交付全部实际/参考XY、失败终点以及进入RB和交回STTW的状态。

**初态标签更正：本轮24车道测试的初差未生效。** 新增initial_*参数被已有start_*初始化覆盖；直接读取保存首帧确认全部y/roll/yaw为零。下列14/24、16/24等只能解释为名义批量重复，不能称扰动验收或独立成功率；原轨迹与失败保留，各run有INITIAL_STATE_CORRECTION.json。已删除重复初始化，准备复用原start_*参数，真正的24初差组合尚未执行。

![统一三模型、主动归位和锁后交STTW的名义XY](physical_homing_xy.png)

三条单名义路线：固定奖励修正阶段75模型3/3通过；父225和修正150各2/3。流程为双轮实际接触远岸且车身就绪后，RB保持车轮控制，原有限伺服主动归位，严格位置/速度条件满足后锁摆再交STTW。无物理状态重置、力矩扩权或最终判据放宽。

![车体速度反馈下24条名义重复，非扰动验收](body_speed_handoff_xy.png)

24名义重复：原轮速反馈14/24；车体真速反馈16/24（直/左/右8/4/4）；严格动态门控1/24；1.5m/s续行12/24；状态就绪再启H75为7/24。批量接触数值变化明显，不能据这些小差异宣称严格因果改善，当前没有稳健采用。23条已锁定轨迹有真实双轮远岸接触、髋膝最大漂移0.0001668rad。

已定位原流程存在任一轮触岸就可交回、未锁摆就交STTW、归位前要求已归位等问题。旧左弯交回后轮中心x10.309m仍在远岸10.6m前；落地真实车速与轮速估计也明显不同。车体速度反馈依赖仿真真值，实车需可靠估计。

**当前限定训练**：从修正75权重/fresh Adam出发，150更新训练真实部署的主动归位阶段。RB持有车轮、原权限内髋膝重力补偿归位、请求关节目标回写观测历史；RK4跳跃/implicitfast归位匹配评测，原网络、Go2随机化、奖励、严格训练成功门槛保持。8环境256步集成实际归位7次、数值有限、0优化；该150轮已完成，九路线结果见上表；旧启动记录保留在run，后续仍须真正初差验收。

训练/合同/全部原始结果：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/rb_adaptation`。真实初差配置true_perturbation_config.json目前只准备、未执行。用户要求续接稳定后按文件继续，文件已明确为上述执行纲要；先完成交接，再推进纲要任务。

## 累计300轮复测与奖励冲突

![225与300全部路线及真实失败终点](continuation300_xy.png)

第二阶段150轮耗时4762.217秒，训练终止成功0/176881。固定225与300各三路线完整链复测共6回合102.408秒：225直线/右弯通过、左弯失败（2/3），300三条均物理失败（0/3）。225两条保持均0.505秒。300右弯驶至x24.483m后失败，图完整保留，不能裁到20m隐藏终点。无统一候选三路线通过，尚未采用。

![阶段训练曲线，不平滑](continuation300_training.png)

奖励代数诊断：固定其他项、匀速前进时，旧进度与速度分项每秒为2v+exp[-4(v-1.8)^2]（v≤5m/s）；1.8m/s为4.6，3.4m/s约6.8。该项明确鼓励超速；旧150模型落地前确有约3.4m/s。它是奖励方向冲突的证据，不证明所有失败由此造成。

下一有限阶段从完整链表现较好的225权重初始化，fresh Adam，进度奖励封顶1.8m/s，另加每秒-2(v-1.8)^2；网络、物理、Go2随机化、地形、原最终门槛及其他奖励不变。配置冻结到新run，150阶段轮数，4096×24，不与旧奖励数值直接排名。预计75–90分钟，超过30分钟暂停目标，后台固定预算与Bark保留。

初次导出缺少PYTHONPATH导致0物理启动失败；补正后实际结果位于[全部逐步结果](/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/rb_adaptation/full_chain300/evaluation_retry/RESULTS.md)。导出器另纠正周期模型的奖励归属：model225附带的是模型224采样分数，不能称225得分；模型权重身份与完整链实际调用未改变。

## 150轮续训与完整链复测

![原Run3、75轮、150轮全部实际/参考XY](adaptation_xy.png)

同两米平地地图、名义摩擦0.8、无高度观测偏置，原模型/75/150各三路线，共9回合145.147秒，零额外优化。原Run3完整目标0/3，75轮0/3，150轮1/3（右弯）。75轮左弯持续10秒，但8.48秒才锁摆，目标区航向误差短暂超过0.2rad，未连续保持0.5秒，仍为失败。150轮右弯5.19秒锁摆、最终保持0.505秒；直线物理失败、左弯越廊，尚未采用。不同路线不能分别挑不同checkpoint拼成统一策略通过。

150轮训练耗时4804.021秒；训练终止事件255245，成功0、物理失败255238、超时7。这是重复训练回合，不是独立测试成功率。采样best仍update0，其初始短批与后续稳态回合不同；最终能力以完整链测试为准。

![全部150轮训练诊断，无平滑](adaptation_training.png)

原始结果和逐步数据：[完整链报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/rb_adaptation/full_chain_recheck/evaluation/RESULTS.md)；[训练日志报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/rb_adaptation/training150_batched/RESULTS.md)。本次仅DEV名义筛选，未通过扰动验收。

后续限定从model150恢复Actor/Critic/normalizer及Adam，追加150轮到300；环境重新初始化，不称逐位精确恢复。地图、奖励、随机化、网络、4096环境/24步保持不变。预计75–90分钟，按用户要求暂停助手目标，后台预算与阶段Bark保留。TensorBoard：http://localhost:41961 。

## 历史标量候选比较

![全部三路线、六候选实际与参考XY](xy.png)

沟前x=8.3–10.3m、沟后x=10.6–12.6m为全宽平地，真实碰撞三角形高度检查通过。旧地图碎石分别侵入0.2m和0.6m，旧结果不能作为本协议验收。沟宽0.3m、两岸同高，远处最高脊坡4cm保留。

固定原高成功率RB Run3 transition_128000（actor ffe98cc0c497）；原平台任务246/250不代表当前串联任务成功率。GPU MJX-Warp物理、Torch H75/ECBC/ESO及JAX RB，DLPack共享，循环内不搬运物理状态或动作到CPU。

本批次18回合、每条最多10秒、零训练，实际耗时108.098秒。**完整目标4/18，均为直线；左右弯均未通过，任务未完成。** 六候选为RB高度观测坐标偏置0.08/0.16m × 接地后转向引导幅度0/0.08/0.15rad；高度偏置只作用于网络观测，不改变物理地形或成功判据。所有失败保留，属于名义DEV筛选，不是独立成功率估计。

|候选|直线60|左弯61|右弯62|
|---|---|---|---|
|0|goal|physical_failure|corridor_or_domain_exit|
|1|goal|physical_failure|physical_failure|
|2|corridor_or_domain_exit|physical_failure|corridor_or_domain_exit|
|3|goal|corridor_or_domain_exit|physical_failure|
|4|goal|corridor_or_domain_exit|physical_failure|
|5|physical_failure|physical_failure|physical_failure|

![速度、姿态与关节](stages_tracking.png)

RB落地恢复期间解锁，满足车身状态后STTW接管车轮并用原有限伺服收摆，回到原构型且速度收敛才锁髋膝；不重置物理状态。轮地显式接触沿用源RB XML的condim6、friction5/.5/.0001，属于声明的新接触域，不复用原0.8地面验收。末端仍要求原位置/路径/航向/速度容差和0.5秒保持。

下一批`stable_gate`要求车身状态连续达标50/100/150ms后才交车轮，超限清零；实际回交时刻可变，不是触地后固定等待。已完成2/18，均为右弯，仍未采用。随后同地图同接触域x11独立锁摆起步的地面诊断：H75直/左/右3/3，零中层1/3（6回合34.876秒）；该诊断不算完整链成功。observer_settle为2/18，joint_settle为2/18，均未达到统一候选三路线通过。home_posture严格归位与缩摆仍为2/18（右弯），没有统一候选通过三路线；停止继续门槛/标量扫描，转向原Run3在当前同高沟任务的适配。随后已接通Playground续训，现有真实优化更新；不代表完整链通过。

完整配置、源快照、逐条轨迹和报告：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/flat_buffers`。历史尝试索引：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/iteration_summary.json`。

## 接触记录更正

安装的MJX-Warp `_get_data_into_warp` 明确跳过contact复制。此前evaluation及run3_native_rb_retry调用mjx.get_data读取CPU接触，故其“无远岸接触、无回交”字段无效，也不能作为正确回交闭环的能力结论。原始状态保留。现在直接使用GPU contact__geom/dist/pos/worldid与nacon有效掩码判断，排除轮自接触；已完成的compact_corrected阶段35处独立NumPy归约一致，后续入口已删除这个重复校验选项。

回放视频（真实终点保留）：[直线通过](/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/flat_buffers/rb_run3_route60_h75_v0.mp4)、[左弯失败](/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/flat_buffers/rb_run3_route61_h75_v0.mp4)。视频为保存状态回放，不是额外物理评测。
