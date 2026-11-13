# STTW → RB → STTW：两米平地地图下的完整链测试

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
