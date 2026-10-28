# H75＋local300离线部署契约核对

2026-10-24。核查范围为当前worktree的MuJoCo ROS包与训练侧实际模型；实际硬件传感器/电机程序位置尚待确认。没有启动ROS节点、实机、电机命令、仿真积分或训练。

## 目的与本轮新增证据

此前已经检查过ROS消息和参考入口，本轮不重复把它算新工作。本轮固定H75与local300实际身份，在左/右弯八＋八个完整真实**仿真**前缀上，重建下一控制时刻的351维中层观测与210维底层观测，验证可移植确定性推理和C++命令合成。另六个有限数值合成案例覆盖位置端点和速度饱和。

|核对|范围|最大绝对差|
|---|---|---:|
|H75原JAX与NumPy参考推理|16条仿真前缀，原α调制保留|2.68e−7 latent|
|local300原JAX与NumPy参考推理|相同16条下一tick真实构造输入|7.08e−8 normalized residual|
|C++ composeCommand与Python实际命令|16真实仿真输入＋6合成限幅输入，匹配模型配置|2.57e−6 rad/s|

另外实际运行既有controller/command四项CPU测试：4 passed/2.24s，包含900步合成变速/ESO序列的C++/JAX核对、reset与命令组合。它们验证软件数值合同，未验证硬件传感器、整套ROS闭环或目标计算机实时性。

[完整收据](../../../../runs/deployment_contract_audit_v1/receipt.json)、[确定性权重NPZ](../../../../runs/deployment_contract_audit_v1/weights.npz)、[实际16条观测及输出](../../../../runs/deployment_contract_audit_v1/real_prefix_inputs.npz)、[解析后的完整契约](../../../../runs/deployment_contract_audit_v1/CONTRACT.json)。权重和原始快照SHA均在收据，真实硬件数据条目数为0。

另外从实际输出的weights.npz重新读取参数，未使用原Actor pickle，重建两网络并复核保存的16条观测/输出：[导出物回读核对](../../../../runs/deployment_contract_audit_v1/export_readback_receipt.json)。这验证交付字节本身可用，不只验证内存中的原权重。

## 三个具体的部署差距

**残差尺度。** 当前ROS YAML为前/后1与5，冻结local300实际为1.5与10；其余200Hz、转角.8、角速度3/60、strength1在本轮核对一致。强制残差fresh的离线合成中，16条真实仿真输入都出现默认尺度与训练尺度的不同，最大约前.08023、后.40425rad/s。这个量是指令差，不是实车速度或抗扰能力差异。ROS launch默认residual_enabled=false，当前也没有因此启动网络。

已生成[完整离线候选YAML](../../../../runs/deployment_contract_audit_v1/controller_local300_candidate.yaml)，解析语义仅改两项残差尺度。原仓库YAML、launch和C++文件保持原样。候选文件没有提供推理节点或传感器估计器，不能作为完整部署方案。

**历史与时序。** H75=16×20原字段＋16mask＋15context＝351；local300=10×20＋10mask＝210。两层历史都在每5ms实际tick更新，中层只每20ms推理一次。16槽名义缓冲长度80ms，最老与最新采样相距75ms；10槽对应50/45ms。继承network.history_dt_s=.02是旧元数据，不能据此把H75历史改成50Hz的320ms缓冲。参考滤波/限速/anti-windup仍每5ms运行，不能只在中层50Hz时更新。

**输入与控制链。** 当前RobotState仅有轮轴速度、侧倾/转向及局部陀螺等字段。实际H75和local300都使用车体前向速度，H75还需世界XY/航向与路径投影、曲率和预瞄点。轮速代理在输入中另有自己的字段，不能把轮速悄悄代替前向速度。当前MuJoCo发布的yaw_vel=gyro_local[2]，训练中用的是展开世界航向的差分率，需要实际姿态/定位映射。

ESO字段也须明确：C++ getEstimatedDisturbance()/getEstimatedEquilibrium()是过滤后的ESO估计及−dhat/a4；getEsoStates()[0]被代码改写为另一条roll_acc模型残差（filtered_disturbance2），二者不能混用。原ROS日志第一ESO列不是可直接替代训练ESO_equilibrium_shift的值。

## 对应到实际软件的差距表

|部分|已核对软件事实|尚缺的实际平台证据|
|---|---|---|
|模型|H75 α调制351→2无最终tanh；local300 ELU210→2有tanh；固定归一化|目标推理进程/计算机与真实运行时延|
|中层输出|有界速度/转向参考修正，经原滤波/变化率限制|实际路径/定位输入及参考接入点|
|底层输出|归一化转向角速度/后轮轴速残差，ECBC后组合|actual local300推理入口；不能把中层latent送ResidualCmd|
|ESO/上一命令|来自控制器内部状态与上次实际施加命令|真实桥接程序如何同步发出这些状态|
|时钟和历史|5ms历史，20ms中层推理，mask与reset语义在原代码|传感器时间戳、定位融合、控制调度与实际掉帧行为|
|坐标|ROS roll在ECBC调用处一次取负，轮速符号在仿真发布/接收链转换|实车IMU/编码器/驱动器实际符号，不能据仿真注释重复翻转|
|物理结果|当前观察来自MuJoCo与明确fallback合同|真实机器人质量/惯量、执行器动态、触地和侧推冲量记录|

本轮没有借未知硬件编写ROS适配节点。后续先拿到现有传感器/电机代码，按上述明确差距连接既有模型，再对保存的真实硬件日志做同样的输入/输出核对。上机与扰动实验应另有真实合同，不能把这个离线包写作sim2real完成。

## 论文路线的含义

已测到α与方向性物理能力的联系；物理特征和固定速度干预没有超越已有策略。当前需要补系统实现与可靠对照，而非再为“新颖性”加网络。这个离线结果补的是可复现实现证据，不补模型创新或硬件性能结论。原H75/local300的candidate/qualification状态保留；没有将它们改标为实机已采用。

两个辅助命令曾误用缺yaml/jax的系统Python，随后用既有项目venv完成，未安装依赖或改变原配置；attempt日志保留。通知初登记使用了相对路径，已只修正本组为绝对manifest/status，保留registration_repair.json，未重启服务或训练。0新物理成本。


## 2026-10-28：已采用协调器的保存状态回放与边界preview

本轮已完成显式`--mode coordinator-replay`，输入为已采用guard_fallback=True的`A/h75_guard_fallback_extension72_v1`，输出为`A/deployment_coordinator_replay_v1`。A为`/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025`。这是零新物理积分、零优化更新的软件审计，未重跑旧16+6审计，未改ROS、模型或原始轨迹。

完整[CPU收据](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/deployment_coordinator_replay_v1/receipt.json)、[五边界误差明细](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/deployment_coordinator_replay_v1/boundary_error_diagnostics.json)、[边界输入/preview NPZ](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/deployment_coordinator_replay_v1/boundary_previews.npz)、[四窗口收据](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/deployment_coordinator_replay_v1/upper_windows/receipt.json)和[失败尝试保留索引](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/deployment_coordinator_replay_v1/attempt_history.json)已保存。最终session18844 exit0；旧49378及数值诊断/序列化失败尝试分别保留，未覆盖源运行。

### 五个完整状态边界

从真实完整carry提取case0@200、case5@1200、case68@1400、case3@800/1000。carry含H75/local300历史、ESO/controller/actuator、filter/offsets、projection/follower及prev_alpha/latched；case3@800先按原记录绑定新path再preview，@1000使用已执行的新path而不重复bind。禁止调用physics._step。

|case/tick|351重建最大差|H75三候选latent最大差|q概率最大差|实际参考最大差(m/s / rad)|
|---|---:|---:|---:|---:|
|0/200|5.96e-08|2.79e-05|1.58e-05|4.77e-07 / 1.64e-07|
|5/1200|5.96e-08|0.00014|2.19e-05|0 / 1.27e-06|
|68/1400|5.96e-08|4.07e-05|5.96e-07|0 / 4.1e-07|
|3/800|1.19e-07|6.11e-05|1.14e-05|0 / 5.96e-07|
|3/1000|5.96e-08|3.69e-05|2.86e-06|0 / 0|

五边界alpha、task mask、permission、prev/latch、degraded/fallback选择，以及q≥.8/.85资格均精确一致。归因隔离显示：观测重建本身对latent的影响≤1.20e−7；用完全相同的保存观测重算，与源记录仍有≤1.405e−4的latent差。当前NumPy/JAX同输入误差仅≤1.79e-07。因此主要差异在当前CPU标量/批量算子与源执行记录的数值路径，而非本次观测重建；未进一步声称已定位某一XLA算子。local300 NumPy/JAX最大差6.33e-08，五条C++/Python命令组合preview最大差9.69e-07 rad/s。

本次工程检查按字段使用绝对界：351归一化输入1e−6、latent2e−4、q5e−5、proxy1e−4；参考速度1e−5 m/s、转角2e−6 rad。latent界经原tanh/缩放最多对应0.2 mm/s与0.04 mrad的未滤波参考差；离散与q门资格不因连续量接近而放宽。这些是事后记录的软件数值兼容界，不是新科学成功标准。

### 四个连续参考协调窗口

窗口为case0[848,860)、case0[188,200)、case1[208,220)、case3[800,1296)，共532个5ms packet、133次20ms决策。以保存351观测、nominal、permission、yaw及path progress作为外部输入，仅起点初始化prev/latch/filter状态，之后递推；四窗均在hold block内部JSON保存/恢复，与不间断回放完全一致。第四窗覆盖一次path绑定、permission撤销及恢复。

|窗口|packet/决策|候选latent最大差|q概率最大差|governed速度差(m/s)|governed转角差(rad)|
|---|---:|---:|---:|---:|---:|
|0: 848–860|12/3|5.12e-05|1.82e-05|0|2.35e-06|
|1: 188–200|12/3|3.36e-05|2.5e-05|4.77e-07|9.31e-07|
|2: 208–220|12/3|9.95e-06|3.32e-06|4.77e-07|2.01e-07|
|3: 800–1296|496/124|0.000137|4.79e-05|3.81e-06|8.23e-06|

四窗所有离散状态和q≥.8/.85资格精确一致；最大proxy差1.055e−4、filtered速度差2.040e−5 m/s、filtered转角差8.233e−6 rad，实际governed最大速度差3.815e−6 m/s、转角差8.233e−6 rad。窗口使用单独记录的整数量级工程界：latent2e−4、q1e−4、proxy2e−4，filter/offset/governed分别按[1e−4 m/s,2e−5 rad]、target按[2e−4 m/s,4e−5 rad]；不贴着每个观测最大值定界。源阈值、权限、控制输出公式、物理和实验结局标准均未改变。

### 覆盖边界与可部署接口含义

这次补足了当前q/协调记忆/permission/path bind/参考滤波链的软件输入输出合同；没有补出日志中不存在的逐tick lower210观测、lower动作、ESO完整更新或issued motor command。五个边界的lower/final command是由完整状态计算的next-control preview：lower actor可做NumPy/JAX核对，最终组合可做C++/Python核对；**没有原issued命令日志对照，也不是连续lower/plant回放或完整ECBC独立实现证明**。四窗则是保存输入驱动的参考协调回放，不是从真实传感器构造351输入的闭环回放。

目前用户没有实机程序；本轮不依赖新增平台路径即可完成。真实定位/速度/姿态估计、时钟/丢帧策略、参考接入与命令发布仍需未来具体平台合同，CPU数值通过不表示目标硬件实时性、实机部署或安全认证。8项相关行为测试通过（含单位分量界、严格离散/q跨界拒绝和中途恢复）；未重跑无关测试或新增实验面板。
