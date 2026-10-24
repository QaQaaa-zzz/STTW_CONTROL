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

生成候选YAML的第一次辅助命令误用了缺yaml的系统Python，随后使用既有项目venv完成，未安装依赖或改变原配置；记录保留在run目录candidate_generation_attempt1.log，0新物理成本。
