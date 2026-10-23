# 容扰度任务环境、训练分布与评测协议

适用已完成的 `tolerance_shaping_train_v1`（baseline/static/dynamic）与 `pulse_curriculum_train_v1`（uniform/guided）。本说明依据执行代码、运行manifest及保存配置核对，不改变既有实验。结果入口：[阶段报告](TOLERANCE_PROGRESS_20261022.md)。

## 1. 机器人实际要做什么

这是一辆带转向和驱动执行器的两轮车，在水平地面跟踪预先生成的世界XY几何路径，同时跟踪外部给定速度。任务包括直行、入弯、转弯、出弯并继续沿出口直线行驶。允许暂时沿时间进度落后；原始路径不随实际车辆位置重新锚定，网络修改后的参考不取代原始任务参考。

任务成功不仅是车没有摔倒：还要求通过弯后目标截面、维持工作姿态并在最后1秒恢复路径、航向和速度。到达目标后继续运行到回合结束，不提前宣布成功。

参考路径由Pure Pursuit产生名义速度/转角，中层网络输出速度和转角参考修正，冻结local300残差底层与ECBC/ESO闭环执行。α0偏路径，α1偏速度；本轮是两个独立中层网络，无动作混合。仿真使用全状态辅助速度/航向测量，不把本实验当作已完成实车状态估计。

## 2. 物理仿真、底层与初始化

| 项目 | 实际设置 |
|---|---|
| 仿真 | MuJoCo3.6.0，MJX的JAX实现，GPU矢量化 |
| 模型 | 仓库 `mujoco_ros_ws20260313/model/scalebike_scene_matlab.xml`，原始资产和接触参数按模型身份固定 |
| 物理步长 | 0.0002s，5000Hz |
| 底层周期 | 25个物理子步合为0.005s，200Hz |
| 中层/路径跟随周期 | 0.02s，50Hz，每次动作执行4个底层tick |
| 任务回合 | 最长20s＝1000中层步＝4000底层tick；失败可提前终止 |
| 控制器 | 动态侧倾参考ECBC＋ESO；`fixed_roll_reference=null`；冻结local300 |
| local300 | 无α输入，210维、10帧历史；Actor SHA `d408a94e886fa7f6309aa2c7635eb62e273c197405293ac9c2f1859014f25bcc` |
| 控制器模型常数 | mass=5.4kg、cg_height=0.2m、cg_forward=0.164m、wheelbase=0.408m、trail=0.024m、caster=25°、gravity=9.8；这是控制器参数，整车物理质量/惯量以XML为准 |
| 执行器 | 无额外延迟，转向角限幅0.8rad；最终转向角速度±3rad/s、后轮轴速度±60rad/s；冻结底层残差权限±1.5/±10rad/s |

源XML SHA `2a0c1376cfd9e6edf9b156b834e8c803bedaa336d2fbb7edffd8405daec2acc6`；便携化XML SHA `9b2002749a24ed47ccc9f88a9eacd044e460cb24ebc9d677ea9900029c40ea8e`。便携化处理包括去除未用torus资产和禁用启动位置servo；没有为本轮训练另调质量、摩擦或solver。

每回合恢复同一已有prepared bank的index3完整物理/控制器快照；不是每次从零速度直立起步，也不承接上一回合末态。随后初始化本回合local300状态、中层因果历史与掩码、路径投影、修正滤波和计数器。每个reset只重新生成路径相对实际初始位姿的锚点和请求速度；没有额外质量/摩擦/传感器噪声随机化。完整bank路径/哈希与控制参数见[身份快照](evidence/tolerance_progress_20261022/environment/PHYSICAL_AND_RESET_IDENTITY.json)。

## 3. 中层网络可看什么、能改什么

Actor351维、Critic352维；16帧×20个历史特征、16个历史掩码、15个当前上下文；Critic额外含剩余时间。上下文包括α、横向/航向误差、当前及前方曲率、车体系预瞄点、外部速度请求、前次修正与滤波状态、预瞄距离。历史按固定物理尺度归一化并clip到±5，非有限数值作为故障处理。MLP128→128→64、ELU；本轮沿用端点网络，不是单网络V3的α调制结构。

动作是两维latent，经tanh映射成Δv/Δδ：减速幅度最多1m/s，加速最多0.25m/s，转角修正最多0.2rad；修正滤波时间常数0.1/0.08s，并限制变化率。合成速度参考限制在1.5–3m/s、转角参考±0.35rad。奖励和评测仍对原始请求/原始路径计算，不能通过改小请求制造成功。

Pure Pursuit以实际弦长分母计算，预瞄时间0.7s、距离0.8–1.8m；名义转角限幅±0.3rad，名义速度slew0.5m/s²、转角slew0.3rad/s。实际控制链/观测顺序见[执行环境](../learning/src/sttw_control/path_command_env.py)。

## 4. 训练环境：随机路径，不是只练两条固定弯

实际使用 `demand_exposure_v2` 的512个环境槽，路径只在reset采样，回合内固定。路径静态网格5001点、间距0.02m（总长100m）；初始直线3m，曲率以0.5m的平滑段过渡，目标在最后弯结束后4m。

| 环境槽 | 数量 | 路径/速度分布 |
|---|---:|---|
| 0–101 | 102 | 旧随机分布：30%直道/缓弯（各半，缓弯30°、R4），50%左右60–100°单弯，20%正反双弯S；速度均匀1.8–2.6m/s，非普通组半径均匀1.8–4m |
| 102–305 | 204 | 高需求左右配对单弯：角度90–100°、半径2–2.2m、速度2.4–2.6m/s |
| 306–511 | 206 | 高需求左右配对S弯：同上幅度范围，两段相反弯之间3m直线 |

左右配对共享相同episode编号时的随机参数；失败后计数独立，不能保证始终逐时刻配对。每组20%reset带路径相对初态偏移：横向±0.1m、航向±0.05rad；改变路径锚点，不注入车辆qpos/qvel。固定route_seed=910201。以上槽数是分配，不是实际轨迹占比，早终止会改变样本权重。[采样实现](../learning/src/sttw_control/path_command_scenarios.py)。

### 两轮训练的区别

| 项目 | 第一轮奖励塑形 | 第二轮真实外扰训练 |
|---|---|---|
| 方法 | baseline/static/dynamic，各α0/1 | uniform/guided，各α0/1 |
| 起点 | 对应Coverage u200，同端点三组父权重一致 | 第一轮baseline u50，同端点两组父权重一致 |
| 初始化 | 复制Actor/Critic/log_std，fresh Adam；不是原优化器精确续训 | 相同 |
| 训练真实外扰 | 无本次外力脉冲 | 每回合50%无扰，50%一次真实侧向力 |
| 奖励 | 原十项任务成本；static/dynamic另加容扰度成本 | 两组相同的原十项成本，公式只指导采样 |
| 轮数/交互 | 6组×50×512×128＝19,660,800策略步 | 4组×25×512×128＝6,553,600策略步 |
| 学习随机种子 | 910301+α | 910401+α；pulse_seed910402 |

本轮PPO每次采128个中层步，不等于128步完整回合；环境跨更新继续到回合结束。Actor/Critic学习率1e-4/3e-4，gamma=.997、GAEλ=.99、最多4epochs×4minibatches、clip=.2、熵系数.001，软KL可能提前结束某次优化。更新数和实际epoch数需分开计。

## 5. 第二轮真实外力如何施加

reset时决定是否施扰；起扰时刻在0.8–3.5s、间隔0.02s的离散网格上均匀采样。脉冲持续0.1s，候选有符号力为−8/−6/−4/−2/+2/+4/+6/+8N。

起扰瞬间读取实际航向ψ，力的世界坐标向量为F·[−sinψ,cosψ,0]，正负表示车辆该时刻的两侧。方向在整个脉冲内固定，不随之后航向旋转；同号力在不同策略轨迹上不保证同一世界方向。力等效施加在车辆subtree COM，通过车架力/力矩表达，并逐物理子步更新力臂；fractional overlap计冲量。不是转向命令扰动、不是关节力矩、不是改状态。

uniform均匀选择8个有符号力；guided使用50%均匀＋50%贴近方向余量的exp软权重。有效余量仅为采样裁到0–8N，非法值回退均匀并计数，不把余量当安全门。两组交互量一致，实际力分布不要求一致，完整暴露统计见[最终结果](evidence/tolerance_progress_20261022/pulse_curriculum_results/training_exposure_summary.json)。

## 6. 评测环境：固定面板与完整状态分支

所有模型使用同一prepared snapshot、物理/控制器/动作接口；固定路径是左右90°弯，曲率半径2m、起始3m直线、0.5m平滑曲率过渡、弯后70m参考直线，目标截面仍为弯后4m。路径名称`left90_R2/right90_R2`在本轮确实对应2m；单网络V3阶段三保留相同名字但实际半径1.8m，不能凭名字混用。

| 维度 | 固定取值 |
|---|---|
| 外部请求速度 | 2.45、2.65m/s（常值请求，名义参考经过slew） |
| α端点 | 0、1 |
| 起扰时间 | 0.8、1.5、2.5、3.5s |
| 有符号力 | 0、±2、±4、±6、±8N |
| 脉冲/回合 | 0.1s / 20s |
| 第一轮比较模型 | parent、baseline u50、static u50、dynamic u50：4×2×2×2×4×9＝1152例 |
| 第二轮比较模型 | parent baseline u50、uniform u25、guided u25：3×2×2×2×4×9＝864例 |

每例使用确定性Actor推理，不采训练高斯探索噪声。对每个策略/α/路径/速度/起扰时间，先用该策略真实运行无扰前缀，在起扰点复制同一完整闭环状态到各力幅值分支；物理状态、ESO、历史、滤波、时间均包含在内。不同策略分别运行自身前缀，并非在同一状态比较动作。记录prefix_valid，起扰前已失败不删除。

固定面板没有复用训练路径序列，但已经被多轮开发使用，不是独立最终TEST。2.65m/s稍高于训练速度上限2.6m/s，应视为小幅速度外推。零外力行按四个起扰时间重复保存，是同一条件重复分组，不应当作独立随机试验。只有一个训练种子，没有独立多种子结论。

## 7. 提前终止、成功与实用诊断

- 物理终止：非有限物理/控制量、车架等不允许部位接地、物理子步峰值|roll|>0.7rad。非有限策略/底层量还会作为工程故障停止任务，而不是正常负样本。
- 跟踪域终止：预瞄点落到车后、路径航向误差达到π/2等Pure Pursuit定义域退出；terminal_contract_v2记录为失败并终止。
- 工作侧倾0.302rad是性能标准，越过会导致原始成功标签失败，但不等于物理摔倒，也不会单凭该阈值立刻停止物理积分。
- 原始成功同时要求：无物理/域失败，全程物理子步峰值侧倾≤0.302rad，完整20s，通过原始目标截面，最后1s全部样本的|速度误差|≤0.1m/s、|横向误差|≤0.1m、|航向误差|≤0.05rad。
- 用户允许小误差后的实用诊断：保持上述完整/目标/物理/域/速度/航向要求，将路径放宽为0.15m、侧倾为0.33rad；原始标签不覆盖。单网络评审曾使用0.20m路径敏感性，不能混进本轮0.15m口径。

速度误差以实际车体前向速度减原始请求计算；轮速×0.1m只是控制器速度代理。路径误差来自固定原始几何路径投影，不是网络修正参考。评测还保存完整轨迹RMSE、峰值侧倾、冲量、目标/保持标记、物理/域失败。提前终止的RMSE按实际样本长度算，需同时读失败数。

抗扰增量分析以各模型自身零力轨迹为参考，取起扰后2秒同一时刻差值RMS；只在所有比较模型都有完整窗口且前缀有效的共同上下文上统计。它包含进度差异，不是同状态策略因果比较；窗口后失败仍留在全任务结果。

## 8. 原始证据与复现入口

- [冻结物理与reset身份](evidence/tolerance_progress_20261022/environment/PHYSICAL_AND_RESET_IDENTITY.json)
- [真实外扰训练α0配置](evidence/tolerance_progress_20261022/environment/pulse_alpha0_config.json)、[α1配置](evidence/tolerance_progress_20261022/environment/pulse_alpha1_config.json)
- [baseline](evidence/tolerance_progress_20261022/environment/baseline_reward_config.json)、[static](evidence/tolerance_progress_20261022/environment/static_reward_config.json)、[dynamic](evidence/tolerance_progress_20261022/environment/dynamic_reward_config.json)奖励配置
- [评测协议](evidence/tolerance_progress_20261022/snapshots/pulse_curriculum_train_v1/comparison_protocol.json)、[评测manifest](evidence/tolerance_progress_20261022/pulse_curriculum_results/manifest.json)
- [评测执行代码](../learning/cli/evaluate_tolerance_comparison.py)、[物理外力实现](../learning/src/sttw_control/teleop_env.py)、[训练采样](../learning/src/sttw_control/path_command_scenarios.py)

**解释配置时注意历史描述字段**：network.initialization的fresh描述是父实验模板，实际新实验权重来源以fork_receipt/plan为准；plant.external_disturbances=false是旧teleop模板，真实脉冲由可选force_pulse接入；stage_B中直道10s是历史基线案例，本轮PPO与主评测实际20s。默认family_fraction只描述legacy子组，完整训练分布以training_sampler和执行采样器为准。

## 9. 与单网络V3任务的区别及后续时长

单网络V3是一个α条件Actor，冻结local300，10s回合、课程路径/速度分布、端点蒸馏初始化和持久路径偏好V3奖励；本轮是双端点、20s、旧奖励和固定高需求采样。不能跨实验直接比较总reward或归因于某单项变化。详见远端单网络分支 `experiment/path-conflict-exposure-v1` 的 `docs/evidence/path_distilled_v3_curriculum_400_20261021/INDEX.md`。

20s沿用了双网络父实验的任务定义，便于保持对照；没有证明它是最优时长。下一轮10s是建议，尚未执行，也没有把旧轨迹裁短重写结果。若实施，应统一全部对照组，并保留必要的20s延长验证。
