# 模型1残差恢复控制

本目录提供 ECBC 的 JAX 实现、256→128 LeakyReLU Actor、两路速度残差、CPU MuJoCo/MJX 环境和冻结策略评估。已接入 PPO 更新循环，并完成首轮随机动态扰动训练；结果与限制见当前验证报告。当前验证见 [`../docs/VALIDATION.md`](../docs/VALIDATION.md)。

## 保存视频与状态图

已有轨迹可直接回放，不重新推进物理：

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python learning/cli/render.py --run runs/verification_20260908/baseline
```

依赖`learning[media]`中的Matplotlib、Pillow、mediapy，以及系统FFmpeg；CLI默认使用EGL无窗口渲染。输出到该run的新`media/`目录：`replay.mp4`、`trajectory.png/pdf`、`states.png/pdf`、`preview.png`、`terminal.png`和`manifest.json`。已存在的media目录不会覆盖。

MP4默认1280×720、30fps：左侧跟随相机，右侧固定俯视，橙色为声明的参考路径，蓝色为实际轨迹。场景线条只是渲染覆盖物，不参与碰撞。HUD显示实际仿真时间、真实纵向速度与侧倾。每帧对应记录中的qpos/qvel，调用mj_forward恢复显示，不调用mj_step；最后一帧必定是轨迹终点。8s轨迹输出241帧，文件时长约8.033s，额外一帧用于保留终点。manifest记录帧索引与源文件SHA256。

图中包含XY路径与误差，以及侧倾、侧倾角速度、真实速度、转向角、转向速度与后轮速度。参考/指令用橙色虚线，实际值用蓝色实线。只有声明了圆或零转向直行时才绘制XY参考路径；普通转向指令不会被误标成直线路径跟踪。残差策略回放按实际声明标注控制器。

## 圆形路径跟踪

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python learning/cli/evaluate.py --config learning/configs/circle_tracking.json --output runs/local_circle --seed 0
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python learning/cli/render.py --run runs/local_circle
```

默认圆半径3m、圆心(0,3)m、逆时针、速度参考2m/s、前视距离2.5m、12s。几何外环使用当前根部XY和车体航向，生成目标转向角，由现有ECBC＋ESO完成姿态控制。它是按空间路径跟随，不强制时间参数化的目标相位；无需新增学习网络、训练或提高执行器限制。当前外环使用仿真定位，实车定位尚未接入。

圆形参数位于唯一的`circle_tracking.json`，实现位于`path.py`；改变实验应修改配置或创建有明确用途的运行声明，不复制模块。当前2.5m前视距离来自三个声明工况的工程比较，结果见验证报告，不代表独立测试集上的最优参数。

## 环境与测试

在仓库根目录执行。当前已验证解释器为 `/home/qy/mujoco_playground/.venv/bin/python`，MuJoCo/MJX 3.6.0、JAX 0.6.2、Flax 0.11.2。可在独立环境安装 `learning/pyproject.toml` 中声明的依赖；版本升级需要重新验证。CPU测试同时需要 `g++`，不需要 ROS。

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python -m pytest learning/tests -q
```

显式运行真实 GPU 检查（两个并行环境，reset 和一次控制步）：

```bash
PYTHONPATH=learning/src STTW_TEST_GPU=1 XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qy/mujoco_playground/.venv/bin/python -m pytest learning/tests/test_mjx.py -q
```

## 运行基线

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python learning/cli/evaluate.py --config learning/configs/baseline.json --output runs/local_baseline --seed 0
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python learning/cli/evaluate.py --config learning/configs/turn_recovery.json --output runs/local_turn_recovery --seed 0
```

输出目录必须不存在。每次运行生成 `declaration.json`、`trace.npz`、`summary.json`、`status.json`，不覆盖旧实验。默认是零残差基础控制器；`--policy PATH` 加载已导出的冻结策略。`--backend mjx` 选择加速后端，需去掉 `JAX_PLATFORMS=cpu`；当前后端工程检查不等于长轨迹 CPU/GPU 等价。

策略导出使用 `network.save_policy`，保存 Actor 参数、归一化、网络结构及完整模型/配置身份。通过 `network.make_policy_identity(env.bundle.identity, asdict(env.config), history_steps)` 构造身份；模型资产变化会拒绝旧策略。初始化网络仅供工程测试，不得把随机网络作为训练完成的策略部署。

## 观测、动作和状态

- Actor：15个状态/指令字段＋每帧有效掩码，默认16维；历史长度为 H 时是15H维帧数据＋H维掩码，按旧→新排序。
- 前四项：`roll-reference_roll`、左倾为正的 roll、roll_rate、后轮轮速×0.1 m。其余字段及顺序由 `observation.FIELDS` 定义。
- `yaw_rate_body` 是本地陀螺仪z轴读数，不宣称等于 Euler 偏航角导数。Critic/恢复评估可使用仿真真值；任务恢复使用车体纵向真实速度，而 Actor 使用轮速估计。
- 输出：`[steer_rate_residual, rear_rate_residual]`，各项在[-1,1]；默认物理补偿幅值分别1 rad/s、5 rad/s。
- 基础控制器先按原算法计算并限幅到±4 rad/s，再经共享执行层限制到XML的±3 rad/s、转角±0.8 rad和后轮±60 rad/s。
- 可配置指令变化率和整控制周期延迟；默认无额外延迟/斜率限制。XML速度伺服和力矩限制依旧参与动力学。不能把指令斜率称为真实机械加速度上限。
- Python与C++原始控制对照包含ESO和输出滤波。旧C++的roll微分器只影响诊断日志，不反馈到控制输出，JAX控制状态不复制这一诊断滤波器。

## 恢复判据与当前限制

示例转弯扰动是世界y方向2 N、4.0 s开始、持续0.1 s；它仅用于工程检查，未经过实车辨识。扰动必须对齐控制周期。恢复保持时间从外力结束后的完整无扰动控制区间开始计数。

默认平衡容差：参考侧倾误差0.05 rad、侧倾速度0.15 rad/s，保持0.5 s。任务恢复再要求真实纵向速度误差≤0.2 m/s、转向角误差≤0.05 rad。失败包括侧倾超限、非轮部件触地及非有限状态；达到8 s视为截断。成功保持后继续运行到终点，后续失败会使最终成功标志为false。轨迹保留第一次满足保持条件的时间，二者含义不同。

当前采用给定前进速度的合成初态，不是已验证的自然启动到达状态。初始侧倾范围可以配置，ESO、动作队列和观测历史从干净状态开始。尚未实现完整困难状态快照恢复、执行器辨识随机化和正式恢复域扫描。空间统计只是整段轨迹根部相对初始位置的世界坐标范围，不是恢复扫掠包络或最小空间。

## ROS 接入（尚未在本机编译）

原工程仍在 `mujoco_ros_ws20260313`。ROS节点使用 `config/controller.yaml`，通过私有参数 `config_path`、`model_path`、`plugin_dir`、`log_dir` 配置路径。默认临时日志目录 `/tmp/sttw_control`；正式实验应将 `log_dir` 指向仓库 `runs/<run-id>` 内的专用子目录。

在已有ROS1/catkin环境中将 `mujoco_ros` 包和相邻 `model` 目录放入工作区 `src/`，构建后可运行 `roslaunch mujoco_ros simulation.launch`。若平台不能加载原 ARM64 torus 插件，先导出不含未用插件声明的模型：

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python learning/cli/export_model.py --output build/scalebike.xml
```

再使用 `roslaunch mujoco_ros simulation.launch model:=/absolute/path/to/build/scalebike.xml load_plugins:=false`。导出的XML引用原始mesh的绝对路径，不复制网格、不替换接触形状。目录移动后需要重新生成。

`residual_cmd` 使用 `ResidualCmd.msg`：Header时间戳及两路无量纲动作，默认残差关闭；单独启动 `control.launch residual_enabled:=true` 才允许新鲜消息进入共享限幅器。超时/非有限残差回到基础控制。状态过期或mode=9输出零指令，重新获得有效状态后重置控制历史。此处只有消息接入层，未提供经过实车验证的神经网络推理节点。ROS适配层当前对应零额外延迟/斜率限制配置，训练中非默认延迟/变化率需在部署适配中对应实现并验证。

## 圆形残差 PPO

保留几何外环、ECBC 和 ESO，Actor 为 256→128 LeakyReLU。
`circle_learning.json` 开启三个附加输入：径向误差、相对圆切线航向误差、参考曲率；默认历史长度1时为18字段＋1掩码。前四个核心输入与原结构不变。附加输入需要定位；目前使用仿真位姿。

```bash
PYTHONPATH=learning/src XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qy/mujoco_playground/.venv/bin/python -u learning/cli/train.py --task learning/configs/circle_learning.json --config learning/configs/ppo_circle.json --output runs/<new-training-run>
```

预算为64环境×256控制步×64次更新，约104.9万个控制步；物理步长及每控制步25个物理子步保持不变。训练初始侧倾±0.02rad，固定4个开发验证种子，更新1、每8次和最后一次保存。`metrics.jsonl`记录更新与验证指标，`status.json`给出最新和最优候选路径；旧实验曾仅按径向RMSE选模；当前规则见下文“恢复选模”，旧实验记录不重新解释为通过新门槛。

超时允许价值bootstrap，跌倒禁止；两者均截断GAE并完整reset控制、ESO、执行器及历史状态。Actor从零确定性残差开始，预tanh高斯探索初始标准差0.15；PPO概率比使用同一个预tanh样本，Jacobian相消。熵项采用潜在高斯熵近似。固定物理尺度归一化随Actor保存。

每个检查点包含可供`cli/evaluate.py --policy <checkpoint>`加载的Actor，以及优化器/critic/RNG的训练快照和摘要。当前没有训练恢复CLI，快照不包含物理环境状态，不能宣称无缝继续同一轨迹。训练声明记录源码摘要、模型和全部参数。视频生成沿用`cli/render.py`，原始数据不覆盖。


比较保存的同条件轨迹（不加载策略、不重新积分）：

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python learning/cli/compare.py --baseline runs/<baseline> --candidate runs/<candidate> --output runs/<comparison> --candidate-label 'Candidate policy'
```

`ppo_circle_smoke.json`为修复后1024环境、单轮32768控制步的完整PPO工程配置。它只检查训练链路，不能用于宣称路径学习收敛。训练日志新增采样、优化、验证、检查点计时与完整墙钟计时。


## 转弯扰动面板

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python learning/cli/disturbance.py --task learning/configs/circle_learning.json --panel learning/configs/turn_disturbance_panel.json --training-run runs/circle_corrected_1024_smoke_20260908 --checkpoint runs/circle_corrected_1024_smoke_20260908/checkpoints/update_0001 --output runs/<new-disturbance-panel>
```

只在冻结训练配置基础上显式覆盖扰动事件，不改变策略观测/动作；先运行每个控制器的无扰动对照，再运行独立正负指令脉冲与质心侧力。`disturbance.recovery_metrics`报告相对参考圆及无扰动轨迹的回轨保持；`disturbance.plot_panel`生成事件标注对比图。单个轨迹的视频仍用`cli/render.py`。

## 动态扰动训练及自动标准评估

```bash
PYTHONPATH=learning/src XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qy/mujoco_playground/.venv/bin/python -u learning/cli/recovery_pipeline.py --task learning/configs/disturbance_learning.json --training learning/configs/ppo_disturbance.json --panel learning/configs/recovery_standard_panel.json --output runs/<new-recovery-run>
```

`pipeline_status.json`显示当前阶段；`training.log`记录主训练，训练内metrics包括实际受扰控制步计数。训练结束选冻结候选、运行独立CPU标准配对面板、保存standard_results.json及带事件窗口的图/视频；不会自动增加训练预算。采样事件完整绑定config及checkpoint，旧跟圆策略不作为这次训练起点。首次JIT计时与稳态采样吞吐应分开解读。

## 恢复选模（2026-09-09）

`ppo_disturbance.json` 的训练预算仍为1024×256×32=8,388,608控制步，训练事件分布不变。开发验证改为4个seed分别交叉无扰动、正负强转向、正负强侧力5个固定工况，每条轨迹再配同策略同初态的无扰动参考，共40条。6.0–6.6s施加扰动，观察至16.6s，单次验证最多132,800控制步；更新前基线及1/8/16/24/32更新共6次，最多796,800开发验证步，独立于训练步预算。

候选逐工况满足无物理失败、真实速度RMSE≤min(0.2m/s,同工况基线RMSE+selection_speed_slack)，且无扰动径向RMSE不比基线增加超过0.01m。之后按“最终联合保持未完成比例、平均保持完成时间、平均额外径向峰值”依次比较。未完成时间按观察窗右删失，不记为0；这里的保持完成率不等价于经过扰动离带后的恢复概率。峰值统一从扰动开始算起，保持时间从扰动结束算起。基线或候选指标非有限时不能通过门槛，JSON用null保留无效证据。

标准面板 `minimum_post_event_seconds=10` 显式延长评估窗口，不改变训练任务或checkpoint身份。当前54条面板最多179,280控制步。41001–41003已经用于分析，重复运行属于回归检查；正式新的独立测试需另行冻结未使用种子，不能继续称为首次留出测试。新配置尚未启动完整训练，旧run内声明和原始12s记录保持不变。

2026-09-09速度优先运行：用户已授权上述预算。当前任务速度惩罚权重为100（上一轮20），ppo_disturbance.json的selection_speed_slack为0，即候选逐工况速度RMSE不得高于基线。完整配置将在runs/launches/dynamic_recovery_speed_20260909冻结，运行输出在runs/dynamic_recovery_speed_20260909；历史运行参数以各自declaration为准。


完整标准面板媒体：
```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu MUJOCO_GL=egl /home/qy/mujoco_playground/.venv/bin/python learning/cli/panel_media.py --panel-run runs/<run>/evaluation/seed_<seed> --output runs/<run>/complete_media
```
输出目录必须是新的；已有单轨媒体必须完整且来源摘要一致才能复用，不覆盖失败或陈旧媒体。每工况生成同步比较视频、带事件阴影的状态图和manifest，统一INDEX.md可打开。recovery_pipeline现在自动调用此步骤覆盖全部工况。

`disturbance_history_learning.json`只把当前速度优先任务的history_steps设为20，形成380维输入（95ms首末跨度）；PPO配置复用ppo_disturbance.json。对照保留所有物理、奖励和事件分布，用相同预算测试历史信息的效果。不同history身份的checkpoint不能互相静默加载。

## 协同动作映射与消融

任务配置可选`action_mapping`，省略时仍直接输出转向角速度/后轮转速残差。启用后Actor的两个tanh输出分别表示`lateral_acceleration_proxy_residual`和`speed_proxy_residual`；`action_mapping.py`用转向角及后轮轮速代理转换成共享执行器接口的有界残差。PPO继续在原latent变量上计算概率，环境内完成确定性映射；动作惩罚使用映射后的电机残差。

| 配置 | 用途 |
| --- | --- |
| disturbance_learning.json | 普通双通道直接残差 |
| disturbance_steering_only_learning.json | 后轮残差尺度为0，隔离驱动补偿作用 |
| disturbance_coupled_learning.json | 含速度—横向响应交叉项 |
| disturbance_decoupled_learning.json | 相同参数，交叉项置0 |

上述配置可通过现有train/recovery_pipeline入口传入；配置就绪不代表已运行正式训练。映射horizon须大于指令延迟，不支持额外指令变化率限制的预测；共享执行器仍执行最终限幅、关节边界和延迟。低于minimum_speed_proxy时映射输出零。参数是局部仿真近似，实车需重新辨识。

**映射策略输出不是ROS电机残差。** `load_policy`返回代理需求，部署必须先调用相同映射再进入ResidualCmd接口；当前ROS未接入此转换，禁止直接发布该策略输出。checkpoint记录action_fields并校验完整配置，旧直接策略身份不因缺省action_mapping字段改变。

标准配对恢复报告增加`path_relative_space`，左正法向误差为`-direction*(distance_to_center-radius)`。左右占用从扰动开始计至观察结束；未恢复保留删失标记。该指标计根参考点，不含车体扫掠包络；旧报告不自动更新，重分析应写入新run目录。

`disturbance_authority_learning.json`启用`authority_aware`有限指令余量分配。优化物理残差u：`sum(w * (J u - demand)^2) + regularization * ||u||²`，其中w为配置权重除以需求尺度平方，边界同时考虑残差幅值、strength、基础指令占用和当前关节位置。横向与速度权重显式配置并绑定checkpoint；当前不是自适应优先级策略。和disturbance_coupled_learning.json比较可隔离余量感知作用。

此模式需给map_action传入base指令，环境已接入；基础指令越界回退零残差。不支持非零指令延迟/额外指令变化率限制。新增评估字段`command_limits`统计最终速度指令达到边界的样本比例（不含reset），不统计电机力矩饱和、关节限位或请求被裁剪次数。ROS部署仍需单独接入映射。
