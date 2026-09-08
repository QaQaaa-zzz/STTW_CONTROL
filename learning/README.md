# 模型1残差恢复控制

本目录提供 ECBC 的 JAX 实现、256→128 LeakyReLU Actor、两路速度残差、CPU MuJoCo/MJX 环境和冻结策略评估。尚未接入 PPO 更新循环；不存在已训练的补偿策略。当前验证见 [`../docs/VALIDATION.md`](../docs/VALIDATION.md)。

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
