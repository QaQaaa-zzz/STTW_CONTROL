# α 条件几何路径恢复：接口、奖励与验证

## 范围与状态

本实现以 `1230ae7f3adee88c134a4d812c9874c3041fc5d4` 为基线。
**α 直接输入残差 Actor**，不是外部参考管理器。α=0 优先路径、α=1 优先速度；
两端均保留被降权目标。α 不改变动作权限、roll 成本、车辆模型或执行器约束。

新增的是显式选择的几何路径实验：`learning/configs/path_recovery.json`。
旧 `command_recovery.json`、旧 reward、旧实验和 checkpoint 不改写。
当前不将 command 的 yaw-rate 跟踪称为路径回归；旧 command 训练仍可用于历史对照。
本次没有训练出新策略，没有 GPU 吞吐、完整物理回归或实车成功证据。

本次交付环境中的验证：28 项 NumPy/JAX 数学、状态机、配置及接口测试通过；
完整 CPU 物理集成测试模块因缺少 Flax/Optax/MuJoCo 而跳过。
所有新增/修改 Python 文件通过语法编译。没有用模拟依赖冒充完整仿真通过。
`learning/tests/test_tracking_integration.py` 已提供给完整项目环境运行。

## 保留的物理与控制接口

- ECBC、ESO、控制周期 0.005 s、原始模型和 25 个物理子步保持原样。
- 前轮是转向角速度 rad/s，后轮是轮轴转速 rad/s；不是转角/车速/转矩输出。
- 动作为 `u = final_limiter(u_base + [1.5,10] * tanh(z))`。
- α 不用于缩小中间档权限，不启用旧 `g=0.2+0.8|2α-1|` 门控。
- 基础控制使用原始几何参考；零残差回归同一基线。新增回报不会改变物理步进。
- 轮速×0.1 只作为 Actor 的速度代理；奖励和评价使用真实纵向速度。
- 未新增机械臂、足端、腾空、步态、能耗或强制瞬时转向角跟踪奖励。

## 观测与身份

新观测顺序：15 个原有字段 + 3 个原始路径字段 + 5 个恢复上下文字段 + α。
10 帧历史与 10 个 mask 共 `(15+3+5+1)*10+10=250` 维。
网络仍为 256→128、LeakyReLU、两维 tanh 输出。α 通过现有 `set_priority` 可在运行时改变；
新实验的训练 α 在每回合随机采样。该实现不声称已验证动态切换效果。

五个上下文字段依次是：上一归一化转向残差、上一归一化驱动残差、路径待回归标志、
归一化离带时间（上限2）、归一化路径保持时间（上限1）。计时来自可观测的几何/姿态误差，
不读取扰动开始/结束标签，也不把仿真真实速度编码进 Actor 上下文。

`network.py` 将新字段顺序和完整奖励配置绑定 checkpoint。`tracking_reward=None` 和
`include_tracking=False` 在旧身份哈希中被移除；旧模型不能当作 250 维新模型直接续训。

## 共享奖励公式

实现在 `tracking_reward.py`，环境、PPO 分项日志和 CPU 轨迹重建均使用它。
所有普通项都是分/秒，最后仅乘一次 `dt=0.005`；真实失败整步替换为 −100。
不对负总奖励截零，不将终止罚再乘 dt。

记速度误差为 ev、原始路径右法向误差为 ey、原始路径航向误差为 eψ。
圆路径也统一为右法向符号；参考是几何路径投影，不是按时刻追逐参考点，不在偏离后重置路径。
第一份配置是非自交的平滑弯道；现有 8 字全局投影的分支连续性未在本次改造。

权重：`wv=(1+9α)/11`，`wp=(10−9α)/11`，和恒为1，端点比10:1。

速度正奖励形状：
`Gv=0.5 exp[-(ev/0.5)^2] + 0.5 exp[-(ev/0.15)^2]`。

路径正奖励形状：
`Gp=0.5 exp[-(ey/0.4)^2−(eψ/0.35)^2] + 0.5 exp[-(ey/0.1)^2−(eψ/0.1)^2]`。

设 `H(z)=z² (|z|≤1), 2|z|−1 (|z|>1)`。

每秒奖励为：

```
1 + 4*wv*Gv + 4*wp*Gp
- 0.1*wv*H(ev/0.2)
- 0.1*wp*(H(ey/0.2)+0.3*H(eψ/0.15))
- 2*H(max(|ev|-(0.5-0.3α),0)/0.2)
- 2*H(max(|ey|-(0.1+0.3α),0)/0.2)
- 100*max(|roll|-0.3,0)^2 - roll_rate^2
- 0.01*sum(action^2) - 0.02*sum((action-previous_action)^2)
- 0.5*pending*min(return_age/3,1)
- 2*pending*I(return_age>3)
```

所有系数只是可复核的第一轮起点，不是已验证最优参数。容忍带是软成本，不是硬约束；
大额终止罚、动作限幅与工作 roll 阈值也不构成稳定性证明。
合理转弯允许侧倾，不追求全程 roll=0，不使用固定正侧倾目标。

## 回归状态机与成功定义

路径离带判据：`|ey|>0.1 m` 或 `|eψ|>0.15 rad`。
离带后计时，连续满足路径阈值、`|roll|≤0.3 rad`、`|roll_rate|≤0.5 rad/s` 满0.5秒，
才清除路径待回归状态。计时与 α 无关，不等待不可观测的扰动撤销标签。
3秒是**从观测到路径离带开始**的回归预算；超时记违规、持续给成本，但不谎称发生物理失败。
本版没有恢复完成加分，避免反复离带/回带刷分。

训练的 `task_recovered` 还要求真实速度误差≤0.2 m/s 连续保持，并且此前确曾离带。
评价使用声明回合末端完整0.5秒窗口，检查路径、航向、速度与姿态，真实失败后不补零。
“从未离带”“最后保持达标”“离带后成功回归”“超过回归预算”分别报告。
名义场景从未离带不计入恢复成功率分母。最终保持或存活都不等于全程任务完成。

## PPO 与速度

仍使用仓库自己的 JAX/Flax/Optax PPO，而不是悄悄换成 rsl_rl。
新配置 `ppo_path_recovery.json`：1024环境×128步×32轮 = 4,194,304训练控制转移；
4 epochs、32768 minibatch，每轮最多16次优化（KL限制下实际可能更少）。
预热另计128×1600=204,800次计算预算，使用真实零残差闭环，不只随机 tick。
学习率3e−4、gamma=.9995、lambda=.99、初始std=.15、entropy=.001及KL=.01保持。
验证在8/16/24/32轮，回合不自动扩展超过10秒，至少留下声明的回归观察窗口。

新训练器记录 `planned_optimizer_steps_per_update`、已有实际更新/KL/编译/采样/优化计时及
逐项有符号 reward。新的几何选模先看物理失败、名义退化、末段保持和离带超时，再看回报；
排名第一也可能不合格，`task_success_verified` 仍为 False。

没有宣称物理采样已经加速。本次没有改物理步长、减少子步、删除 forward、切换后端、
伪造状态池或移植 rsl_rl。应在相同任务质量下用日志测实际墙钟收益。

## 运行方式

在仓库根目录、项目已安装依赖的解释器下先执行测试：

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu python -m pytest \
  learning/tests/test_tracking_reward.py \
  learning/tests/test_tracking_contract.py \
  learning/tests/test_tracking_integration.py -q
```

有限预算训练（命令存在不表示本次已经执行）：

```bash
PYTHONPATH=learning/src XLA_PYTHON_CLIENT_PREALLOCATE=false python learning/cli/train.py \
  --task learning/configs/path_recovery.json \
  --config learning/configs/ppo_path_recovery.json \
  --output runs/path_recovery_declared_run
```

从该运行真实 checkpoint 做配对评估，输出原路径、每步总奖励/有符号分项、roll、
横向/航向误差与目标/真实/轮速估计速度的 PNG/PDF、NPZ、摘要与索引：

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu python learning/cli/tracking_evaluate.py \
  --task learning/configs/path_recovery.json \
  --training learning/configs/ppo_path_recovery.json \
  --checkpoint runs/path_recovery_declared_run/checkpoints/update_0032 \
  --output runs/path_recovery_declared_evaluation \
  --alphas 0 0.25 0.5 0.75 1 --seeds 47011
```

不传 `--checkpoint` 只评估基础控制器。新任务使用本评估入口，不把旧 command 或旧
reward_breakdown 流水线当作已验证的新几何任务诊断。所有输出目录必须是新的，旧证据不覆盖。

## 尚未完成的实验范围

没有新训练结果、CPU/MJX全链实测、GPU预检、跨训练种子/左右弯/变速曲率课程、
物理吞吐剖析或实车测试。本次只是可测试的条件路径任务、奖励与训练/评价接口实现。
不能据这份实现声称α单调性、鲁棒性、恢复域扩大或可靠部署已经成立。
