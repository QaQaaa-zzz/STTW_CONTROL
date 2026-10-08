# STTW_CONTROL R196 / R244 冻结残差底层调用契约

固定入口：`/home/qy/STTW_CONTROL/docs/FROZEN_LOWER_CONTROLLERS.md`。
机器可读注册表：`/home/qy/STTW_CONTROL/learning/configs/frozen_lower_registry.json`。
检索关键词：**STTW_R196_ALPHA1、STTW_R244_ALPHA1、R196、R244、冻结底层**。
当前研究事实与采用决定仍以 [PROJECT_STATE.md](/home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md) 为准。本文件描述接口，不另建研究状态台账。

## 1. alpha到底是多少

|使用场合|R196|R244|
|---|---|---|
|原训练|每回合等概率抽取0/.5/1，回合内固定|相同|
|2026-10-08原生路径复测|分别测试0/.5/1|相同|
|共同速度＋前轮转角底层测试|**固定1.0**|**固定1.0**|
|本文件两个固定调用别名|`STTW_R196_ALPHA1`：**1.0**|`STTW_R244_ALPHA1`：**1.0**|

α是Actor观测条件，不是模型自动估计的输出，不是电机增益，也不是权重文件固有的单一常量。α=1偏重速度；α=0偏重路径；α=.5居中。两模型具体奖励权重不同，因此相同α不代表行为相同。

上层以后若有自己的任务偏好，命名`upper_alpha`，不要自动传给这里的`lower_alpha`。这两个ALPHA1别名的lower_alpha固定1。若需要其它α或回合内动态变化，显式另声明调用模式；旧训练没有验证任意动态α调度。改α时当前帧保存当前值，旧历史帧保留当时值，禁止回填覆盖历史。

## 2. 精确模型与依赖

- R196：`/home/qy/STTW_CONTROL/runs/precision_speed_ecbc1_20260920/training/checkpoints/update_0196`
- R244：`/home/qy/STTW_CONTROL/runs/soft_budget_ecbc1_20260920/training/checkpoints/update_0244`

每个目录同时保留`actor.msgpack`及`identity.json`。注册表钉住Actor和identity的SHA-256、来源训练声明、α、字段顺序及缩放。不能改成某次训练的latest、不能拿另一个目录的normalizer配权重。两个Actor均为310→128→128→128→2，ELU，输出端tanh；10帧历史。

可用解释器：`/home/qy/mujoco_playground/.venv/bin/python`；普通Actor加载依赖`/home/qy/STTW_CONTROL/learning/src`。模型路径使用绝对路径，从任何工作目录均可定位；这不是把权重复制到全局目录，也没有新增后台服务。

## 3. 观测接口：不能只输入速度和转角

每步30个原始字段；将10帧按**旧→新**展平为300维，再追加10个有效mask，共310维。α在每帧零基下标18，因此观测中的α位置为`18 + 30*k, k=0..9`，不是最后一维。无效历史帧保持零及mask=0；只在真实采样的新帧写α=1，mask=1。

|零基下标|冻结字段名称|
|---:|---|
| 0 | `roll_error` |
| 1 | `roll` |
| 2 | `roll_rate` |
| 3 | `speed_estimate` |
| 4 | `steer` |
| 5 | `steer_rate` |
| 6 | `yaw_rate_body` |
| 7 | `rear_rate` |
| 8 | `front_rate` |
| 9 | `steer_reference` |
| 10 | `speed_reference` |
| 11 | `base_steer_rate` |
| 12 | `previous_steer_command` |
| 13 | `previous_rear_command` |
| 14 | `estimated_disturbance` |
| 15 | `path_lateral_error` |
| 16 | `heading_error` |
| 17 | `path_curvature` |
| 18 | `speed_priority` |
| 19 | `previous_steer_residual` |
| 20 | `previous_rear_residual` |
| 21 | `return_pending` |
| 22 | `return_elapsed` |
| 23 | `return_hold` |
| 24 | `return_credited` |
| 25 | `return_ever_left` |
| 26 | `return_deadline_missed` |
| 27 | `longitudinal_error` |
| 28 | `yaw_rate_reference` |
| 29 | `yaw_rate_world` |

- `roll_error = roll - reference_roll`；参考侧倾仍为ECBC动态参考。
- `speed_estimate = rear_rate * 0.1m`，其中rear_rate是控制内部完成符号转换后的值，不把MuJoCo负轮轴原始qvel直接代入。它是轮速估计，不是真实前向速度。
- 转角与角速度用rad、rad/s，速度m/s，位置误差m，曲率1/m，回归elapsed/hold为秒。
- 横向误差采用参考曲线**右法向**，航向误差为实际减参考并wrap；沿程为实际减时间参考沿切向投影。world yaw-rate与body yaw-rate不能互换。
- `previous_*_residual`为上一归一化动作，`previous_*_command`为上一实际执行命令；勿把二者混淆。
- 字段生成复用`observation.make_frame / advance_history`与`tracking_reward.return_observation`，不要自己省略恢复时钟或以零长期代替路径误差。
- `load_policy`内部使用identity.json里的mean/std归一化并执行tanh；传入**未归一化**的310维观测。禁止重复归一化、重复tanh、擅自添加clip或交换字段。

## 4. Actor层最小加载示例

以下只加载冻结Actor，并非完整车辆控制器。传入的obs必须由上述真实状态链生成。

```python
import sys, json, hashlib
from pathlib import Path
import numpy as np

ROOT = Path('/home/qy/STTW_CONTROL')
sys.path.insert(0, str(ROOT / 'learning/src'))
from sttw_control.network import load_policy

registry = json.loads((ROOT / 'learning/configs/frozen_lower_registry.json').read_text())
alias = 'STTW_R196_ALPHA1'  # 或 STTW_R244_ALPHA1
entry = registry['models'][alias]
checkpoint = Path(entry['checkpoint'])
for filename, key in [('actor.msgpack', 'actor_sha256'), ('identity.json', 'identity_sha256')]:
    assert hashlib.sha256((checkpoint / filename).read_bytes()).hexdigest() == entry[key]
source = json.loads(Path(entry['source_declaration']).read_text())
policy = load_policy(checkpoint, expected=source['policy_identity'])

def infer(obs):
    obs = np.asarray(obs, dtype=np.float32)
    assert obs.shape == (310,) and np.isfinite(obs).all()
    frames, mask = obs[:300].reshape(10, 30), obs[300:]
    assert np.isin(mask, [0, 1]).all()
    assert mask[-1] == 1 and np.all(frames[mask == 1, 18] == entry['alpha'])
    assert np.all(frames[mask == 0] == 0)
    action = np.asarray(policy(obs))  # 已tanh，shape=(2,)
    assert action.shape == (2,) and np.isfinite(action).all()
    residual = action * np.asarray(entry['action_scales'])
    return action, residual
```

来源identity检查证明“这是原来的模型”；它本身不证明另一个仿真或实车的物理/观测契约相同。普通路径任务可使用当前RecoveryEnv生成观测；速度/转角接口须采用下一节的适配链。

## 5. 作为速度＋前轮转角底层的完整链路

对外参考顺序是`[v_ref(m/s), delta_ref(rad)]`；Actor输出顺序为`[前轮角速度残差, 后轮轴转速残差]`。

```text
已发布的速度/转角参考＋车体测量＋定位
  → 原ECBC/ESO输出预览
  → 因果参考路径与时间误差＋回归状态＋10帧观测（lower_alpha=1）
  → 冻结Actor输出归一化残差a
  → 基础前轮角速度 + 1.5*a[0]
     基础后轮轴转速 + 10*a[1]
  → 原执行器限幅/位置保护与物理接口
  → 下一步更新回归状态、历史、世界yaw-rate与参考路径
```

控制周期固定0.005s。基础后轮命令为`v_ref/0.1m`，不是车体加速度；前轮输出是角速度，不是转角。当前总角速度上限前轮±3rad/s、后轮±60rad/s，转角位置上限±0.8rad。必须复用执行器逻辑，不只对上式简单clip。ROS/仿真与内部符号转换保持现有闭环约定且只做一次。

已实际验证的速度/转角适配入口（诊断实现，并非已安装的通用生产类）：

- `/home/qy/STTW_CONTROL/runs/tracking_candidate_review_20261008/command_replay.py`
- 类`Transfer('precision196')`或`Transfer('soft244')`，其中α在prepare和after_step均固定1。
- 宿主`/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/learning/src/sttw_control/direct_command_env.py`，使用完整准备状态reset，再调用低层prepare/after_step；200Hz推进。
- 原工作树`FrozenLowerController`只接受旧280维/256→128接口，**不能直接把这两个310维checkpoint路径塞给它**。本次Transfer是明确适配，不是更名后的相同接口。

适配只根据已发布参考构建历史曲线及端点切线延长，不泄漏未来指令。运动学关系`yaw_ref=v*cos(caster)*tan(delta_ref)/wheelbase`在此仅用于定义参考，不是学习模型的动力学预测器。原生路径评估则使用原任务的参考和路径控制外环；两种场景结果不能混称。

每个机器人/环境/模型实例必须独立持有ECBC/ESO、执行器上一命令、观测历史/mask、参考位姿和路径缓存、回归状态、上一残差与yaw历史。硬reset全部清理；接续完整准备状态时保留对应控制器/执行器状态，明确重置任务参考与策略历史，不能只清空Actor历史而保留上一回合ESO。模型切换也不能复用另一模型的回归状态，两者回归规则不同。

当前Transfer路径容量2001点，只支持本次10s诊断；不能直接作为无限时长常驻底层。运行中α变化、长时间连续控制、移植后真实速度/定位估计均需单独处理。Actor内部速度来自轮速，但适配回归状态还使用仿真真实前向速度和位姿；实车调用必须提供对应估计，不能宣称仅凭轮速已等价复现。

## 6. 可复现范围与状态

现有对照：六场景、seed77001、共同准备状态、α1、每回合10s，无新增训练。R196和R244原始权重不变。注册这两个别名不会自动更改上层配置或接管车辆。

- R196：共同指令末段保持6/6，但急转峰值roll约0.413rad。
- R244：共同指令末段保持5/6，急转峰值roll约0.435rad。

这是可调用的冻结模型身份与已验证诊断接口，不是稳定性/实车/任意时长保证。结果：[共同指令对照](/home/qy/STTW_CONTROL/runs/tracking_candidate_review_20261008/commands/INDEX.md)；训练说明：[模型清单](/home/qy/STTW_CONTROL/runs/tracking_candidate_review_20261008/model_inventory.md)。

从任意工作目录查阅：

```bash
cat /home/qy/STTW_CONTROL/docs/FROZEN_LOWER_CONTROLLERS.md
cat /home/qy/STTW_CONTROL/learning/configs/frozen_lower_registry.json
```
