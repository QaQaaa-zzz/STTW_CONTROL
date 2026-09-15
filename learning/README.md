# 模型1条件残差恢复控制

本目录实现ECBC＋ESO基础上的有界两路残差、CPU MuJoCo/MJX、PPO训练、标准恢复评估和完整媒体。当前研究状态见[PROJECT](../PROJECT.md)，逐轮证据见[VALIDATION](../docs/VALIDATION.md)，完整公式与框图见[论文草稿](../docs/paper/MANUSCRIPT.md)。所有命令在仓库根目录执行；以下启动命令是操作说明，不代表已执行或自动追加预算。

## 当前配置（2026-09-12）

| 项目 | 当前值 |
|---|---|
| 任务 | 左转R4m、2.1m/s、30s、200Hz |
| 基础控制 | ECBC＋ESO动态参考保留；后轮速度前馈 |
| 学习侧倾目标 | 固定6.84°；只改观测/奖励/训练恢复判据 |
| Actor | 19字段×10帧＋10mask=200维，256→128 LeakyReLU，tanh两输出 |
| 动作 | 前轮角速度残差±1rad/s、后轮轴速残差±5rad/s |
| α | 训练逐回合U[0,1]；开发/标准0/.5/1 |
| 奖励 | 生存5/s，速度100、横向20、航向1、越界80；固定成本尺度；端点比199 |
| 默认动作映射/风险 | 不启用映射；无显式姿态风险输入/门控 |
| 训练配置 | 8192环境×256步×32更新，67,108,864训练步；此预算最近已完成 |

最新运行`runs/priority_fixed_roll_20260912`已complete。三α各9/12恢复、各3次正转向失败；不是可直接部署或已验证优于基线的策略。旧队列error是DVGC依赖失败的历史，之后用户授权直接启动并完成，勿误重训。

## 环境与测试

已用解释器：`/home/qy/mujoco_playground/.venv/bin/python`。依赖入口为`learning/pyproject.toml`，媒体需FFmpeg。模型资产和仿真依赖版本进入运行声明，升级后须重新验证。

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python -m pytest learning/tests -q
```

真实GPU工程检查只查编译/reset/一步，不证明恢复性能：

```bash
PYTHONPATH=learning/src STTW_TEST_GPU=1 XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qy/mujoco_playground/.venv/bin/python -m pytest learning/tests/test_mjx.py -q
```

## 完整条件训练流程

明确新的预算和未存在的输出目录后使用：

```bash
PYTHONPATH=learning/src MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qy/mujoco_playground/.venv/bin/python learning/cli/recovery_pipeline.py --task learning/configs/priority_conditioned_learning.json --training learning/configs/ppo_priority.json --panel learning/configs/priority_standard_panel.json --output runs/DECLARED_NEW_RUN
```

入口冻结三配置和源码摘要，依次训练、标准评估、媒体、奖励诊断，最后标complete。条件策略固定末次checkpoint；默认不自动续训。90标准回合、12配对视频/状态图，45残差轨迹组成15组奖励图。失败轨迹照常保留，视频短侧可冻结末帧但显式标注TERMINATED；数值不延长。

`--resume`只用于训练完成后的受支持后处理续跑，必须传原frozen三文件并满足源码/输入身份检查；不是恢复训练参数的入口。禁止把更改奖励后的配置交给旧策略冒充原实验。

## 等待外部任务后启动

`learning/cli/deferred_training.py --plan <已声明计划>`只读监视指定队列身份、完成状态和进程，配置/源码变化即停，独占launch_claim避免重复。当前旧queue已经消费或停止，不能作为新实验计划复用。显式立即启动的manual_launch与旧错误记录分开保存；实际训练进度以目标run/pipeline_status.json为准。

## 只评估和回放

基线示例（历史circle_tracking.json有独立R3m/2m/s设置，不等于当前条件任务）：

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python learning/cli/evaluate.py --config learning/configs/priority_conditioned_learning.json --output runs/DECLARED_BASELINE --seed 47001
```

评估输出declaration、trace.npz、summary、status；checkpoint与模型、观测顺序/归一化、动作及时序绑定。回放现有记录不调用物理step：

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu MUJOCO_GL=egl /home/qy/mujoco_playground/.venv/bin/python learning/cli/render.py --run runs/DECLARED_BASELINE
```

已有完整条件面板独立补奖励图，无新增仿真：

```bash
PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python learning/cli/reward_breakdown.py --run runs/priority_fixed_roll_20260912
```

重建器读取冻结配置和真实reference_roll，核对每步reward；支持当前直接残差、无风险门控的圆/8字路径，其他模式显式拒绝。可变α回合不能冒充固定α比较。输出analysis/reward_breakdown下PNG/PDF、逐帧分项NPZ、summary与INDEX，全部场景/种子均保留。惩罚率图是正成本每秒值；真实奖励要乘dt，终止−100为替换，不是再加全部惩罚。

## 观测与恢复语义

观测具体顺序由`observation.FIELDS/PATH_FIELDS`定义。10帧跨度45ms，包含上一最终命令和α，非完整独立动作历史。Actor用轮速×0.1代理，奖励/评价用仿真真纵向速度。当前没有俯仰角或真实扰动输入；定位仍依赖仿真路径状态，实车接入未验证。

ECBC动态θ参考不变；`TaskConfig.learning_roll_reference`仅覆盖学习侧。trace.reference_roll记录学习目标。None恢复原动态学习参考且保持旧checkpoint默认身份；固定正6.84°仅用于当前左转圆诊断，不能声称适用于右转/8字。

训练恢复加分只看侧倾误差/角速度/速度/转向误差，标准联合恢复还看绝对路径、同策略名义额外偏差、航向与终点保持。两者不一致是已记录限制。每个标准种子为开发回归初态，非独立训练重复。失败短轨迹与完整30s不作无条件RMSE/能量排名。机械功不等于电池能耗。

## 代码入口与历史实验

| 模块 | 用途 |
|---|---|
| controller.py / actuator.py | 原状态反馈/ESO、命令合成/边界/队列 |
| env.py / observation.py / events.py / path.py | CPU/MJX共享控制步骤、历史状态、随机扰动、圆/8字 |
| priority.py / network.py / training.py | 外部条件权重、checkpoint身份、PPO |
| disturbance.py / evaluation.py / media.py | 标准指标、状态记录、机械功和回放 |
| pipeline.py / reward_breakdown.py / deferred_training.py | 固定流程、完整奖励图、排队 |
| action_mapping.py / screening.py | 已有响应映射/余量分配实验分支；当前默认不启用 |

ROS C++与插件仍位于`mujoco_ros_ws20260313/`，本轮文档不修改。ROS接口和训练侧符号转换按AGENTS执行；实车推理、定位/俯仰观测、时延与驱动接口需单独验证。旧映射、8字筛查、速度优先实验的命令及解释以各自冻结声明和VALIDATION日期记录为准，不用旧示例代替当前任务参数。

## 论文与科研图导出

唯一稿件`docs/paper/MANUSCRIPT.md`，配套HTML/PDF、SVG/PDF/PNG图和输入哈希。`docs/paper/build.py`只读取现有证据，不训练或步进仿真。图生成使用项目NumPy/Matplotlib；导出还需Markdown、WeasyPrint（仅文档依赖）。本次将这两个包及依赖装在临时文档目录，没有改训练环境。

```bash
PYTHONPATH=learning/src /home/qy/mujoco_playground/.venv/bin/python docs/paper/build.py --figures-only
# 安装文档导出依赖到隔离位置后：
PYTHONPATH=learning/src:/tmp/sttw_paper_tools /home/qy/mujoco_playground/.venv/bin/python docs/paper/build.py --export-only
```

PDF含嵌入式公式/图片可直接阅读；HTML依赖同目录figures，分享时保留整目录。原始runs默认不入Git，没有本地原始记录时不能完整重算。论文中未实现的机制和待补实验必须保持标注。

## TensorBoard训练过程

安装本项目依赖时会安装TensorBoard。所有调用`sttw_control.training.train`的训练都会把事件写到`<训练输出>/tensorboard`，每轮JSON日志落盘后flush，异常退出也会关闭写入器。项目`runs/`内的事件目录自动链接到`runs/tensorboard/`索引；服务只扫描该索引，不遍历模型和worktree目录。

在项目根目录启动查看服务：

```bash
/home/qy/mujoco_playground/.venv/bin/python -m tensorboard.main --logdir runs/tensorboard --host 127.0.0.1 --port 6006 --reload_interval 5 --samples_per_plugin scalars=100000
```

访问 http://127.0.0.1:6006 。已有服务时直接打开页面，不重复启动；当前服务PID、参数与日志在`runs/tensorboard_service/`。`scalars=100000`为每曲线最多返回10万个点；此版本设为0会返回空曲线，不代表无限制。

主要分组：

- `train/mean_step_reward`：随机训练采样的平均每控制步奖励；横轴step是**全局PPO轮次**，不是控制步数。`train/control_transitions`单独记录控制交互数。
- `reward_components/`：有符号生存、速度、yaw、姿态、侧倾角速度、动作及失败奖励；没有的旧字段不补0。
- `loss/`和`kl/`：策略损失、价值损失、潜变量高斯熵、尝试更新时的近似KL。它们是尝试优化的小批次统计，不代表最终保留策略的固定任务效果。
- `optimizer/`：最终精确KL、整轮回退、尝试/局部接受小批次数；`retained_minibatches`在整轮回退时为0。
- `validation/`、`paired_episode_return/`：实际执行过的开发评估回报、失败、速度/yaw误差和末段保持，以及配对基线回报和差值；缺少评估的轮次不生成点，不宣称任务成功。命令面板标签`case_00/alpha_0.5/seed_46001`按冻结声明的序列顺序解析。
- `timing/`、`memory/`：采样、优化、验证、墙钟、吞吐和原日志显存统计。

已有100轮记录已导入`runs/command_balanced_100_20260914/training/tensorboard`，包含父运行1–8轮和续训9–100轮。它们是同一训练链；导入时间不是历史训练时间，因此查看Step轴及原日志墙钟字段。开发评估仅在1、8、100轮有点；本面板case_00为gentle、case_01为tight_turn，标准测试12组另见原运行analysis索引。平滑只是显示选项，分析原值可把Smoothing设为0。

历史导入示例（目标必须尚不存在，重复/倒序轮次会被拒绝）：

```bash
PYTHONPATH=learning/src /home/qy/mujoco_playground/.venv/bin/python learning/cli/tensorboard_logs.py --metrics <父训练目录>/metrics.jsonl --metrics <续训目录>/metrics.jsonl --output <续训目录>/tensorboard
```

保留来源SHA与JSON，导入不重新执行训练或评估。显示与运行参数参考[TensorBoard官方说明](https://www.tensorflow.org/tensorboard/get_started)。


### Command yaw tracking bonus and sequential runs

`command_yaw_reward.json` adds a positive `yaw_tracking` component: `5 * exp(-(yaw_error / 0.1)**2)` reward per second. It is independent of alpha; all existing costs remain. `MotionCommands` defaults disable it, preserving old policy identities. Failed transitions replace all terms with the existing terminal penalty. JSON, TensorBoard and per-step evaluation plots retain its positive sign.

`TrainingConfig.validation_updates` optionally declares stage-local development evaluation indices (final always evaluated). For 50 updates resumed from 100, `[25, 50]` evaluates global 125/150. This does not change optimizer behavior or resume physical state.

`deferred_training.py` accepts `pipeline_kind: command` with `dependency_status` and `dependency_launch` (PID and optional process_starttime). It waits only for that successful pipeline and its process exit, not unrelated GPU jobs. Frozen `input_sha256`, the exclusive lock and one-time launch claim prevent input drift and duplicate starts. `notify: true` launches the existing stage/completion/error monitor for the child; queue errors notify separately. The default recovery pipeline behavior remains available.

### 固定command场景的跨方法图

`PYTHONPATH=learning/src python learning/cli/command_comparison.py --manifest comparison.json --output runs/<run>/analysis/trajectory_comparison`

manifest格式为`{"runs":{"方法名称":"/absolute/run/root"}}`，每个root具有evaluation/alpha_*/seed_*/*/{baseline,residual}。每场景/种子输出一张全部alpha的XY图和一张每步总奖励/累计回报图，附PDF、NPZ及来源哈希。参考使用速度与世界yaw-rate精确分段积分；失败记录不延伸。自动command诊断会生成单模型+基线版本，已有manifest则保留其跨方法比较清单。

精简TensorBoard投影：`PYTHONPATH=learning/src python learning/cli/tensorboard_core.py --manifest runs/tensorboard_core/manifest.json --output runs/tensorboard_core/events --watch --resume`。manifest的runs值为有序metrics.jsonl文件列表；首次运行去掉--resume。保留训练每步平均reward、开发episode总回报均值/基线/差值、关键奖励分项、policy/value loss、KL、保留minibatch数和开发失败比例。详细审计仍在原始JSON中。默认服务：http://127.0.0.1:6006。


## α输入Actor的几何路径恢复（新增，2026-09-15）

### 入口与运行边界

此任务复用ECBC+ESO与原执行器；α=0路径优先，α=1速度优先。α进入每帧观测，不控制动作权限；三种优先级共同约束roll和最终回归。配置是待训练第一阶段，不是已经成功的策略。

```bash
export PYTHONPATH=learning/src
# 只执行训练；目录必须不存在。该命令不是本次已经执行的记录。
python learning/cli/train.py --task learning/configs/path_priority_recovery.json --config learning/configs/ppo_path_priority.json --output runs/DECLARED_PATH_RUN/training
# 完整独立流水线（不要与上面共用已经存在的输出目录）
python learning/cli/recovery_pipeline.py --task learning/configs/path_priority_recovery.json --training learning/configs/ppo_path_priority.json --panel learning/configs/path_priority_standard_panel.json --output runs/DECLARED_NEW_PATH_PIPELINE
```

前者预算4,194,304训练转移；额外预热89,600计算转移。后者还包含开发面板及标准5α×2初态seed×6场景×2控制器的评估、图表/视频；它们不是独立训练种子。训练seed65、开发48001、标准49001/49002相互分离，但本轮没有运行这些长实验。

### 精确奖励

设ev为真实前向速度减**本转移发生前的请求速度**，ey为原始路径横向投影误差，ep为包角航向误差，a为两维有界残差，alpha使用转移前值。所有误差使用物理单位，不是网络归一化观测。

```
wv=(1+9*alpha)/11; wp=(10-9*alpha)/11
Gv=.5*exp(-(ev/.5)^2)+.5*exp(-(ev/.15)^2)
Gp=.5*exp(-(ey/.4)^2-(ep/.35)^2)+.5*exp(-(ey/.1)^2-(ep/.1)^2)
H(z)=z^2                    (|z|<=1)
    =2*|z|-1                (|z|>1)
bv=.5-.3*alpha; by=.1+.3*alpha
Ctail=.1*(wv*H(ev/.2)+wp*(H(ey/.2)+.3*H(ep/.15)))
Cbudget=2*H(max(|ev|-bv,0)/.2)+2*H(max(|ey|-by,0)/.2)
Croll=100*max(|roll|-.3,0)^2+roll_rate^2
Caction=.01*sum(a^2)+.02*sum((a-a_previous)^2)
Creturn=.5*pending*(1+min(elapsed/3,1))   # 仅初始化1s以后
r=.005*(1+4*wv*Gv+4*wp*Gp-Ctail-Cbudget-Croll-Caction-Creturn)
```

共同回归带为|ey|≤.1m、|ep|≤.15rad、|ev|≤.2m/s、|roll|≤.3rad、|roll_rate|≤.3rad/s；连续保持.5s才完成。时钟从真实离带起计时、包括扰动期，3s未回归记为deadline_missed，不能靠晚到清除。曾离带后完成可每回合加2分一次；失败转移所有分项清零，仅−100。常规奖励不截成非负。完整可改参数均在配置tracking段，不引入机械臂、脚/轮腾空、抓取或无依据能量奖励。

### 接口、兼容与诊断

新Actor输入280维：27字段×10帧+10mask。观测中的α、原始路径、速度参考、上一残差和回归状态均有显式顺序及归一化；checkpoint绑定新身份，不能用旧190/200维模型冒充。`tracking=None`、`include_tracking=False`和无speed_schedule保留旧身份。软件力矩/物理模型未改，前轮仍是rad/s转向角速度而非转角。

单条evaluate会自动重建14项有符号奖励并存`analysis/tracking/summary.json`与components.npz；面板提供配对XY、每步总奖励/分项、横向/航向/roll、目标/真实/轮速代理速度、估计误差、回归计时的PNG/PDF及索引。失败曲线在实际失败时终止。新恢复判断和旧command yaw诊断分开。

新开发选模必须无失败、末段共同保持、无超时，且名义速度RMSE不比配对基线高.05m/s、名义横向/航向RMSE不高.03（各自单位），成熟阶段速度/路径过程容忍超限比例均≤10%。这些只是声明的开发门槛，不是安全证明；不合格时best为空，流水线last仅作为标记清楚的诊断候选。

### 本轮没有承诺的结果

没有将PPO替换成RSL-RL，没有宣称16个大mini-batch一定优于512个小mini-batch；主要采样瓶颈仍需在目标GPU实测。没有改物理步长/接触参数换取速度。当前弯道/速度配置不是所有左右转与极端指令的覆盖；固定的原路径跟踪与恢复机制先通过工程测试，再另行做课程、消融、多训练seed和硬件状态估计验证。
