# STTW_CONTROL 当前项目说明

整理日期：2026-09-26。本文只维护当前有效状态与入口。旧实验完整过程见[方法台账](docs/METHODS_AND_RESULTS.md)及Git历史；重写前的839行版本保存在提交`3e48a6f`中，可用`git show 3e48a6f:PROJECT.md`查看。

想先看普通话总结，打开[最近几次实验报告](docs/RECENT_EXPERIMENT_REPORT.md)。操作规则见[AGENTS.md](AGENTS.md)。

## 1. 我们在解决什么

小车使用完整ECBC＋ESO维持基础控制，强化学习只加有限修正。希望α=0更重视原始几何路径、必要时降速；α=1更重视真实速度、允许临时偏离。所有α都必须在规定时间内回归，并满足共同最终保持要求。

当前使用几何路径，允许沿路径暂时落后。网络没有未来随机指令，α不改变动作权限。普通场景不强迫三条轨迹不同。当前尚无模型通过全部声明开发条件。

## 2. 当前用哪个目录

| 工作区 | 分工 |
|---|---|
| `/home/qy/STTW_CONTROL` | 原工程与早期实验，包含软预算V1原0244模型 |
| `/home/qy/STTW_CONTROL_mode_isolation` | 继承原0244 Actor的三个独立α实验，以及20轮和250轮复核 |
| `/home/qy/STTW_CONTROL_priority_v2` | **当前工作区**：V1/V2冷启动比较、补评与本轮200更新训练 |

当前分支：`experiment/priority-return-v2`。原目录和旧工作区未搬走，不在此静默合并它们。不同工作区的源码版本和运行目录分开核对。

Python：`/home/qy/mujoco_playground/.venv/bin/python`。运行CLI时从当前工作区设置`PYTHONPATH="$PWD/learning/src"`。本项目不管理JIT或Isaac实验。

## 3. 已启动的唯一新正式训练

**运行：[`priority_v2_cold200_repaired_20260926`](runs/priority_v2_cold200_repaired_20260926/INDEX.md)。**

本轮最初为修复评估程序后的同条件V2完全冷启动，未加载原0244、独立模式或旧V2第100轮。2026-09-26检查发现进程已退出：本轮第100轮快照已保存，第100轮开发评估未完成，逐轮日志写到99。退出原因尚未确定，不能归因于旧离线精度错误。用户授权恢复本轮完整第100轮快照，再做100更新，总计200；奖励、物理和停止门槛不变。

当前恢复入口：[attempt_0001](runs/priority_v2_cold200_repaired_20260926/recovery/attempt_0001/INDEX.md)。恢复Actor、Critic、Adam、std和RNG；车辆重新闭环准备、错相初始化，不称逐位不中断续跑。原始快照和中断记录保留。

| 项目 | 冻结设置 |
|---|---|
| 网络 | Actor/Critic均128→128→128、ELU；两路tanh输出 |
| 基础控制 / 残差 | ECBC＋ESO输出1.0；前轮±1.5rad/s，后轮±10rad/s |
| α | 每回合等概率抽取0/0.5/1，回合内固定，Actor/Critic都可见 |
| 时序 | 200Hz；正式回合10秒；闭环准备3.5秒，准备状态完整保留 |
| 观测 | 10帧历史；字段/归一化/物理身份绑定模型；新增速度字段当前用后轮转速×0.1m |
| PPO | RSL-RL 3.2.0；1024环境×128步；200更新；seed66 |
| 优化参数 | 2 epochs，minibatch32768，固定学习率1e-4，gamma .9995，GAE .99，clip .1，entropy .001，初始std .1 |
| 正式训练预算 | 26,214,400控制转移；准备、错相初始化和评估另记 |
| 模型检查 | 第0/100/200轮完整固定场景；不按训练片段奖励选择合格模型 |
| 停止条件 | 错误；连续三次整轮KL回退；三个开发检查点的物理失败或路径指标持续恶化；预算结束 |

- [冻结任务](runs/priority_v2_cold200_repaired_20260926/frozen/task.json) / [冻结PPO](runs/priority_v2_cold200_repaired_20260926/frozen/ppo.json) / [声明](runs/priority_v2_cold200_repaired_20260926/declaration.json)
- [恢复实时状态](runs/priority_v2_cold200_repaired_20260926/recovery/attempt_0001/training/status.json) / [恢复记录](runs/priority_v2_cold200_repaired_20260926/recovery/attempt_0001/training/resume.json) / [恢复启动来源](runs/priority_v2_cold200_repaired_20260926/recovery/attempt_0001/launch.json)
- [原中断状态](runs/priority_v2_cold200_repaired_20260926/training/status.json) / [原逐轮日志](runs/priority_v2_cold200_repaired_20260926/training/metrics.jsonl)；这是保留的历史现场，不再作为实时状态。
- [开发选模](runs/priority_v2_cold200_repaired_20260926/training/model_selection.json) / [监视器](runs/priority_v2_cold200_repaired_20260926/monitor/status.json)
- [TensorBoard](http://127.0.0.1:6006)：选择`resume100/training/tensorboard`查看恢复阶段，`repaired200/training/tensorboard`查看原阶段；服务记录在[tensorboard_service.json](runs/priority_v2_cold200_repaired_20260926/tensorboard_service.json)。

启动源码是`2c1e34a`；后续文档提交不冒充训练启动版本。第0轮有35条旧初始评估轨迹经完整参数、任务身份和奖励重建核验后复用，其余7条正常生成；见[复用记录](runs/priority_v2_cold200_repaired_20260926/initial_evaluation_reuse.json)。这只节省评估计算，没有继承训练权重。

本文不持续写入易过期的轮数；以status、metrics和checkpoint交叉核对。恢复日志单独保存，并继承原第0轮开发结果、补完第100轮检查，仍在总第200轮做终评。训练奖励缺失的第100轮不补造。

## 4. 当前训练环境的实际范围

`priority_v2_screen=true`时，正式训练抽取**三种固定模板**：

| 比例 | 模板 |
|---|---|
| 30% | 普通加速：速度参考从2.3到2.5m/s，保持直行 |
| 35% | 核心左转：1秒请求2.5m/s、+1.8rad/s，1.95秒撤销转弯请求 |
| 35% | 核心右转：上项左右镜像 |

普通参考变化率为速度0.5m/s²、yaw 0.6rad/s²；核心冲突分别1.0m/s²、2.4rad/s²。这是参考变化率，不是机械加速度限制。任务开始时的实际车速以准备状态日志为准，不能直接用参考2.3m/s代替实际值。

它不是早期1.7～3m/s全范围随机指令训练，也不是早期40%名义＋40%冲突＋20%外扰混合。当前训练没有侧向扰动；侧扰动与附件单弯属于扩展开发测试。配置中的宽范围字段不代表这个模板分支实际遍历了那些组合。

配置中`timed_reference.mode=time`表示参考的生成方式；当前`tracking.geometric=true`，奖励仍按几何路径而非时间沿程位置评分，不能只看mode名称判断任务。

源码入口：[timed_reference.py](learning/src/sttw_control/timed_reference.py)。原始请求经平滑后积分生成固定路径，投影不会跟随车辆平移，也不向Actor泄漏尚未发布的未来指令。

## 5. 当前V2奖励与回归规则

权威实现：[priority_return_v2.py](learning/src/sttw_control/priority_return_v2.py)。旧V1保留在共享tracking奖励模块，不能用新规则改写其原始回报。

- 路径/速度权重：α=0为8/.1，α=.5为4/4，α=1为.1/8；保留共同预算、姿态、动作及平滑成本。
- 普通成本不使用V1的总成本饱和映射，控制周期只乘一次；物理失败整步-200，任务未完成的一次性罚为-20，不设恢复正奖金。
- 当前已发布转弯请求触发机动阶段。已生效参考曲率≤.05、速度参考变化率≤.1并连续稳定.25秒后确认出口，不以原始请求撤销时刻替代。
- 截止时间为出口确认后3秒，最多到回合结束。到达出口弧长或出口确认后1秒时开始收紧容忍；最终.5秒前收紧完成；新指令不能清除旧债务。
- 共同最终要求：速度误差绝对值≤.05m/s，横向误差绝对值≤.10m，航向误差绝对值≤.15rad，侧倾绝对值≤.30rad，侧倾角速度绝对值≤.30rad/s，连续保持≥.5秒；发生机动时还需走到出口后至少1米。
- 工程合格要求完整回合、无物理失败、未错过截止时间且最终完成；严格合格另查全程侧倾绝对值≤.30rad、超速≤.05m/s。最终保持、按期回归、全过程要求不能互相替代。
- 10秒是有限任务终点，当前V2不在正常结束后继续自举未来价值。速度奖励对齐动作前已生效的外部参考。

V2与早期“不对称速度”方案不是同一套公式；不能把旧priority_ratio=34.2、旧速度带或旧奖励best要求默认为本轮规则。所有版本按冻结配置解释。

## 6. 开发评估与选模

入口：[priority_v2_campaign.py](learning/cli/priority_v2_campaign.py)，六场景为：ordinary_accel、core_left、core_right、disturbance_left、disturbance_right、synthetic_turn。

每个checkpoint有18个确定性策略条件（六场景×三个α）、18个固定动作噪声序列诊断和6条共享物理基线；seed49001。固定噪声序列不能解释成随机策略失败概率。当前评估实际由Priority V2入口执行，不能根据旧通用`development_scenarios`列表误报只测四场景。

混合α按等权平均路径成本Jp、速度成本Jv选择诊断候选；共同合格模型为空时明确写null。比较范围是已评估的第0/100/200轮，不是所有模型。历史训练reward-best与当前开发选模分开。

重复评估已有模型时先核对缓存身份和来源，再运行有限声明的范围。示例：

```bash
cd /home/qy/STTW_CONTROL_priority_v2
export PYTHONPATH="$PWD/learning/src"
JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python \
  learning/cli/priority_v2_campaign.py --stage evaluate --execute \
  --task runs/priority_v2_cold200_repaired_20260926/frozen/task.json \
  --checkpoint /实际需要复核的模型目录 \
  --output /新的评估输出目录
```

不要直接复用旧错误目录，也不要把此示例当成自动再启动评估的指令。历史五场景15条件面板仍保留，需明确指定时单独运行，不与当前六场景混算。

## 7. 已有结果：哪些可以确定

| 运行 | 状态与可靠结论 | 来源 |
|---|---|---|
| 原软预算V1，250轮 | 奖励best为0244；单弯纠偏明显，α分化弱，超速/侧倾工作带不合格 | [原0244诊断](/home/qy/STTW_CONTROL/runs/soft_budget_ecbc1_20260920/single_turn_review/diagnosis/REPORT.md) |
| 三独立α各250轮 | 继承0244 Actor；最终保持5/9、按期且保持2/9；全程工作带0/9 | [独立模式报告](/home/qy/STTW_CONTROL_mode_isolation/runs/mode_isolation/analysis/REPORT.md) |
| 三独立α第20轮补评 | 没有新训练；最终保持9/9，但按期且保持3/9，核心左右转仍超时 | [20轮报告](/home/qy/STTW_CONTROL_mode_isolation/runs/mode_isolation/analysis/UPDATE20_REPORT.md) |
| V1冷启动200轮 | 200更新已执行；第100轮Jp6.945/Jv.746、3/18通过，第200轮10.161/1.557、2/18通过；后期退化 | [V1报告](runs/priority_v1_v2_cold200_20260921/v1/analysis/REPORT.md) |
| 旧V2冷启动 | 第100轮模型已保存，评估报错退出；原日志99条，不是完成200轮 | [原状态](runs/priority_v1_v2_cold200_20260921/v2/training/status.json) |
| 旧V2第100轮补评 | Jp7.082/Jv.932，工程和严格均3/18，仅普通加速通过；暂未优于同100轮V1 | [补评报告](runs/priority_v2_review_20260926/analysis/REPORT.md) |
| 新V2同条件200轮 | 冷启动已保存100轮后进程中断，按用户指令恢复剩余100轮；最终能力尚未确定 | [运行入口](runs/priority_v2_cold200_repaired_20260926/INDEX.md) |

独立模式的9个条件与新对照的18个条件不同，不直接比较成功百分比。原0244训练环境、观测及选择方式也与后来的V1冷启动组不同，不能将跨代差异都归因于奖励。

## 8. 本次修复与验证

旧V2第100轮的离线重建按双精度累计弧长，在线为float32。约3.5e-5m的差别使精确跨越回归起点晚一控制步，触发状态审计错误。提交`2c1e34a`对齐离线累计精度和记录时间，未改在线奖励、物理或门槛，未放宽审计阈值。

相关31项测试通过；第100轮完整补评完成，18条确定性轨迹的奖励重建最大绝对误差约9.4e-7。新第0轮缓存复用也经任务和完整模型参数核对。详细范围见[VALIDATION.md](docs/VALIDATION.md)、[补评验收记录](runs/priority_v2_review_20260926/review_receipt.json)。

已知状态呈现限制：在最终开发检查触发趋势停止时，即使200更新预算已用完，status仍可能是stopped/complete=false。V1正是这一情况；必须看更新数和stop_reason。初始化末期首次编译期间，status也可能暂时保留phase_spreading，以进程、评估输出和新标量辅助核对。

## 9. 下一步及未执行事项

1. 继续已授权的新V2到声明预算或停止条件，不再开并行正式组。完成后按同场景比较0/100/200，保留负结果，不强迫最后一轮获胜。
2. 若仍无改善，优先考虑只更换新增速度输入的单因素诊断，区分轮速估计与真实速度影响；尚未启动true-speed oracle正式训练。
3. 随后再分别检查恢复阶段学习信号和共同要求的可实现性。没有证据支持同时扩大网络、动作权限和奖励权重。

旧A/B/C各16轮、每模式80轮、仅α=0、连续α、五场景15条件或训练reward-best等条款属于各自历史方案，不能自动恢复为当前配置。早期48候选与有限失败边界检查仍保留原证据；不把用户后来授权冷启动200轮理解为这些检查已经全部通过。
