# STTW_CONTROL：单网络直接速度—转向参考策略 V3

版本 V3.0；配套 `STTW_Direct_Command_V3.json` 与 `STTW_Direct_Command_V3_reference.py`。
本文件是待实施规格，不是已训练结果；超参数为首轮可审计初值。V3明确替代V2参数网络方案，不同时实现两套学习器。

## 0. 唯一主线与本轮交付

一个共享Actor接收α∈{0,1}、车辆短历史、原始速度/转角指令及累计航向误差，直接输出两个参考修正，得到修改后的速度和转角，再通过原ECBC与执行器。

- 网络不输出Phi、侧倾参考界或航向反馈增益。
- 禁止解析α分配器、解析降速/减转向规则、q(v)参考投影、候选搜索、MPPI、世界模型、扩散、专家穷举和示范蒸馏。
- q(v)只允许用于原ECBC、观测中的原始平衡需求特征、奖励的原始指令分类；不得利用它修改网络动作。
- α同时影响Actor输入和奖励权重，不影响模型、安全阈值、动作范围、执行权限或观测质量。
- 不是纯电机策略：ECBC保留姿态反馈。网络输出的是速度m/s与转向角rad的参考，不是转向角速度或后轮转矩。
- 本轮从零训练一个共享网络，默认最多20次PPO更新；一次主场景与一次新随机指令配对展示后停止。20轮是pilot，不是预设收敛轮数。不得自动续训100/200轮。
- 不以“有两种可行解需要证明”为前置；只做必要接口检查，不重跑温和大面板。

## 1. 仓库与已有代码保护

已核对远端相关分支HEAD为`90d6a8435da1f1cbcec585e8cf51a9536ffdc053`，该分支记录的是旧动态ECBC实验，不是V3已完成。
读取本地AGENTS.md、git status、HEAD、现有teleop模块和正在运行任务。新建独立worktree/分支`experiment/direct-command-policy-v3`；若已有未推送teleop物理kernel、参考积分、日志，优先复用并记录实际父提交/路径映射。不删除、覆盖或reset已有工作，不停止其他任务，不自动push。

冻结XML、质量惯量、摩擦、物理步长、求解器、原ECBC/ESO系数；新任务`fixed_roll_reference=null`、`learning_roll_reference=null`。不改旧reward/config/checkpoint语义，不载入V1/V2/旧残差策略权重或优化器。

底层dt=0.005 s；策略dt=0.020 s；一Actor动作持有4个底层tick。原执行合同：additive，base_output_scale=1，strength=1，project_base=false；前轮角速度残差±1.5 rad/s、后轮轴角速度残差±10 rad/s，最终命令±3/±60 rad/s，关节转角±0.8 rad。原延迟、伺服与力矩限制照旧。工作侧倾0.30与真实失败0.70 rad、触地/非有限判据区分。读取值不符时报告，不默改物理。

## 2. 动作：网络最终决定速度、转向，不增加另一个决策层

Actor输出2维高斯潜变量`z=[z_v,z_delta]`；训练采样，评估用均值。令a=tanh(z)：

```
Dv_target = where(a_v >= 0, 0.25*a_v, 1.00*a_v)  # m/s，范围[-1,+.25]
Dd_target = 0.20*a_delta                        # rad，范围[-.20,+.20]
```

网络是相对**当前原始指令**作修正，不是相对上一次修改指令作累加。正速度补偿用于纠正真实速度跟踪误差，不等于允许真实超速；真实超速另计成本。

每5ms只对修正量进行常规变化率限制：
```
Dv_tmp = Dv_prev + clip(Dv_target-Dv_prev, -1.0*dt, +0.5*dt)
Dd_tmp = Dd_prev + clip(Dd_target-Dd_prev, -.8*dt, +.8*dt)
v_g = clip(v_c + Dv_tmp, 1.5, 3.0)
d_g = clip(d_c + Dd_tmp, -.35, .35)
Dv_prev = v_g-v_c    # back-calculation，避免限幅后的隐藏积分
Dd_prev = d_g-d_c
```
滤波状态是修正量，不是电机角度，不重写真实物理状态。初始化两个偏移为0。

这不是解析偏好分配：所有α使用同样的映射/限幅/滤波，决定让哪个通道以及让多少的是网络。
零输出、零初始滤波状态应对任意允许的原始指令保持原ECBC身份。不要对完整原始指令再加一个更慢的滤波导致零网络也改变基线。

## 3. 与ECBC后的旧残差权限兼容

从同一旧controller state、同一测量计算两次：
```
u_nom  = [ECBC(measured_state, raw_steer).steer_rate, raw_speed/R]
u_goal = [ECBC(measured_state, governed_steer).steer_rate, governed_speed/R]
a_motor = clip((u_goal-u_nom)/[1.5,10.0], -1,1)
final = apply_residual(old_actuator, u_nom, a_motor, actual_steer)
```
R沿用0.1m代理。两次ECBC内部速度都来自现有rear_rate*R，不能用v_c或v_g代替。ESO只提交一次；不得把第一次preview更新后的state传给第二次。旧`_prepare`若已经推进ESO，不得同时再用新kernel推进。

这里的2维电机残差只是输出接口转换，不是第二个策略。保留最终MuJoCo轮轴ctrl符号，不再对a_motor套tanh。记录u_nom/u_goal、requested_difference、bounded_additive_difference、final_command与clip标记。最终clip之后的actual-command变化不得与前级有界残差混为一谈。

## 4. 输入与网络结构

Actor：`345 -> 128 ELU -> 128 ELU -> 64 ELU -> 2 linear`。
Critic：`346 -> 128 ELU -> 128 ELU -> 64 ELU -> 1 linear`，独立参数；Actor输入后只加`remaining_policy_steps/800`。Actor已经包含α，Critic不要再重复拼一个α。

Actor推理参数69,186个，训练另有2个log_std。无GRU/Transformer/Dropout/BatchNorm/在线running normalization。隐藏层orthogonal(sqrt2)、bias=0；Actor末层weight=0、bias=0；Critic末层orthogonal(1)、bias=0。

20维帧×16=320，按从旧到新展平；再拼16个valid mask；再拼9个context，共345。history以50Hz更新，首末跨度0.30秒。每项x/scale后有限值裁至[-5,5]，统计裁剪比例；布尔和mask保持0/1。非有限观测或网络动作是policy_fault，不能改零假装正常。

|frame索引|字段|scale|
|---:|---|---:|
|0|侧倾phi|.3|
|1|侧倾角速度|1.5|
|2|实际转角|.35|
|3|实际转向角速度|3|
|4|前向速度估计|3|
|5|世界偏航速度估计|2|
|6|后轮正向线速度代理|3|
|7|前轮正向线速度代理|3|
|8|原始限速后v_c|3|
|9|原始限速后delta_c|.35|
|10|原始v_c导数|.8|
|11|原始delta_c导数|.45|
|12|上一真实执行周期的v_g|3|
|13|上一真实执行周期的delta_g|.35|
|14|上一最终前轮角速度命令|3|
|15|上一最终后轮轴角速度命令|60|
|16|上一前级已限幅前轮附加残差|1.5|
|17|上一前级已限幅后轮附加残差|10|
|18|ECBC可获得的ESO等效侧倾修正|.3|
|19|原始指令对应phi_raw=delta_c/q(v_c)|.3|

以上previous字段不能泄漏本次Actor将选的动作。第16/17项是final actuator clip之前已通过1.5/10限制的附加量，不是最终命令减去未限幅baseline；后者可能超此范围。

Context顺序：`alpha/1, epsi_unwrapped/pi, sin(epsi), cos(epsi), chi/1, g/1, settle_clock/.8, Dv_prev/1, Dd_prev/.2`。

前向速度/航向首轮声明`simulation_state_assisted`：使用仿真现有world velocity前向投影与pose yaw，未来实机需相应估计器。Actor不访问未来schedule、场景ID、绝对XY、真实外扰、回合时间；这是有历史的部分观测策略，不冒称完全Markov。长期欠账用积分器，不要求0.3秒历史记住几秒前动作。

## 5. 原始参考与奖励上下文（不生成控制动作）

原始输入是目标速度型油门v_c和转角delta_c，不是转矩油门或独立yaw-rate。
外部目标先按每回合固定的slew逐5ms更新，得到实际下发c_k。未来随机目标由环境测试器持有，policy只能读当前已发布指令和过去变化。

参考用`omega_c=v_c*cos(caster)*tan(delta_c)/L`，区间恒定twist精确积分：theta=omega_c*dt；XY加`v_c*dt*sinc(theta/2)*[cos(psi+theta/2),sin(...)]`，sinc=sin(x)/x；yaw加theta。只在任务开始把参考设为共同实际初态，过程中不重定位。实际yaw用相邻wrapped增量累计；epsi=psi_reference_unwrapped-psi_actual_unwrapped。XY只记录，不参与目标。

q(v)复用现有controller ratio；在正向范围未触发floor时`q(v)=A/(v²-B)`，A=gL/cos(caster)，B=cg_forward*trail*g/cg_height。用于奖励分类时v=v_c；不得把这个分类当作“已证明真实危险”或用它投影动作。

原始指令冲突程度：
```
phi_raw = delta_c/q(v_c)
chi = clip((abs(phi_raw)-.18)/(.30-.18),0,1)
```
这是固定指令需求指标，不是学习的可行性判定。真实姿态风险独立计成本。

航向评价权重仅由过去原始指令计算：
```
eligible = abs(delta_c)<=.05 and abs(phi_raw)<=.18 \
           and abs(dv_c_dt)<=.10 and abs(ddelta_c_dt)<=.02
settle_clock = min(.8, settle_clock+dt) if eligible else 0
g = clip((settle_clock-.4)/.4,0,1)
```
唯一时序：在区间开始使用已发布c/导数/clock计算chi、g；区间结束用本区间eligible推进clock，然后发布下一c。若新发布指令使eligible=false，即使旧clock=.8，本区间g也立即为0，next_clock=0；eligible=true时才按旧clock计算当前g，并推进next_clock。Actor与reward对同一个区间使用同一g，不能在二者之间重复推进时钟。mask/reset测试覆盖边界。

g只是训练/观测中的航向评价阶段权重，不是控制开关、不执行任何kpsi反馈，不根据实际epsi大小、是否让步、网络动作或模型预测改变。这样网络无法通过“不承认进入恢复”逃避航向成本。
先等待0.4秒再0.4秒渐增，给普通转向瞬态留出空间；第一轮训练的充分恢复窗口是低曲率/回正后的平稳指令，不承诺持续快速变化时也能消除历史误差。小偏差有0.03rad死区。

## 6. 精确奖励：全部基于真实运动对原始请求

定义`H(z)=z² if |z|<=1 else 2|z|-1`，所有成本非负；系数是成本/秒，不能每个5ms直接扣整项。每个区间使用本区间原始c和alpha，与该区间推进后的物理状态比较。

```
ev = actual_forward_speed - original_speed_command
ed = actual_steer - original_steer_command
ep = reference_yaw_next_unwrapped - actual_yaw_next_unwrapped
wv = 8.0 - 7.2*chi*(1-alpha)
wd_base = 8.0 - 7.2*chi*alpha
wd = (1-g)*wd_base + g*1.0
```

|项|计算公式|目的|
|---|---|---|
|速度跟踪|`wv*H(ev/.10)`|α1冲突时强保护，α0仍有非零代价避免过度降速|
|转角跟踪|`wd*H(ed/.05)`|α0冲突时强保护；恢复时松开原转角要求以允许补航向|
|航向恢复|`4*g*H(max(abs(ep)-.03,0)/.15)`|不追时间位置；不奖励恢复开关本身|
|侧倾工作范围|`40*H(max(abs(phi)-.26,0)/.04)`|给0.30工作范围提前代价；不处罚正常倾斜转弯|
|侧倾角速度|`1*H(max(abs(phi_dot)-.8,0)/.8)`|抑制剧烈倾倒，不把所有正常roll rate压成0|
|额外超速|`2*H(max(ev-.05,0)/.10)`|防止α0利用低速度权重长期超速；小正参考补偿仍可用|
|低速|`4*H(max(1.5-v_actual,0)/.20)`|防止停车/近停替代任务完成|
|参考修正正则|`.02*((Dv_executed/1.0)^2+(Dd_executed/.20)^2)`|弱约束，不压制必要让步；Dv=本步v_g-v_c|
|最终命令变化|`.10*sum(((u_final-u_final_prev)/[3,60])^2)`|约束最终合成命令，不只约束网络潜变量|

C_raw为以上之和，`C=min(C_raw,100)`。正常5ms奖励=`-.1*dt*C`，一个50Hz转移奖励为四个区间之和。无alive、无一次恢复奖金、无额外最终位置/航向罚、无离带倒计时收紧、无减速正奖金、无跨α动作差异奖金、无固定侧倾跟踪罚。

全成本cap只用于极端状态及失败计分有界化；保留每项原始成本、有效成本与cap_fraction。有效分项按相同factor=min(1,100/max(C_raw,eps))缩放，仅为总和可重建，不改变实际reward定义。不能声称cap完全没有影响：若普通/恢复稳定段cap_fraction>5%，标reward_resolution_warning并报告，不自动重调系数或把奖励上升解释为成功。

偏好是有限权重软优先，不是严格词典序保证。网络直接输出，不允许偷偷用解析规则强行制造两端曲线分离。

## 7. 失败、有限时长与bootstrap

任务定义为有限16秒，50Hz恰好M=800个区间。Critic可见剩余时长，Actor不见。真实物理失败沿用0.7rad/触地/非有限等判据，不把0.30工作界直接改成终止；工作越界单独统计。

正常区间奖励下界是`-s*Dt*Cmax=-.1*.02*100=-.2`。若第k个策略区间发生失败（k从0计），N=800-k包含当前区间，**用以下值替换整个当前策略区间奖励**：
```
R_failure = -5 - .1*.02*100 * (1 - gamma**N)/(1-gamma)
```
之前已完成区间奖励不更改，本区间普通分项清零并记为failure_cost；余下仿真不实际执行。该计分把剩余任务视为高成本吸收状态，避免策略用提前结束逃避后续负成本。不能再叠旧-200，不能再加当前普通reward或终端bootstrap。

这里gamma=.997，最早失败约-65.64，最后区间失败-5.2，精确数值由脚本计算。它只处理计分激励，不是安全定理。

- 物理失败：terminated=true，bootstrap=0；4个底层tick内首次失败后冻结该环境到此策略步结束，不在同一Actor动作内切到新episode。
- 16秒任务正常结束：finite_task_end，terminated=true，bootstrap=0；不要按旧时间截断补gamma*V。
- 128步采样片段结束但episode未结束：保持物理/历史继续，正常bootstrap。
- 外部预算中断：不得把人为截断伪造失败或成功；未完成优化的rollout不强行publish新checkpoint。
- final observation必须在reset之前保存。物理非有限按失败记录，神经网络/优化器非有限停止运行并保留最后有限模型。

## 8. 训练工况：第1轮起同时覆盖跟踪、冲突和恢复

所有episode16s，无外扰，无摩擦/质量随机化，无倒车，不扩大到低速平衡课题。每回合独立抽speed_slew~U[.3,.8]m/s²、steer_slew~U[.15,.45]rad/s，回合内固定；目标幅值v∈[2,2.6]、delta∈[-.30,.30]。
抽样由JAX PRNG执行，seed66001，env_id与episode_index fold_in；随机命令发生器只在环境侧。全程保存case manifest，类型只作日志，不输入policy。

固定混合比例：25%温和、50%有限冲突、25%连续随机。不是先用全部预算训练直行，再从旧checkpoint补训急弯；本轮不改变混合比例，避免回报因分布变化失去可比性。

### 8.1 温和族（25%）
任务开始原始命令=prepared实际设定速度、0转角。第一次目标变化U[.5,1.5]秒，目标速度U[2,2.6]；每次转角有40%概率0，否则U[-.08,.08]。目标保持U[1,2]秒，重复至t<8。t=8统一目标(2.3,0)，保持到16。
目标：普通工况尽量跟踪、不要为了α差异人为减速；尾段可校正小偏差。

### 8.2 有限冲突族（50%）
任务t=0目标(v_hi,0)，v_hi~U[2.4,2.6]，真实速度通过共同slew升高；turn_start~U[1,2]。
转角幅值a~U[.22,.30]、符号均匀±1。
80%：turn_start目标(v_hi,sign*a)，保持U[2,3]秒，然后回正。
20%：先保持sign*a，时长U[1.5,2.5]；再请求-sign*a，保持U[1.5,2.5]；然后回正。
回正目标速度再抽U[2,2.4]、转角0，保持到16秒。
所有目标仍经过slew；反向目标不一定在有限保持期内完全达到，必须以实际下发delta_c记录，不把scenario标签当“发生了最大反向”。最大tail_start=7秒，最慢回正可到9秒，留下恢复时间。
目标：固定同一网络学习两端取舍、预见可见的指令变化趋势、冲突后的航向恢复。

### 8.3 连续随机族（25%）
第一次目标U[.5,1]秒，每次独立抽v~U[2,2.6]、delta~U[-.30,.30]、保持U[.8,1.6]秒，重复至t<8；t=8目标(U[2,2.6],0)，保持到16。
目标：不依赖特定弯道或固定冲突时刻；给动态流配备明确恢复机会，而不是要求永远冲突又必须即时消除欠账。

## 9. α采样与初态

512环境前256固定α0，后256固定α1；训练全程一个共享Actor。每条episode内α不变，其他值配置报错。初始命令种子可以成对，但任一环境失败后各自reset，因此训练不是始终严格配对，不能虚称每条随机轨迹一一配对；严格配对只用于评价。

建立8个真实prepared snapshot：初始(v,roll)见JSON，v为2.0/2.3/2.6，roll为0或±.01；用原动态ECBC零修正运行3.5秒，让原ESO启用规则完成。保留完整物理积分状态、ESO、执行器和真实历史；不只存qpos/qvel。
准备末端无失败、|phi|≤.05、|phi_dot|≤.1、速度误差≤.15、|delta|≤.03。失败报告初始化问题，不静默延长/拼理想状态。
每次reset从bank抽完整状态，任务参考在该实际pose初始化，策略修正滤波=0、context时钟=0；不重置已准备的ESO。历史可保留真实prepare帧，或不足帧标mask0，不复制同一帧伪造历史。
禁止旧phase-spread百万步预热；bank共5600个底层控制步，初始化成本单独列账。

## 10. PPO精确设置与旧接口改造

- 使用现有RSL-RL 3.2.0/PyTorch与MJX采样栈，不升级依赖，不写第二套全功能训练框架。
- 新task-local adapter采用2潜变量，但与旧2电机动作**语义不兼容**；旧export、tanh和identity不能直接沿用。新identity绑定观测字段、动作映射、dt、奖励、模型。
- num_envs=512，num_steps=128，一轮65536策略转移，最多262144底层步；epochs=4，minibatches=4，batch_size=16384。
- Actor与Critic分离；Adam actor lr=1e-4（含log_std），critic lr=3e-4，betas(.9,.999)，eps1e-5，weight_decay0。固定学习率；关闭会覆盖所有param_group lr的adaptive schedule。
- gamma=.997，GAE lambda=.97，clip=.2，value_loss_coef=.5，value clipping off，entropy_coef=.001。
- 初始潜变量std=[.2,.2]，学习log_std并限制std∈[.05,.5]；先投影std，再计算新分布KL。部署使用mu；不采样。
- 全batch advantage标准化一次，actor/critic梯度范数分别clip1.0，不用critic大梯度共同缩小actor。
- 每epoch之后在本轮所有采样观测上计算old/new高斯精确mean KL及分α KL；mean>.01就停止本轮剩余epoch；mean>.03或非有限时恢复本epoch前模型/优化器并停止pilot，标优化步长异常。不得反复重试同一批到“通过”。实际优化epoch数量如实记录。
- PPO storage保留采样前tanh的z与其old_logprob、old_mu/std；概率比在同一潜空间计算。不能对有限幅后的v_g/delta_g算Gaussian概率，不能将±1电机残差当PPO action。
- JAX物理每个策略步内部scan4个控制tick；与Torch交互以batch为单位，避免逐环境/逐子步CPU传输。
- Actor权重导出到Flax时隐藏激活、矩阵转置、末层和动作映射都绑定新合同；导出前后mu在同一批观测上比较，不能只比较最后动作clip后“都饱和”。

物理动作限速和有界输出不意味着安全；KL限制也不是物理风险保证。

## 11. 本轮实施顺序与计算预算

(1) 阅读本地代码与记录身份；(2) 实现新观测、直接动作映射与独立reward；(3) 8环境×16策略步×2更新短测；(4) 全新初始化共享Actor，最多20次更新；(5) 用最后完成checkpoint展示主场景与一条新随机序列；(6) 停止、报告、本地commit。

训练第5/10/15/20轮保存checkpoint与TensorBoard；第10轮报告真实梯度、权重变化、分项与分α统计，不自动插入额外完整面板。不要按混合平均训练reward挑一个幸运checkpoint当最终结果，默认统一展示last_completed_update。原reward-best可作为候选文件但不替换约定展示模型。

硬预算：编译300s，准备/短测120s，pilot训练1200s，展示180s；累计新增计算1800s，先到任何对应上限即保存并停止。单完整展示回合60s上限。默认训练最多1,310,720策略转移、5,242,880底层步（失败冻结后实际物理步可少），另计工程与8状态准备。代码编写时间不计计算预算，但计算等待不能伪称编写时间。

阶段预算不是必须用满；超预算不自动调小物理精度、改dt、升级引擎或开新账本。没有GPU或512环境不适合时报告硬件阻断，不能静默在CPU跑几小时。

20轮只诊断学习是否有信号，不足以否定方法或宣称收敛。进一步实验计划可提出同配置累计100轮、后续独立种子，但本轮不自动执行；续训时必须恢复完整优化器/RNG并声明环境重新初始化边界，不做跨任务checkpoint接续。

## 12. 最小必要测试，不重复全库长仿真

纯函数测试：α输入严格0/1；345/346输入维度和字段；动作单位/映射/clip/back-calculation；零输出任意原始slew下不改变原指令；未展开参考航向与sinc；奖励分项重建；合成偏好排序；g仅依赖raw历史；失败吸收界/最后一步/episode与rollout终止；导出mu一致；同状态改变α仅改policy输入和reward、不改limits。

真实模型只做：零策略直行与温和转向各最多4秒短回放，确认ECBC符号、一次ESO、相同原始/修正参考时零残差；不要求原ECBC通过高难冲突。8环境短测确认Actor与Critic有真实更新、PPO存储z、gamma/四子步正确、无未来指令泄漏。可以运行现有快速CPU单元测试防回归，不重跑昂贵历史训练/大面板。

合成成本测试（chi=1,g=0，忽略相同的共同项）：
A：ev=-.40,ed=.01；B：ev=-.04,ed=-.08。alpha0成本A=5.92、B=17.728；alpha1成本A=56.032、B=3.04。只证明评分方向，不称物理解存在性实验。
恢复测试：同速、同安全条件，允许少量delta偏离但epsi明显下降的样本应能优于坚持delta=0而保留大欠账；测试不生成强制恢复动作。

## 13. 展示：默认6条完整回合，不做大面板

主场景原始目标：t0=(2.6,0)，t1.5=(2.6,+.25)，t4.5=(2.6,0)，任务16秒，speed_slew=.5、steer_slew=.3。共同prepared speed2.3、同一完整状态、同一指令，运行B0、共享网络α0、共享网络α1。

新随机展示：NumPy PCG64(88001)，初始目标(2.3,0)，先抽first_switch~U[.5,1]，每段依次抽v~U[2,2.6]、delta~U[-.30,.30]、duration~U[.8,1.6]至t<8，t8目标(2.3,0)至16；speed/steer slew仍.5/.3。先冻结CSV再3方法复用，不能根据结果改seed。与训练集PRNG及随机种子隔离。

原始B0不是固定侧倾版本；可以复用身份相同的既有B0轨迹，模型/config/初态/输入任一不同就不混拼。

主冲突统计固定窗口[2.5,4.5)，检查实际转角与实际速度，不拿修改后目标计算“优越跟踪”。比较G0/G1：期望E_delta0<E_delta1，E_v1<E_v0；建议明确优势门槛为主误差比≤.85，同时绝对差delta≥.01rad、v≥.05m/s；主目标绝对质量delta0 RMSE≤.05、v1 RMSE≤.15。实际G0相对[1.0,1.5)平均速度下降≥.15m/s且持续.2s，G1平均转角幅值相对raw小≥.03。没有达到就如实报告，不用更换窗口/奖励制造通过。

共同安全门槛：无真实失败；工作界0.30，展示数值容差0.002，仅作指标判断不改物理或reward阈值。
恢复：原始转角降至|delta_c|≤.005之后计时，到episode末寻找|epsi|≤.05、|ev|≤.10、|ed|≤.04连续.5s；报告最后2秒RMSE与首次恢复时间。不能以policy自行宣告恢复开始来拖延计时。XY只画不判位置回归。

随机流普通段两端相近可正确；不要求所有时刻人为拉开。若主指令基线未表现冲突，只报告not_informative_case，不要求网络故意牺牲，不自动加大指令直到失败。

只有6条不代表泛化；后续需镜像左右、独立种子、未见slew和更多输入序列，但本轮先交付能判断方向的图。

## 14. 日志、失败定位与接口文件

每轮统计分alpha、分任务族、普通/冲突/恢复窗口：actual速度/转角RMSE、raw/effective各成本、侧倾峰值/越界时长、失败率、恢复误差、动作/潜变量std、ref/motor限幅率、cost_cap_fraction、value/advantage、actor/critic梯度、mean及分α KL、真实学习率、采样/优化/编译时延。

每个完整展示回合逐5ms落盘raw target/c、proposal/g、实测v/delta/phi/phidot/yaw、累计参考与actual heading、epsi、chi/g/clock、两个filtered offset、潜变量均值、nom/goal/final控制、前级/后级clip、reward、终止码。至少每1秒仿真时间增量保存，异常输出partial，不把所有轨迹只留内存。

同图B0/α0/α1：XY、速度、转角、侧倾、航向误差、两通道实际残差；不加入不存在的方法图例。参数网络Phi/kpsi曲线彻底取消。另做同一批真实观测仅替换α的mu/action响应诊断（无新物理rollout），观察条件是否被利用；动作差异不是成功判据。

最小新增或复用映射建议：
- `direct_command_env.py`：独立任务/四子步/参考与cost时序；复用已有physics kernel。
- `direct_command_policy.py`：345/346网络、潜变量到参考偏移；不借用V2三参数映射。
- `direct_command_reward.py`：单一可审计cost实现，CPU/MJX/离线共用公式。
- `direct_command_ppo.py`：任务适配、独立lr、epoch KL、有限时长返回、导出。
- `direct_command_scenarios.py`：三族因果输入。
- 新config、短测/训练/展示CLI；旧模块不改含义。不因名字不同再写第二份同功能实现。

CLI至少支持`--config --output --updates --compute-wall-budget --dry-run`；默认updates20且硬预算，不提供默认all巨大评估。无参数解析歧义、不将v/delta误读为v/yaw。schema拒绝未知字段，Python参考脚本只作数学校验，不直接作为环境代码运行。

结束交付README、真实源码commit/配置hash/模型hash（各一次）、status/budget、checkpoint/optimizer/RNG、观察与动作schema、测试摘要、主场景与随机对比图、指标和未完成项。本地commit，不自动push。

常见诊断顺序：alpha条件被忽略→训练采样/奖励计算检查；所有policy停滞→权限/限幅/低速代价；都不修正→潜变量探索/梯度/旧映射误用；高reward但摔→失败计分/成本cap/安全成本；不恢复→训练尾段/g/累计yaw/ed成本未松开；策略步训练正常但展示极慢→设备/JIT/传输，不能回到完整模型搜索。

## 15. 为什么这些选择与本任务匹配

相对raw输出让零网络就是基线；两通道都可改而不是alpha硬锁通道。非对称速度范围允许必要降速和小正补偿；转向仍可正负调整。历史MLP提供短期动态，显式航向负责长期欠账。原始指令指标只调奖励，不限制动作，避免再引入参数分配层。普通阶段双目标强、冲突阶段10:1偏好、稳定回正后放松瞬时转角而补航向，分别对应用户的三个要求。安全成本独立、动作权限一致，但没有理论安全或词典序保证。

所有数值待短训证据检验，不能承诺“按本文件就必然成功”。需要后续改单项时新版本、新身份、说明证据；不在本轮边跑边偷偷调参。

## 16. 方法与接口来源（不作为具体系数已验证的依据）

- PPO原始论文：https://arxiv.org/abs/1707.06347
- 有限任务终止与截断：https://gymnasium.farama.org/main/tutorials/handling_time_limits/
- JAX计时与同步：https://docs.jax.dev/en/latest/benchmarking.html
- 本仓库当前`controller.py`、`actuator.py`、`rsl_training.py`，读取基准`90d6a84`。已有RSL桥仍对旧几何任务和旧2电机动作绑定，动作维数相同不代表语义兼容。

方法来源仅支持通用训练/接口原则。本文动作范围、任务分布、奖励、恢复评价门和预算均为这次明确的设计提案。
