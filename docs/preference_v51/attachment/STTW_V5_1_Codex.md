# STTW V5.1：主目标超容差强惩罚＋完整航向恢复

## 0. 本轮决策（替代V5的两阶段运行安排，不替代控制架构）

证据基准：`QaQaaa-zzz/STTW_CONTROL`，`experiment/r196-preference-v5`，提交`6a264f77468cc4090f8e79919027b333740e5dfa`，训练实现`68996da`。证据目录`docs/evidence/r196_preference_v5_20261009`。

本轮不是V4/SmoothV4，不应拿旧裁剪比例、旧底层、旧250轮结论解释当前结果。V5实际为Stage1 40更新＋Stage2再80，累计120/端；Stage1 gate未通过后，用户明确授权进入Stage2。两端独立。当前已证实alpha0能持续降速且保留更多转角；不足是alpha1速度及侧倾、两个端点的航向恢复和普通工况干预。

保留R196（lower_alpha=1）＋ECBC/ESO、现有`lower_reference_centered=true`接口。保留两个独立上层，alpha在run内固定0/1；本轮不合并、不双头、不蒸馏。上层仍输出相对原始指令的两个参考残差。保持物理、模型、执行器、动作映射、slew、50Hz/200Hz、345/346维网络与既有历史字段。

正式主方案：两个上层Actor/Critic/优化器/标准差全新初始化，不加载V5上层续训。V5模型仅为对照。原因是新训练从第一批即覆盖完整恢复，而不是继承“5秒转弯结束即终止”的课程起点；这是本轮明确的选择，不声称重训普遍优于热启动。

不得执行候选搜索、MPC、参数网络、解析alpha动作分配或外部航向控制器。下文偏航速率目标只用于损失计算，不得下发为控制命令。不得停止R244/JIT等其他任务，不自动push。

## 1. 当前证据与解释边界

V5 update120 fast_turn [2,4)：
- alpha0 governed(v,delta)=(2.244,.235)，actual约(2.177,.233)，速度RMSE .423914、转角RMSE .020736；峰值侧倾.289488。
- alpha1 governed约(2.543,.205)，actual约(2.469,.197)，速度RMSE .132297、转角RMSE .053202；峰值侧倾.314970，越.30共1.755s。
- 10秒末航向RMSE .848558/.655855；16秒末 .604181/.350704，均非合格恢复。
- 10秒六场景三项末保持：alpha0 0/6，alpha1 1/6。
- alpha0前10秒恢复段4.76s，航向成本积分401.5896，速度.81824、转角.19038。现有航向成本已经占主导，cap为0。不得再称“航向没有奖励”或“本轮总成本100裁平”。
- 本次lower_reference_centered=true；u_nom/u_goal已以governed为中心，不能机械套用早期raw-centered上下层共享预算的诊断。现有链路证据表明持续减速已经执行，没有接口抵消。

分析依据为已推送源码、CSV和JSON，不是本助手重新运行车辆。本附件仅运行新奖励增量数学检查；PNG/NPZ未在本工作环境成功获取。不得补写不存在的仿真结果。

## 2. 最小代码修改范围

读取本地AGENTS及override；基于真实V5工作分支建立新实验分支/目录，不改写旧run。复用现有：
- `direct_command_reward.py`：新增一个actual primary_excess项，替换yaw_damping的数学定义。
- `direct_command_env.py`：仅支持训练用初始航向误差任务；现有状态/ESO/执行流程不改。
- `direct_command_scenarios.py`：四种训练族，其中新增初始方向误差族。
- `smooth_command_config.py`：新增显式V5.1解析器，引用阶段2配置但以scratch从更新1开始；不是恢复Stage1模型。
- 现有训练/汇报模块：支持新分项、任务族统计，保留KL和已有PPO。

新模式建议标志`priority_recovery_v51`，旧V3/V4/V5默认行为不改。不要同时创建第二套训练器或泛化框架。

## 3. 新强主项：惩罚实际误差，不是惩罚指令没有修改成某个答案

定义e_v=actual_forward_speed−original_limited_speed，e_d=actual_delta−original_limited_delta，e_psi=reference_unwrapped_yaw−actual_unwrapped_yaw。H(z)=z²（|z|≤1），否则2|z|−1。

chi、g保留V5原始指令/计时定义。新增：

```
C_primary_excess = 40 * chi * (1-g) * (
  (1-alpha) * H(max(abs(e_d)-0.03, 0)/0.05)
  + alpha    * H(max(abs(e_v)-0.05, 0)/0.10)
)
```

必须满足：
1. V5已有实际speed/steer costs全部保留。本项是超出软容差后的额外惩罚。
2. alpha0重点保转角，alpha1重点保速度；不能两项无区别地都乘40。
3. 使用本物理区间实际测量与本区间原始发布指令。不是governed误差，不是proposal误差。
4. 恢复时g趋1，本额外主项退出；否则用户回正后还强迫实际轮角为0，会妨碍航向恢复。
5. 容差.03rad/.05m/s是训练损失转折点，不改原来任何验收阈值。
6. 系数40是本轮固定初值，不是已验证最优值。不要自行扩大到1000，不自动扫系数。

数值：
- alpha0 abs(e_d)=.02时新增0，.05时6.4，.08时40。
- alpha1 abs(e_v)=.03时新增0，.10时10，.15时40，.20时80。
- alpha1 .15m/s误差：旧主项16，新主项合计56；alpha0 .08rad：旧17.6，新57.6。

新增分项cap=400，仅保护严重离群；不对全成本再次clip。新增项clip从alpha1速度误差约.60m/s、alpha0转角误差约.305rad开始，应记录。已有13个caps维持，总上界从1680变2080。

失败吸收计分由runtime实际caps求和生成：
```
R_fail = -5 - .1*.02*2080 * (1-gamma**N)/(1-gamma)
```
N包含当前政策区间；真实物理失败替换该区间奖励一次。policy fault/NaN停运行而非伪装零损失。正常终点16秒no bootstrap；rollout切段正常bootstrap。不得继续使用旧1680。

## 4. 恢复不是把转角罚到零：替换旧近目标yaw_damping项

原heading误差成本仍保持V5：`6*g*H(max(abs(e_psi)-.01,0)/.10)`。它已经很大，不再盲目翻倍。

用下式替换旧指数门控的yaw_damping（不能两者叠加）：
```
r_c = v_c*cos(caster)*tan(delta_c)/wheelbase
r_debt = clip(.8*e_psi, -.4, .4)          # rad/s，仅损失中的参考
C_yaw_recovery = 2*g*H(((yaw_rate-r_c)-r_debt)/.20)
```

日志新名`yaw_recovery`，cap仍20（用它替换原caps中yaw_damping键，因此不会增加上界）。所有existing成本重建脚本同步改名/算法。不要把r_debt写入raw/governed、不要用反解公式自动替网络产生转角。网络继续自行选择两项残差。

理由：e_psi正时，实际偏航速度需要比参考更正，才能减少欠转；负时对称。近零处希望e_dot≈−.8 e。大误差限幅.4rad/s避免损失无限要求猛转。这只是更明确的即时闭环学习目标，没有硬安全或指数收敛保证。当前yaw_rate必须使用与unwrapped yaw一致的世界偏航角速度，不能拿body gyro z直接替代。

使用同物理区间的raw参考速度、post-state debt和实测区间yaw_rate，与现有时序一致；每5ms积分。既有参考积分不变。例子：e_psi=.8，r_c=0，实际r=0/.4/−.4时，新增替换项分别6/0/14，两个alpha一致。仍保留实际侧倾/侧倾角速度成本和速度跟踪，损失中的建议偏航不会覆盖安全反馈。

## 5. 其余奖励冻结，避免继续叠加不相关变化

保留V5的：normal、conflict_alpha0、conflict_alpha1、recovery heading/speed/steer（除替换yaw项）、safety、magnitude、upper_rate(.03)、upper_acceleration(.01)、reference_priority(4)、command_compatibility(2)。q(v)仍仅参与已有参考需求成本及ECBC，不做动作投影。

保留V5时间正则：前20policy更新0，20→40线性至.002，仅已跟稳且同回合相邻状态；不要用minibatch邻行配对，不跨reset；不新增后处理低通。

注意：提高全reward scale不能等价于提高优先级；保持scale=.1。本轮也不降低侧倾权重来换速度分数。安全与跟踪分别汇报。

## 6. 从第一批开始训练完整任务，取消5秒Stage1→16秒Stage2切换

每端全部run只有一个16秒任务、一个PPO循环、同一奖励。Actor/Critic/optimizer新初始化；Actor末层零权重零偏置，旧R196不动。V5 actor/checkpoint120不得导入。两个端点分别训练，固定各自alpha。所有输入、输出维度保持345/346/2；alpha为本run常数，不能宣称这两个独立Actor已经支持单个Actor跨alpha切换。

训练族在reset独立采样（同两个run使用同一生成规则和随机种子）：
- 30% nominal：复用V5完整任务普通指令，速度[2,2.6]、转角±.08、60%直行，8秒以后回正。
- 40% conflict：速度[2.4,2.6]、转角幅值[.22,.30]、正负各半；转向起点[1,2]秒。80%单弯、20%反转；到达目标后保持[.8,1.5]秒。回正后50%保持本次高速、50%选[2,2.6]。仍16秒。
- 10% random：复用V5随机原始目标与slew，前8秒变化，后8秒回正。
- 20% heading_recovery_start：见下一节。明确这是训练分布新增任务，不计入原六场景成功率。

所有raw指令继续在[2,2.6]m/s、±.30rad内，速度slew每回合U[.3,.8]，转角slew U[.15,.45]，Actor不得读未来目标表。物理/初态准备库不变，不加入外扰、摩擦随机化或位置任务。

### 6.1 初始航向误差训练族（不得偷换评价参考）

从已有R196完整稳定快照开始，实际物理状态、ESO、执行器、轮速等完全不改。将该训练任务的虚拟参考初值设置为：
```
reference_xy(0) = actual_xy(0)
reference_yaw(0) = actual_unwrapped_yaw(0) + e0
```

e0符号各半，绝对值50% U[.03,.15]rad、50% U[.15,1.0]rad。raw speed取所抽准备快照自身的speed，raw steer=0，整个回合保持。初始settle_clock仍0，g按原规则自然在.4～.8秒启用；不要直接设置g=1。与普通任务一样建立有效history mask，生成Actor observation前，reference/debt/sin/cos必须一致。

lower的初始参考/历史仍按真实actual pose初始化，接收网络修正后的governed；不要把上层e0暗中当作底层路径纠偏目标，这会变成底层替上层做恢复。

该任务是明确的“有初始方向误差的参考跟踪”训练数据，不是假装复现了真实扰动后的相同完整状态。它让网络无需先经历一次复杂冲突才能看到有符号的恢复误差。初值只在reset定义一次，之后原始指令积分连续，不得再锚定/清零。训练记录family、e0与参考初值。

所有原六场景和独立评价必须e0=0、初始参考等于actual、原始指令/窗口完全不变。不得把这个训练族变得容易后的成功直接宣布为真实冲突后恢复成功。

## 7. 超参数与预算

保持与V5同量级设置：每端512env×128策略步；Actor1e−4，Critic3e−4；4epochs、4minibatches（16384/批）；gamma .997、GAE .99；clip .2；value_coef .5；entropy .001；Adam(.9,.999),eps1e−5；分开clip actor/critic grad=1；速度/转向latent std初值[.30,.10]、min[.08,.03]、max[.40,.20]。不用AR、OU、额外教师或新网络。

每端最多100次有效更新，总13,107,200策略转移、52,428,800底层tick（准备/评价另记）。不自动续到250/500。第25更新看straight_hold和fast_turn；第100看原六场景，fast_turn保留10秒与16秒完整窗口。已有B0同协议同物理同快照可复用，不重新跑纯ECBC基线；必要的配对身份检查不等于逐位一致性宣称。

用户此前取消额外墙钟硬截止，本轮不擅自恢复一个30分钟总门槛。报告实际wall、采样吞吐和ETA，保留更新预算与非有限/KL停止；没有用户许可不自动追加。不要因为运行时间较长替换成不完整5秒评价。

KL：本run有效alpha数据计算；不存在另一个alpha不是错误。soft .01保留epoch并结束本批后续epoch；hard .03回滚该epoch、结束本批、actor LR减半到最小1e−5、重新采样；连续3批hard拒绝才整run停止；nonfinite立即停。

## 8. 只做这些必要检查

1. 运行附件reward_delta_reference.py，并对运行时代码同输入逐项匹配。新增/替换成本是cost rate，乘.1*.005；4tick汇总不能重复四遍20ms成本。
2. heading e0正负、实际yaw_rate世界坐标的符号匹配；可用保存数据/现有工具判断，不启动参数扫描。
3. 新训练族只在reset设置虚拟参考初值，普通任务与评价无变化；lower不使用e0作弊。
4. 一次8环境×16策略步×2更新工程小批（不记入正式100更新），确认固定alpha、训练loss、梯度、reset有限；正式Actor之后重新初始化。

不重跑全仓测试，不验证人类知道的两种取舍是否存在，不做网格求解，不重新比较R196/R244底层。

## 9. 报告和决策：主目标不合格不能用次目标或return掩盖

必须保存各相位、各分项cost、primary_excess原始与有效值/cap比例、原始/修改后/实际三种指令语义、真实低层控制链。相位由raw指令决定；初始直行g>0不算post-turn recovery。

主fast_turn [2,4)：报告alpha0转角RMSE，alpha1速度RMSE，两端mean actual speed/steer、峰值侧倾与越界秒数。保持既有门槛：alpha0转角≤.05rad，alpha1速度≤.08m/s；两端侧倾≤.302为数值容差，不将.315写成通过。实际减速以转前基线及连续时长计算，不仅靠“低于正在升高的速度指令”。

最终保持：每步|ev|≤.1、|ed|≤.04、|e_psi|≤.05连续.5秒；10秒与16秒分别报告，不用16秒通过覆盖10秒失败。R196上层任务没有XY cost；XY继续展示但不把位置落后或平行偏移伪装成已完成位置恢复。

第25更新只判断有无明显回退/数值故障；无单次效果不佳自动重抽seed。第100更新保存并停止，以物理指标评价，非training reward-best。若alpha1仍欠速且新primary_excess已明显主导，检查已记录的governed目标与actual差异，再考虑单独调整alpha1系数；不得同时改所有系数/物理。

重要结果区分：
- 主项增大且alpha1实际速度改进：支持这次定向修改。
- 主项增大但方向仍不恢复：不能继续只加转角主项；检查恢复族与yaw_recovery分项。
- 只reward改善或零摔倒：不等于任务成功。
- 先确保安全＋主目标与恢复，再比较平滑与少干预。

## 10. 交付

代码与原V5的准确差异、本次解析后的两份配置、lower/actor身份、25/100节点的必要物理图、各分项表、训练统计、完整主要轨迹。不得只上传“源码未推送”的commit名而无改动内容；push需用户明确授权。不要分享权重或外部二进制大文件，除非另有许可。

本轮提出的是固定容差强惩罚＋即时恢复成本＋恢复任务覆盖，并非CPO、安全证明或严格词典序控制。需要硬约束时应另行设计约束学习，不把有限权重称为必然优先。
