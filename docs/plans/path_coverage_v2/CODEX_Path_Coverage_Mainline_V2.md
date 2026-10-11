# STTW 几何路径主线：仅修正有效训练覆盖，不再优化底层

## 0. 本轮决策与证据边界

基准仓库 `QaQaaa-zzz/STTW_CONTROL`，已审阅提交：
`e6af0cae12790320b4d42cef2e582bcc7bade701`（experiment/path-feedback-v1）。

当前远端最新结果是两个上层各200更新已经完成，包含奖励峰值附近候选补评；不是旧INDEX中的17:16/18:27启动快照。实机目录是否另有新进程，执行前只读查一次。不停止、热改、覆盖另一个已启动run；若存在更晚的同任务运行，先核对它是否已包含本方案，避免重复启动。

结果依据：
- `docs/evidence/path_feedback_v1_20261010/phase_C_results_20261011/INDEX.md`
- 同目录 `training_summary.json`、`paired/identities.json`、`selection.json`
- `learning/src/sttw_control/path_command_scenarios.py`
- `path_command_statistics.py`、`path_command_env.py`、`path_command_reward.py`
- `path_command_training.py`、现有runner、`path_command_selection.py`
- `learning/configs/path_feedback_phase_C_run.json`、`STTW_Path_Feedback_V1.json`

已证实：几何反馈外环运行，两个best均6/6到达并最终保持；高速左右弯各违反侧倾工作界，取舍不稳定。local300对best策略的全程速度跟踪RMSE约0.0212–0.0292 m/s、转角0.00182–0.02134 rad，尚无理由将其作为本轮阻断项。当前更新200的策略有保持能力退化，不应从last200继续。

本轮唯一研究干预：改变训练reset时的路径/速度联合采样分布，使真正启用偏好的需求区间得到足够曝光。保持20秒任务、正常奖励、控制结构、物理和PPO超参数不变；从已验证的本端best完整学习状态分叉。

这是修改建议，不代表已验证新训练会成功。所附geometry_budget_check.py只验证代数和名义几何覆盖，不运行物理、不寻找控制解。

## 1. 锁定主线，不再扩展工程

最终系统：
固定且不可重定位的路径 + 外部期望速度
→ 实际位姿反馈的Pure Pursuit（不按曲率替alpha减速）
→ 两个暂时独立、alpha=0/1固定的上层MLP，各自直接输出Δv/Δδ
→ 已有一阶修正滤波与slew/幅值限制
→ 冻结local300 + ECBC/ESO + 既有执行器
→ 状态反馈。

本轮禁止：
- 再训练local300、换400、恢复R196、加入新的下层alpha/path/return任务。
- 为追求底层零误差重复局部七场景、旧6.315秒恢复或四条组合诊断。
- 增加规划神经网络、世界模型、扩散、候选枚举、解析alpha动作分配器。
- 合并上层、双头、蒸馏、扩大网络或改变输入351/352维。
- 更改物理子步0.0002s、25子步/底层周期、底层200Hz、外环及上层50Hz。
- 修改lookahead、滤波常数[0.10,0.08]s、动作范围、下层权限或工作界。
- 改正常奖励、突然放大主项、增加动作差异奖励、恢复实际转角主目标。
- 因正常有限PP域退出而终止整批训练。保留terminal_contract_v2；物理/域失败只结束对应环境并计罚，NaN/Inf等工程错误才停止运行。
- 以底层/零上层完全合格作为训练前置；底层不是本轮论文目标。

## 2. 关键诊断口径：不能把chi>0误叫“所有弯道样本”

当前statistics中的bend定义是 `chi > 0`，strong_demand是`chi >= .5`。
`ordinary`即`chi<=0`，可能仍有真实几何弯道。不要写成“98%都在直行”。
当前末期偏好有效区间约2.2%，强需求约0.6%，出口直线约67%。
因为w_v=10-9chi(1-alpha)、w_path=10-9chi*alpha，chi=0时两个端点任务权重相同。
这说明区别偏好的训练时刻很少；不证明它是唯一失败原因。

增加统计别名，保留旧字段兼容，不重写旧产物：
- geometric_turn: abs(log.path_curvature)>1e-4
- demand_positive: chi>0（旧bend的准确名字）
- strong_demand: chi>=.5
- exit_straight: 原空间定义不变
- 左/右：以当前局部曲率符号分，不把S弯第二段沿用第一弯符号。
- 每个统计用active_tick为分母；窗口可重叠，不能相加。
- 按上述窗口分别记录ev、ey、eh、peak_roll越界时长/峰值、Δv/Δδ有符号均值、下层actual-governed误差、分项成本与触顶比例。
- 同时记录reset家族比例与有效时间比例，二者不能互相替代。

这些统计在原训练rollout中设备侧累积，每10/25更新输出；不是增加场景评价。

## 3. 新采样器：20%旧分布保持，约80%有明确较强需求的左右转弯

### 3.1 512环境固定分组

每个端点使用同样的分组与采样规则：
- env_id 0..101：102个，使用原sample_route的完整旧分布（约19.92%）。
- env_id 102..305：204个，强需求单弯组（约39.84%）。
- env_id 306..511：206个，强需求S弯组（约40.23%）。

原分布保持组保留原30% ordinary/gentle、50% single、20% S及原速度/半径/初始偏差规则。
强需求组使用已有单弯/S生成公式，而不是新建迷宫、随机地图或在线规划。

强需求单弯/S参数：
- 期望速度 U[2.4,2.6] m/s。
- 半径 U[2.0,2.2] m。
- 每段转弯角 U[90,100]°。
- 前缀3m、曲率渐变0.5m、S中间直线3m、goal=最后弯结束后4m。
- 100m/5001点固定网格、0.02m间距与原实现一致。
- S为两个相反方向的同半径/同角度弯，不改几何表达。
- 保留原20%路径初始偏移概率，横向±.1m、航向±.05rad。
- 仅reset时锚定一次，车辆实际完整准备状态不变，整回合路径不再重定位。
- horizon仍20s；出弯后继续路径反馈与保持，不能成功后提前结束、删掉出口评价。

左右成对：强需求组按相邻两个env组成一对，速度/半径/角度共享随机量，初始弯向相反。路径锚定横向/航向偏差也按局部坐标做符号反射。不得镜像真实物理状态或宣称机构完全对称。第i对每个episode按pair_id与episode采样；两成员episode编号因失败不一致时按各自实际编号处理，不为凑同步强制reset。上下两个alpha用相同route seed，但保持各自真实闭环。

采样种子910201。算法采样/探索RNG来自各自父模型，路径RNG用新版本key避免复用旧训练路径。元数据记录sample_group与实际速度/半径/符号。

为何这样选：按当前ECBC ratio、名义曲率1/R计算，2.4m/s、R2.2m平台的chi约.622，其余组内平台更高。保留减速、入弯、曲率过渡和出口，不是从已经转好的状态起步。高需求不等于物理必然失稳或有严格安全可行性证明。

### 3.2 代码实施

不要直接修改现有sample_route默认行为。新增可选择的sampler_version=`demand_exposure_v2`，默认仍走legacy。
复用路径积分函数/Path结构，只参数化family、v/R范围、角度与group。
`load_plan()`当前硬编码sampler_resolution和比例；增加显式版本分支与严格新参数校验，不删除旧配置防误读断言，也不允许JSON写了新比例而实际仍走旧逻辑。
固定DEV仍用原`cases/build_path`，不经过训练sampler，不修改六个DEV的速度、路径、20s窗口及参考。

### 3.3 有效曝光目标

本轮希望strong_demand有效控制步占比至少5%（原约0.6%），demand_positive也相应增加。
名义几何计算在假设s=vt下，仅两个新组就约贡献7.8%的strong样本；这不是实际训练结论。
每10更新读取真实active样本。连续两个完整10更新窗口低于5%时，先检查新采样配置是否真的加载、域失败是否过早、路径进度统计是否正确；不要擅自改reward或底层。一次必要诊断后暂停新增更新、保存并报告，不在同run中静默修改目标分布。
不要求出口直线比例降到任意漂亮数值；保持20s是为了控制变量并保留出口能力。
不能把路径尾部70m改短就宣称20s中直行时间减少。

## 4. 正常奖励保持原样，并明确其任务意义

H(x)=x²(|x|≤1)，否则2|x|-1。
ev=actual_forward_speed-v_user。
ey/eh来自固定路径上的连续局部投影，不使用时间移动目标、不追赶纵向时间位置。

保留：
C_speed=(10-9chi(1-alpha))*H(ev/.10)
C_path=(10-9chi*alpha)*[H(ey/.10)+.5H(eh/.10)]
C_primary=20chi*[(1-alpha)H((|ey|-.10)+/.10)+alpha H((|ev|-.05)+/.10)]
C_roll=40H((peak_roll-.26)+/.04)
C_work=120H((peak_roll-.30)+/.02)
C_rollrate=H((|rollrate|-.8)+/.8)
C_overspeed=2H((ev-.05)+/.10)
C_low=4H((1.5-v_actual)+/.20)
C_offset=.02*sum((offset/[1,.2])²)
C_rate=.03*sum((d_offset_dt/[1,.4])²)，20ms端点计算。
分项caps、合计5080.2失败上界、scale=.1及terminal_contract_v2全部保留。

本轮必须用相同人工样本核对新旧cost函数逐项相等；此检查不运行物理。
不增加左右动作不同奖励、不奖励减速本身、不强迫alpha0在可兼得工况比alpha1更差。
共同安全必须相同；lower训练包络不是此轮在线动作投影，也不能当作精确可达集合。

## 5. 从哪个模型继续：本端best，不是last200，更不是从零重新学出口

这次只改训练分布，任务语义、正常奖励、输入输出与底层不变。
采用已达到6/6最终保持的几何上层best作为起点：

alpha0源update100：
- checkpoint SHA c1246d7bcae1568d923e2b819cf87a4c10c7a128c0fdfd9e14b73126ff75c20a
- Actor SHA 99ed5c769bcf80ac0ca5b5be912c629ea2df5e6631673934a8ff08ca0aa82422
alpha1源update150：
- checkpoint SHA 2df605339a892e180399be9647ece2aba181b601e62d7c7966e97f5a59732261
- Actor SHA 77af68f4b45427cc309b2ef51b30ab1d60363c7af616293331fec71de9a90809
local300 Actor SHA：
d408a94e886fa7f6309aa2c7635eb62e273c197405293ac9c2f1859014f25bcc

路径从paired/identities.json与本地真实不可变receipt解析，不按文件名猜测。
显式fork恢复同一checkpoint里的Actor/Critic/log_std/Adam，确认策略及优化器同代；不得拿actor100配optimizer200。若保存文件不包含必要状态则停止并准确报告，不偷偷随机初始化。

原restore_checkpoint将task/lower/config identity全部严格相等，这一轮sampler不同：
实现显式fork白名单，仅允许sampler_version/seed/分布与run预算、日志字段变更。
其余reward、network、PP、filter、model、lower、timing、terminal/eval identity必须严格一致。
原resume函数仍用于完全相同身份的中断恢复，不删其保护。
建立新run、新config/source/parent/counter/RNG溯源，不覆盖旧best或旧run。
物理从原完整prepared bank重新开真实回合，history/filter按原reset合同初始化；不声称逐帧续跑。
父best重新作为该新run初始验证候选登记，原六条验证证据匹配SHA后复用，新增基线仿真0。

## 6. 训练参数和预算

保留：每端512环境×128策略步；Actor/Critic 351/352、128-128-64 ELU；alpha回合内固定，两个端点独立；Actor1e-4、Critic3e-4；gamma .997、lambda .99；4epochs、4minibatches、clip .2；entropy .001；std继承父模型，范围.05–.5；KL .01/.03、原回滚/学习率处理不变。
保持当前版本DirectCommandPPO，不新增按阶段经验回放或把旧数据当新on-policy rollout。mini-batch只是从本轮真实rollout正常取样，不能从小段困难状态重复采样却假装目标未变。

默认每端新增100更新：
13,107,200策略转移（两端合计）、52,428,800底层控制步；不是每端这些数字。
新run `new_update=0` 对应其父best；lineage分别100+new_update、150+new_update。
保存每10更新，异常退出保存最后完整已接受边界。
新增50、100更新做原六场景固定DEV；这是已有成熟best上的适应验证，不新增20/25评价。
沿用已编译批量DEV和匹配B0缓存。两节点两端总24回合、96,000底层步；最终可引用已验证best的对应不可变轨迹，不为重新绘图再跑六条。

第100时只有满足下列条件才允许按50更新扩展，最多新增200/端：
- 强需求曝光已达到目标，工程/域失败没有出现新的系统性恶化；
- 出口/目标仍保持6/6，不用恢复退化换取过弯改善；
- 固定DEV共同侧倾失败数下降，或者在失败数不增加时，最差高速弯的本端主误差较父best至少改善5%，且最坏侧倾峰值/累计越界没有恶化；
- 最近两个25更新强需求窗口的相应成本或误差仍显示同向改善，而不只总reward提升。
若没达到，不自动追加100/200/500，也不自动重训底层。
扩展验证新增150、200，最终记录因何继续/停止。没有额外默认墙钟强制截止；只记录实际训练/验证耗时，不抢占其他GPU作业。显式用户停止优先。

## 7. best_task与“完成主线”的判据

保持现有path_task_best_v2排序：物理/域任务失败→工作界失败→主目标超带→目标与最终保持→连续质量。保留qualified=false的best。
本轮同6DEV只是开发选择，不是泛化。不能拿训练reward峰值直接替best_task，也不再补评一圈峰值附近模型；没有保存的精确权重只能如实说未保存。
每个checkpoint保存其来源更新。final export从磁盘best_task重载，打印SHA；final报告引用匹配的验证轨迹；不得用内存last。

必须分别报告：
1. 同一固定路径和v_user下B0/alpha0/alpha1。闭环PP名义转角可以不同，不强迫nominal流完全相同。
2. 在相同空间弯段比较两端实际路径和速度误差；保留各自到达时间，不平移时序。
3. 普通段两目标均跟踪；弯后完整1秒保持。
4. 两端共同0.302rad工作界，不能放宽以制造取舍。
5. alpha0路径优势与速度代价、alpha1速度优势与路径代价。只有二者在共同约束内存在明确且可重复的对应关系，才称已证明偏好；一方两项都优于另一方，属于另一个策略尚未优化好，不是目标完成。
6. 若两项均可跟好，不强迫分化；若安全跟踪需要取舍，不用无意义故意劣化来制造曲线。

## 8. 下层只作旁证，不再成为无休止前置任务

在当前rollout/DEV中继续记录actual-governed，不新增下层专门测试。
“低层非零误差”“高速越界”“两个alpha的nominal不同”都不是重训底层的充分理由。
仅当日志中出现：下发参考已稳定≥.5s、参考处于已知温和范围、姿态瞬态已经较小，而速度误差>.15m/s或转角误差>.05rad又持续>.5s，并在不止一个已有案例重复，才提出底层阻断诊断。
不能把全程平均RMSE当作任意时刻都准确；同时也不能要求瞬态零误差。
若主要是上层给出大幅/快速不相容指令或新上层没有减速，应继续处理上层，不把任务丢回下层。

## 9. 最小实施清单与执行顺序

1. 读取AGENTS与工作区，保护运行/旧结果；确定最新结果确已完成且不存在重复新run。
2. 增加版本化sampler，保留legacy；参数真实接线，不复制trainer。
3. 加准确的需求/几何弯道统计与组别计数；利用现有设备侧累积，避免每5ms打印或CPU同步。
4. 允许显式best-learning-state fork，只放行本轮声明的采样字段差异。
5. 必要检查限于：生成形状/有限域/符号、reward不变、parent身份、有限失败单环境reset、一次8×16工程小批。没有必要重复已完成的16/11/7项整套旧检查。
6. 工程批不得更新正式父权重；正式从同样完整parent重新加载。
7. 启动两个端点各100新更新；10/25为采样统计而非固定评价，50/100使用既有DEV。
8. 完成后输出主线结果，按第6节决定是否有限扩展；不自动push、部署、蒸馏或重训底层。

建议实现位置：path_command_scenarios.py、path_command_statistics.py、path_command_training.py及当前runner/CLI的加载与fork接口。path_command_env/reward/PP/filter及local300不做行为变更。
配置由附带plan生成新resolved design/run JSON；附带JSON是执行规格，不可直接冒充原生CLI run配置。将新参数读入及生效值写进manifest。

## 10. 交付格式

先给一页结论：本轮有效强需求占比、两个alpha真正做了什么、工作侧倾/最终保持是否改善、选用哪个best、为什么继续/停止。
提供原六场景XY和实际速度/路径/roll/修正/下层误差图；相比父best而非last200。
推送时（获得授权后）包含实际运行源码差异、版本化配置、best SHA、训练统计与固定验证指标；不只推图片/状态快照。
不以“训练完成”“reward增加”“更平滑”“底层不完美”代替主线结论。
