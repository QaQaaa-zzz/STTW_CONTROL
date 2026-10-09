# STTW V5.2：基于V5.1结果的最小修改、底层逐时刻审计、最佳模型与收敛管理

## 0. 本轮结论与唯一实施主线

证据分支：`experiment/r196-preference-v5-1-strong-primary`。
本次审阅提交：`26c3fedac76b099ae98ff153f3f50f09f2730da9`。
证据目录：`docs/evidence/r196_preference_v51_20261009/`。

保留两个独立上层、每个run固定alpha=0或1；最终合并是后续任务。本轮不变更为共享Actor/双头/蒸馏，不扩大网络或动作空间，不替换R196。
保留网络直接输出速度/转角参考修正；保留R196+ECBC+ESO、`lower_reference_centered=true`、物理、时序、参考积分和实际限制。不加搜索、世界模型、扩散或解析偏好分配器。

本轮只做：
1. 从已保存NPZ计算上层让步与底层执行误差的精确分解，不产生新仿真；
2. 增加明确的侧倾工作界越界成本，允许alpha1为实际速度跟踪作正参考补偿，略缩小普通/恢复速度死区；其余V5.1主项与恢复逻辑保持；
3. 保留当前100轮Actor作为有用起点，重置Critic/优化器适应新回报，追加训练；
4. 实现真实best model、后期稀疏验证和基于收敛趋势的扩展。禁止第20/25轮评测，禁止最后权重自动冒充best。

不要改回V5或V4奖励。新版本模型身份、配置hash、父checkpoint与新旧更新计数必须分开。

## 1. 已核实证据以及解释边界

V5.1每端100次更新，没有证明收敛。公开CSV的每25更新平均step reward：
- alpha0：-0.05931381, -0.05317716, -0.04799679, -0.03734911；
- alpha1：-0.06801755, -0.05641128, -0.05393193, -0.04142840。
51–75→76–100负成本幅值下降约22.18%/23.18%。这是训练分布下的进步信号，不是固定策略收敛证明。16s/0.02s=800步，128步/批产生6.25批相位周期，必须看完整回合/长窗口和后期固定验证，不能看某一个step reward值。

fast_turn稳定请求[2,4)原始为[2.6m/s,.25rad]：
- alpha0：governed=[2.3637216,.2385200]，actual=[2.2858405,.2321528]；
- alpha1：governed=[2.6435819,.2222811]，actual=[2.5504348,.2103310]。
由均值差可得lower speed bias=-.0778811/-.0931470 m/s，lower steer bias=-.0063672/-.0119501rad。不能把这些bias称为RMSE。
利用公开窗口极值保守推导：alpha1在[2,4)每个时刻的actual-governed速度误差均介于[-.1094561,-.0845275]m/s；两端在该窗口的转角跟踪绝对误差分别不超过.019322/.021224rad（保守界，不是实际测得峰值）。
alpha0全16秒的actual最小转角-.1029488，而governed全程最小-.0348310，故至少存在一个时刻下层转角误差绝对值>=.0681179rad。发生时刻须用NPZ定位，不能从汇总凭空给出。

fast_turn峰值侧倾alpha0=.3154、alpha1=.3676，均越工作界；稳定窗口主误差delta0=.0186153rad、speed1≈.0500m/s已明显改善。应优先补共同安全，不继续一味放大主项。
原始fast_turn在t=1同时加速与转弯。低于上升后的速度请求可能是不继续加速，并非一定比转弯前真实减速。持续制动指标只在预先稳定高速的附加场景报告，不把原场景的此指标失败自动等同协调失败。

助手本轮读取源码、CSV与JSON，计算了所列均值/界；容器未取得NPZ字节，未做新物理仿真。下面逐时刻审计由Codex在已有本地NPZ执行。不得声称助手已经复算所有逐步误差。

## 2. 冻结项与来源定位

读取AGENTS.md与本地真实训练工作区，使用独立分支和新run目录，不覆盖现有结果、不停止其他任务、不自动push。
父run优先从已推送identity/status.json和本地manifest解析，不靠猜目录扫描所有项目。
已知父run：
`/home/qy/STTW_CONTROL/runs/worktrees/r196-preference-v5-1-strong-primary/runs/preference_v51_strong_primary_100_20261009`。
各端从父run自己的`last_completed.json`解析对应update100 checkpoint。必须读取其实际update和alpha，不将alpha0权重当alpha1。路径缺失则报告，不无声改用V4/V5权重。
R196：`STTW_R196_ALPHA1`，lower alpha恒1；payload SHA256：
`a2e5bf112e0cb986b4ec207fa9f20dcf90f48af6d7855b0abda32bfa13206672`。
同一既有prepared_bank和注册适配器复用，不重做百万步预热。

动作、观测：Actor345，Critic346，MLP128-128-64 ELU，两输出；50Hz上层、200Hz底层；tanh→delta_v[-1,+.25]m/s、delta_delta[-.2,+.2]rad；reference speed[1.5,3]、steer±.35；修正速度下降1/回升.5m/s²、转角.8rad/s；底层残差±1.5/±10rad/s、最终±3/±60rad/s，物理机械±.8rad。

重要：`lower_reference_centered=true`时u_nom、u_goal都以governed计算。不能恢复旧版“raw ECBC+upper差值”的合成，也不能从当前相等的u_nom/u_goal反推上层没作用。上层作用已体现在governed中，R196残差另加。

## 3. 底层跟踪审计——先做已有数据后处理

运行附件：
```bash
python audit_lower_tracking.py \
  --input docs/evidence/r196_preference_v51_20261009/data/update100_timeseries.npz \
  --output runs/<new_run>/audit_parent_lower
```
脚本按compact key适配。如本地完整原始NPZ字段更丰富，直接使用，不重新仿真。

每个200Hz区间同时记录：
- 原始发布指令c_raw；
- 网络保护前请求c_proposal；
- 实际下发参考c_g；
- 实际v/delta；
- e_upper=c_g-c_raw；
- e_lower=actual-c_g；
- e_task=actual-c_raw=e_upper+e_lower。
三个误差不可混为一谈。必须断言逐时刻恒等式成立。日志时间为区间起点、actual为该区间结束状态；对比的是同区间保持的指令，不把下一时刻新raw拿来评分。

分析全部真实保存时刻，不只用均值。至少分别报告全程、[1,6)、[2,4)、governed连续.3秒低变化率段、governed变化段、转弯后恢复段：bias/RMSE/P95/最大误差及时间、超阈值比例、每个连续越限区间与最长时长。转角诊断阈值.04rad、速度.10m/s；原始波形仍保存，不能只保存异常片段。平稳标准用governed而非raw：|dv_g/dt|<=.1m/s²，|ddelta_g/dt|<=.02rad/s。

参考阶跃/限速变化后出现短暂误差不等于底层永远不能跟。可以额外报告滞后诊断，但正式误差必须零时移；不能把曲线向前/后平移制造合格。

电机链单独报告：final_rear*.1→wheel_speed_proxy→actual_forward_speed。当前轮速代理基本跟住最终电机命令，而代理与车体速度有明显差异。`slip_proxy`只是二者差值，不是经过接触运动学验证的真实滑移率。不要先改摩擦或电机增益。

底层决策：
- 跟踪偏差稳定在约.08–.10m/s但可由适量参考补偿、且实际控制余量存在，不据此更换底层；上层继续以actual vs raw优化。
- 若在governed平稳>=.5秒后，|e_lower_speed|>.15m/s或|e_lower_delta|>.04rad仍连续>.5秒，结合模型/姿态/历史定位实际区间；至少检查同段下发参考是否本身激进、是否执行器限幅。
- 若两条独立已存在轨迹都在温和可行请求下出现上述持续错误，先报告LOWER_TRACKING_LIMITED并暂停扩大上层训练；只允许针对其中一个已定位段补一次短复现，不能批量参数搜索，不能擅自换底层。把“该接口在这些状态失败”与“整个系统物理无解”分开。
- 诊断e_lower默认不加大权重到reward。否则有可能反过来禁止上层为已知下层偏差作正补偿。

## 4. V5.2奖励：仅以下差异，其余沿用V5.1

H(z)=z²(|z|<=1)，否则2|z|-1。现有真实主误差和primary_excess保持：
`40*chi*(1-g)*[(1-alpha)H((abs(ed)-.03)+/.05)+alpha H((abs(ev)-.05)+/.10)]`。
原有heading=6*g*H((abs(epsi)-.01)+/.1)、yaw_recovery目标clip(.8*epsi,±.4)及其weight2、兼容性weight2、offset/rate/acc/Actor时间正则全部保留，不再新增另一套航向控制器。

### 4.1 新增共同工作界超限成本

使用该5ms控制区间内所有物理substep的峰值：
```
phi_risk = max(abs(roll_at_interval_end), peak_abs_roll_within_interval)
C_roll = 40 * H(max(phi_risk-.26,0)/.04)    # 原风险公式；现在采用区间峰值
C_work = 120 * H(max(phi_risk-.30,0)/.02)  # 新项 working_roll_excess
```
两个alpha相同，所有阶段生效。原工作界.30、验收数值容忍.302、摔倒.70均不变。
新项单独cap3000；旧roll cap900不变；总上界2080+3000=5080。禁止sum后整体clip。
在phi=.3154时新项约71.148，旧项70.8；phi=.3676时新项691.2，旧项175.2。此处目的是不允许用较好的速度/转角结果持续交换工作界越界，不是强求所有转弯都直立。
这是学习软成本，不声称硬安全保证。不能把该成本达标当作无限时域安全证书。

`teleop_env.py`已经计算peak_roll；把它显式传入interval_cost/smooth_state_cost的新可选参数，不需要改物理引擎或再展开预测。旧模式缺参数时按旧行为；V5.2强制要求peak值存在。记录endpoint与substep peak两者。

### 4.2 alpha1参考偏好改为只惩罚主动降速，不惩罚必要正补偿

旧速度参考项H((abs(dv)-.05)+/.20)改为：
```
C_ref_priority = 4*chi*(1-g) * (
  (1-alpha)*H(max(abs(dd)-.01,0)/.05)
  + alpha*H(max(-dv-.05,0)/.20)
)
```
速度实际超速成本与真实速度主误差不变，正补偿不是额外奖励。+0.10m/s参考修正可以没有这个“让步成本”，但actual超速仍然受到处罚。
原因：当前alpha1中上层+.04358与底层-.09315相加才得到actual欠速-.04957；若大幅处罚所有正修正，就会压制对下层偏差的合理补偿。不能把“速度优先”理解成“速度参考绝不允许超过原始值”。

### 4.3 普通和恢复速度死区略缩小

`normal.speed_deadband_m_s`和`recovery.speed_deadband_m_s`由.03改为.01；权重8、尺度.1不变。其它deadband与验收阈值不变。目的是修正直行/变速时实际速度偏高的无益干预，而不是处罚所有残差幅度、消除有效补偿。

### 4.4 计分与失败一致性

正常控制区间仍：r=-.1*.005*sum(independently_capped_components)；四个区间构成一个20ms策略奖励，rate/acc仍只按20ms定义计算一次。
物理失败在首次发生的策略区间替换为：
`r_fail=-5-.1*.02*5080*(1-gamma**N)/(1-gamma)`。
N包含当前区间，按真实16秒终点计算；不能在训练rollout128步切段当作任务结束。新的高位失败目标可能较大，记录Critic梯度/价值损失并执行既有clip，但不能把失败数据删除。
独立分项cap总和自动检查，不能散落硬编码。

本轮不加完整XY目标、不奖减速本身、不奖两端动作差异、不处罚底层为了平衡的反打，不把真实速度替换成wheel proxy以制造达标。

## 5. 初始化与训练——不把已经学出的100轮策略再丢弃

这是新reward分支的Actor热启动，不是旧reward无改动续跑：
- 各自加载父V5.1 update100 Actor与log_std；父权重不覆盖；
- Critic新初始化、Adam新初始化；新状态下先各4个rollout只拟合Critic（8epochs/rollout，Actor固定），然后正常PPO；这些不算策略更新；
- 真实物理/ESO/history用原prepared_bank开始新回合；没有完整物理checkpoint时，不声称逐物理步精确续接；
- alpha0/1各自独立优化；记录parent_updates=100、additional_updates、lineage_updates=100+additional，不把新训练200写成从零300。

原因：当前100轮后段仍明显改善，主目标已具有意义，这轮是共同安全边界和补偿目标的有限修订，不需要再次从零发现减速。新Critic不承接旧奖励价值。保留当前Actor不等于保证后续不遗忘，best永远单独保护。

PPO保持512env、128policy steps、4epochs、4minibatches=16384；Actor1e-4、Critic3e-4、gamma.997、GAE.99、clip.2、entropy.001；两个grad norm各1；载入各自logstd，保持速度[.08,.4]和转角[.03,.2]边界。不另加噪声或新网络。
保持V5.1从一开始就16秒混合任务和20%初始航向误差族，不再5秒→16秒课程切换。

本轮默认每端追加200次策略更新；若后期仍在改善且未达到任务目标，按第7节最多再追加两组100，即每端最多400新增策略更新。相当于旧100+新200～400的学习历史；没有固定20分钟超时把未收敛run截断，但必须累计并如实报告墙钟和计算步数，不影响其他项目，不自动超过新400。
初始4个value-only rollout与评测预算另计，不混入PPO回合数。

KL不改：soft .01结束本批后续epoch；hard .03回滚该epoch、Actor LR减半（最低1e-5）、新采样；连续3批硬拒绝停，非有限立即停。不存在的另一alpha组不能制造NaN。

## 6. best model：保存候选、晚期验证、最终用best

### 6.1 不再20/25轮评价

删除当前run/evaluator里硬编码evaluation20/evaluation25/阶段通知。训练小批数值检查不是场景评价，最多一次8环境×16步×2更新。
新训练前99次不跑固定工况评测；parent100已保存结果可只读复用。
验证发生在additional=100,150,200；如扩展，250,300,350,400。只跑数值、不逐次渲染大片PNG/视频，最终best才出完整图。

### 6.2 保存规则

每10次新增更新保存一次完整learner；每次中断/最后另存last。保存Actor/Critic、Adam、std、RNG、LR、更新数、配置hash、lower/观测/动作schema身份；注明是否保存物理状态。
`last_model.pt`仅用于恢复训练。禁止将最后checkpoint直接复制成best而不给指标。
允许额外保存`train_best_candidate.pt`，依据最近至少256个完整回合、分族加权的回报/物理统计，用于防止临时回退；它不是正式best。回合跨多个策略更新时标明区间版本，不能将混合版本平均回报虚报为单一checkpoint成绩。

正式`best_model.pt`来自下述确定性固定验证；对新模型与incumbent用完全相同的协议。parent100可作为初始候选：已有相同轨迹全部字段足够时离线重评分及物理指标复用即可，无新增仿真；新reward下的训练reward不能与旧reward直接比较。未实际验证完整协议的parent不可冒充best。

### 6.3 固定验证与精确选择

沿用原fixed_command_six_v1六场景、seed77001、同prepared bank[3]；五个场景10秒，fast_turn一次跑16秒同时保留0–10与10–16窗口。各alpha是其本端任务，每次只执行该端Actor。B0同协议已有数据可复用；不要重新跑六遍底层。
输入始终为原始发布指令，参考不重定位。评测动作取Actor均值、不采样；不进行online fine-tuning。

保存每case：真实失败、substep peak是否>.302、越界区间，真实主误差、次误差、末.5s联合保持、航向恢复时间、下层跟踪和上层平滑。
主误差考核在[1,6)中raw已静稳.3s的样本：如果chi>=.8，alpha0转角RMSE<=.05、alpha1速度RMSE<=.08；其余普通静稳样本同时速度<=.10、转角<=.04。不满足样本数量50个时N/A而非误差0。原[1,6)、[2,4)RMSE照常另报，保持前后可比。
末[9.5,10)每步要求abs(ev)<=.10、abs(ed)<=.04、abs(epsi)<=.05、peak_roll<=.302；fast_turn16秒末保持另记，不覆盖10秒失败。

每候选计算：
1. physical_failure_cases（包含任何nonfinite/policy_fault/真实物理失败）；
2. working_limit_failure_cases（任一保存substep peak>.302的case）；
3. primary_failure_cases（上述本端主要误差不合格的case）；
4. joint_final_hold_failure_cases（10秒末保持失败case）；
5. physical_quality_score Q。
按此元组从小到大比较。Q是各case的均值：
`H(primary_rmse/primary_tol)+.2H(secondary_rmse/secondary_tol)+H(final_heading_rmse/.05)+H(final_speed_rmse/.10)+H(final_steer_rmse/.04)+2*working_violation_fraction/.01`。
冲突secondary alpha0 speed尺度.60，alpha1 steer尺度.15；普通case主项为两种归一化RMSE之和，secondary记0；含混合普通/冲突mask时分别按各有效样本数加权（先分别归一化，不能改变本端物理容差）。H同reward定义，仅用于排序，不反向训练。
前4项相同后，Q至少改善.5%才替换incumbent；否则保留旧best，不因更晚的update覆盖。若Q近似相同只报告平滑差异，不制造更优结论。附件helper只定义元组比较，repo adapter负责从完整轨迹计算上述输入。
`qualified=true`须前4项全部0；没有合格模型仍可保存best，但`qualified=false`、记录每项失败原因。最佳不等于合格。

原子保存best_model.pt + best_actor.pkl + best_model.json + validation_metrics.json；JSON至少有源ckpt路径/sha256、parent/additional/lineage更新数、配置hash、protocol_id、固定种子、score tuple、qualified、选择候选集合。验证结束前失败/部分轨迹不得参与best替换。
最终绘图、部署导出和独立测试必须重新从best_model.json解析并加载best权重；不得偷偷使用内存中的last；打印SHA与step并断言一致。不要声称“整个训练过程每个checkpoint中的全局最优”，只能称“已验证候选中的best”。

## 7. 收敛判断与追加规则

“达100/200轮”“KL小”“零摔倒”都不是收敛。每25新增更新统计真实完整回合（分训练族），报告reward、主/次误差、工作界时长、恢复成功与时间、lower稳定跟踪、Actor/Critic梯度和KL。
保持raw step reward用于诊断，不以其6批相位起伏选best。未从完整回合采集到样本的族标no_samples，禁止补0。

在new200、new300做续训决策，new400必须停止并报告：
- 若连续两次晚期验证合格：可按任务目标完成停止，不宣称数学收敛。
- 若相同前4项下，最近50–100更新验证Q改善>=2%，或前4项有改善，且训练同族物理任务成本窗口改善>=3%、没有数值发散：未收敛，自动再+100（不超过400新更新）。
- 若最近3次验证前4项不变、best Q改进<1%，且最近4个25更新完整回合任务成本相对变化<2%：经验平台期。若不合格，标plateau_unqualified并停止盲加，不称训练成功。
- 若训练回报改善但验证持续无改善/更差：不自动无限追加；允许最多一个额外100块确认，之后报告训练/验证分布差异；历史best保持。
- 主要目标有改善但共同安全明显退化时，不用“总reward提升”批准更差策略；仍用安全优先best，并诊断工作界项。

新reward起始的适应期不能和旧reward片段直接拼成一条“收敛曲线”。图中清楚标parent100与V5.2开始。

## 8. 需要修改的代码范围

已推送实际源码可读；复用当前：
- `direct_command_reward.py`：working_roll_excess、reference_priority单侧改法、普通/恢复速度deadband；峰值传参；
- `direct_command_env.py`：把physics log的peak_roll传reward；诊断与终止保持；
- `smooth_command_config.py`：独立resolve_v52/白名单patch，验证reward cap总和5080，不覆盖父配置；
- `smooth_command_training.py`：合法Actor-only热启动、Critic warmup、追加更新计数、best回调、后期稀疏验证、收敛扩展；
- `preference_command_reporting.py`：lower误差全时间线、post-governed tracking、best加载身份和新reward audit；
- `learning/cli/train_smooth_command.py`：新增明确的`--preference-v52 --repair-parent --additional-updates --max-additional-updates`，避免旧V51入口无声忽略预算。

可采用新入口（这些参数须先实现，不能假定现有版本已有）：
```
python learning/cli/train_smooth_command.py \
 --preference-v52 \
 --repair-parent <已核实V5.1父run> \
 --additional-updates 200 --max-additional-updates 400 \
 --output runs/preference_v52_tracking_best_<timestamp>
```
没有传错schema就不重新构建整个PPO或物理系统。训练best选择和收敛判断应独立于画图流程，不因图像预算中断训练或伪报完成。

## 9. 仅必要测试与最终交付

只新增四类必要检查：
1. e_task=e_upper+e_lower的逐步恒等式、时间对齐；
2. reward_delta_reference.py的公式/cap/failure检查；区间峰值不漏罚；
3. best-selection safe beats unsafe、partial不参与、later-worse不覆盖、final加载best SHA；
4. 最多一次8环境×16策略步×2更新，确认finite梯度和新入口参数；不跑第20/25轮场景评价。

已有物理、底层身份、ESO契约直接复用。不要重新搜索“是否存在减速解”。不要不加判断地罚所有e_lower，否则可能压制必要的速度正补偿。

最终交付：best与last各自身份/指标；晚期候选表与收敛/扩展决策；实际raw/governed/actual三层对比；两通道lower误差全程和异常时段；工作界峰值/持续时间；10秒与16秒恢复；本端主要偏好是否达标。末次最佳固定验证后的独立检查最多加一条预先稳定高速再转弯的指令，和一条未见随机指令，不据此宣称大范围泛化。

只有转向+速度+航向才是本轮目标；世界XY可以落后，无XY成本时不得把大XY距离单独当作训练无效的证据。所有源码/配置/预算差异以小commit交付，不自动push；不停止R244/JIT/用户其他任务。
