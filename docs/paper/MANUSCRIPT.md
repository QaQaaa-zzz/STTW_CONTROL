# 容扰度引导的单轨机器人偏好区间规划与反馈恢复

**Disturbance-Tolerance-Guided Preference Range Planning and Feedback Restoration for a Single-Track Robot**

> TCST方向内部研究主稿；证据更新至2026-10-29。当前版本为U170固定中层、43特征后续策略预测器与10Hz几何接回。尚未投稿就绪，人类作者核验、独立重复与实车验证未完成。旧H75三候选原型不作为本版性能证据；[旧主稿快照](history/pre_return43_revision2737/MANUSCRIPT.md)和[EVIDENCE_DETAILS.md](EVIDENCE_DETAILS.md)保留此前方法、负结果与近邻文献研究。研究决策唯一入口为[项目台账](/home/qy/STTW_CONTROL/research-hub/STTW_PROJECT_STATE.md)。

## 摘要（中文审阅）

单轨机器人在低速转弯时，路径跟踪与速度保持的偏好会改变后续抗扰表现；较小的名义跟踪误差或侧倾角并不直接代表较大的扰动余量。本文研究冻结控制行为下的容扰度引导偏好区间规划。固定中层U170提供路径—速度偏好条件下的参考修正，一个小型监督预测器联合评价候选偏好、准备时间和后续偏好恢复策略的不触地、原目标到达及原路径误差。非学习协调器优先保留用户偏好，在模型筛选的候选集合中选择执行值；准备后依据当前姿态、速度和参考权限平滑恢复用户偏好，10Hz几何接回保持原任务目标。低速左弯的同一24N等效阵风下，固定α=0触地而α=1存活，但后者未到原目标。模型冻结后的五条新仿真覆盖两个低速弯道、三个来源状态，其中四条实际使用网络选择，另一条验证预测区间上端；全部存活并到达原目标。结果支持有限开发域中的控制机制与实现可行性，不构成连续区间安全认证、预测概率校准或独立泛化证明。[E250–E258]

## Abstract

Low-speed turning exposes a coupling between tracking preferences and disturbance tolerance in single-track robots. We study preference-range planning while keeping the reference-generating middle policy and the lower controller frozen. A compact supervised predictor evaluates candidate preferences, preparation time, and a specified feedback-restoration policy through no-contact, original-goal, and path-error outcomes. A nonlearned selector retains the user preference when predicted feasible and otherwise selects a queried candidate. Sensor-based restoration subsequently returns execution toward the original preference, while geometric replanning preserves the original goal. In a low-speed turning example, an identical equivalent gust caused contact for the path-prioritized setting, whereas the speed-prioritized setting survived but missed the goal. Five new simulations after model freezing, covering two low-speed geometries and three source states, all survived and reached the original goal; four used neural range selection and one tested an estimated range endpoint. The evidence supports a finite-domain control prototype. Continuous interval safety, calibrated probabilities, independent generalization, and hardware performance remain unestablished.

## 1 研究问题与贡献定位

用户给定α_user，α=0偏路径、α=1偏速度。研究问题不是如何让所有任务指标同时最优，而是：当用户偏好对应的未来抗扰能力不足时，何时以及向哪些偏好暂时调整，才能保留完成原任务并恢复偏好的机会？用户偏好、原世界目标和原路径评分始终保留；执行α可以临时不同。[E250,E259–E261]

拟向TCST呈现三项控制层面的贡献。第一，以低速转弯的同扰动实验刻画偏好、准备时间与闭环抗扰结局的耦合。第二，把“候选α及其后续返回策略”作为预测对象，联合筛选容扰与任务表现，输出可多段的候选区间。第三，将区间选择、传感器反馈返回和原目标几何接回组合，展示原任务完成的闭环实例。MLP、分层控制或加入α本身不作为首次性主张；正式新颖性论证还需与最接近的偏好控制、参考治理及学习恢复预测文献逐项对照。旧稿的文献记录可作入口，但不同旧物理/策略下的基线结论不能直接支持本版优势。

TCST强调具体控制工程问题中的技术贡献，涵盖分析、设计、仿真及硬件。当前整理按完整Paper组织，版面尚未套用IEEE双栏模板；官方投稿上限为Paper 16页、Brief Paper 8页，含图。此为2026-10-29核对的编辑要求，并不意味着本文已经达到录用条件。[R1,R2]

## 2 问题定义与实际控制系统

当前固定中层为STTW_MIDDLE_H75_LOW_SPEED_MIXED_U170，由H75加入低速小弯混合场景续训后选出。后续容扰学习不再更新中层或ECBC、ESO、local300底层。中层50Hz，底层200Hz，每底层步含25个0.0002s物理子步；前后轮摩擦系数均0.8。重规划模块按10Hz检查/更新接回路径，原世界目标不变；这不是小网络以10Hz连续重新决策的证据。[E259,E261]

```text
原路径、原目标、速度请求、alpha_user
       ↓ 几何跟随 / 10Hz因果接回
当前观测、前方曲率、候选中层参考修正、假设扰动和准备时间
       ↓ 小型联合结果预测器
通过抗扰/到达/路径阈值的候选集合 → 区间表示
       ↓ 用户优先选择 + 当前传感器反馈恢复
alpha_exec → 冻结U170（50Hz）→ 原参考滤波与权限
       → ECBC + ESO + local300（200Hz）→ 车辆
```

物理完整快照包括车辆、ESO、执行器、参考滤波与历史观测，分支接续不重置这些量。预测器只使用当前可构造观测的子集和候选查询，不读取完整未来轨迹或真实起扰日程。当前观测含真实车体速度；实车应提供相应估计，后轮轴速乘半径不能无验证替代。所有新结果均为仿真。[E259–E261]

本文的风为固定世界水平向量上的质心等效外力：方向(-sin0.8, cos0.8, 0)，持续0.4s。20N或24N是力幅值，不是气动标定风速。低速检验使用完整直立观察窗真实前向速度绝对值不超过2.1m/s。该门槛是本研究工况定义，不是稳定性定理。[E250,E254–E256]

## 3 后续策略条件的容扰—任务预测

### 3.1 预测对象与监督标签

定义 f_ψ(z,a,d,τ,α_user; π_return) → (q_no-contact, q_goal, ê_path, v̂_peak)。其中z取源时刻最新状态与路径上下文，a为候选偏好及U170对应参考修正，d为假设扰动方向、幅值、持续时间，τ为准备时间；π_return为冻结的传感器返回规则。q_goal标签要求到达原目标且完整窗口存活。窗口截止原任务10s，触地轨迹在真实失败处截止；到达为接触/故障前进入原目标1m圆，不要求终端停车、航向对齐或驻留。[E251,E259–E261]

43个特征包括最新观测34维、候选/假设查询8维及原用户α一维。观测含速度、姿态、参考滤波以及当前/前方曲率等信息。架构为43→16ELU→16ELU→4，三个随机初始化成员分别预测后平均；每成员1044参数。两分类头使用BCE，路径与速度头学习log1p目标，损失权重1/1/0.25/0.25，失败样本不参与连续量回归，避免把早失败短轨迹误当低代价。AdamW学习率及权重衰减均0.001，三个成员各5000次CPU监督更新，固定末轮。训练拟合良好不等于校准或泛化。[E251,E252,E260,E268]

训练数据为232条查询、188条独特物理轨迹：185条“原用户α等于执行α”的恒定行为等价查询；44条在允许返回前已经触地的因果前缀失败查询；3条真实反馈返回轨迹。44条派生查询不计作新增物理实验或独立重复。新验证轨迹未用于该次拟合；来源状态和邻近恒定策略数据属于已知开发域，不称独立holdout。[E251,E252]

### 3.2 区间与实际选择

对预先声明的假设族D，在每个候选取最小不触地/到达分数和最大路径误差预测。候选集合定义为：

```text
A_ok = {a in A_query : min_d q_no-contact(a,d) >= eta_q,
                       min_d q_goal(a,d) >= eta_g,
                       max_d e_path(a,d) <= e_max}
```

本文实例采用eta_q=0.8、eta_g=0.5、e_max=4m。4m是开放平面强扰动实例的原路径RMSE门槛，不是道路边界；RMSE从源时刻算至首次到达，未到达存活轨迹用剩余观察窗。速度头仅作诊断，不作为已证明上界。不能将上述分类分数直接解释为具有覆盖率保证的安全概率。[E254–E258,E260]

原五档0/.25/.5/.75/1保留，当前研究另查询.875。区间连接相邻合格查询点，遇到不合格点断开；尚未查询的连续内部未验证。α_user在集合内时优先执行原值，否则从已查询的合格点最小化 (a−α_user)² + 0.1(a−α_previous)²。原方案中完整J_user目标目前落实为任务/路径预测门及偏好距离项，未实现长时域J_user优化，不将其写成现有算法。空集合或参考禁止发布时保持前值并标记退化，不宣称退化安全。[E260,E261]

### 3.3 执行与反馈返回

当前闭环实验在声明的源时刻查询一次区间。候选执行至少2s后，当当前侧倾绝对值≤15°、侧倾角速度≤0.35rad/s、前向速度≥1.5m/s且参考允许发布，连续满足0.3s后以每秒1α的上限返回α_user；条件变坏时暂停。定时从方案起点计算，不读真实外力的开始/结束标签。用户α用于原任务奖励，实际α用于中层查询，切换不清空闭环状态。[E253–E256,E261,E265]

预测器必须与这一后续策略一致：旧42特征恒定α模型不能直接评价反馈返回方案。新的manifest绑定43特征、返回参数和实现身份。这种一致性是方法复现要求，不是稳定性证明。全程连续重选区间、改变返回规则或改变用户偏好均需要对应验证。[E251,E260,E261]

## 4 仿真证据

### 4.1 同一扰动下的偏好差异

![五档固定偏好的实际与原始参考轨迹](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/u170_low_speed_five_alpha_load_boundary_v1/analysis/left90_R1p3_v2p0_t2.2_F+24_fixed_candidate_xy.png)

图1：U170，2m/s请求、R1.3m左90°弯，24N等效力在原任务2.2s开始、持续0.4s，五档α各一条相关确定性轨迹。实际轨迹在真实终点截止，原目标不变。[E250]

| 固定α | 10s物理结局 | 原目标到达 |
|---|---|---|
| 0 | 2.675s触地 | 否 |
| .25 | 触地 | 否 |
| .5 | 存活 | 8.680s |
| .75 | 存活 | 否 |
| 1 | 存活 | 否 |

α0与α1受扰前1–2.2s最低真实速度分别约1.448和1.660m/s；全直立窗最高速度均低于2.1m/s。α0名义侧倾角幅值并不更大，因此不能将“姿态角更大”或速度单一因素当作已证明原因。结果显示同一偏好产生的闭环状态/参考历史与抗扰结局相关；固定α=.5也能完成这个事件，不能据此宣称所有固定α均失败或必须使用学习调度。[E250]

### 4.2 返回用户偏好对原目标的作用

三条开发轨迹从同一完整早状态准备α=.875，在实际起扰1.9/2.0/2.1s时均存活并到达，时间6.165/8.645/6.225s。认证复用同源同扰动的恒定.875轨迹，后者三条都存活但仅一条到达。返回规则改善了这组原目标结果；不同起扰条件共享源状态，不能作为三个独立重复估计成功率。[E253]

### 4.3 冻结网络后的五条新物理验证

![网络选择与返回的实际轨迹](/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/u170_return_policy_nn_confirmation_v1/analysis/left90_R1p3_v2p0_source_t1p0_a0_prep0.925_F+24_fixed_candidate_xy.png)

图2：网络在原任务1s查询固定假设族并选.875，真实起扰1.925s未传入选择器，反馈恢复原用户0。明显绕行仍保留，不将到达称为原路径精确跟踪。[E254]

| 验证类型 | 速度请求 / 半径 | 真实起扰 / 力 | 模型区间；执行α | 到达时间 | 最终α |
|---|---|---|---|---:|---:|
| 新时刻NN选择 | 2m/s / 1.3m | 1.925s / 24N | [.875,1]；.875 | 6.935s | 0 |
| 新时刻NN选择 | 2m/s / 1.3m | 2.075s / 24N | [.875,1]；.875 | 6.980s | 0 |
| 区间上端实测 | 2m/s / 1.3m | 2.000s / 24N | [.875,1]；固定测试1 | 6.735s | 0 |
| 另一来源状态NN选择 | 2m/s / 1.3m | 2.000s / 20N | [0,.5]；0 | 4.930s | 0 |
| 不同低速弯道NN选择 | 1.6m/s / .9m | 2.000s / 20N | {0}；0 | 6.960s | 0 |

五条均完整窗口存活、到达原目标并通过全直立窗速度≤2.1m/s检查；其中四条由网络实际选择，一条为预测区间端点测试。前三条执行返回，后两条始终保持用户0。共两个几何、三个来源状态，不能写成五个独立随机重复或统计可靠性。全部逐步奖励、实际α、原路径误差与真实失败截止均保留。神经选择前的实际未来事件输入检查为False。[E254–E258,E264]

相同早状态的模型分数阈值从.8升至.9999时，输出从[.875,1]收缩为{1}，说明接口响应阈值；该数字不表示99.99%真实不触地概率。不同低速来源的24N假设查询可返回空集，空集没有被删除。[E262,E266]

### 4.4 预测局限与已知失败

旧恒定策略V2在新准备时间上的四条实际推荐中三条触地，尽管之前32条α插值查询表现较好。增加监督更新改善训练拟合，也未自动修复时间间隙。新43模型将原用户及返回策略明确纳入预测对象；这些先前失败保留在研究台账和独立run，不能从新五条成功回写成旧模型可靠。[E263,E267]

新五条在默认筛选门槛下未出现预测可行而触地/未到达的案例，但路径预测存在误差；更严格2m路径门槛下，先前两条时刻验证存在一次误判。速度峰值头在上端案例低估约0.276m/s，因此不能用其作约束证书。五条正例不足以校准失败风险。[E257,E258]

## 5 面向TCST的论点与最小后续工作

现有数据已经形成可写作的应用控制故事：路径优先可能降低未来容扰能力，过晚切换可能失效，提前准备与反馈恢复可以同时服务存活和原目标。论文价值应落在这一闭环问题、可执行的策略条件预测及实际效果上，不要求所有场景全面领先。但目前五条成功轨迹仍不足以支撑宽泛可靠性或神经预测器不可替代性。

后续按以下顺序推进，均为尚未执行的工作，而非当前结果：

1. **冻结后的独立小面板。** 保持模型、规则、阈值及物理，预先锁定少量新的初态/左右弯/起扰组合，报告全部成功与失败及同一统计单位的结果；不在看到结果后改条件仍称独立测试。不为凑成功扩大训练，也不穷举checkpoint。
2. **任务与方法归因。** 先整理已有同源恒定α缓存，展示准备、保持与返回的差异，分开到达、触地、原用户代价和路径偏离。现有固定.5成功反例必须保留。新的速度支持或外部强基线按用户当前决定暂不启动；是否需要补充由最后主张决定，而不是要求大幅胜过某个指定基线。
3. **实车与论文定稿。** 在明确实车授权及传感器/执行接口后验证关键低速转弯—扰动—恢复过程；同步完成最近邻文献对照、作者核验和IEEE格式。实车尚未执行，不把仿真触地或等效力结果写成硬件验证。

当前方法不包含已证明的UCEDT牛顿阈值。UCEDT可作为平衡潜力的物理解释候选；若在正文使用其公式，需要另行核验原文、适用条件与当前闭环差异。本版容扰依据为实际闭环不触地标签，避免将解析势能量直接当真实外力安全界。

## 6 结论

本文实现了面向低速单轨机器人、以冻结偏好控制行为为基础的容扰—任务区间规划原型。共同扰动下存在固定偏好的触地差异；在明确的后续反馈策略条件下，小型预测器可以生成依赖状态与阈值的候选区间，并支持新时刻及相关不同低速弯道中的任务完成。现有证据支持有限域可行性及继续面向TCST开展验证，不证明连续区间保证、独立泛化、概率校准或实车性能。

## 证据与流程说明

E250–E268为本版当前证据，路径与SHA由本地builder解析；每项标记AI artifact_checked，人类核验未记录。旧E001–E249保留在历史主稿与证据明细，不能与当前物理/策略混算分母。本文仅在本地整理与导出；未上传未发表稿件、未投稿。作者、基金、利益冲突、数据公开许可与最终AI披露尚待人类确认，不编造声明。

[R1] IEEE Control Systems Society. Transactions on Control Systems Technology: Aim & Scope. https://www.ieeecss.org/publication/transactions-control-systems-technology （访问2026-10-29）。

[R2] IEEE Control Systems Society. Author Information: Transactions on Control Systems Technology. https://www.ieeecss.org/publication/transactions-control-systems-technology/author-info （访问2026-10-29）。

流程软件：Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. 2026. https://doi.org/10.48550/arXiv.2609.00065 （当前arXiv v2元数据核对2026-10-29）。scientific-writing用于本地证据组织与表述检查，不是控制方法或实验性能依据。
