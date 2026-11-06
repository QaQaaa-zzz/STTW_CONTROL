> 第三章台账只读交付快照；实时维护入口为 research-hub/CHAPTER3_PROJECT_STATE.md。本快照不构成第二维护入口。

# 第三章研究台账：能力感知规划与动态技能协同执行

> 用户于2026-11-06明确要求新建，只维护第三章。此文件为第三章详细事实与后续行动的维护入口；第一章STTW论文/实验及第二章RecoveryBridge独立研究不在此记账。STTW_PROJECT_STATE.md保留既有历史与本入口索引，不搬走或删除历史证据。

```yaml
schema_version: 1
project: STTW_CONTROL_CHAPTER3
state_revision: 14
parent_history: STTW_PROJECT_STATE.md
updated_utc: 2026-11-06T15:05:56.582536+00:00
```

## 维护规则

- 每次实验启动/结束/失败、代码或配置变化、采用/否决、结论更正、用户新要求、交付/通知/push，都在这里增量维护。
- 更新时持同目录 `.write.lock` 独占锁，锁内重读本文件、递增本文件 `state_revision`。项目总账只留跨章节决定与入口索引；不把第一章并行线程的状态覆盖成本章状态。
- 每项记录包括：目的、方法/奖励、模型与配置、代码版本、预算与数据角色、实际结果/负例、采用状态、证据路径、下一步。没有训练则明确“无新奖励/0优化更新”。
- 执行、评估、采用分别记录；计划不算结果，存活不算完整任务成功。实验图以全程XY实际/原参考为首，失败数据保留。
- 索引中的旧记录属于历史证据，本次不是逐实验重跑认证；近期明确核验的结果与运行状态另列。找不到的身份或证据标为待核对，不补造。

## 当前目标与用户最新约束

第三章：已知地图与自然语言任务下，根据残余动态状态选择路线、速度、alpha、连接及技能交接，并在受扰后修订剩余任务。四模块：任务规划、几何与能力工具、执行管理、已有运动技能。

**当前优先打通：起伏路段 → 转弯对准平台 → 平整起跳区 → 跳跃 → 落地续行。** 单纯直线接跳台已通过，但用户明确认为任务太简单，不作为当前复杂任务完成证据。

- 跳跃区域不叠加起伏；当前成功配置出弯至台沿保留9.6m平地（平台整体后移4m）。
- 暂停大模型部署，无模型下载/新服务；不以LLM作为当前前置。
- 完整复杂任务打通以后Bark通知；精简当前情况、必要代码/配置/结果，推送现有远端。远端为 `https://github.com/QaQaaa-zzz/STTW_CONTROL.git`，工作分支 `experiment/sttw-offroad`。截至本次建账，本章新交付尚未push。
- 复用成熟实现，底层够用即推进；重仿真排队，不并发抢GPU。新RL训练需有界预算与实际需要；当前无RL训练。
- 原STTW负责地面连续任务；带摆原RB车负责跨技能，Go2仅为软件/训练处理参考。后续若训练保留已授权Go2随机化。

## 路径与身份

- 工作树：`/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad`（下文相对代码/runs路径以此为根）。
- 数据盘：`/media/qy/6A7E3DB07E3D75C3/STTW_runs`。
- 原STTW：冻结H75 u77与用户指定的橙色原Local300，原.0002s物理/.005s底层/.02s中层。
- 跨技能：原RB整车5.091kg、轮半径.07m/全宽.05m、21圆柱组合圆角轮、摩擦.8；当前复杂链.0025s物理/.005s底层/.02s策略（早期简单链.005s物理）；地面锁摆，跳跃释放。
- RB地面：`learning/configs/rb_sttw_rounded_controller.json`，ECBC+ESO；H75使用当前RB几何配置，**zero_lower=true，不是Local300残差在执行**。
- 跳跃：`runs/rb_sttw_platform/selected_jump_candidate.json`中的update2，actor SHA256 `80529e5c6ffdc4036da9ee08a4218193d706ae55e23ce02e9cc7cd840c01be39`。具体适配/能力限制以各run协议为准。

## 当前工作与运行（持续更新；旧尝试见下方历史）

|项目|执行/评价状态|配置与预算|实际证据/下一步|
|---|---|---|---|
|反向6cm出口→左/右弯6候选|执行/评价完成，12/12|0优化，444.61s|前瞻选择v2.3、alpha.5左右弯均完成；所有固定候选也通过，尚不能证明能力选择优于固定参数。|
|起伏→右弯→跳台→续行|单名义DEV完整通过22s|2cm双起伏、R2右弯、9.6m平地、0.16m平台；8800物理步，0优化|`rb_offroad_platform_shift/evaluation`：全程路径最大误差.79024m，严格双轮落台1.26s，交回地面后7.64s/11.295m；此前失败全部保留。|
|简单同车链|执行/评价完成，名义工程接口采用|`learning/configs/isaac_rb_ground_jump_ground.json`；10s/2000physics|`rb_rounded_continuous_chain`完整10s、严格落台1.7s、交回后5.72s/8.959m、全程横偏最大.28541m。仅单场景DEV。|
|大模型|暂停|无下载、无启动、无费用任务|按用户要求暂不推进。|
|远端精简交付|证据包完成，待提交/push|`docs/evidence/chapter3_motion_chain_20261106`|代码、配置、全程XY/轨迹、指标和负结果；离线重算通过；实际push及Bark回执待记录。|

## 已做方法与结果（按问题组织）

|ID|目的/尝试|结果及采用状态|主要证据入口|
|---|---|---|---|
|C3-01|原STTW加摆共同车模型、Local300/H75接口验证|完成早期移植及标准工况；随后用户改用原RB整车，不与现平台混作同模型比较|旧总账3337–3490；`learning/configs/rb_sttw_platform.json`记录后续模型来源|
|C3-02|Phase U/RB Torch接口、奖励移植、Go2 PPO/GPU训练适配|做过Actor/Critic及21项奖励移植、分阶段续训；早期重车结果不冒充原RB能力。具体训练身份见历史索引|旧总账3385–3490及所引run；`pendulum_skill.py`|
|C3-03|原RB底座ECBC、输入输出与轮地掉速诊断|驱动输入和轮轴输出跟上时车体仍掉速；试过SDF密度、接触参数、解析轮及模型响应，不把掉速一律归因网络|旧总账3491–3591，`runs/rb_sttw_platform`|
|C3-04|圆柱与圆角轮、ECBC/ESO适配|单圆柱转弯响应不理想；最终21圆柱圆角轮保持原轮尺寸，采用wc8/wo10/filter8|`rb_sttw_rounded_controller.json`；旧总账3592–3605|
|C3-05|残差从零及偏航奖励训练，累计1500更新|比较训练采样best242、episode峰值713、last1500；残差主动降低速度输入，未采用为当前底层；额外600轮设想已取消|`runs/rb_scratch_yaw1500`、`runs/rb_ground_standard1500`、旧总账3606–3620。非无条件采用best。|
|C3-06|RB锁摆ECBC+H75左右弯五alpha|10个H75条件完整10s并到达；支持地面接口可用，不代表所有地形通过|`runs/rb_h75_ecbc_ground/{left90_R2,right90_R2}/evaluation/xy_all_alpha.png`|
|C3-07|RB接3cm起伏及地面6候选|external/replicated入口失败；平面+局部凸起改善入口但仍在凸起失败，未采用为已通过地形能力|数据盘`rb_connection6`、`rb_connection6_replicated`、`rb_connection6_baseplane`|
|C3-08|原STTW出口配对6候选、经验能力工具|轻/正向6cm各旧接口10/12，固定参数与工具无明确增益；不可拿小样本Brier0声称泛化|`runs/connection_candidates_covered/collection_after_reboot`、数据盘`sttw_connection_bump6_positive`|
|C3-09|修复到达后无限直线造成原路偏离|恢复原路线后正向6cm12/12完成10s，保留原2m走廊；接口采用|数据盘`sttw_connection_bump6_original_exit/xy_candidates.png`及records.jsonl；`goal_replanning.py`|
|C3-10|能力预测与任务管理接线|已有轻量评价、未知返回、任务程序/剩余目标维护；前瞻选择已接采集器；10项核心测试通过，在线全系统未完成|`empirical_capability.py`、`continuous_task.py`、`collect_connection_candidates.py`；提交54b36b03|
|C3-11|ECBC接近→冻结update2跳跃|短接近后上台，8s存活、台上连续严格支撑2.27s；后续尚由jump控制|数据盘`rb_rounded_ground_jump`|
|C3-12|落地直接交回零转向ECBC|8s存活但末端y2.887m，未采用|数据盘`rb_rounded_full_chain`|
|C3-13|落地H75原路恢复|首次因从s0投影导致预瞄在车后，修复为当前位置初始化；早交接续行通过但严格落台仅.5s；晚交接满足1.01s但4.8s侧倾失败|`rb_rounded_h75_chain`、`rb_rounded_h75_landing_projection`、`rb_rounded_h75_settled_chain`|
|C3-14|支撑+速度约束交接|晚交接speed .59565接近ECBC增益公式奇异点.59710；先保留jump姿态控制缓升速，>=1.2m/s再H75接管。旧地图x15末端掉落，非低侧倾成功|数据盘`rb_rounded_speed_handoff`；不把模型奇异性解释当单因素因果证明|
|C3-15|延长末端平地后的完整简单链|10s无终止、严格落台1.7s，采用作为工程接口；已Bark，不代表新复杂任务或论文完整完成|数据盘`rb_rounded_continuous_chain/{xy_jump.png,landing.json,chain_summary.json,bark_receipt.json}`；提交2db66b38|
|C3-16|当前起伏+右弯完整任务|前段2cm双起伏，R2右转，出弯到台沿5.6m平地；执行结果待返回|`prepare_rb_offroad_chain.py`、数据盘`rb_offroad_turn_chain`；提交4113386d/9a04f71b|

所有当前验证均为仿真DEV；没有新增实机验证。原STTW接续与原RB跨技能分开记录，模型、时序、控制器不同，不直接混合奖励。后续训练奖励、域随机化及更新数必须按实际新协议记，不由本表推定。

## 当前复杂任务的完成定义

在运行前配置中声明：完整18s无物理/控制终止；依次通过起伏前后、弯入口、弯出口和起跳前检查点（半径1m）；全程距离原固定路径≤1m；严格双轮台上连续支撑≥1s；交回地面控制后继续≥2s且前进≥2m。保存原始XY、接触与全程时序，不以单末端状态取代。

`analyze_isaac_pendulum_landing.py`检查接触，`analyze_rb_continuous_task.py`汇总任务阶段。二者结果尚未返回时不标完成。任务几何、窗口及门槛若后续改动，必须写清原因并保留旧结果。

## 未完成与下一步

1. 收取当前两个已登记任务；复杂链失败则定位真实阻断，保持用户要求的起伏和转弯，不用简单直线替代。
2. 复杂链通过后精简代码/配置/轨迹结果推送现有远端，检查实际远端提交，并Bark通知。
3. 第三章后续仍缺：足量出口状态的能力评价证据、6/36候选收益比较、受扰任务修订与完整技能对照。LLM和实机当前不执行；不得将工程链打通宣布为整章完成。

## 历史索引（旧总账保留，按关键词回填，非重新复现实验）

以下仅索引与第三章模型/接口/接续相关的历史记录；旧台账的第一章论文和独立STTW实验仍由其原线程维护。较早阶段逐run身份若需要精确比较，应读链接原合同与原始产物。

|旧revision|历史事项|来源|
|---|---|---|
|3649|复杂链执行前明确全轨迹完成条件（revision 3649）|[旧记录](STTW_PROJECT_STATE.md)|
|3648|复杂运动链场景已排队，绘图保留未起跳失败（revision 3648）|[旧记录](STTW_PROJECT_STATE.md)|
|3647|用户增加起伏转弯跳台完整任务，准备场景并排队（revision 3647）|[旧记录](STTW_PROJECT_STATE.md)|
|3646|前瞻选择采集已启动，核心任务接口测试通过（revision 3646）|[旧记录](STTW_PROJECT_STATE.md)|
|3645|推进反向出口6候选，运行前能力决策与实际分支结果接通（revision 3645）|[旧记录](STTW_PROJECT_STATE.md)|
|3644|单场景运动链打通通知已发送（revision 3644）|[旧记录](STTW_PROJECT_STATE.md)|
|3643|同车连续运动链首次通过完整10s，准备Bark通知（revision 3643）|[旧记录](STTW_PROJECT_STATE.md)|
|3642|场地生成路径错误更正并启动正确续行地图（revision 3642）|[旧记录](STTW_PROJECT_STATE.md)|
|3641|速度交接后续行正常至有限地图末端，补足续行场地（revision 3641）|[旧记录](STTW_PROJECT_STATE.md)|
|3640|定位ECBC低速奇异区，采用状态约束交接试验（revision 3640）|[旧记录](STTW_PROJECT_STATE.md)|
|3639|延后交接失败保留，完整链尚未验收（revision 3639）|[旧记录](STTW_PROJECT_STATE.md)|
|3638|落地路径接管完成8s，延长稳定支撑后再交接（revision 3638）|[旧记录](STTW_PROJECT_STATE.md)|
|3637|落地H75接入定位到进度初始化错误并原位修复（revision 3637）|[旧记录](STTW_PROJECT_STATE.md)|
|3636|落地直行接口偏航积累，复用H75原路跟踪接回（revision 3636）|[旧记录](STTW_PROJECT_STATE.md)|
|3635|短接近跳跃上台通过，接入落地后ECBC续行（revision 3635）|[旧记录](STTW_PROJECT_STATE.md)|
|3634|原路线恢复12/12完成，继续同车地面到跳跃实测（revision 3634）|[旧记录](STTW_PROJECT_STATE.md)|
|3633|原路线恢复复核结束；调通底座后复用短接近跳跃接口（revision 3633）|[旧记录](STTW_PROJECT_STATE.md)|
|3632|用户明确先打通第一部分，暂停语言模型准备（revision 3632）|[旧记录](STTW_PROJECT_STATE.md)|
|3631|到达后恢复原路线，复用同出口快照复核12候选（revision 3631）|[旧记录](STTW_PROJECT_STATE.md)|
|3630|6cm来源12候选完成：当前无优于固定参数证据，尾段管理需修正（revision 3630）|[旧记录](STTW_PROJECT_STATE.md)|
|3629|能力查询与任务管理复用入口已接通，采集按原预算继续（revision 3629）|[旧记录](STTW_PROJECT_STATE.md)|
|3628|能力评价加入同组固定参数对照，原STTW强扰动采集继续（revision 3628）|[旧记录](STTW_PROJECT_STATE.md)|
|3627|回归原STTW地面主实验，执行已准备强扰动接续来源（revision 3627）|[旧记录](STTW_PROJECT_STATE.md)|
|3626|采用平面承载与局部凸起网格组合，保留原高度任务（revision 3626）|[旧记录](STTW_PROJECT_STATE.md)|
|3625|复制地形未解决入口失稳，保留两次真实失败不扩训（revision 3625）|[旧记录](STTW_PROJECT_STATE.md)|
|3624|首个地形入口失败保留，采用既有复制地形布局继续接续（revision 3624）|[旧记录](STTW_PROJECT_STATE.md)|
|3623|当前锁摆底座6候选接续采集预算（revision 3623）|[旧记录](STTW_PROJECT_STATE.md)|
|3622|当前ECBC加冻结H75完成左右弯五alpha，进入接续阶段（revision 3622）|[旧记录](STTW_PROJECT_STATE.md)|
|3621|冻结H75接当前RB底座ECBC地面验证启动合同（revision 3621）|[旧记录](STTW_PROJECT_STATE.md)|
|3620|用户明确底层够用即推进，取消未启动600轮设想（revision 3620）|[旧记录](STTW_PROJECT_STATE.md)|
|3619|七工况指令响应诊断：残差主动掉速，旧车XY不能代替新底座请求跟踪（revision 3619）|[旧记录](STTW_PROJECT_STATE.md)|
|3618|地面1500训练与原七工况回收，复用及吞吐要求继续执行（revision 3618）|[旧记录](STTW_PROJECT_STATE.md)|
|3616|七工况四候选评测进入训练之后的GPU队列（revision 3616）|[旧记录](STTW_PROJECT_STATE.md)|
|3615|完整七工况复用入口准备，修正历史半径及子步硬编码（revision 3615）|[旧记录](STTW_PROJECT_STATE.md)|
|3614|阶段末地面配对图复用统一入口，不新增仿真（revision 3614）|[旧记录](STTW_PROJECT_STATE.md)|
|3613|累计1500路线真实恢复并运行，监视按预计时长（revision 3613）|[旧记录](STTW_PROJECT_STATE.md)|
|3612|地面从零训练路线扩大至累计1500轮，复用Go2恢复入口（revision 3612）|[旧记录](STTW_PROJECT_STATE.md)|
|3611|偏航奖励候选保留改善与退化；不按last采用（revision 3611）|[旧记录](STTW_PROJECT_STATE.md)|
|3610|偏航奖励scratch150完成，进入同工况轨迹测试（revision 3610）|[旧记录](STTW_PROJECT_STATE.md)|
|3609|底层奖励缺少真实转弯响应约束，新增偏航角速度项候选（revision 3609）|[旧记录](STTW_PROJECT_STATE.md)|
|3608|第三章优先复用成熟模块与仿真吞吐（用户明确）（revision 3608）|[旧记录](STTW_PROJECT_STATE.md)|
|3607|从零残差三个候选固定地面轨迹评估（revision 3607）|[旧记录](STTW_PROJECT_STATE.md)|
|3606|从零残差150轮完成，初始零输出已核实（revision 3606）|[旧记录](STTW_PROJECT_STATE.md)|
|3605|ESO wo10候选改善直行；从零残差150轮合同（revision 3605）|[旧记录](STTW_PROJECT_STATE.md)|
|3604|直行漂移定位到ESO补偿放大；降低observer带宽（revision 3604）|[旧记录](STTW_PROJECT_STATE.md)|
|3603|用户聚焦地面底层并允许从零训练；先分离ESO贡献（revision 3603）|[旧记录](STTW_PROJECT_STATE.md)|
|3602|用户明确要求消除RB横向偏移；残差150轮候选暂不采用（revision 3602）|[旧记录](STTW_PROJECT_STATE.md)|
|3601|圆滑轮地面150轮完成，比较三个实际候选（revision 3601）|[旧记录](STTW_PROJECT_STATE.md)|
|3600|新轮域跳跃落地成立；启动有界150轮地面残差适配（revision 3600）|[旧记录](STTW_PROJECT_STATE.md)|
|3599|圆滑轮域跳跃候选配对，检查统一平台可行性（revision 3599）|[旧记录](STTW_PROJECT_STATE.md)|
|3598|圆滑原生轮左右/直行与Local300配对完成，基础运动可继续适配（revision 3598）|[旧记录](STTW_PROJECT_STATE.md)|
|3597|圆滑原生轮候选恢复ECBC转弯，进入Local300配对测试（revision 3597）|[旧记录](STTW_PROJECT_STATE.md)|
|3595|侧倾参考适配保留负例，不以存活代替轨迹合格（revision 3595）|[旧记录](STTW_PROJECT_STATE.md)|
|3592|标准圆柱整车保持速度，转向跟踪尚未通过（revision 3592）|[旧记录](STTW_PROJECT_STATE.md)|
|3591|同尺寸标准圆柱单轮通过滚动诊断，接入整车短评测（revision 3591）|[旧记录](STTW_PROJECT_STATE.md)|
|3590|同外径同宽解析圆柱轮单轮试验（revision 3590）|[旧记录](STTW_PROJECT_STATE.md)|
|3589|标准轮替换保持原外径与宽度（用户明确）（revision 3589）|[旧记录](STTW_PROJECT_STATE.md)|
|3586|原解析MuJoCo单轮对照合同（revision 3586）|[旧记录](STTW_PROJECT_STATE.md)|
|3585|滚动损耗隔离完成；平滑轮形尚未使整车ECBC可用（revision 3585）|[旧记录](STTW_PROJECT_STATE.md)|
|3584|高分辨率SDF单轮减小掉速，回到整车验证（revision 3584）|[旧记录](STTW_PROJECT_STATE.md)|
|3583|RB同轮形高分辨率SDF候选用于隔离离散滚动损耗（revision 3583）|[旧记录](STTW_PROJECT_STATE.md)|
|3581|单轮复现滚动掉速，比较同轮形SDF表示（revision 3581）|[旧记录](STTW_PROJECT_STATE.md)|
|3580|最新STTW结果已按用户要求推送（revision 3580）|[旧记录](STTW_PROJECT_STATE.md)|
|3579|RB轮地推进原因隔离：同轮单体滚动（revision 3579）|[旧记录](STTW_PROJECT_STATE.md)|
|3578|RB输入输出诊断结论与滚动容差负结果（revision 3578）|[旧记录](STTW_PROJECT_STATE.md)|
|3577|RB驱动已输出，试验官方滚动接触容差修正（revision 3577）|[旧记录](STTW_PROJECT_STATE.md)|
|3576|响应用户输入输出诊断：轮速给上且跟踪，车体未跟上（revision 3576）|[旧记录](STTW_PROJECT_STATE.md)|
|3574|RB配对模型加车速环结果归档（revision 3574）|[旧记录](STTW_PROJECT_STATE.md)|
|3571|RB配对ECBC延长直行但掉速，接回受限车速环（revision 3571）|[旧记录](STTW_PROJECT_STATE.md)|
|3569|RB配对响应模型用于下一ECBC候选（revision 3569）|[旧记录](STTW_PROJECT_STATE.md)|
|3567|RB速度反馈候选失败后进行配对局部响应辨识（revision 3567）|[旧记录](STTW_PROJECT_STATE.md)|
|3565|RB新增闭环车速调节候选以避免轮速恒定但车体减速（revision 3565）|[旧记录](STTW_PROJECT_STATE.md)|
|3564|RB反馈滤波短测完成，仍未达到底层可用条件（revision 3564）|[旧记录](STTW_PROJECT_STATE.md)|
|3563|RB实际关节无异常锁定，测试ECBC速率反馈滤波（revision 3563）|[旧记录](STTW_PROJECT_STATE.md)|
|3560|RB俯仰诊断结果及锁摆后关节参数实读（revision 3560）|[旧记录](STTW_PROJECT_STATE.md)|
|3559|STTW远端证据交付完成（revision 3559）|[旧记录](STTW_PROJECT_STATE.md)|
|3558|RB补充俯仰响应定位地面失稳（revision 3558）|[旧记录](STTW_PROJECT_STATE.md)|
|3557|STTW投稿标准与精简证据交付（revision 3557）|[旧记录](STTW_PROJECT_STATE.md)|
|3556|RB重配ECBC两组短测收取（revision 3556）|[旧记录](STTW_PROJECT_STATE.md)|
|3555|RB经验系数候选未改善，独立设计侧倾阻尼反馈（revision 3555）|[旧记录](STTW_PROJECT_STATE.md)|
|3554|RB响应模型ECBC候选短测（revision 3554）|[旧记录](STTW_PROJECT_STATE.md)|
|3553|RB按新车从零配置ECBC后逐层适配（revision 3553）|[旧记录](STTW_PROJECT_STATE.md)|
|3551|RB反馈增强失败与积分步长短对照（revision 3551）|[旧记录](STTW_PROJECT_STATE.md)|
|3549|RB Local150轨迹未改善，转向有界ECBC恢复增益适配（revision 3549）|[旧记录](STTW_PROJECT_STATE.md)|
|3548|RB Local150完成与原模型及训练候选轨迹比较（revision 3548）|[旧记录](STTW_PROJECT_STATE.md)|
|3547|RB地面150训练实际更新与TensorBoard记录（revision 3547）|[旧记录](STTW_PROJECT_STATE.md)|
|3545|RB Local300地面适配150轮启动预算（revision 3545）|[旧记录](STTW_PROJECT_STATE.md)|
|3543|RB候选平台接触证据完成，优先转地面适配（revision 3543）|[旧记录](STTW_PROJECT_STATE.md)|
|3542|短地面交接失败与标准初态落地证据补录（revision 3542）|[旧记录](STTW_PROJECT_STATE.md)|
|3540|RB update2短地面到跳跃交接启动（revision 3540）|[旧记录](STTW_PROJECT_STATE.md)|
|3538|RB150训练结束与四候选轨迹结果：保留update2（revision 3538）|[旧记录](STTW_PROJECT_STATE.md)|
|3536|RB150训练后候选评测入共享队列（revision 3536）|[旧记录](STTW_PROJECT_STATE.md)|
|3535|RB150训练进入实际更新且TensorBoard加载（revision 3535）|[旧记录](STTW_PROJECT_STATE.md)|
|3533|原RB整车150轮适配训练预算与启动（revision 3533）|[旧记录](STTW_PROJECT_STATE.md)|
|3532|RB随机化训练接口短集成通过（revision 3532）|[旧记录](STTW_PROJECT_STATE.md)|
|3531|三策略结果与RB训练原生驱动随机化接线修复（revision 3531）|[旧记录](STTW_PROJECT_STATE.md)|
|3528|RB原整车三策略名义配对回放启动（revision 3528）|[旧记录](STTW_PROJECT_STATE.md)|
|3527|旧RB整车历史策略已有效起跳但落地侧倾失败（revision 3527）|[旧记录](STTW_PROJECT_STATE.md)|
|3525|RB初态加载转换为PhysX张量精度（revision 3525）|[旧记录](STTW_PROJECT_STATE.md)|
|3524|RB跳跃初始化边界精度修复（revision 3524）|[旧记录](STTW_PROJECT_STATE.md)|
|3523|原RB整车历史跳跃策略接入Isaac Lab（revision 3523）|[旧记录](STTW_PROJECT_STATE.md)|
|3521|RB运行时刚体树核对完成，无安装坐标错误证据（revision 3521）|[旧记录](STTW_PROJECT_STATE.md)|
|3519|Local300提前接管结果及运行时姿态核对（revision 3519）|[旧记录](STTW_PROJECT_STATE.md)|
|3517|惯量与落差排查完成，提前接入冻结Local300实测（revision 3517）|[旧记录](STTW_PROJECT_STATE.md)|
|3515|RB运行时惯量与初始落差对照（revision 3515）|[旧记录](STTW_PROJECT_STATE.md)|
|3514|RB转向响应脉冲完成，约化模型拟合不足以替换控制器（revision 3514）|[旧记录](STTW_PROJECT_STATE.md)|
|3512|RB求解迭代未改善，开始受控转向响应测量（revision 3512）|[旧记录](STTW_PROJECT_STATE.md)|
|3511|RB实际碰撞属性已核实，恢复历史求解迭代（revision 3511）|[旧记录](STTW_PROJECT_STATE.md)|
|3510|核对RB导入碰撞API挂载层级（revision 3510）|[旧记录](STTW_PROJECT_STATE.md)|
|3509|RB接触诊断阶段完成且尚未解决直行失稳（revision 3509）|[旧记录](STTW_PROJECT_STATE.md)|
|3508|RB分段轮胎未单独解决，核对接触距离（revision 3508）|[旧记录](STTW_PROJECT_STATE.md)|
|3507|复用历史RB十六段轮胎凸包并启动地面测试（revision 3507）|[旧记录](STTW_PROJECT_STATE.md)|
|3506|RB刚性SDF接触未改善，转查历史凸包表示（revision 3506）|[旧记录](STTW_PROJECT_STATE.md)|
|3504|RB轮胎弹性接触继承问题单因素诊断（revision 3504）|[旧记录](STTW_PROJECT_STATE.md)|
|3503|RB反馈四条件小仿真完成（revision 3503）|[旧记录](STTW_PROJECT_STATE.md)|
|3502|纠正ESO启用说明并检查轮速反馈（revision 3502）|[旧记录](STTW_PROJECT_STATE.md)|
|3501|RB三档ECBC带宽比较完成且均未稳定（revision 3501）|[旧记录](STTW_PROJECT_STATE.md)|
|3500|RB ECBC带宽小批量比较启动（revision 3500）|[旧记录](STTW_PROJECT_STATE.md)|
|3498|RB原生驱动复测完成，伺服爆振消除但ECBC仍需适配（revision 3498）|[旧记录](STTW_PROJECT_STATE.md)|
|3497|RB起步失败定位显式速度伺服，恢复原生驱动对照（revision 3497）|[旧记录](STTW_PROJECT_STATE.md)|
|3496|用户明确允许小规模RB地面对照直接执行（revision 3496）|[旧记录](STTW_PROJECT_STATE.md)|
|3495|RB底层训练入口沿用Go2并适配轮半径（revision 3495）|[旧记录](STTW_PROJECT_STATE.md)|
|3494|按RB锁摆整车重新建立ECBC模型系数（revision 3494）|[旧记录](STTW_PROJECT_STATE.md)|
|3492|RB路径接口轮半径贯通与预热状态保存（revision 3492）|[旧记录](STTW_PROJECT_STATE.md)|
|3491|RB原整车导出与首个地面配对实验排队（revision 3491）|[旧记录](STTW_PROJECT_STATE.md)|
|3490|用户改为RB原整车并重新适配STTW（revision 3490）|[旧记录](STTW_PROJECT_STATE.md)|
|3488|用户确认共同车型并授权提高轮驱动与摆臂能力（revision 3488）|[旧记录](STTW_PROJECT_STATE.md)|
|3487|接续12候选完成与队列误等待修正（revision 3487）|[旧记录](STTW_PROJECT_STATE.md)|
|3486|共同车增重来源逐部件核实（revision 3486）|[旧记录](STTW_PROJECT_STATE.md)|
|3484|历史成功01966080原权重接入共同车评测队列（revision 3484）|[旧记录](STTW_PROJECT_STATE.md)|
|3482|用户要求摆臂限幅诊断与历史成功sim2sim找回（revision 3482）|[旧记录](STTW_PROJECT_STATE.md)|
|3479|600训练完成；标准和提前接管仍未过台，排队比较训练地形表示（revision 3479）|[旧记录](STTW_PROJECT_STATE.md)|
|3478|最高奖励best451完整Isaac视频交付，台前失败保留（revision 3478）|[旧记录](STTW_PROJECT_STATE.md)|
|3477|用户要求最高奖励模型完整视频，明确允许小视频不等待队列（revision 3477）|[旧记录](STTW_PROJECT_STATE.md)|
|3476|发现训练准备阶段与接管时机差异，安排小型配对诊断（revision 3476）|[旧记录](STTW_PROJECT_STATE.md)|
|3474|新奖励阶段曲线出图与600阶段运行证据（revision 3474）|[旧记录](STTW_PROJECT_STATE.md)|
|3472|将600阶段best与last加入同条件接管比较（revision 3472）|[旧记录](STTW_PROJECT_STATE.md)|
|3471|续训队列已启动真实GPU更新（revision 3471）|[旧记录](STTW_PROJECT_STATE.md)|
|3470|用户授权继续训练：补齐450后追加150到600（revision 3470）|[旧记录](STTW_PROJECT_STATE.md)|
|3468|用户澄清仅任务间串行，训练内部保持GPU并行（revision 3468）|[旧记录](STTW_PROJECT_STATE.md)|
|3465|恢复source-only出口快照，避免重跑上游A（revision 3465）|[旧记录](STTW_PROJECT_STATE.md)|
|3464|排队期间完成剩余任务修订接口（revision 3464）|[旧记录](STTW_PROJECT_STATE.md)|
|3463|统一锁串行交接已实测，仅一个CUDA任务（revision 3463）|[旧记录](STTW_PROJECT_STATE.md)|
|3462|串行纠正：让出GPU并等待其他会话评测（revision 3462）|[旧记录](STTW_PROJECT_STATE.md)|
|3460|串行队列恢复训练真实运行（revision 3460）|[旧记录](STTW_PROJECT_STATE.md)|
|3459|死机后持久化结果更正与全项目GPU串行队列（revision 3459）|[旧记录](STTW_PROJECT_STATE.md)|
|3454|原300阶段完整训练曲线与选模证据（revision 3454）|[旧记录](STTW_PROJECT_STATE.md)|
|3453|跳跃诊断补齐速度、后轮目标与台沿距离（revision 3453）|[旧记录](STTW_PROJECT_STATE.md)|
|3450|能力评价器声明尺度与下一组实际出口条件准备（revision 3450）|[旧记录](STTW_PROJECT_STATE.md)|
|3447|首组右弯失败场景边界问题与覆盖修正预算（revision 3447）|[旧记录](STTW_PROJECT_STATE.md)|
|3444|减速惩罚续训实际启动（revision 3444）|[旧记录](STTW_PROJECT_STATE.md)|
|3443|推进奖励及减速惩罚：从300追加150的有界阶段（revision 3443）|[旧记录](STTW_PROJECT_STATE.md)|
|3437|源地形实测侧倾很小，首组仅作为轻扰动接续基线（revision 3437）|[旧记录](STTW_PROJECT_STATE.md)|
|3434|同车四候选有效轨迹：best88更高但last150仍台前停滞（revision 3434）|[旧记录](STTW_PROJECT_STATE.md)|
|3433|首组左弯六候选轨迹：到达与后续跟踪区分（revision 3433）|[旧记录](STTW_PROJECT_STATE.md)|
|3432|冻结150阶段best88和last150，同车四候选接管比较启动（revision 3432）|[旧记录](STTW_PROJECT_STATE.md)|
|3431|恢复重启中断的同出口实验，仅补右弯六候选（revision 3431）|[旧记录](STTW_PROJECT_STATE.md)|
|3430|从150模型恢复Adam后续训已154轮（revision 3430）|[旧记录](STTW_PROJECT_STATE.md)|
|3429|续训初始化进程已消失，保留未更新记录后重新启动（revision 3429）|[旧记录](STTW_PROJECT_STATE.md)|
|3425|用户授权dt005最后模型150继续训练150轮（revision 3425）|[旧记录](STTW_PROJECT_STATE.md)|
|3423|53轮训练核实与存储模型奖励归属筛选（revision 3423）|[旧记录](STTW_PROJECT_STATE.md)|
|3421|同车接口实现与首跑无效；任务进度接口完成（revision 3421）|[旧记录](STTW_PROJECT_STATE.md)|
|3420|同一dt005实例接入H75加Local300后跳跃配对（revision 3420）|[旧记录](STTW_PROJECT_STATE.md)|
|3418|dt005正式训练实际4轮，旧训练已停止并保留（revision 3418）|[旧记录](STTW_PROJECT_STATE.md)|
|3417|旧训练确认停止；dt005初始化OOM后调整容量重启（revision 3417）|[旧记录](STTW_PROJECT_STATE.md)|
|3416|用户授权切换跳跃训练至0.005秒（revision 3416）|[旧记录](STTW_PROJECT_STATE.md)|
|3414|六候选采集已进入仿真，准备同模型跨alpha轨迹交付（revision 3414）|[旧记录](STTW_PROJECT_STATE.md)|
|3413|启动首个真实同出口左右弯六候选采集（revision 3413）|[旧记录](STTW_PROJECT_STATE.md)|
|3411|dt005短测有效完成：采样更快但落地响应变化（revision 3411）|[旧记录](STTW_PROJECT_STATE.md)|
|3409|用户要求立即测试，缩减两lane接触缓冲后重跑dt005（revision 3409）|[旧记录](STTW_PROJECT_STATE.md)|
|3408|第三章六候选固定场景与采集命令准备完成（revision 3408）|[旧记录](STTW_PROJECT_STATE.md)|
|3406|0.005秒首试显存不足，结果无效而非动力学失败（revision 3406）|[旧记录](STTW_PROJECT_STATE.md)|
|3404|用户提出0.005秒物理步长，启动有界配对尝试（revision 3404）|[旧记录](STTW_PROJECT_STATE.md)|
|3403|Phase U正式训练已8轮，TensorBoard已加载并给出耗时估计（revision 3403）|[旧记录](STTW_PROJECT_STATE.md)|
|3402|同出口六候选的可访问闭环状态接续接口完成（revision 3402）|[旧记录](STTW_PROJECT_STATE.md)|
|3401|训练等待期推进接续评价的按来源运行留出接口（revision 3401）|[旧记录](STTW_PROJECT_STATE.md)|
|3400|准备训练候选冻结评估并纠正Go2中途iter元数据解释（revision 3400）|[旧记录](STTW_PROJECT_STATE.md)|
|3399|Phase U 4096训练已启动并接入既有TensorBoard（revision 3399）|[旧记录](STTW_PROJECT_STATE.md)|
|3397|完整跳跃环境接线通过并声明4096x150训练预算（revision 3397）|[旧记录](STTW_PROJECT_STATE.md)|
|3395|原Phase U完整GPU环境接线与有界集成检查（revision 3395）|[旧记录](STTW_PROJECT_STATE.md)|
|3392|四动作域随机化适配与GPU原生DelayBuffer接线通过（revision 3392）|[旧记录](STTW_PROJECT_STATE.md)|
|3390|短坡全组结束；随机重复启动字段错误已修复准备恢复（revision 3390）|[旧记录](STTW_PROJECT_STATE.md)|
|3388|原Phase U事件终止及固定106维critic状态投影完成（revision 3388）|[旧记录](STTW_PROJECT_STATE.md)|
|3386|保留原Phase U Actor critic结构，原Go2 PPO接口更新通过（revision 3386）|[旧记录](STTW_PROJECT_STATE.md)|
|3385|原Phase U 21项奖励批量Torch移植及回合回报语义（revision 3385）|[旧记录](STTW_PROJECT_STATE.md)|
|3382|对齐接近航向仍失稳，跳跃诊断减负实测完成（revision 3382）|[旧记录](STTW_PROJECT_STATE.md)|
|3381|同策略对齐高台航向并移除未用逐物理步诊断（revision 3381）|[旧记录](STTW_PROJECT_STATE.md)|
|3379|Phase U与RB接管首轮完成及后续效率约束（revision 3379）|[旧记录](STTW_PROJECT_STATE.md)|
|3376|用户允许Phase U与RB先配对试跑，保留自身模型网络（revision 3376）|[旧记录](STTW_PROJECT_STATE.md)|
|3372|锁摆H75左右弯完成，原策略与RB权重原样导出并GPU核对（revision 3372）|[旧记录](STTW_PROJECT_STATE.md)|
|3371|锁摆H75左弯五alpha完成并到达目标，RB观测数值接口已实现（revision 3371）|[旧记录](STTW_PROJECT_STATE.md)|
|3370|复用原STTW备用位置servo打通RB动作接口，未宣称闭环接管（revision 3370）|[旧记录](STTW_PROJECT_STATE.md)|
|3369|摆关节独立更新已实际通过，推进锁摆H75左右弯（revision 3369）|[旧记录](STTW_PROJECT_STATE.md)|
|3368|将锁摆限位更新改为仅两个USD摆关节，不回写轮子（revision 3368）|[旧记录](STTW_PROJECT_STATE.md)|
|3366|锁摆首版等上下限被PhysX拒绝，修为极小游隙制动后继续（revision 3366）|[旧记录](STTW_PROJECT_STATE.md)|
|3365|用户认可当前效果，继续推进而非重复调参（revision 3365）|[旧记录](STTW_PROJECT_STATE.md)|
|3364|用户要求地面锁摆跳跃释放：采用显式理想关节锁止，短测实际摆动（revision 3364）|[旧记录](STTW_PROJECT_STATE.md)|
|3363|从标准原轨迹辨明回正后累计航向偏差，继续等待H75（revision 3363）|[旧记录](STTW_PROJECT_STATE.md)|
|3361|七工况完成：带摆Local300仍有米级回正偏离，启动原H75路径闭环评测（revision 3361）|[旧记录](STTW_PROJECT_STATE.md)|
|3360|准备带摆H75三个标准路径与五alpha，保持先评Local300的顺序（revision 3360）|[旧记录](STTW_PROJECT_STATE.md)|
|3358|带摆标准评测继续，新增全过程原路径距离诊断（revision 3358）|[旧记录](STTW_PROJECT_STATE.md)|
|3356|带摆Local300原标准七工况实际运行中（revision 3356）|[旧记录](STTW_PROJECT_STATE.md)|
|3355|用户要求按STTW全过程轨迹验证带摆Local300与H75，登记七工况配对评测（revision 3355）|[旧记录](STTW_PROJECT_STATE.md)|
|3354|带摆左右转弯结果完成，经验接续查询逻辑可用但未建实库（revision 3354）|[旧记录](STTW_PROJECT_STATE.md)|
|3351|四lane带摆转弯原ECBC与Local300实际运行中（revision 3351）|[旧记录](STTW_PROJECT_STATE.md)|
|3350|登记带摆左右转弯原ECBC与冻结Local300配对诊断（revision 3350）|[旧记录](STTW_PROJECT_STATE.md)|
|3348|Isaac共同带摆车三秒原ECBC直行完成（revision 3348）|[旧记录](STTW_PROJECT_STATE.md)|
|3347|整段粗糙路首个条件完成，维持完整预算继续（revision 3347）|[旧记录](STTW_PROJECT_STATE.md)|
|3346|登记Isaac共同带摆车三秒ECBC闭环诊断（revision 3346）|[旧记录](STTW_PROJECT_STATE.md)|
|3345|完成带摆非原点膝轴与圆柱URDF导出，原车回归通过（revision 3345）|[旧记录](STTW_PROJECT_STATE.md)|
|3343|第三章共同车体候选编译并确认原STTW物理参数保留（revision 3343）|[旧记录](STTW_PROJECT_STATE.md)|
|3339|用户更新跨技能平台：Isaac Lab共同车体及必要策略适配获授权（revision 3339）|[旧记录](STTW_PROJECT_STATE.md)|
|3337|采用第三章修订任务书并登记带摆地面兼容首测（revision 3337）|[旧记录](STTW_PROJECT_STATE.md)|

## 更新记录

### revision 1 — 新建并回填

用户明确要求第三章独立台账。完成当前目标、身份、运行、主要尝试与负例、代码/结果入口、待办和历史索引回填；旧总账只追加入口与分工声明。未创建训练、未改变正在运行的任务、未push。

### revision 2 — 建账后即时回收运行终态

本轮实际systemctl确认两service均inactive。反向出口12条采集完成；复杂链0.4s侧倾失败、未起跳，不能发完成Bark或推送完成包。结果入口已更新于上表，等待原始XY诊断。历史索引按revision定位旧台账，取消会随插入漂移的行号锚点。此更新只涉及第三章。

### revision 3 — 复杂链起步状态不一致，采用真实地面快照

实读失败0–.4s轨迹：尚未到凸起中心即roll45.46deg。原入口使用jump reset快照：车体速度2m/s但前后轮轴速度均0，rootz.15m；同时导入地面warmup的ESO/前指令2.3m/s。不是同一完整地面闭环状态，且H75在此首次从起点执行。明确修正为原rb_ground_standard1500真实完整地面快照（轮轴约-32.8rad/s），初始平移/绕竖轴旋转到(-4,-7,pi/2)，模型/地形/原路径/控制器/18s门槛不变。该初态变更有记录，旧reset失败不覆盖。
输出rb_offroad_warm_start，配置runs/rb_sttw_platform/offroad_warm_start.json，单lane18s/3600physics/0optimizer。增加保存已有ground.step lower_trace用于查看真实参考/速度，不新增控制逻辑。若仍失败再由实际输入输出判断，不盲目训练。当前两个旧任务均已终态，可排队执行新试验。

UTC=2026-11-06T14:44:23.293221+00:00

### revision 4 — 真实地面初态改善启动但起伏仍失败，复用平面承载接触

warm_start已终态0.9s失败，轮速参考/实测一致但车体速度在原平地网格即有振荡，进入起伏后roll失稳，未起跳。起步不一致修正未单独解决复杂地形，保留失败。下一复用已做过terrain_base_plane接口：原平地以无限解析平面承载，网格仅保留非零高程三角形；本场景原跳台box存在负z底面，独立terrain_plane将地下点clip到0，地上起伏/台阶高度形状不变，明确为接触表示变化。仍2cm双起伏/R2弯/原门槛/18s3600physics/0optimizer，输出rb_offroad_plane_support。没有减小起伏/改奖励或训练。
反向6cm12候选已完成12/12；前瞻左右均选v2.3alpha.5且实际分支通过，目前无优于固定v2.3alpha.5证据。XY已生成待完整报告，相关结果不冒充复杂跨技能任务通过。

UTC=2026-11-06T14:46:00.450139+00:00

### revision 5 — 平面承载试验回收：起伏仍阻断，未冒称通过

rb_offroad_plane_support实核service inactive/results complete但1.22s侧倾失败，244physics/2.52s墙钟，位置(-3.866,-4.688)，peakroll44.21deg，未起跳。相较warm_start可走过更长起伏，但不足以完成，不采用为成功方案；0.9s和1.22s均保留，不能归因为仅初态修复。代码dfba52bb本地已提交，尚未push。原两次失败XY已生成。
下一保持2cm非零起伏与R2转弯、平整跳跃区，将原短促双凸起改成较长波长起伏，明确几何变更并保留旧场景；不降低高度到零、不以直线任务替代、不新增训练或大模型。复杂任务仍未完成，不发完成Bark。第三章台账现revision5。

UTC=2026-11-06T14:47:21.286903+00:00

### revision 6 — 保持高度的长波起伏复杂链试验

复用prepare入口将几何参数显式化。本次起点(-4,-9)，起伏区y[-8,-4.8]长3.2m包含两波，峰高仍.02m(旧长1.6m两波)；R2右转/出弯5.6m平直起跳段/原平台均不变。沿用真实地面warmup完整初态、解析平面+非零高程网格、H75+ECBC和update2、.005physics、18s完整窗/3600physics/0optimizer、原工程门槛。输出rb_offroad_long_wave/evaluation，场景未以零起伏或直线路线替代；旧短波失败保留。声明为几何调整试验，不是同任务控制改进因果对照。

UTC=2026-11-06T14:47:54.769372+00:00

### revision 7 — 长波仍失败，单改现有角速度滤波诊断

long_wave终态1.98s/396physics失败，位置(-3.973,-5.308)，peakroll40.37deg，未到弯道。trace显示1.2–1.3s rollrate从-1.29至1.80rad/s尖峰，转向后续摆幅增大；速度当时1.8m/s，不能套用.597m/s低速解释。原地形两波高2cm/长3.2m保持。下一只启用既有TorchController roll_rate_filter_hz=10（原0），其余wc/wo/feedback/ESO/执行器/网络不变；假设接触冲击反馈被滤波改善，未证明。配置runs/rb_sttw_platform/offroad_filtered_rate.json，输出rb_offroad_filtered_rate；单lane18s3600physics/0optimizer。若不改善保留负例，不自动大规模调参/训练。

UTC=2026-11-06T14:49:27.667822+00:00

### revision 8 — 10Hz滤波无明显改善，做一次接触步长对照

filtered_rate终态2.0s/400physics侧倾失败，近似原1.98s，不采用10Hz滤波为解决方案。保留同地形。下一恢复原无滤波ECBC，仅physics .005→.001，控制仍.005/H75与jump仍.02，通过既有substeps实现；明确短工程物理步长诊断而非静默改变正式训练合同。单lane18s最多18000physics/0optimizer，输出rb_offroad_contact_dt001。目的区分小轮起伏接触的步长误差与实际底层不足，记录墙钟；不能提前称加速或模型通过。原失败图/数据保留。大模型与push仍未进行。

UTC=2026-11-06T14:50:38.232909+00:00

### revision 9 — 1ms通过起伏与弯道但跳跃失败，检查统一2.5ms步长

contact_dt001终态8.435s/8435physics/49.90s墙钟，7.2s实际经过起伏右弯抵达jump交接x2.0239,y-.01698，随后上台附近侧倾109.67deg失败、未地面交回。原5ms同任务1.98s在起伏失败，表明步长对小轮接触/执行响应有明显影响；不宣称全部不一致由步长单因素解释。完整XY已生成，保留负例。
下一仅在整个同车任务统一physics=.0025（不按技能改变物理步长），地面.005=2子步，jump/H75.02不变，原模型/地形/网络/奖励/执行器保持。名义DEV18s最多7200physics，0optimizer，输出rb_offroad_contact_dt0025。目的是找到两技能共同可执行的时间离散，不将“地面能用/跳跃不能用”作为完成。当前无训练/push/Bark完成通知。

UTC=2026-11-06T14:53:00.728264+00:00

### revision 10 — 统一2.5ms通过前段，交接仍带残余横向运动

contact_dt0025终态8.745s/3498physics/24.69s墙钟，完成起伏和R2转弯、7.24s到x2.028,y-.00491交接，跳跃后侧倾53.11deg失败；不采纳为完整成功。对照实际handoff：简单通过例speed1.923、roll.00143、rollrate-.000615、世界vy.00674；复杂2.5ms例speed2.042、roll.00858、rollrate-.08624、世界vy.15177。虽然横向位置接近零，残余航向/角速度尚不同。说明只按x接管不能代表准备充分，不宣称步长单独解决全部问题。
下一保留非零起伏/R2转弯/统一2.5ms，扩大平台前平整连接长度，为原H75提供消除转弯残余状态空间，同时按实际交接状态核验；不拼接不同时域轨迹、不reset交接状态。复杂任务未通过，未Bark/未push。

UTC=2026-11-06T14:54:43.879011+00:00

### revision 11 — 扩展平整接近段，保持起伏和转弯

新几何仅将起伏/右弯横移4m，弯末x=-6，至台沿x3.6平整长度9.6m（原5.6m）；起伏仍2cm/3.2m两波，R2转弯，原平台与跳跃技能不变。初始(-8,-9)朝北，真实地面warmup；统一2.5msphysics、5ms底层/20ms策略。为额外4m路程完整观察，评估窗22s/最多8800physics，0优化。prepare入口显式turn-end-x和seconds参数，输出rb_offroad_long_approach。判断交接残余状态与实际全链；原18s失败不裁剪或覆盖，仍不通知完成、不push。

UTC=2026-11-06T14:55:17.520126+00:00

### revision 12 — 保留原已通过起伏弯道，后移平台独立检验连接距离

long_approach终态4.28s控制域退出、rollmax10.17deg非典型摔倒；横移起伏/弯道后其出口状态也变化，不能视为只测连接长度，旧run保留。下一保持原起点(-4,-9)、原起伏与R2弯(已在2.5ms通过)，仅将原平台几何向前+x4m，前沿7.6/后沿11.6，起跳交接x6；出弯至台沿9.6m。原source mesh全体刚性平移但平面承载不受影响；新增起伏顶点保持原位置/网格，更新距离观测/触发区/严格landing/绘图的显式平台坐标配置，旧无字段默认3.6/7.6不变。
run rb_offroad_platform_shift，统一2.5ms/5ms/20ms、22s最多8800physics、0optimizer。完整原路门槛不放宽，不改变jump网络/力限/摩擦，不reset交接状态。明确工程几何变化，旧复杂失败不覆盖。尚未push/Bark。

UTC=2026-11-06T14:57:42.557236+00:00

### revision 13 — 复杂起伏转弯跳跃续行通过，进入精简交付

rb_offroad_platform_shift/evaluation实核service inactive/results complete，完整22s/8800physics/60.05s墙钟，0优化。原固定路径全程XY已目视：2cm两波起伏→R2右弯→9.6m平整接近→0.16m平台跳跃→落地H75+ECBC续行。统一physics .0025、lower .005、upper/jump .02，原网络和执行器保持。analyze_rb_continuous_task完整7项均true：原路最大误差.79024m(<预先1m)、RMSE.22321m、台上严格双轮连续支撑1.260s、地面交回后7.64s/11.295m；全程峰值roll12.223deg，无终止。末端y-.07855m。不是依末端生存判完成，未声称鲁棒或整章完成。
采用该名义场景工程链作为当前可用接口。较长平整接近段改善与此前失败有关联，但几何/时间离散经历过变更，不能宣称已完成严格接续消融。随后按用户授权精简代码、配置和关键原始结果图，推送既有origin/experiment/sttw-offroad，并Bark通知；此条记录时尚未push。大模型仍暂停。

UTC=2026-11-06T15:00:57.204489+00:00

### revision 14 — 刷新第三章总览，消除历史运行状态歧义

只维护第三章。当前总览同步最终成功配置、反向候选结果和交付状态；旧失败原记录保留。无新增训练或仿真。
