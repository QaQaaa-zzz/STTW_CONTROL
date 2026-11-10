# STTW保留实验索引

这是唯一研究台账保留实验节的版本化交付快照；当前活动状态以[STTW_PROJECT_STATE.md](/home/qy/STTW_CONTROL/research-hub/STTW_PROJECT_STATE.md)为准。机器可读目录及报告发现清单见[catalog.json](catalog.json)。本次未删除任何原始实验。

## 保留实验总索引

本次按用户要求将积极结果和可复用机制证据置于主台账。每行指向原报告及其中完整实际/参考XY、逐工况结果；同组失败和对照随证据保留，不跨实验累加样本量。来源为repo_artifact，核验为artifact_checked（本次读取报告及核对路径；未重跑全部原始轨迹）。本表是实验主题索引，不是独立实验总次数。

### 底层与模型基础

|编号/实验|模型|保留的结果|用途与范围|原证据/XY入口|
|---|---|---|---|---|
|L01 局部底层300轮验收|Local300 / R196|七类速度/转向任务7/7合格，对照R196为3/7；Q200→300改善38.89%。|保留强底层的能力依据。旧300轮报告的采用门槛属当时快照；当前采用身份以冻结注册及运行contract为准。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/evidence/local_lower_noalpha_v2_0300/RESULTS_0300.md)|
|L02 同完整状态恢复|Local300 / R196|Local300在0.445s进入联合误差带并保持至2s；R196在2s内未进入。|底层局部恢复能力。单一保存状态，不是总体恢复率。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/evidence/local_lower_parent_recovery_20261010/INDEX.md)|
|L03 物理域底层续训|Local-domain U200|四路线×两初态×五α，父/163/200均40/40到达并观察满10s；U200低速转角RMSE2.002°→1.552°，高速1.147°→0.950°。|保留U170等历史实验使用的底层谱系。不是当前Isaac橙色Local300；路径误差有代价。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/README.md)|

### H75训练与路径偏好

|编号/实验|模型|保留的结果|用途与范围|原证据/XY入口|
|---|---|---|---|---|
|H01 完整尾段25→75 PPO|H75谱系|强扰动任务186→191/200；低速52→56/60；中等200/200、直线10/10保持；合计448→457/470，9改善0退化。|训练足量带来任务恢复改善。含重复零力记录；不称9次防摔。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/evidence/tolerance_progress_20261022/full_tail_length75/FINAL_RESULTS.md)|
|H02 五α名义任务保持|原H75|四名义工况×五α为20/20原任务保持；Zero4/4到达且不摔，但末段航向约4–6°未满足原保持条件。|中层改善末段跟踪及偏好表达。Zero也能到达，不能说无网络不能行驶。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/deliverables/sttw_review_bundle_20261024/REPORT.md)|
|H03 四路线冻结中层比较|原H75；对比U170/W400/u325|H75 α0窄任务8/8，α.5为4/8，α1为0/8；三档平均转弯速度2.202/2.339/2.452m/s。|路径优先与速度优先取舍，保留H75作为路径优先主模型。同Local200底层；开发集选型，不将W400等成绩归H75。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/frozen_middle_comparison/README.md)|
|H04 新任务组合五α|H75 / U170|80条均到达并观察满10s；H75 α0窄任务8/8；U170选α0窄6/8对中档4/8，选α1宽8/8对中档6/8。|按任务选偏好的收益。不同模型分列；既有规则下的新组合开发证据。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/README.md)|
|H05 低中高速/单弯/S弯/直角矩阵|H75 / U170 / corrected PP|504条完整10s；H75到达212/216，U170209/216，PP72/72。低速受推S弯中H75 α1在两初态均有局部速度/路径优势。|覆盖1.90/2.50/3.05m/s与+36N扰动，保留广覆盖行为证据。PP各速度/外力组平均路径误差更小；局部正例保留全矩阵分母。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/ras-route-matrix-20261104/docs/results/sttw-task-matrix-20261104/README.md)|
|H06 示范教学强任务与低速改善|u325派生教学模型 / H75|固定400步教学强任务195/200、低速58/60；H75为191/200、56/60。|保留教学能改善任务恢复的候选证据。训练预算不等、侧倾性能有代价；见证筛选本身未有稳定增益。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/evidence/tolerance_progress_20261022/full_tail_length75/FINAL_RESULTS.md)|

### 低速、推力与恢复

|编号/实验|模型|保留的结果|用途与范围|原证据/XY入口|
|---|---|---|---|---|
|D01 低速真实触地力档扫描|冻结底层；无中层Actor|12工作状态108分支：1.5m/s参考下±32N通过、±48N触地；1.8下±48通过、±64触地；2.2/2.55通过已测±64N。|低速工作状态更脆弱的实测基础。0.1s侧推、4s观察；不同速度由真实准备形成不同姿态，不是单速度因果隔离。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/deliverables/sttw_review_bundle_20261024/evidence/low_speed_physical.md)|
|D02 同快照速度参考干预|冻结底层|120分支；24非零条件hold19无触地，提前加速23、同期加速22；24零力分支均完整无触地。|只调整参考即可改变物理结局。已知起扰时刻诊断；扩检有加速负例，非普适加速规则。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/evidence/tolerance_progress_20261022/low_speed_reference_authority/REPORT.md)|
|D03 完整任务因果ESO参考规则|原任务策略+简单参考规则|24条全任务；非零8配对无触地5/8→7/8，新增2、退化0；零力4配对不干预。|不读取真实起扰时刻的参考调节正例。两个相关初态的DEV；任务完成和无触地分别计。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/deliverables/sttw_review_bundle_20261024/evidence/causal_reference.md)|
|D04 不同偏好前缀的方向抗扰能力|原H75 / Zero|旧几何受扰完整无触地：Zero4/16、α0为0/16、α1为6/16；新几何Zero和α1均4/16。|偏好改变进入状态及方向性能力。旧几何局部优势；新几何反例一并保留，不称稳定泛化。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/deliverables/sttw_review_bundle_20261024/REPORT.md)|
|D05 ±36N侧推任务比较|U170 / 几何规则|48条均完整10s；正36N低速S弯/高速右单弯子集，α1宽任务4/4、α0与规则0/4；全带符号矩阵分别4/16、2/16、2/16。|特定方向推力下速度优先档的任务用途。正例为事后子集，完整分母保留；这是U170。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/README.md)|

### 历史扩展能力

|编号/实验|模型|保留的结果|用途与范围|原证据/XY入口|
|---|---|---|---|---|
|U01 物理域迁移任务|U170 / 几何规则|48条均到达并完整10s；两个摩擦极端域α1宽任务各8/8，规则6/8和7/8。|物理参数变化下偏好能力。开发选定极端域及对应初态，不称独立泛化。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/README.md)|
|U02 有限半径90度转弯|U170 / 几何规则|48条均到达并完整10s；α0→1转弯速度2.122→2.346m/s，窄任务6/8→2/8、宽任务2/8→8/8。|可解释路径—速度取舍。规则两任务均8/8；是平滑有限半径弯，非尖角。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/README.md)|

### 越野、石子与选档

|编号/实验|模型|保留的结果|用途与范围|原证据/XY入口|
|---|---|---|---|---|
|T01 五速度×四地形重复|原H75+Local300 / Zero|100工况600条：五α H75500/500完整10s，Zero71/100；覆盖平地及2/4/6cm斜脊，1.5–2.6m/s。|无越野专项续训的基础通过能力及五档响应。低速请求从2.3m/s暖态减速；完整观察与到达/保持分开。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_tradeoff_repeats/RESULTS.md)|
|T02 代价/约束下的选档|原H75|校准二次代价η=.533选.75；同时满足2/4cm的.40m峰值路径约束选.5；重复矩阵保留各地形速度选择。|没有普适最优α，选择由任务决定。不是部署在线选档器。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_tradeoff_repeats/ALPHA_SELECTION.md)|
|T03 6cm/2.6m/s既定1→.5切换|原H75+Local300|五参数重复：30条H75均到达并完整10s，Zero2/5；相对固定1，切换共同到达前窗口路径RMSE降低均值.053691m，5/5改善；速度4/5改善。|动态参考偏好有可重复局部收益。比固定.5路径更差、速度更好；包含共同重规划，非全局最优切换。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_bend_switch_repeats/RESULTS.md)|
|T04 密集石子及10cm单脊|原H75+Local300 / Zero|72条：H7539/60完整，Zero0/12；α.5到达保持11/12。10cm/2.6m/s α.5任务C为3/3；石子任务B α1为3/6、α.5为2/6，成功案例互补。|强化地形下中层作用及α互补。同样保留21条H75终止；11/12不是所有任务都合格。|[报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_dense_corridor_ridge10/evaluation/RESULTS.md)|
|T05 石子高度/密度/聚集布局|原H75+Local300，P1|高度、嵌套密度与聚集方式改变实测接触段及不同任务的α排名。|地形结构影响通过方式。保留全60工况入口；20mm石子是能力边界，不挑成总体优势。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_gravel_closeout/CONCLUSION.md)|
|T06 前后轮接触重叠与残余响应|原H75+Local300，P2|0.4m间距前后轮接触重叠均值18.49%，0.2m为9.22%，0.8m为0%；43个配对触障条件中0.4m第三簇残余更大。|石子排列影响连续响应的机制线索。不是普适共振规律。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_gravel_closeout/CONCLUSION.md)|
|T07 连续石子任务条件选档|原H75+Local300，P4|同40上下文：Task-only15/40，固定α0/.5/1为5/10/11；Task+Phase16/40。|保留任务选档较最佳固定档多4例的收益。两张图；200逻辑结果来自152条不同物理轨迹；相位只净增1例，未采用。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_gravel_closeout/CONCLUSION.md)|
|T08 接触几何及有效拖距量化|原H75+Local300|4/6cm越脊负拖距时长中位8.0/53.3ms，对应侧向力矩峰值.367/.788Nm，绝对角冲量.001229/.008757Nms。|保留真实接触机制数据供后续解释。接触补偿未提升任务合格率，不将几何现象写成补偿成功。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_contact_g5/CONCLUSION.md)|

### 随机越野续训

|编号/实验|模型|保留的结果|用途与范围|原证据/XY入口|
|---|---|---|---|---|
|R01 共同起点150轮、无外部推扰|原H75 / continued150|13工况×五α：完整44/65→56/65，救回14、丢失2；共同窗口路径RMSE .436→.332m。|越野续训减少终止并改善路径的正面效果。速度RMSE .156→.404m/s，65条均变差；保留速度代价，不全面替换原H75。|[报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_random_terrain_continue/common_start/evaluation_no_push_paired/RESULTS.md)|
|R02 共同起点150轮、复合推扰|原H75 / continued150|同65条件完整24→25；共同窗口路径RMSE .5831→.4616m，47条件改善。|保留有推扰条件下的路径改善与强推力边界。速度.1774→.3523m/s；完整率提升很小，不包装成显著鲁棒性提升。|[报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_random_terrain_continue/common_start/evaluation/RESULTS.md)|

### 当前机制证据

|编号/实验|模型|保留的结果|用途与范围|原证据/XY入口|
|---|---|---|---|---|
|M01 同前缀通道分叉|原H75+Local300|20组×5分支：完整H75与仅转向均20/20满10s、19/20保持；仅速度7/20；Zero10/20。|转向参考通道在这些条件下承担主要跟踪收益。Zero实为4独立上下文重复；s2.5在障碍前；未证明两通道协同必需。|[报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_alpha_mechanism/RESULTS.md)|
|M02 ECBC名义公式独立复算|固定速度名义控制器|82速度点核验特征多项式；2.25→2.567m/s同单位冲量侧倾峰值+4.87386%，0.1rad侧倾控制项需求−20.40145%。|速度改变控制需求与扰动响应的理论解释。数值/公式复算，不是新物理实验或真实地形冲量增大证明。|[报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_alpha_mechanism/REVIEW.md)|

