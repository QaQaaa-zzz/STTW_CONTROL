# STTW关键结果速查

当前事实以[唯一研究台账](/home/qy/STTW_CONTROL/research-hub/STTW_PROJECT_STATE.md)为准；[详细模型、范围和来源目录](catalog.json)。

## 关键结果速查

各链接进入原报告及实际/参考XY。满10s、到达、任务合格、触地分别计；不同模型/协议不合并成功率。以下为报告级核验，完整对照与限制见原文。

|实验|关键结论|证据路径|
|---|---|---|
|L01 局部底层300轮验收|Local300七场景合格7/7，R196为3/7；保留为底层依据。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/evidence/local_lower_noalpha_v2_0300/RESULTS_0300.md)|
|L02 同完整状态恢复|同完整状态：Local300在0.445s进入误差带并保持，R196在2s内未恢复。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/evidence/local_lower_parent_recovery_20261010/INDEX.md)|
|L03 物理域底层续训|Local-domain U200：低速转角RMSE2.002°→1.552°；三候选均40/40到达。与Local300分开。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/README.md)|
|H01 完整尾段25→75 PPO|H75 25→75 PPO：强扰动186→191/200，低速52→56/60；汇总448→457/470，9改善0退化。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/evidence/tolerance_progress_20261022/full_tail_length75/FINAL_RESULTS.md)|
|H02 五α名义任务保持|H75五α名义保持20/20；Zero4/4能到达，但末段航向不满足原保持标准。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/deliverables/sttw_review_bundle_20261024/REPORT.md)|
|H03 四路线冻结中层比较|H75 α0/.5/1：窄任务8/4/0（各8例），转弯速度2.202/2.339/2.452m/s，体现取舍。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/frozen_middle_comparison/README.md)|
|H04 新任务组合五α|新任务组合80条均到达；H75 α0窄8/8；U170选端点窄6/8、宽8/8，固定中档4/8、6/8。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/README.md)|
|H05 低中高速/单弯/S弯/直角矩阵|1.90/2.50/3.05m/s，含+36N：H75到达212/216，U170209/216，PP72/72；PP总体路径更好，H75有低速受推S弯局部优势。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/ras-route-matrix-20261104/docs/results/sttw-task-matrix-20261104/README.md)|
|D01 低速真实触地力档扫描|底层108分支：1.5m/s下±32N无触地、±48N触地；1.8下±48无触地、±64触地；2.2/2.55通过已测±64。0.1s侧推、4s观察。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/deliverables/sttw_review_bundle_20261024/evidence/low_speed_physical.md)|
|D02 同快照速度参考干预|同快照24受扰条件：保持参考19例无触地，提前加速23例、同期加速22例；不是普适加速规则。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/evidence/tolerance_progress_20261022/low_speed_reference_authority/REPORT.md)|
|D03 完整任务因果ESO参考规则|因果ESO参考规则：非零8配对无触地5→7，零力4配对不干预；两个相关初态的DEV。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/deliverables/sttw_review_bundle_20261024/evidence/causal_reference.md)|
|D04 不同偏好前缀的方向抗扰能力|H75旧几何受扰无触地：α1为6/16，α0为0/16，Zero4/16；新几何α1与Zero均4/16。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/deliverables/sttw_review_bundle_20261024/REPORT.md)|
|D05 ±36N侧推任务比较|U170 ±36N全矩阵宽任务：α1为4/16、α0和规则各2/16；正推力局部子集4/4对0/4。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/README.md)|
|U01 物理域迁移任务|U170两摩擦域α1宽任务各8/8，几何规则6/8、7/8；48条均到达。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/README.md)|
|U02 有限半径90度转弯|U170有限半径90°弯：α0→1速度2.122→2.346m/s，窄任务6→2/8、宽任务2→8/8；规则两任务均8/8。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/README.md)|
|T01 五速度×四地形重复|平地及2/4/6cm、1.5–2.6m/s：H75满10s为500/500，Zero71/100。低速从2.3m/s暖态减速。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_tradeoff_repeats/RESULTS.md)|
|T02 代价/约束下的选档|校准二次代价选α.75，2/4cm共同0.40m路径约束选α.5；没有通用最优档。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_tradeoff_repeats/ALPHA_SELECTION.md)|
|T03 6cm/2.6m/s既定1→.5切换|6cm/2.6m/s，1→.5切换对固定1：路径RMSE五组均改善，均值−0.053691m；对固定.5路径更差、速度更好。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_bend_switch_repeats/RESULTS.md)|
|T04 密集石子及10cm单脊|密集石子+10cm：H75满10s为39/60，Zero0/12；α.5到达保持11/12；石子任务B α1为3/6、α.5为2/6，案例互补。|[报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_dense_corridor_ridge10/evaluation/RESULTS.md)|
|T05 石子高度/密度/聚集布局|石子高度、密度及聚集改变接触段和α排名；20mm仍为能力边界。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_gravel_closeout/CONCLUSION.md)|
|T06 前后轮接触重叠与残余响应|P2：0.4/0.2/0.8m间距前后轮接触重叠18.49%/9.22%/0%；0.4m第三簇残余较大。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_gravel_closeout/CONCLUSION.md)|
|T07 连续石子任务条件选档|P4任务选档15/40，固定0/.5/1为5/10/11；相位选档16/40，额外增量未采用。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_gravel_closeout/CONCLUSION.md)|
|T08 接触几何及有效拖距量化|4/6cm负拖距中位时长8.0/53.3ms、侧向力矩峰值.367/.788Nm；补偿29→28/50无任务增益。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad/runs/h75_contact_g5/CONCLUSION.md)|
|R01 共同起点150轮、无外部推扰|共同起点150轮无推扰：满10s 44→56/65，救回14丢失2；路径RMSE .436→.332m，速度.156→.404m/s。|[报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_random_terrain_continue/common_start/evaluation_no_push_paired/RESULTS.md)|
|R02 共同起点150轮、复合推扰|同续训有推扰：满10s 24→25/65；路径RMSE .5831→.4616m，速度.1774→.3523m/s。|[报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_random_terrain_continue/common_start/evaluation/RESULTS.md)|
|M01 同前缀通道分叉|20组分叉：完整/仅转向均20/20满10s、19/20保持，仅速度7/20、Zero10/20；未证明速度通道必需。|[报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_alpha_mechanism/RESULTS.md)|
|M02 ECBC名义公式独立复算|名义复算：2.25→2.567m/s，同冲量侧倾峰值+4.87%，侧倾控制项需求−20.40%；不是地形冲量增大证明。|[报告](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_alpha_mechanism/REVIEW.md)|
|H06 示范教学强任务与低速改善|教学模型强任务195/200、低速58/60，H75为191/200、56/60；训练成本不同，非筛选机制证明。|[报告](/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/evidence/tolerance_progress_20261022/full_tail_length75/FINAL_RESULTS.md)|
