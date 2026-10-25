# 提前与全程同接口速度支持：160逻辑条件结果

2026-10-25；STTW_CONTROL；开发期机会验证，不是独立泛化或最终投稿结果。

## 目的与操作条件

检验固定速度支持能防摔却损害任务时，提前支持后释放是否有用。新增普通task64和父Actor各自的EARLY/CONSTANT两个过程，原用户alpha=.5/.75，原左右R1.1请求1.8/1.5m/s，无扰及1.9/2.9秒×世界方向±32N、持续.1秒，全部从认证完整task0/physical700连续运行10秒。底层local300/ECBC/ESO、真实历史、滤波、执行器与参考权限冻结；同16个完整有效策略块后交回原alpha。

两过程都只通过原两维修正接口：dv_support=max(dv_actor,1.5-v_nom)，保持原steer latent，再经原反映射、滤波/变化率/限幅。EARLY在既有1.5m入弯准备区平滑启动，从入弯起释放，到几何弯中点归零；CONSTANT同目标全程启用。不是跳改配置下限，也不保证实际速度达到1.5m/s。仅用当前进度/原路径/名义参考/Actor输出，不读取实际未来事件；原351Actor未新增输入，进度来自既有几何follower。

原计划EARLY可能40条；在新物理前补40条同接口CONSTANT用于排除硬clip实现差异，合80新。code5360dde，35CPU通过，左右完整状态GPU shape各2batch20×50×4、0积分；11关键runtime模块SHA及plant step/交回资格AST核对。80新实际160000个5ms底层步/160000槽位、0优化更新，session77853 exit0；80匹配旧轨迹另计157096历史步，合160逻辑唯一记录，不算独立样本。分析session26697 exit0，0新物理/训练、0UNKNOWN，160raw/成本/几何标签与新80最终alpha/时钟/累计离域核对。

## 完整结局

任务=实际未触地且完成原目标/末窗路径、航向、速度要求；历史工作侧倾合规另报，不混为触地。

|配方|名义任务/4|受扰未摔/16|受扰任务/16|
|---|---:|---:|---:|
|父Actor|1|15|1|
|父Actor+硬floor|1|16|3|
|原task64|3|15|5|
|task64+硬floor|2|16|4|
|父EARLY|1|16|2|
|父CONSTANT|1|16|3|
|task EARLY|3|16|5|
|task CONSTANT|2|16|4|

- task EARLY相对同接口CONSTANT：受扰任务1新增/0丢失，名义任务1新增/0丢失，实际未摔相同。相对原task：未摔1新增/0丢失、受扰任务1新增/1丢失，总任务同为5；不是保留所有原正例。
- 父EARLY相对父CONSTANT：受扰任务1新增/2丢失，净退1；同一阶段规则并非普遍更好。task EARLY相对父EARLY任务3新增/0丢失，不能将父负结果删掉或把全部差异归于新阶段机制。
- CONSTANT与相应硬floor的全部20条任务/触地分类分别完全相同；保留连续数值差，不追位级重放。EARLY的任务变化不是只在对比硬clip时出现。

## 关键正例、反例和原alpha代价

**新任务正例：右alpha.75、1.9s负力。** task EARLY/CONSTANT均未摔；EARLY末路径.05863m、航向.02623rad，CONSTANT为.77808m、.70708rad，所以不是微小门槛变化。EARLY在4.68s交回原Actor并完成原任务。但全过程最大路径偏差2.50922vs.77808m、RMSE1.19885vs.34754m，原alpha成本57.35574vs35.52752（高21.82823）。收益是末段恢复，不能写成全过程更贴线或更低代价。

**丢失原任务：右alpha.5、2.9s负力。** 原task成功，EARLY仍未摔但末路径.51186m、航向.73374rad、成本73.12494，原task为.06154m/.01515rad/65.31079。原task会几何失效后交回，EARLY本例没有交回而仍末窗失败；不是所有任务失败都能靠交回解决。

**名义与扰动成本方向不同。** task EARLY相对CONSTANT在无扰平均成本差：alpha.5为−27.17507，alpha.75为−10.90670；在受扰为+10.90488、+3.43826，分别8例中7例/5例成本更高。原成本包含路径、速度、侧倾及工作带等全部项；没有只截取有利时段。父EARLY名义与受扰均值均更差。

这支持“保留普通策略的名义任务，同时改变抗扰—任务恢复取舍”的局部机会，不支持无条件代价占优。实际未摔也不等于不偏离道路或一般环境安全；父EARLY若干轨迹有很大绕行，完整XY保留。

## 额外敏感性图的边界

mixture_sensitivity展示Delta(p)=(1-p)*名义均值差+p*受扰均值差。p是假设的每回合受扰比例，两个路线及各路线4个力/时刻分支在本开发面板内等权，不是实测事件频率或部署分布。task两alpha成本交点约.714/.760；图展示全部p∈[0,1]，不选择有利p作为接受门槛或宣称未知分布下最优。它仅解释为何名义和受扰成本必须分开报告。

## 继续与停止的裁决

1. 保留task EARLY为开发候选，CONSTANT/硬floor和原task为强基线，不采用新权重、不包装阶段规则为独有理论。父EARLY负结果保留。当前有有限增量，值得一次新时刻与完整alpha的前瞻检查；不要求无敌或每个案例都改善。
2. 固定5360dde的阶段与目标，不扫释放点/1.5/16块/幅度/时长，不用这组160继续训练选择器。下一具体设计见early-support-prospective-confirmation计划：新增3个时刻、5个alpha及零中层强对照；仅设计，未准备/启动。新路线/真实新初态与机制分离仍是后续缺口。
3. 当前阶段规则在实际脉冲前后都可能通过进度反馈改变参考，尚未用同后续控制的2×2分支隔离“进入状态”与“事后支持/恢复”。不得将全部增益归于提前状态塑造，也不根据未来事件选择离线最优后冒充部署。
4. 旧Emin=(z+KF)^2/W未参与控制，不能由本轮结果称其得到验证、牛顿边界成立或余量具有保守保证。后续容扰对象仍是给定闭环、可执行参考过程、扰动合同与观察窗下的实测能力。
5. 参考治理和最小干预本身已有文献；目前贡献仍需从偏好—动态平衡耦合的机制、强规则不能覆盖的控制增量及新情境证据建立，不宣称RAL已就绪。

## 可查看的证据与审查范围

- [全部方法](../../../../runs/early_speed_support_v1/analysis/by_method.csv)、[逐例配对](../../../../runs/early_speed_support_v1/analysis/paired_support.csv)、[全部160记录](../../../../runs/early_speed_support_v1/analysis/all_cases.csv)
- [新任务正例及同接口对照XY](../../../../runs/early_speed_support_v1/analysis/xy_right_t380_F-32.png)、[丢失原任务的XY](../../../../runs/early_speed_support_v1/analysis/xy_right_t580_F-32.png)、[名义右路XY](../../../../runs/early_speed_support_v1/analysis/xy_right_zero.png)
- [成本取舍](../../../../runs/early_speed_support_v1/analysis/cost_tradeoffs.csv)、[假设暴露比例敏感性图](../../../../runs/early_speed_support_v1/analysis/mixture_sensitivity.png)、[声明与公式](../../../../runs/early_speed_support_v1/analysis/mixture_sensitivity.json)
- [逐例接口及重点反例复核](../../../../runs/early_speed_support_v1/analysis/focused_review.json)

全部10共享XY已目视，覆盖两route/五event/两alpha/八方法；另看new_trace066/072/073与敏感性图。剩77张新增细图和80缓存细图本轮未逐张重审，保持visual_review_complete=false。所有80新轨迹steer latent保留、profile6 gate0精确恢复原latent、速度支持target不降低原输出（仅浮点容差）通过事实日志审计；没有设置新的物理bitwise门槛。

根采集/分析中文Bark accepted1792946351.555109/1792946415.3661041，heartbeat11.68s/scan空。一次派发使用已有pidfd fallback，观察真实进程退出再分析，无进度轮询、重跑物理或新增常驻watcher。当前本运行无活动物理/训练/分析。
