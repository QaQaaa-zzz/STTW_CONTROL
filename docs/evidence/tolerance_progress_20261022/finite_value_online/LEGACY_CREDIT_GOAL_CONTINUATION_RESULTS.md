# 旧物理信用策略在共同20秒接续下的用途裁决

2026-10-27；正式90条已完成，主分析exit0及独立90 raw重计一致。内部DEV报告，不是独立测试或新方法认证。

## 先看真实任务轨迹

下图到首次进入原目标1m圈为止；菱形为首达，叉为真实触地，方块为观察截止未达。全20秒尾迹另存，避免把到达后继续行驶的末点误当任务失败。每图包括两α及全部五角色，zero是同一真实轨迹共享展示。

![右弯1.9秒负向脉冲的任务前缀](../../../../runs/legacy_credit_goal_continuation_v1/diagnostics/task_prefix_right_t380_F-32.png)

![左弯2.9秒正向脉冲的任务前缀](../../../../runs/legacy_credit_goal_continuation_v1/diagnostics/task_prefix_left_t580_F+32.png)

## 目的：回答什么，为什么值得做

旧hidden10s中credit对ordinary有条件性到达新增，但未新增防摔。用户提出时间太短、偏离后应重新规划去原目的地。本组在同一合理接续合同下比较冻结模型，问旧增益是否仍有用，还是主要被普通规划/规则覆盖。不是为了让旧解析式重新生效，也不要求策略无敌。

实验前沿scientific-brainstorming1.4收益分析及非提出者反审：保留一次固定比较，普通基线覆盖或全过就封闭，不扩力、改窗口、加规划次数或新训练追差异。完整条件见[启动前协议](LEGACY_CREDIT_GOAL_CONTINUATION_PROTOCOL.md)。

## 条件与具体做法

- 路线左右90°、R1.1m，请求速度左1.8/右1.5m/s；用户α=.5/.75。冻结ECBC/ESO/local300及原执行器、中层三组旧Actor，不重训底层或上层。
- parent=H75原父Actor；task_only/physical_credit=online_preference_survival_v1_attempt3两匹配64终点，Actor SHA分别d1064208…/2d053213…/4ac4cfbb…。floor是parent Actor+1.5m/s参考下限；zero是零神经修正，不是关掉底层或路径跟随。
- 原参考修正速度[-1,+.25]m/s、转向±.2rad及原滤波/速率，最终速度[1,3.5]、floor独自收紧下限1.5。未混入后来.1rad几何反馈/短速度支持/goal-preview。
- 每个原完整初态task0、physical700连续推进；不在受扰点重置、合成状态或切α。每角色在相同原来源与同α下配对，实际轨迹可因控制不同而分化。
- 无扰及task1.9/2.9s×±32N/.1s，原固定世界方向/施力点沿原栈。控制器/规划器看不到本次起扰时刻和力方向。90真实（父/task/credit/floor各20，zero10），不是100独立样本。
- 每条最多20s，真实contact/fault停；共同既有16有效控制块后Actor交回。普通1s边界检测invalid或abs(ey)>1m，最多一次PythonRobotics Dubins R1.8重规划到**原世界目标**；不换用户α、速度或目标。只替换projection/follower，其余物理/ESO/历史/滤波/clock/clean/domain记录保留。
- 主到达=接触或故障前首次进入原goal1m；不加末速/末航向/保持窗口 veto。整20s不触地另报。.5/.75和10s切片不替换1m主口径。

## 结果

|方法|同批轨迹前10s到达|20s到达|20s存活|
|---|---:|---:|---:|
|parent|8/16|15/16|15/16|
|task_only|3/16|9/16|15/16|
|physical_credit|7/16|11/16|15/16|
|floor|8/16|16/16|16/16|
|zero_middle|5/8|8/8|8/8|

无扰parent/task/credit/floor各4/4、zero2/2在20秒全部到达且存活。90条UNKNOWN0。全体到达者也整窗存活；本批没有“先到后触地”。

**用户对时间的判断有实际支持**：同一批连续轨迹从10s看到20s，parent多7次到达，ordinary多6次、credit多4次、floor多8次、zero多3次。这说明晚到不能直接当不可恢复。这里是新共同接续轨迹的时间切片；旧hidden10s与新合同还同时相差交回/规划，不能把旧新差异全部归因延时或规划。

三条触地都在右弯α.5、1.9s、−32N条件：parent在3.010s，ordinary在2.470s，credit在2.560s。它们不能靠单纯把观察时间加长而继续到目标。floor和zero同事件未触地且到达。

### credit具体新增、丢失了什么

对ordinary的20s/1m：到达3增1失，存活0增0失。

|事件（credit对ordinary）|credit最近距|ordinary最近距|结局|
|---|---:|---:|---|
|左α.5，2.9s，+32N|.912m|1.048m|credit新增到达|
|左α.75，1.9s，+32N|.832m|1.522m|credit新增到达|
|右α.5，2.9s，−32N|.862m|1.485m|credit新增到达|
|右α.75，2.9s，−32N|1.043m|.143m|credit丢失，ordinary在12.180s到达|

对parent：到达0增4失、存活0增0失。对floor及逻辑共享zero：到达0增5失、存活0增1失。zero配对的16行是8条真实受扰轨迹跨α共享，不作16独立样本。

credit未达的四条存活轨迹最近距为1.300/1.971/1.012/1.043m，后两条非常接近主阈值，不能形容为根本失稳。连续值保留；即使适度放宽，这组简单基线的主终点仍全过，不靠事后挑半径证明必要性。

### 半径敏感性（受扰20s到达）

|方法|.5m|.75m|1m|真实分母|
|---|---:|---:|---:|---:|
|parent|13|15|15|16|
|ordinary|9|9|9|16|
|credit|8|8|11|16|
|floor|12|15|16|16|
|zero|7|8|8|8|

较小半径下credit没有维持对ordinary的到达计数优势；不能只报11对9。这里没有独立训练重复和独立TEST，不将相关案例数包装成显著性证据。

## 实现、核验和成本

一次90路vmap/scan50/共享编译缓存，349608实际lower ticks、360000分配槽，0训练/拟合，用时277.70s含初始化/编译/I/O；75次因果规划。全部原raw约73MiB在HDD。CPU主机/既有规划器14项通过，真实GPU零积分shape通过后正式执行。封存源/184归档模块与完整初态合同在run中。装配预检错误及修正保留engineering_attempt1/2/3，均0积分，不改旧实验。

主分析对90raw、时钟、配置/原目标/事件、参考权限和总冲量核对；独立AI不读取主结果、不import其helper，直接从全部raw重计540个时窗×半径记录和480个逻辑配对，主要结局与计费一致。逐tick脉冲误差沿既有1%substep容差处理，不追求bitwise一致。未独立物理重演或人类逐例验证。

另输出10张全20s XY、10张任务前缀XY及20张控制/奖励诊断图。当前路线V3运行奖励由实际effective components×.005×scale重建，最大残差4.77e−7；因此本组没有“只算一次0.005但当成0.02”的运行成本漏计。失败余项单列。这个核对仅本组，不能代替全部历史训练dt审计；重规划后的V3奖励也不能当原用户canonical Jα。

## 推论与去留

1. **保留资产和条件收益**：credit对匹配ordinary仍有3增1失；旧hidden合同里floor未覆盖的到达用途也不撤销。
2. **本新合同封存**：floor和zero覆盖已测1m首达与20s存活，credit未新增防摔，不足以作为当前主方法的控制增量。不自动恢复credit训练，不扩力/时刻/路线/窗口/候选库。
3. **容扰对象继续保留**：这组没有估计最大不倒推力，也没有使用旧Emin选择参考；不能宣称容扰公式已有效。已有低速不触地数据、偏好生成状态差异仍是可复用资产。
4. **论文需要的仍是可解释的联合增量**：在原用户偏好代价下协调参考与闭环防摔，必须和普通跟随/规划/速度规则比较。该比较已排除“只是ordinary不够好就足以构成创新”的解释，但没有证明所有偏好控制无用。重规划路径效用定义仍待用户，不代选Jα或借诊断reward挽救故事。

## 紧凑证据入口

- [完整分析与540行记录](../../../../runs/legacy_credit_goal_continuation_v1/analysis/review.json)、[逐例CSV](../../../../runs/legacy_credit_goal_continuation_v1/analysis/records.csv)、[逐例配对](../../../../runs/legacy_credit_goal_continuation_v1/analysis/pairs.csv)。
- [独立核数](../../../../runs/legacy_credit_goal_continuation_v1/independent_review.md)、[固定执行合同](../../../../runs/legacy_credit_goal_continuation_v1/execution_plan.json)、[运行终态](../../../../runs/legacy_credit_goal_continuation_v1/evaluation/status.json)。
- [奖励核对/图索引](../../../../runs/legacy_credit_goal_continuation_v1/diagnostics/receipt.json)、[分项积分](../../../../runs/legacy_credit_goal_continuation_v1/diagnostics/cost_integrals.csv)。全部原数据不重复打包，不上传外部服务。

方法审查采用Scientific Agent Skills（scientific-brainstorming1.4、scientific-visualization1.4、scientific-writing2.3）；技能论文已在主稿参考文献[4]记录。AI核验不等于人类稿件批准，submission_ready仍false。
