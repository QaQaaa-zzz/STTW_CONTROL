# 无alpha局部底层：300更新结果分析

![七场景实际／原始参考XY](XY_ALL_0300.png)

[全部逐时跟踪、奖励分项与轮速估计图](INDEX.md)。覆盖七个声明场景，各2000控制步/10秒、200Hz、bank3、env88001–88007。原始发布指令与R196逐值相同。XY为描述性证据，局部任务没有路径奖励。

**结论：best300在声明的七场景局部跟踪协议下7/7合格，R196为3/7。尚不能称底层完成采用：只有一次后期验证合格，旧完整状态恢复未运行，冻结上层组合未运行。**

正式累计300/300更新、300实际接受更新、39,321,600控制转移；本续训阶段新增100更新/13,107,200转移，完整状态延续200→300，无重新初始化。阶段elapsed含编译/验证为2890.96秒。best=update300，last=update300，两者角色分开；随机训练奖励候选是update281（由282采样计分），不用于本结论。

best SHA `8aa8a90e76dce8c8ad34a8cbca3b4602015926f3d18e24af3000ac2bc26b3577`。最终从磁盘重新加载best完成七场景评价。selection Q=1.739090418，重载Q=1.739100906；float32重复数值差最大XY=0.000130653m、转角=0.000016659rad。四类失败计数及合格判断相同，selection与重载原始轨迹均保留；不称逐位复现。

|候选|局部合格数/7|主跟踪失败|末段联合保持失败|物理/工作界失败|质量Q，越低越好|
|---|---|---|---|---|---|
|100|1/7|6|3|0/0|6.242981|
|150|1/7|6|3|0/0|4.458282|
|200|2/7|5|2|0/0|2.846016|
|250|3/7|4|1|0/0|2.426835|
|300|7/7|0|0|0/0|1.739090|
|R196|3/7|4|0|0/0|3.506210|

固定质量Q200→300改善38.89%，250→300改善28.34%。本轮选择范围仅100/150/200/250/300，不能称每更新全程物理最优。

|场景|best300平稳速度RMSE m/s|R196速度|best300平稳转角RMSE rad|R196转角|小角窗口best/R196 rad|结果|
|---|---|---|---|---|---|---|
|straight_speed_change|0.021661|0.018629|0.001172|0.002933|—|通过|
|steady_positive|0.044913|0.039452|0.003075|0.005197|—|通过|
|steady_negative|0.030179|0.051859|0.012445|0.004140|—|通过|
|return_positive|0.043112|0.039408|0.004546|0.018099|—|通过|
|return_negative|0.036653|0.056439|0.006841|0.022107|—|通过|
|post_return_small_positive|0.038606|0.058569|0.007433|0.021723|0.003186/0.014099|通过|
|post_return_small_negative|0.043969|0.043467|0.005942|0.018265|0.004913/0.014545|通过|

正负小转角实际平台后的有效窗口与方向均通过；门槛为.01rad，目标±.04rad。一般平稳段速度/转角门槛为.05m/s/.02rad，末段连续.5秒联合保持；全程峰值侧倾最大.239191rad，低于.302工作界。速度和转角两项都计分，不以零摔倒或奖励上升代替验收。

## 持续失配与切换瞬态

200时steady_negative转角误差>.02rad连续区间[2.165,10.000)，7.835秒；300时该场景仅[1.040,1.515)，.475秒的入弯瞬态，后续长期偏差已消除。这是当前局部协议的改善，不能冒充旧B0[6.315,8.745)的2.43秒原失配已修复。

所有逐时超差区间如下。一般场景转角阈值.02rad，小角场景按整条轨迹报告.01rad（验收仍按实际小角平台窗口）。切换段超差保留，协议允许稳定前的瞬态；没有裁掉失败区间。

|场景|转角超差区间s|速度超差区间s|
|---|---|---|
|straight_speed_change|[]|[]|
|steady_positive|[{'start_s': 1.034999966621399, 'end_exclusive_s': 1.5, 'duration_s': 0.465, 'start_index': 207, 'end_index_exclusive': 300}]|[{'start_s': 1.6100000143051147, 'end_exclusive_s': 1.7149999141693115, 'duration_s': 0.105, 'start_index': 322, 'end_index_exclusive': 343}]|
|steady_negative|[{'start_s': 1.0399999618530273, 'end_exclusive_s': 1.5149999856948853, 'duration_s': 0.47500000000000003, 'start_index': 208, 'end_index_exclusive': 303}]|[]|
|return_positive|[{'start_s': 2.0349998474121094, 'end_exclusive_s': 2.995000123977661, 'duration_s': 0.96, 'start_index': 407, 'end_index_exclusive': 599}, {'start_s': 4.545000076293945, 'end_exclusive_s': 5.505000114440918, 'duration_s': 0.96, 'start_index': 909, 'end_index_exclusive': 1101}, {'start_s': 5.614999771118164, 'end_exclusive_s': 6.050000190734863, 'duration_s': 0.435, 'start_index': 1123, 'end_index_exclusive': 1210}]|[{'start_s': 2.544999837875366, 'end_exclusive_s': 4.519999980926514, 'duration_s': 1.975, 'start_index': 509, 'end_index_exclusive': 904}]|
|return_negative|[{'start_s': 2.0399999618530273, 'end_exclusive_s': 3.0, 'duration_s': 0.96, 'start_index': 408, 'end_index_exclusive': 600}, {'start_s': 4.574999809265137, 'end_exclusive_s': 5.480000019073486, 'duration_s': 0.905, 'start_index': 915, 'end_index_exclusive': 1096}, {'start_s': 5.565000057220459, 'end_exclusive_s': 6.03000020980835, 'duration_s': 0.465, 'start_index': 1113, 'end_index_exclusive': 1206}]|[{'start_s': 2.674999952316284, 'end_exclusive_s': 3.0, 'duration_s': 0.325, 'start_index': 535, 'end_index_exclusive': 600}, {'start_s': 3.0899999141693115, 'end_exclusive_s': 4.53000020980835, 'duration_s': 1.44, 'start_index': 618, 'end_index_exclusive': 906}]|
|post_return_small_positive|[{'start_s': 2.0199999809265137, 'end_exclusive_s': 3.049999952316284, 'duration_s': 1.03, 'start_index': 404, 'end_index_exclusive': 610}, {'start_s': 3.869999885559082, 'end_exclusive_s': 4.505000114440918, 'duration_s': 0.635, 'start_index': 774, 'end_index_exclusive': 901}, {'start_s': 4.545000076293945, 'end_exclusive_s': 5.494999885559082, 'duration_s': 0.9500000000000001, 'start_index': 909, 'end_index_exclusive': 1099}, {'start_s': 5.539999961853027, 'end_exclusive_s': 6.130000114440918, 'duration_s': 0.59, 'start_index': 1108, 'end_index_exclusive': 1226}, {'start_s': 7.519999980926514, 'end_exclusive_s': 7.779999732971191, 'duration_s': 0.26, 'start_index': 1504, 'end_index_exclusive': 1556}, {'start_s': 7.87999963760376, 'end_exclusive_s': 8.204999923706055, 'duration_s': 0.325, 'start_index': 1576, 'end_index_exclusive': 1641}]|[{'start_s': 2.674999952316284, 'end_exclusive_s': 3.0, 'duration_s': 0.325, 'start_index': 535, 'end_index_exclusive': 600}, {'start_s': 3.0899999141693115, 'end_exclusive_s': 4.53000020980835, 'duration_s': 1.44, 'start_index': 618, 'end_index_exclusive': 906}]|
|post_return_small_negative|[{'start_s': 2.0199999809265137, 'end_exclusive_s': 3.0299999713897705, 'duration_s': 1.01, 'start_index': 404, 'end_index_exclusive': 606}, {'start_s': 4.519999980926514, 'end_exclusive_s': 5.525000095367432, 'duration_s': 1.0050000000000001, 'start_index': 904, 'end_index_exclusive': 1105}, {'start_s': 5.579999923706055, 'end_exclusive_s': 6.15500020980835, 'duration_s': 0.5750000000000001, 'start_index': 1116, 'end_index_exclusive': 1231}, {'start_s': 7.514999866485596, 'end_exclusive_s': 7.795000076293945, 'duration_s': 0.28, 'start_index': 1503, 'end_index_exclusive': 1559}, {'start_s': 7.929999828338623, 'end_exclusive_s': 8.135000228881836, 'duration_s': 0.20500000000000002, 'start_index': 1586, 'end_index_exclusive': 1627}]|[{'start_s': 2.544999837875366, 'end_exclusive_s': 4.519999980926514, 'duration_s': 1.975, 'start_index': 509, 'end_index_exclusive': 904}]|

turn_return切换时约.9–.96秒的>.02rad瞬态仍存在。XY图也显示正向回正/负向小角场景的原始参考位置偏差仍存在；局部合格不意味着世界航向或路径恢复，不意味着新模型逐项优于R196。下一步组合必须保留原始任务参考，不能用修小指令重定义成功。

## 训练证据与状态审计

[奖励、loss、KL与完整回合误差](training_0300.png)。训练批次是变化任务/变化策略，不当作固定配对物理结果。

|完整训练回合窗口|回合数|平均每步成本|速度RMSE m/s|转角RMSE rad|
|---|---|---|---|---|
|151–175|1536|0.0031060|0.028273|0.041424|
|176–200|1536|0.0027831|0.027559|0.038438|
|251–275|1536|0.0020644|0.025407|0.033001|
|276–300|1536|0.0018636|0.024270|0.030790|

审计：metrics更新号完整1–300，无重复或缺号；完整update300与resume_boundary策略一致，Adam非空，保存512×10×20局部历史及完整环境pickle/RNG。R196七份缓存与冻结upper两端checkpoint/Actor哈希均未变化。reward逐帧从冻结本地配置重建，与候选日志数值核对；R196同奖励重评分，不跨upper reward比较。没有新增源码修改、全仓测试或仿真评测。

## 当前决策与剩余门槛

不因一次7/7而自动注册、安装或启动upper。附件要求连续两次后期验证合格，目前250未过、300首次通过；需保留此证据缺口。依照既定顺序，下一个控制能力检查是一次原6.315秒完整plant/ESO/actuator状态的2秒恢复比较：局部[2.3,0]，1秒内进入|ev|≤.10/|ed|≤.04、连续保持.5秒、后续不重新离带、不新增工作界超限。新历史只由先前十个真实状态/命令/执行动作构造，不拼日志伪造闭环快照。

若旧恢复点失败，只能针对该类有效恢复状态定向补训，不能自动追加更多普通直行到400。若通过，仍需确认连续后期合格条件，再进入冻结upper0best新增200/upper1best新增250的直行、fast_turn组合，10秒与16秒分开，逐时分解upper修改/lower执行/最终任务误差。当前恢复比较、组合比较均not_run，新底层not_adopted，无上层训练或push。

本次用户请求分析已完成；未自行发起下一训练阶段。总预算上限400保持不变。
