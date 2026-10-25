# 学习策略与速度下限组合：完整结果与裁决

2026-10-25；STTW_CONTROL；相关开发分支，不是最终独立确认。

## 为什么做、如何做

前一轮普通task64同权限完成5/16受扰任务但未摔15/16；父策略加floor任务3/16、未摔16/16。本轮检验：将同样的简单速度下限套在学习Actor上，是否已能兼顾两者，避免把未比较的简单组合误当新方法空间。

仅新增task64+floor、physical64+floor：左右R1.1，请求1.8/1.5m/s，原alpha=.5/.75，无扰及1.9/2.9秒×世界方向±32N、持续.1秒，共40条。全部从认证完整task0/physical700状态连续运行10秒，冻结local300/ECBC/ESO，保留历史、滤波、执行器和时钟。各方法同16个有效完整策略块后交回原alpha；future pulse不进入控制器。floor沿用现有1.5m/s参考下限机制，原动作/参考权限与限速不变，不叠加ESO规则。

code7f549d9；33CPU检查通过，左右完整状态GPU shape各20、0积分。实际40/40完成，80000新5ms积分步/80000槽位，0优化更新；controller session33659 exit0。80匹配旧轨迹只复用，120逻辑样本都核raw SHA、原alpha成本与任务标签；新40终态核alpha/时钟/累计离域。分析session80618 exit0，0新积分/训练、0UNKNOWN。原父策略训练和历史轨迹成本不算本轮新增。

## 相同交回权限下的完整结果

任务=实际未触地且原目标及末窗路径/航向/速度达标；工作侧倾合规performance_success另报，不混为触地。

|方法|名义任务/4|受扰未摔/16|受扰任务/16|
|---|---:|---:|---:|
|父策略|1|15|1|
|父策略+floor|1|16|3|
|普通task64|3|15|5|
|普通task64+floor|2|16|4|
|physical-credit64|4|15|3|
|physical-credit64+floor|2|16|4|

逐对结果：task+floor对task未摔1新增/0丢失、任务0新增/1丢失；physical+floor对physical未摔1新增/0丢失、任务1新增/0丢失，但名义任务丢失2例。两组合对父floor均任务1新增/0丢失、未摔相同。两组合的受扰任务与未摔标签完全一致，不能由此归因物理信用训练更有效。

## 改善和代价发生在哪

1. **补上原摔倒**：右alpha.5、1.9s负力，task/physical原实际触地，两个floor组合完整10秒未触地，但均未完成任务。task+floor末路径.8219m、航向.8351rad；这是真实防摔收益，不是完整恢复。
2. **丢失普通策略原正例**：右alpha.5、2.9s负力，task无floor成功；加floor仍不倒但末路径.06154→.11742m、航向.01515→.21605rad，路径RMSE .75963→1.1708m，原alpha成本65.311→69.820，交回5.22→5.92s。不是几毫米级标签变化。最大偏离1.7330→2.4652m，不能只报末窗。
3. **对父floor的剩余任务增益很有限**：左alpha.75、2.9s正力，两组合任务成功，父floor因航向.056996rad略超.05带失败。task+floor航向.02733rad，路径RMSE .6354vs1.0467m，成本35.929vs40.724。有连续误差改善，但二元任务只多一个相关开发案例，不能据此宣布方法突破。
4. **名义退化不能隐藏**：两个floor学习组合右路两alpha均名义失败。普通名义3→2，physical名义4→2。较高防摔不能自动抵消对正常任务的损害。

## 只读状态诊断：不能只看速度

右alpha.5早起扰前最后一帧，task无floor/有floor实际速度1.2887/1.4616m/s、侧倾−.0870/−.0909rad、实际转角−.1897/−.1634rad；晚起扰前速度1.4426/1.5198、侧倾−.1024/−.1692、实际转角−.2412/−.3364rad。因此floor同时改变进入姿态和转向，不能从这组数据推断单独提速的因果作用或通用临界速度。world-direction脉冲仍按原合同，不冒充同body方向力。

pre_pulse_contexts.csv包含所有120轨迹的边界参考占用和96条受扰前200ms状态。参考恰在下限只叫占用代理，不当作控制必要性标签；没有拟合触发阈值或利用已知未来事件选动作。

## 研究决策

- 保留普通策略的任务恢复、floor的防摔、同权限交回作为可复用资产；两个floor组合是后续必须保留的强对照，不等同新控制方法已被采用。
- 固定floor无法保留全部原偏好任务，值得研究“何时、以何种可执行参考过程保留抗扰能力”；当前还没有证明状态相关方案一定能实现二者兼得。
- 停止同一floor数值、16块等待、physical信用权重和训练长度搜索。没有新的独立机制证据，不再扩physical训练。
- 下一步返回规划书第7节的提前参考机会，只检验一个有物理动机、只依赖当前路径阶段的平滑干预过程，先核能否通过原幅度/变化率/滤波实现。不能用实际起扰时刻切换，不能根据每个事件事后挑动作。
- 机会检验必须同时比较原task、固定floor学习组合、父floor和同样干预的父Actor，兼顾无扰与完整双方向/时刻；若简单阶段规则覆盖收益，将其保留为强基线。当前不开发新的大Critic或增加抗扰需求输入，不把这一想法称首次方法。具体新实验须另行封存，当前未启动。
- 旧E_min=(z+KF)^2/W仍只作为局部理想输入的解释/基线；本轮不验证其牛顿阈值或安全余量。真实闭环能力与原alpha任务代价才是后续选择对象。

## 图与审查范围

全部10共同XY已目视，覆盖2route×5event×2alpha×6方法；已另看new_trace021/023的完整实际速度/轮速代理/转向/姿态/奖励/交回时序。其余38新增细图和80缓存细图本轮未逐张重审；保持visual_review_complete=false。全部原数组、奖励分项和图可定位。

- [早起扰：原接触与floor未摔](../../../../runs/learned_floor_factorial_v1/analysis/xy_right_t380_F-32.png)
- [晚起扰：floor丢失原任务正例](../../../../runs/learned_floor_factorial_v1/analysis/xy_right_t580_F-32.png)
- [名义右路：固定floor的任务代价](../../../../runs/learned_floor_factorial_v1/analysis/xy_right_zero.png)
- [全部方法](../../../../runs/learned_floor_factorial_v1/analysis/by_method.csv)、[逐对新增与丢失](../../../../runs/learned_floor_factorial_v1/analysis/paired_floor.csv)、[全部案例](../../../../runs/learned_floor_factorial_v1/analysis/all_cases.csv)、[状态诊断](../../../../runs/learned_floor_factorial_v1/analysis/pre_pulse_contexts.csv)、[目视记录](../../../../runs/learned_floor_factorial_v1/analysis/visual_review.json)

正常采集/分析中文Bark已accepted1792943514.720075/1792943716.142658；heartbeat1.1s、scan空。一次只读派发attempt1遇当前Python无os.pidfd_open，原错误/源码/trace保存，真实错误Bark accepted1792943684.0119271；复用现有Linux syscall fallback、3真实进程事件CPU测试通过后明确只读attempt2完成。没有重跑物理、重装环境或改变监控服务。
