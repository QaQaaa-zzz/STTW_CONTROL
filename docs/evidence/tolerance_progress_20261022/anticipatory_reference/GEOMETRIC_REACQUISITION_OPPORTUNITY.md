# 存活后的几何再捕获：下一项有界机会验证

2026-10-25。状态：控制器与专用后缀runner已实现（ef574aa），10项相关CPU检查与实际完整状态GPU shape通过；六条物理后缀已启动，结果待收回。服务于原问题：在冻结底层和原偏好任务代价下，把防摔之后的可用控制能力兑现出来。不是另起硬件项目或把普通路径跟随包装成创新。

## 为什么换问题

旧ESO规则在本右低速弯受扰4/4未摔、1/4任务，floor4/4未摔、0任务；新固定sequence26仍3/4与0，常值slow0/4与0。因此停止两个固定目标过程，不加结点或继续调旧能量代理。

已完成shadow_reacquisition.py：70逻辑结果中的43条safe_task_failed末1秒，有6条按原方程复算的preview始终在车后。其中多数实际转向并不为0，所以不是“域外控制被关闭”这一个原因；不能把6条解释成全部失败。

已有两条完整真实t10秒状态：floor/alpha0/t420负向及zero/t520正向。均task tick2000、physical tick2700、未触地无fault；真实ESO、执行器、滤波、底层及中层历史、投影、时钟整体保存；末XY与源原数组核对。来源、lane与SHA见runs/geometric_reacquisition_inventory_v1/inventory.json，提取0物理。

## 已有静态证据与公式边界

对当前真实位置p/yawψ和原路径查询点q计算：b=R(−ψ)(q−p)，原pure pursuit κ=2b_y/(b_x²+b_y²)。仅在原路径局部为直线、eψ=h−ψ且e_y为原横向误差时，有 b_x=L cos(eψ)+e_y sin(eψ)。因此增大L有时能使目标回到前方；cos(eψ)≤0时不能据此保证改善。

实际14个点查询：floor状态原L约1.05m时b_x=−1.442m，已测L=6m第一次回到原body-x有效阈值前方；zero状态从原L到24m全部无前方点，不能用“增大lookahead”统包，需显式转向过程。它们是目标点可用性，绝不是闭环可恢复或物理连接轨迹证明。preview_authority.csv还列原有±.2rad修正clip、filter/slew后四tick参考变化；没有自由实际转向速率假设。

当前TeleopEnv把fixed_roll_reference设为None，ECBC参考侧倾由转向参考/ratio(v)给出，ESO平衡偏移保留。旧6.84°模式不是本运行的硬侧倾目标，不用它解释这两状态。

## 拟定最小对照

先只问两真实状态有没有合法几何修复机会。三个后缀共用当前完整状态、原路径/请求1.5/原alpha状态0、下层与原修正权限：

1. 原live_zero继续原追踪，不调用失效域Actor。
2. 固定6m路径点产生几何转向修正；点仍在车后时使用共同的有界航向转向基线。
3. 从[原L,2,4,6,10,16,24]m查询中选择最近的前方点；均不可用时用与2完全相同的航向转向基线。

原Pursuit/nominal publisher不改，新几何目标仅转成两维残差输入：速度修正0，转向修正=所需转向参考−当前nominal steer，再走原map/filter/rate/reference bounds。航向转向基线拟固定目标转角0.18rad×sign(wrap(路径航向−当前yaw))，也经过同一修正范围；不是原地旋转、直接电机命令或安全保证。没有候选前方点时如实记不可用，不伪造可行标签。全部原Actor查询保持关闭，本次还不证明偏好策略交回。

两状态×三规则=6新后缀。拟统一12秒后缀、原任务10+12=22秒新延伸协议；最长14400新5ms ticks、0optimizer。延伸较8秒的原草案更长，因为一条状态距原直线约8m且朝向背离，需要真实转向和前进；同一新时域给全部规则。原10秒失败保留，不回填。原全程成本和goal/末窗判断应拼接真实10秒前缀与新12秒后缀，不能用新连接路径重定义目标。控制若提前触地/故障立即停止。

这三条只作机会/强基线筛查，控制量与回退需TDD验证、封存后登记启动；当前无新物理。若固定6m已覆盖，则作为共同工程模块保留，不称新方法。若自适应确有额外空间，再在完整任务起点验证检测、稳定、接入和恢复原alpha策略，并补固定/状态相关接入的匹配比较；若都无机会，不反复调L和0.18rad。

## 文献核对与新颖性

有界原始来源核对，非系统综述：

- [Macenski等，Regulated Pure Pursuit](https://arxiv.org/abs/2305.20026)：作者摘要已包含自适应变体上的速度调节，条目列Autonomous Robots 2023。自适应lookahead/减速本身不是新贡献。
- [Ohnishi与Takahashi，DWPP](https://arxiv.org/abs/2601.15006)：本次只读作者摘要，明确在v−ω空间考虑速度/加速度约束。尚未读全文或核对正式出版记录，不照搬其理论到当前速度/转角参考、动态平衡及冻结神经底层。
- [Nav2当前官方RPP说明](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/controller_plugins/configuring_regulated_pp/)：已支持速度相关lookahead和dynamic window；rotate-to-heading针对能原地旋转的平台。我们的真实参考速度下限≥1m/s，不能直接用停住原地转向当恢复动作。

因此本阶段主要是补足强对照并测可利用空间。若最终只有修追踪器，没有超简单规则的抗扰协调，不能借“容扰度”名称抬成新方法。后续若发展方法，仍须体现真实参考权限、原alpha代价、实际防摔后果和强对照外增量。

工具：scientific-critical-thinking、experimental-design、systematic-debugging；软件出处沿相邻DESIGN的Scientific Agent Skills引用。几何公式为局部坐标直接展开，不是新理论或安全认证。


## 后缀接口实施时必须保留的时钟语义

通用anticipatory_run.execute要求task tick0并按初始physical time重新加pulse_onset，不能直接用于这些t10快照；否则会重复施加外力。新后缀runner应从真实task tick2000/physical tick2700继续，到task tick4400/horizon22秒；不重置任何时钟，并显式使用零新外力。source原10秒前缀仅从原数组拼接分析，不再模拟。各规则使用同一继承min-reference floor（floor状态1.5，zero状态1.0），不以改变输入权限换收益。须真实检查父状态不failed、不fault及数值有限，触地和故障状态不可恢复推进。

## 已封存运行与分析入口

运行：runs/geometric_reacquisition_extension_v1；plan.json保存模型、配置、原始前缀及源码身份。extension/status.json为六条采集状态，monitoring.json记录900秒检查间隔，原10秒结果不改。CLI --phase analyze负责原22秒任务/成本/共同XY；analyze_suffix.py补充逐条原前缀不变核对、零新外力与时钟连续、后缀原任务误差和成本、目标→限幅→滤波→最终参考→实际转向的差别。当前只完成分析脚本准备与语法检查，未读取或宣称新结果。首次进入路径/航向带不等于持续接回，更不等于恢复原偏好Actor。

## 已完成裁决（2026-10-25）

6/6后缀已完成，14400实际5ms步、189.782s、0训练/新外力。六条无触地且均未满足原联合任务；00 zero本来已能靠近路径，adaptive末误差改善但原alpha总成本222.345→231.839，01 fixed6比adaptive更接近却仍偏离6.04m。停止本候选参数/时长扩展和评分网络；不据本批宣称接回无解。完整六条/前缀44字段不变/共享XY与六条机制图见run comparison/DECISION.md。后续回到原64物理策略真实两个任务正例做隐藏时刻开发确认；18既有轨迹进入状态审计表明正例并非只是提速，仍待因果验证。
