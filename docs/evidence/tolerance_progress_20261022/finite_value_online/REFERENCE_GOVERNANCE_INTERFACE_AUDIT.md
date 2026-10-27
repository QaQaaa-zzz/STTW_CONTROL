# 冻结接口的参考治理前提审查

2026-10-27。目的：判断现有几何跟随器与冻结反馈能否直接采用时间治理理论，避免重新实现已有比较器或增加用户未要求的停车任务。0新物理、0训练、0拟合、0权限修改。

## 怎样检查

实读`path_reference.py`、`geometric_path.py`、`path_command_env.py`、`path_command_policy.py`、`frozen_local_lower.py`和`path_reference_governor.py`。新CLI `audit_reference_governance_interface.py`只调用纯接口函数，不构造env、不加载Actor、不积分物理。

最终v2将native三份归档源码SHA核对到原执行plan的native_expected_sources，三份生效配置分别核到各原plan的config或sealed/input SHA。当前工作树与native归档的pursuit/publish/project/at/map_latent/correction_tick六函数AST一致；整个policy文件仍有其它差异，不称完整运行时一致。

实际检查45个correction合成参数组与9个goal-marker参数组。它们不是45条机器人轨迹或停车测试。独立AI重算纯函数、六AST与全部SHA一致；v2新增执行身份绑定另核，不把v1旧脚本SHA冒充最终版本。5个既有预测治理CPU测试通过。

## 结果与含义

|实际实验配置|速度参考范围m/s|当前权限能否发布精确0速度参考|
|---|---|---|
|旧能力90|[1,3.5]|否|
|旧重规划72|[1,3.5]|否|
|native80|[1.5,3.5]|否|

`correction_tick`在滤波/速率限制后限幅，即使合成nominal=0或负latent，发布结果仍受下限约束。**命令下限不是实际速度下限，不证明静止必摔。**实际速度取决于物理与完整反馈。

`pursuit`速度目标直接取v_user，预瞄点取实测投影progress+lookahead。九个检查保持几何、pose/progress=10m、v_user=1.5，仅改goal标记5/10/15m，命令均[1.5,0]；说明goal标记本身没有触发停车。用户本来只要求首入目标圈，不应为套理论额外加入停车条件。

`PathCommandEnv.tick`每个底层步按实测post XY更新project，后者使用局部搜索窗和单调几何进度。当前PathState没有另一个独立积分、可直接冻结的虚拟路径时间变量。冻结几何progress或任务时钟，不是现有API中能自动使速度归零的治理动作。

local300包含实际历史、ESO相关输入及参考/执行器carry，本轮没有证明其固定参考点稳定性。`controller._system`内部minimum_speed用于增益计算，也不是物理最低速度或停车稳定证明。本轮不更改该正则化、不测试低速物理。

## 理论边界和已有比较器

此前已原文核对[Arslan, Time Governors for Safe Path-Following Control](https://arxiv.org/html/2212.01444) III/IV：其框架含固定路径参数时的点稳定前提，示例为全驱动高阶系统。当前接口未证明这些条件，不能直接承接其安全/收敛结论。**这不排除所有参考治理，也不证明我们的新颖性。**

此前只核摘要的[Wang等EABC trackstand研究](https://arxiv.org/abs/2410.19615)已研究单轨车静止平衡；不能将本接口缺零速请求扩成单轨车物理上不能静止，也不冒称本项目已有其控制能力。

当前仓库已有`path_reference_governor.py`。它在前一/当前神经proposal物理偏移间取五点，完整carry和现有反馈/滤波/下层预测1秒，无实际未来事件输入，假设未来外力零。它不依赖点稳定来作有限预览，但也没有终端不变集或认证backup；全部预测触地时previous proposal仅是不保证安全的操作fallback。有限候选全部失败不证明物理不可恢复，有限预测通过也不保证递归可行。

回读历史`path_governor_numerical_5/numerical.json`：五预测均存活、选fraction1，1000实际预测ticks/1000slots，首次batch执行12.2068s，**governor_control_episodes=0**。它不是在线闭环治理、20ms实时或抗扰收益证据。应复用现有比较器，不能把重新实现它列为新路线。

## 去留：本轮不启动新RG实验

沿scientific-brainstorming v1.4区分接口事实、理论前提与方法假设，非提出者独立反审赞同：前提未证只定位适用性缺口，不能自动授权改floor、增加候选、加长预测或新训网络。用户当前任务也不要求停车，不能将停车作为新贡献制造的任务。

还实际回看了优先重规划前暂停的`reference_scenario_search.py`/958b8ed。它尚未回答“同6维参考偏置按多个扰动后果生成候选，是否补足旧force0生成目标错配”，不同于旧池受扰筛选，但属于普通scenario/CEM，非新算法。**当前保持未执行，不标已经失败。**新72/80中zero覆盖二元结果，且改路后的原用户效用尚未统一；直接优化旧scored_cost可能只改善改路后的诊断reward，当前信息收益不足。

以后若重新讨论，先需同20秒、原goal、同因果规划/权限和明确原用户评价；保留scenario-random、F0优化、原Actor、zero/静态规则强对照，设计与冻结评价分开，候选不读实际未来事件，实际ticks/槽分别计费。阴性只关闭固定表示/搜索/设计集，不扩knots、force或轮数补考。这里只记重新讨论条件，不是新实验计划。

**下一项具体工作**：核清改路后原用户α评价的定义与可重建范围。保留原goal首达和整窗触地两项独立事实，原路径/原请求速度/参考变化代价分别可审查；当前路径reward、原固定路径效用和任务成功不混用。先查已有信号与评分实现，不再重算同批终点表或用事后赢家训练选择器。若不能忠实重建则明确UNKNOWN并收缩对应优势主张。

## 交付与异常

最终HDD入口`runs/reference_governance_interface_audit_v2`，含review.json与independent_review.json/md。v1原输出和独立复核保留；v2补身份绑定后纯函数重跑45/9，仍0物理，不能将两次检查计成新仿真。本次前提审查没有增加论文结果章节。

首次脚本将native source root重复拼接learning/src导致FileNotFoundError；按plan准确路径修正，attempt1/error.json留档并调用中文Bark。只是新只读脚本路径错误，没有训练/物理异常或重启。随后v1与最终v2实际exit0。
