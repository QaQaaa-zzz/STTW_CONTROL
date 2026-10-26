# 冻结策略的离线价值拟合诊断

目的：判断已测Critic末段数值失配能否由普通MC重拟合解决，或需要明确非正回报/有限终点结构；不声称价值拟合改善会提高Actor或形成论文新方法。

数据固定matched_live_ppo_v1的第64次采集：真实采样的是plain/support各自冻结第63终点，不是final64。核原数组/采样身份/原63checkpoint SHA。每臂60完整历史回合，replicate0的30回合训练、replicate1的30回合开发验证；各含左右/五alpha/三力档。整回合分区，不跨时间点泄漏；仍是同初态祖先的相关DEV，非独立路线测试。

MC目标按实际20ms时间gamma=.997，含全部fallback任务成本，只给合法决策的352维观测拟合。不读取validation设初值，不拟合Actor/物理Critic、不推进物理或改奖励。

四模型每臂同数据/同mini-batch顺序、Adam3e-4、batch1024、32epoch、grad_clip1，固定终点不挑最佳：
1. inherited_refit：现有Critic原样初始化，普通MC MSE重拟合。
2. reset_free：原隐藏特征、输出头清零，V=-rD(t)+NN(s,t)。
3. time_free：相同隐藏特征、头常值-r，V=D(t)NN(s,t)。
4. time_nonpositive：相同隐藏特征、头初值inverse_softplus(r)，V=-D(t)softplus(NN(s,t))。

D(t)=(1-gamma^(500*remaining_fraction))/(1-gamma)，D(T)=0。r仅由train最小二乘决定。新2–4初始函数逐点相同；1是保留继承初值的常规重拟合对照，不能将1对其它的差全归参数化。所有loss均在原价值单位计算，不改目标或加尾段权重。2与3/4比较区分相同初始函数下的条件化/边界；3与4区分非正约束。不同参数化的优化几何亦变化，不能声称完全等效更新。

报告train/dev整体与末1秒/末20ms的MC误差、正值预测比例、初始函数一致性、全部32epoch曲线、实际optimizer步及耗时。真实历史平均reward写TensorBoard并核HTTP，明确非新增rollout；Bark中文完成/错误，不启邮件/autofix。两臂四模型顺序执行，不抢占或停止其它项目。

若普通MC已覆盖，不开发新Critic结构；若结构改善dev但未验证Actor收益，只记预测候选，再设计匹配在线试验；若都不改善则保留不确定性，不扫epoch/权重。冻结原Actor、Critic与全部历史；不更改任何best/alias，不把此工程诊断当RA-L独立贡献。
