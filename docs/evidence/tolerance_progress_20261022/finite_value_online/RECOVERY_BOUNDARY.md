# 主机重启后的原64轮边界恢复

2026-10-25，原训练PID3548929及一次接续3629632均不存在，状态仍留running；机器boot ID从145f054f-fdd7-4b52-a58e-0a5d2c93204b变为5fa5773a-c76f-4231-9a96-c691c831c5d2，当前开机时间18:37:52 PDT。程序stdout无traceback，重启原因未知。不能据此诊断PPO或模型出错。原两个状态保存快照后登记host_reboot error，由现有中文STTW Bark通知；不启邮件或自动修复。

## 已保留与已核验

原`runs/finite_value_online_v1`保留为中断历史，不覆盖其日志、17轮权重或第18轮未完成目录。第18轮只有sampling_identity和initial_carry，无chunk/完整回合/learner结果。

三臂最新完整边界是17，checkpoint和Actor哈希、配置、value contract、fingerprint、next_collection17、Adam及更新计数[60,68,68]均已核验。恢复后freeze_collection可逐字段复现原保存的pending18身份。17个已留存数据批的decisions/physical哈希、每臂60回合及全已知结局、实际5805780/分配6120000个5ms lower转移槽已复核。原始审计在旧run/boundary17_recovery_audit.json。

未落盘的预重启物理和optimizer工作准确值为UNKNOWN。按一次scan50×180×4只会有一个待记录块，36000是没有已完成写入在重启中丢失时的条件上界；文件写入并非持久fsync事务，故不能把它当无条件总成本上界。恢复与论文保留已记录成本，并把额外未知开销单列，不能声称完整生命周期费用已精确计清。

## 恢复合同

这是前台获授权的显式恢复，不是自动重试。新目录计划`runs/finite_value_online_v1_recovery17`；新CLI `resume_finite_value.py`/模块finite_value_recovery。只允许恢复源状态最新已完成17，拒绝选16重复训练或18未完成状态。

- 原64轮、五alpha、两路径、脉冲与原采样seed完全不变。恢复各臂各自17轮完整Actor/Critic/logstd/Adam，不初始化新Adam或重做warmup。
- 仅执行原全局索引17..63（第18..64轮），共47轮；采样fold_in仍用原全局索引，不能从0重新编号或重播已完成17批。
- 新run计数含认证原前缀，且另列新物理/分配槽和原前缀来源；保留17+47个完整采集/优化批。23040000是原声明保留轨迹的分配上限；重启丢失工作的未知开销不被隐藏为0。
- 新TensorBoard仅记录恢复后的第18..64轮真实数据，原1..17事件文件不改；先180路零积分shape，首个恢复批的真实reward必须通过HTTP核验。
- 原奖励、alpha、参考权限、底层、gamma/lambda/熵/学习率/epoch及64终点不变；不以本中断挑best、扩训练或加方法。
- 原接续owner保留历史，不重启其死PID；新源目录使用新的实际PID/boot绑定与owner，原300条件表哈希不变，仍只执行一次固定300评价，0optimizer/最多600000 lower槽。

新的运行代码与输入在prepare时封存。恢复后有新错误则保留并停止，不内置反复恢复。先核模型/形状/注册，再启动并实际核验PID与真实reward；未完成步骤不能由本计划代替。
