# 磁盘中断、数据复用与恢复边界

2026-10-25，STTW_CONTROL 三臂原64轮实验。目的仍是判断价值估计改动是否在同预算下改善原偏好任务与抗扰表现；本目录仅解决工程中断，不构成新方法收益。

## 已核验的事实

- recovery17 的训练105538及接续110558均已消失；boot ID未变。系统盘曾为0用户可用空间，journal出现ENOSPC，程序traceback与status.tmp为空；不虚构程序异常栈，也不归因于算法或GPU。
- 最后完整配对边界45，三臂Adam计数均为 `[172,180,180]`。checkpoint/Actor SHA、完整learner fingerprint及采样身份已恢复认证。
- 第46批10个chunk均通过ZIP CRC与SHA；决策形状每块50×180、lower200×180；通过原 `assemble_live_arm` 验证全部180回合真实时钟、奖励掩码、完整结局、critic输入与数据有限性。三臂各60回合全窗存活只是本批训练数据，不是终点独立效果。
- `decisions.npz`完整，`physical.npz`损坏；原始chunk完整。保留损坏文件，用chunk新建派生汇总，不重跑46物理。
- 边界45实际15,773,063/分配16,200,000个lower步；缓存46实际及分配均360,000，恰与失败status的16,133,063/16,560,000吻合。前次主机重启前未记录开销仍UNKNOWN。
- 只读认证脚本 `audit_cached_collection.py`；原始证据 `runs/finite_value_online_v1_recovery17/cached46_audit.json`。0新增物理、0optimizer。

## 原始文件保留与迁移

只迁移本任务停止的execution，不移动权重或修改其他项目。HDD归档根：
`/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025`。

首次原17目录394文件/6,950,015,360字节经双SHA后迁移，逻辑路径由symlink保留。初版目录fsync缺口是在其删除SSD源后才收到审查意见；无中途崩溃，随后完成目标目录树与receipt父目录fsync，时序已如实记入original17_archive_receipt.json。不能声称初版已在删除前完成目录fsync。

后续 `safe_archive.py` 在删除前完成文件与目录fsync及manifest持久化；校验源集合/size/mtime，拒绝路径嵌套、receipt/.tmp写入payload、源symlink、已有目标以及不存在的归档父目录。6项测试包括校验损坏、目录fsync失败均保留源。第二次迁移结果以recovery17_archive_receipt.json终态为准，非仅凭进程启动宣称完成。

## 恢复方案与实际接续

独立 `finite_value_storage_recovery.py`，旧reboot恢复模块不改。恢复45完整Actor/Critic/std/Adam；前缀递归引用原1..17和恢复18..45；缓存46重新认证、原目标下仅优化一次；新物理仅47..64。最终仍固定64，原300终点评价合同不变，不重做warmup、不挑best、不扩训练。新原始数据写HDD，防止再次占满SSD。

恢复适配器code e92d449，61项相关CPU测试及6项迁移测试通过，独立AI审查问题已修。实际prepare完成，180路零积分shape完成；新训练PID460647和一次评价接续460648均已核验真实argv。新run `runs/finite_value_online_v1_recovery45` 全部写HDD，原逻辑路径由symlink保留。第二迁移662文件/12,871,849,325字节完成，receipt终态archived_original_path_preserved。

首批46已完成且三臂完整checkpoint重新载入认证 `[176,184,184]`、next_collection46，均warmup=false；cached_physics_receipt证明新增物理0，配对计费恰为实际16,133,063/分配16,560,000。当前新物理47正在运行。HTTP真实reward46为free_GAE−65.7701、finite_GAE−64.8009、free_MC−66.9563；只证明训练可见性，不作控制优劣结论。TensorBoard http://localhost:48611 ，进度 `runs/finite_value_online_v1_recovery45/status.json`；monitoring记录下次+900秒。详细认证 `first_resumed_collection_receipt.json`。

新的storage sequence按pidfd等待真实训练退出，training_status_reads0，不轮询训练曲线。仅完整64终点触发原300评价；其余错误停止保留，未增加优化或采用。旧两个失败run/owner及原始错误完整保留。恢复训练成功不等于控制方法有效或RA-L目标完成。


## 后续主机重启：55边界

原recovery45在55轮后、56的3块采样后再次主机重启。已认证最新完整55及三臂[212,220,220]，原56采样身份复现；3块ZIP/SHA完整，共实际/分配108000 lower步，但没有中间完整carry，不能直接拼接续跑。旧状态和全部文件保留，root状态已如实更正为host_reboot_interrupted。

原位扩展恢复协调器支持递归原1..17/recovery18..45/recovery46..55来源；完整缓存46虽无本地chunk，按缓存receipt计费一次。只允许恢复协调器变更且核验旧source_at_launch SHA，其余数学、奖励、物理和优化源码仍须与旧封存SHA逐项一致。

下一从55完整Adam重做未完成56的物理采样并执行原56..64各一次优化，不再warmup或重优化55。原已完成/当前有效采样槽上限23040000保留；已失败部分108000单列，并在recorded_total中合计，故已记录总上限23148000，未记录的重启费用仍UNKNOWN。不能将这次工程重试伪称在原全成本内。训练后原固定300不变。新prepare/shape/运行状态需见共享台账，不凭本节计划宣称启动。


### 55恢复实际首批验证

新run `runs/finite_value_online_v1_recovery55`（全HDD），code482648b；prepare/180路零积分shape完成，训练323815与一次终点评价接续323816均实际核验。首56完整更新后新checkpoint载入确认三臂[216,224,224]/next56/warmup=false；原pending56采样身份和事件表逐字段复现。HTTP http://localhost:59393 已核三臂实际reward56，数值仅作可见性、不作为方法收益。当前第57批新物理，终点仍64；monitoring为900秒。

第56完成的有效采样累计20160000槽，加旧中断108000后已记录总20268000槽，工程开销未隐去；其它未记录重启开销UNKNOWN。首次认证见first_resumed_collection_receipt.json。新评价接续waiting_training_exit/训练状态读取0，不重复旧owner。当前系统盘约64.56GiB可用，达到50GiB约定。实际终点及方法效果仍待回收，不代表Transactions目标完成。


### 原固定终点已收回（2026-10-26）

recovery55训练64/64、固定评价300/300与只读出图均完成；最终有效实际22,610,292/分配23,040,000 lower步，含失败108,000后已记录总22,718,292/23,148,000，额外未落盘费用UNKNOWN。评价600,000 lower步、0optimizer。工程接续没有形成新独立训练重复；[科学裁决](../DECISION.md)为价值结构未超过普通MC及父策略，不采用新权重，不再扩该训练。所有旧中断/损坏数据和来源保留。
