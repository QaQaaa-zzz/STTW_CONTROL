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
