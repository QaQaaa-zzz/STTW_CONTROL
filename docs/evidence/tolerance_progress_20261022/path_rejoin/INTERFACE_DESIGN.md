# 路径接回：实时原路径跟随基线与接口实施计划

目标：先排除“只因为旧评测离域后锁参考而失去任务”的解释，再判断是否需要稳定/连接规划。当前仅实现与CPU验证，不启动新物理或训练。来源为用户持续推进授权、规划书第5–6节、固定488终点DECISION；旧结果及其终止合同不变。

三种方案比较：

1. 实时原路径跟随＋零中层修正：最小工程变化、强必要对照；可能在原preview失效处仍不能回到原路径。
2. 稳定后几何连接：可能处理原路径点位于车后的情况，但需要新的接入任务定义和相对原nominal的权限核对；不能直接用任意转角参考扩大±.2rad修正权限。
3. 学习恢复/参考评分：现有数据还未证明前两者解决不了的问题，后置。

本阶段选择1。新make_live_path_step(env,horizon_s)在每个实际5ms tick调用原env.tick(state,zeros(2),原path,原请求速度,pulse)。唯一终止为物理失败、工程故障、固定时域结束；domain_exit保留累计标志但不再终止新合同。原follower仍在20ms时钟上更新，原projection不重定位，原correction_tick滤波/slew/限幅不改；从既有状态接管时zero目标通过原滤波逐渐退出旧修正，不能硬清offset。

每个50Hz区间结束仅对真实推进过的状态调用record_frame，更新实际351维历史；不能对已停止的padding伪造新历史。新基线不调用Actor，不能借此宣称旧Actor已经能够在域外合法恢复。ECBC、ESO、执行器和local300状态全部由原tick推进，不重置。观测/控制数值失败仍终止。

返回逐lower-tick原日志和新合同scored_tick_reward。offset_rate项沿原policy_step计算，物理失败沿原failure_reward，原domain退出不再加终止罚——这是新恢复任务合同，不与旧488累计回报直接当同一奖励排名。原任务路径/航向/速度连续成本始终按原alpha、原请求、原路径计算，未来需要另定义接回成功保持/必经段；本阶段不从domain标志修改历史标签。

实施任务（inline执行，无委派）：

- [x] 新建learning/tests/test_live_path_continuation.py，用toy env验证域退出后4真实tick继续、history只在有真实步时更新、接触/故障不能复活、剩余时域padding不推进、零目标不覆盖已有offset/下层内部状态。
- [x] 先观察缺少模块的红灯，再实现learning/src/sttw_control/live_path_continuation.py；不修改PathCommandEnv、survival_collection或旧评测源码。
- [x] 新行为测试及现有physical_continuation/survival_endpoint_collection测试；CPU结果不称GPU动力学验证。
- [x] 记录代码身份、已完成/未完成部分及下一步小验证参数草案。新仿真另在台账登记、Bark注册后执行；不重跑488或扩大训练。

下一物理验证的推荐最小问题：同一完整任务初态下held尾与live原路径跟随是否有任务新增/防摔损失。优先确定性非零左右弯和少量alpha，10s共同观察窗；原始完整初态/模型/事件身份沿既有资产认证，prefix需要实际重放时单独计费，不宣称bitwise重放。先有该强对照结果，才定义几何连接候选。具体回合矩阵尚未锁定，因此本文件不启动运行。

设计自审：不更改冻结底层/权限，不清域退出历史，不向控制器透露未来扰动，不把旧held尾当实时上层历史，不在本阶段引入新的可学习模块；尚无防摔/恢复效果声明。

## 实际验证

先运行新增测试得到ModuleNotFoundError红灯，然后实现模块。CPU命令：`PYTHONPATH=learning/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python -m pytest learning/tests/test_live_path_continuation.py learning/tests/test_physical_continuation.py learning/tests/test_survival_endpoint_collection.py -q`，实际20 passed in 1.40s。toy fixture只检查状态推进/终止/记账，不充当真实动力学验证。

真实环境零积分审计：`runs/live_path_interface_audit_v1/audit.py`实际加载已认证local300、原完整bank3/配置，左右两条原路径均通过jax.eval_shape，状态树和四lower-tick日志形状匹配。审计status completed，0physics/0optimizer；source/config/lower身份见receipt.json，原日志及Bark登记保留。这个检查未执行JIT物理，不证明接回成功或运行耗时。

旧488的六张XY总图现均已逐张实际查看；重复绕圈/远离原目标的held尾可见，图形与指标一致。120张详细时序图仅先前一张完成目视，其余仍待查，不把总图目视当详细图全审。
