# 六候选批量执行配对

同根、同模型、同 400/200/50 Hz 时序；0 次训练更新。实际轨迹见 [XY 对比](batch_vs_single_xy.png)，逐候选判据和交接时刻见 [比较数据](batch_comparison.json)。

采样耗时 224.478 秒，旧单环境累计 462.524 秒，观测比 2.0604；机器共享负载不同，不是独占性能基准。双方均 3/6 完整成功，但只有 2/6 逐候选结果一致，新增与损失各 2。部分差异出现在上游地面阶段；不能将本批替代旧配对数据，暂不采用为正式采集入口。

原始轨迹、执行输入、接触与运行日志：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_jump_batch_pilot/evaluation_repaired_scope/`。首次启动闭包作用域异常保存在同级 `evaluation/`，不是技能失败。比较命令：`python learning/cli/compare_jump_batch.py --run <上述目录>`。

当前地面根状态采集和两阶段连接实验继续；本批仅为工程性能与行为比较，不是独立 TEST 或方法提升证据。

## 单环境新旧执行器对照

新入口单环境完整22秒已完成。1635个状态样本一致，XY和速度RMSE均0，T=14.90秒；出口5.22秒、交权9.16秒、落台稳定12.02秒、交回12.72秒均一致。见 [单环境XY](single_lane_control/batch_vs_single_xy.png) 和 [逐项数据](single_lane_control/batch_comparison.json)。这支持当前六环境差异与多环境执行条件有关，具体原因尚未确定；不能据此宣称任意批量等价。

单环境新入口213.792秒，旧97.331秒，当前共享机器负载不同且新入口继续计算跳跃期间地面历史，未实现单环境加速。正式跨技能候选使用原单环境入口推进。

## 零平移、外部地形模式：未采用

六条全部在2.24秒提前结束，尚未到出口决策。采样22.816秒来自提前失败，不能当有效加速。各lane相互一致但与原单环境不同；具体原因待查。保留[逐候选对比](coincident_origins/batch_comparison.json)和[实际XY](coincident_origins/batch_vs_single_xy.png)。下一诊断复用已有原生replicated地形模式，其结果尚未收取。
