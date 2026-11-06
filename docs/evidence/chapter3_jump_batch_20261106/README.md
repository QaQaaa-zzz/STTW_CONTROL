# 六候选批量执行配对

同根、同模型、同 400/200/50 Hz 时序；0 次训练更新。实际轨迹见 [XY 对比](batch_vs_single_xy.png)，逐候选判据和交接时刻见 [比较数据](batch_comparison.json)。

采样耗时 224.478 秒，旧单环境累计 462.524 秒，观测比 2.0604；机器共享负载不同，不是独占性能基准。双方均 3/6 完整成功，但只有 2/6 逐候选结果一致，新增与损失各 2。部分差异出现在上游地面阶段；不能将本批替代旧配对数据，暂不采用为正式采集入口。

原始轨迹、执行输入、接触与运行日志：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_jump_batch_pilot/evaluation_repaired_scope/`。首次启动闭包作用域异常保存在同级 `evaluation/`，不是技能失败。比较命令：`python learning/cli/compare_jump_batch.py --run <上述目录>`。

当前地面根状态采集和两阶段连接实验继续；本批仅为工程性能与行为比较，不是独立 TEST 或方法提升证据。
