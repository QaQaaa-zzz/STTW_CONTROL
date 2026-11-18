# P5 条件历史桥

三个种子各600更新完成，墙钟17.77秒，控制策略更新0次。输入为固定跳前条件预测模型的状态/隐状态、时钟与初始执行历史；预测537维控制器/RB历史，不读取未来真实历史。

使用phase0_continue600/residual_seed1（固定1s DEV选择）；12个地图/α来源，每个来源仅保留canonical downstream0路线，TRAIN/DEV按地图分离。α覆盖0/.5/1。归一化只用TRAIN，每25次存档；已知DEV不是独立TEST。

|seed|最佳更新|入口归一化RMSE|控制器RMSE|RB RMSE|
|---|---:|---:|---:|---:|
|0|575|0.188067|0.201696|0.149651|
|1|450|0.208959|0.222360|0.171894|
|2|575|0.197720|0.210399|0.162650|

这些数值无物理单位，不是厘米误差。400→600入口RMSE分别改善3.15%、退步0.61%、改善3.70%；保留最佳及反弹，不能据此认定正式收敛或规划可靠。此阶段无新物理轨迹，实际/预测地面XY见../phase0_continue600/RESULTS.md；跳跃链路收益尚待验证。

原始模型、逐存档及TensorBoard：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p5_alpha_models/conditional_history_bridge`。Bark回执sent/HTTP200。全部存档指标见checkpoint_metrics.json，来源与归一化合同见contract.json。
