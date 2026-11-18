# 同数据结构模型与直接模型

本报告的模型训练与预测评测只覆盖跳前STTW阶段0，不代表其他阶段或RB预测模型已完成。

[图保留在本地完整证据包：DEV条件1秒实际/预测XY第一页]（见 compact_delivery_manifest.json；远端省略重复分页图）

[图保留在本地完整证据包：可用DEV条件3秒实际/预测XY第一页]（见 compact_delivery_manifest.json；远端省略重复分页图）

[图保留在本地完整证据包：本批初始化训练与开发曲线]（见 compact_delivery_manifest.json；远端省略重复分页图）

同数据M0线性基线初始1秒/3秒平均最大XY误差为0.06728/0.21449m。M0用相同粗糙TRAIN跳前STTW阶段0转移拟合，区别于此前平地名义模型。M1为M0＋地形/历史修正，M2用相同输入、GRU64/128×128头及姿态积分直接预测动态增量；本报告覆盖种子[0]，两个学习模型使用相同采样和更新预算；其余计划种子不计为完成。

|模型|种子|最佳已评估更新|时域s|窗/例|平均最大XY m|P95 m|最坏m|末25次相对改善|
|---|---|---|---|---|---|---|---|---|
|residual|0|600|1.0|3115/216|0.02913|0.06748|0.18842|1.68%|
|residual|0|600|3.0|955/216|0.06640|0.13228|0.20139|2.06%|
|direct|0|600|1.0|3115/216|0.04233|0.10904|0.26960|0.77%|
|direct|0|600|3.0|955/216|0.10367|0.21521|0.28909|-9.29%|

最佳按固定DEV1秒均值在每25次存档中选择，3秒不单独选模；不是所有更新全程最优。重叠窗口非独立样本，各时域覆盖不同；逐例最坏值保存在cases.csv。损失是监督预测损失，不是PPO，没有策略reward/KL指标。学习模型读取真实381维历史作初始化，未来只用预测状态及自身隐状态查询固定地形，不读未来真实历史。

地形描述是预测位置附近15点的高度、坡度、局部高度变化及支撑mask；沟与图外未标为平地支撑。双线性栅格只是模型输入，最终碰撞几何不替换。地图划分、初态及速度以本轮manifest为准；开发预测比较尚无独立TEST或规划收益证明。没有用平均误差宣称全程净空合格、训练正式收敛或完整P3已完成。

原始存档/标量/逐窗口：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p5_alpha_models/phase0/pairs/seed0`；M0、M1、M2均未修改RB/H75或物理。

本轮M0拟合41134条有效TRAIN转移；来源明细与训练预算见manifest.json。所有模型保留每25轮存档。

[分α预测误差及真实窗/例分母](alpha_summary.csv)。不同α对应真实执行样本，不是给同一轨迹改标签。
全部逐例XY分页：
- [1s 第1项](prediction_xy_50.png)
- [1s 第2项](prediction_xy_50_page02.png)
- [1s 第3项](prediction_xy_50_page03.png)
- [1s 第4项](prediction_xy_50_page04.png)
- [1s 第5项](prediction_xy_50_page05.png)
- [1s 第6项](prediction_xy_50_page06.png)
- [1s 第7项](prediction_xy_50_page07.png)
- [1s 第8项](prediction_xy_50_page08.png)
- [1s 第9项](prediction_xy_50_page09.png)
- [3s 第10项](prediction_xy_150.png)
- [3s 第11项](prediction_xy_150_page02.png)
- [3s 第12项](prediction_xy_150_page03.png)
- [3s 第13项](prediction_xy_150_page04.png)
- [3s 第14项](prediction_xy_150_page05.png)
- [3s 第15项](prediction_xy_150_page06.png)
- [3s 第16项](prediction_xy_150_page07.png)
- [3s 第17项](prediction_xy_150_page08.png)
- [3s 第18项](prediction_xy_150_page09.png)

本批真实耗时654.08秒，两个模型各600更新、25个已评估存档；只完成seed0。残差末段仍改善，直接3秒从575到600反弹9.29%，未认定收敛；暂不采用为可靠全链预测器。[判断与下一阶段](decision.json)。
