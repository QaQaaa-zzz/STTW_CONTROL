# 同数据结构模型与直接模型

本报告的模型训练与预测评测只覆盖RB回交后STTW阶段2，不代表其他阶段或RB预测模型已完成。

[图保留在本地完整证据包：DEV条件1秒实际/预测XY第一页]（见 compact_delivery_manifest.json；远端省略重复分页图）

[图保留在本地完整证据包：可用DEV条件3秒实际/预测XY第一页]（见 compact_delivery_manifest.json；远端省略重复分页图）

[图保留在本地完整证据包：本批初始化训练与开发曲线]（见 compact_delivery_manifest.json；远端省略重复分页图）

同数据M0线性基线初始1秒/3秒平均最大XY误差为0.06896/0.27221m。M0用相同粗糙TRAINRB回交后STTW阶段2转移拟合，区别于此前平地名义模型。M1为M0＋地形/历史修正，M2用相同输入、GRU64/128×128头及姿态积分直接预测动态增量；本报告覆盖种子[0, 1, 2]，两个学习模型使用相同采样和更新预算；其余计划种子不计为完成。

|模型|种子|最佳已评估更新|时域s|窗/例|平均最大XY m|P95 m|最坏m|末25次相对改善|
|---|---|---|---|---|---|---|---|---|
|residual|0|1175|1.0|8398/618|0.01882|0.03948|0.32483|-1.00%|
|residual|0|1175|3.0|2511/553|0.04415|0.08033|0.28129|3.13%|
|residual|1|1125|1.0|8398/618|0.01872|0.03971|0.27463|0.56%|
|residual|1|1125|3.0|2511/553|0.04337|0.07980|0.27346|-2.44%|
|residual|2|1200|1.0|8398/618|0.01860|0.03993|0.23693|1.74%|
|residual|2|1200|3.0|2511/553|0.04487|0.07960|0.27992|0.66%|
|direct|0|1200|1.0|8398/618|0.02215|0.04598|0.26807|4.31%|
|direct|0|1200|3.0|2511/553|0.04791|0.09103|0.43770|14.64%|
|direct|1|1200|1.0|8398/618|0.02143|0.04607|0.27039|2.29%|
|direct|1|1200|3.0|2511/553|0.04754|0.09330|0.49353|6.32%|
|direct|2|1150|1.0|8398/618|0.02246|0.04751|0.24828|4.92%|
|direct|2|1150|3.0|2511/553|0.05024|0.09090|0.48135|5.80%|

最佳按固定DEV1秒均值在每25次存档中选择，3秒不单独选模；不是所有更新全程最优。重叠窗口非独立样本，各时域覆盖不同；逐例最坏值保存在cases.csv。损失是监督预测损失，不是PPO，没有策略reward/KL指标。学习模型读取真实381维历史作初始化，未来只用预测状态及自身隐状态查询固定地形，不读未来真实历史。

地形描述是预测位置附近15点的高度、坡度、局部高度变化及支撑mask；沟与图外未标为平地支撑。双线性栅格只是模型输入，最终碰撞几何不替换。地图划分、初态及速度以本轮manifest为准；开发预测比较尚无独立TEST或规划收益证明。没有用平均误差宣称全程净空合格、训练正式收敛或完整P3已完成。

原始存档/标量/逐窗口：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p5_alpha_models/phase2_continue600`；M0、M1、M2均未修改RB/H75或物理。

本轮M0拟合110892条有效TRAIN转移；来源明细与训练预算见manifest.json。所有模型保留每25轮存档。

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
- [1s 第10项](prediction_xy_50_page10.png)
- [1s 第11项](prediction_xy_50_page11.png)
- [1s 第12项](prediction_xy_50_page12.png)
- [1s 第13项](prediction_xy_50_page13.png)
- [1s 第14项](prediction_xy_50_page14.png)
- [1s 第15项](prediction_xy_50_page15.png)
- [1s 第16项](prediction_xy_50_page16.png)
- [1s 第17项](prediction_xy_50_page17.png)
- [1s 第18项](prediction_xy_50_page18.png)
- [1s 第19项](prediction_xy_50_page19.png)
- [1s 第20项](prediction_xy_50_page20.png)
- [1s 第21项](prediction_xy_50_page21.png)
- [1s 第22项](prediction_xy_50_page22.png)
- [1s 第23项](prediction_xy_50_page23.png)
- [1s 第24项](prediction_xy_50_page24.png)
- [1s 第25项](prediction_xy_50_page25.png)
- [1s 第26项](prediction_xy_50_page26.png)
- [3s 第27项](prediction_xy_150.png)
- [3s 第28项](prediction_xy_150_page02.png)
- [3s 第29项](prediction_xy_150_page03.png)
- [3s 第30项](prediction_xy_150_page04.png)
- [3s 第31项](prediction_xy_150_page05.png)
- [3s 第32项](prediction_xy_150_page06.png)
- [3s 第33项](prediction_xy_150_page07.png)
- [3s 第34项](prediction_xy_150_page08.png)
- [3s 第35项](prediction_xy_150_page09.png)
- [3s 第36项](prediction_xy_150_page10.png)
- [3s 第37项](prediction_xy_150_page11.png)
- [3s 第38项](prediction_xy_150_page12.png)
- [3s 第39项](prediction_xy_150_page13.png)
- [3s 第40项](prediction_xy_150_page14.png)
- [3s 第41项](prediction_xy_150_page15.png)
- [3s 第42项](prediction_xy_150_page16.png)
- [3s 第43项](prediction_xy_150_page17.png)
- [3s 第44项](prediction_xy_150_page18.png)
- [3s 第45项](prediction_xy_150_page19.png)
- [3s 第46项](prediction_xy_150_page20.png)
- [3s 第47项](prediction_xy_150_page21.png)
- [3s 第48项](prediction_xy_150_page22.png)
- [3s 第49项](prediction_xy_150_page23.png)
- [3s 第50项](prediction_xy_150_page24.png)
- [3s 第51项](prediction_xy_150_page25.png)
- [3s 第52项](prediction_xy_150_page26.png)
