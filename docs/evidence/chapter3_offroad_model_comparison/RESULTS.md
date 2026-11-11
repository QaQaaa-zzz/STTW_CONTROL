# 事件滚动与直接成败预测的冻结对照

[采集实际／参考XY](../chapter3_offroad_event_training/RESULTS.md) · [测试实际／参考XY](../chapter3_offroad_new_source/RESULTS.md)

相同状态/历史编码、高程、动作候选与整根划分，两模型600次更新、每100次仅用DEV选模。事件模型按32粒子滚动预测完整任务，直接模型预测整程成败；此表不把单事件条件预测当整程预测。TEST为既有4个未拟合的来车初态，曾用于技能比较，不作为完全未查看的确认性测试。

|分组|模型|条数/独立初态|Brier|平均预测|实际完成率|
|---|---|---:|---:|---:|---:|
|TRAIN|event|480/8|0.0985|0.1198|0.1625|
|TRAIN|direct|480/8|0.1061|0.0970|0.1625|
|TRAIN|constant|480/8|0.1361|0.1625|0.1625|
|DEV|event|240/4|0.1165|0.1208|0.1833|
|DEV|direct|240/4|0.1300|0.1013|0.1833|
|DEV|constant|240/4|0.1502|0.1625|0.1833|
|TEST|event|240/4|0.0885|0.1173|0.1417|
|TEST|direct|240/4|0.0975|0.0933|0.1417|
|TEST|constant|240/4|0.1220|0.1625|0.1417|

|分组|初态|事件候选/结果|直接候选/结果|TRAIN固定候选/结果|事后可完成|
|---|---:|---|---|---|---|
|TRAIN|0|51/True|2/True|48/True|True|
|TRAIN|1|48/True|2/True|48/True|True|
|TRAIN|2|48/True|2/True|48/True|True|
|TRAIN|3|48/True|2/False|48/True|True|
|TRAIN|4|48/True|2/True|48/True|True|
|TRAIN|5|48/True|2/False|48/True|True|
|TRAIN|6|48/True|2/False|48/True|True|
|TRAIN|7|48/True|2/True|48/True|True|
|DEV|8|48/True|2/False|48/True|True|
|DEV|9|48/True|2/False|48/True|True|
|DEV|10|48/True|2/True|48/True|True|
|DEV|11|48/True|2/True|48/True|True|
|TEST|0|48/True|2/False|48/True|True|
|TEST|1|48/True|2/False|48/True|True|
|TEST|2|48/True|2/True|48/True|True|
|TEST|3|51/False|2/True|48/True|True|

候选编号与原协议一致。按完整任务概率选最大值，不用TEST标签选阈值、checkpoint或固定档。以上选档只查已保存结果，候选入口存在克隆物理差异，不能称真实在线增益；应另跑共同前缀的在线执行。

测试离线选档：事件3/4、直接2/4、TRAIN固定绕行4/4。事件模型整程Brier更好，但没有超过固定档；不能把校准改善直接当规划收益。随后单独声明新初态在线验证，冻结两模型和固定候选48，不调整此测试集。
