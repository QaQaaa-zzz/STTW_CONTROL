# P3 supervised full-joint model: true-entry DEV assessment

[Checkpoint curves](learning_curves.png). Full data and model source: `/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p3_world_long`. No policy updates.

- [All36 map06, seed0 XY](map06_seed0_xy.png)
- [All36 map07, seed0 XY](map07_seed0_xy.png)
- [All36 map06, seed1 XY](map06_seed1_xy.png)
- [All36 map07, seed1 XY](map07_seed1_xy.png)
- [All36 map06, seed2 XY](map06_seed2_xy.png)
- [All36 map07, seed2 XY](map07_seed2_xy.png)

|Seed|Checkpoint|Mean full-segment max XYZ (m)|Worst XYZ (m)|Missing handoff /72|
|---|---|---|---|---|
|0|1100|0.1342|0.2063|72/72|
|1|1200|0.1228|0.1917|0/72|
|2|1150|0.0938|0.1570|18/72|

All72 DEV sequences per seed retained. Same-domain generated terrain layouts; not novel terrain types or independent TEST. Full-record duration is used only to score prediction, not as model input. Controller/RB histories are learned estimates after initialization. This report does not establish planning utility or P3 completion.

事件错误：seed0漏报全部72次交回；seed1预测72次但平均提前0.964秒；seed2预测54次且平均提前0.638秒。事件时序不合格，不能进入P4。1秒窗口误差不能替代整段预测。已冻结下一阶段完整入口序列训练合同，结果尚未回收。


后续完整入口训练及预测事件停止评测见 [完整入口结果](full_entry/README.md)。旧窗口训练失败不覆盖；新数据集/物理/网络结构不变，改变训练窗口采样与DEV选存档范围。


[损失尺度及交回损失修正结果](handoff_loss/README.md)：种子0原始几何4/4可行准入，但旧冻结裕量仍0/4，短接触事件抽样缺口尚待修正。P3未完整验收。


## P3单段阶段收束与P4入口

[短接触事件及固定时序负对照](events/README.md)。5ms原始事件转换为累计20ms标签；选定事件读出seed1/update600无漏报，但同20ms网格的TRAIN中位时刻基线更好，不宣称事件学习优势。真实GPU检查点100=40+序列化+60逐位一致，附加事件读出不改变物理预测。

P3范围内已具备：既有phase0/phase2地面模型；全关节RB分支三种子训练、存档与DEV选择；单步/多步/整段自由预测；姿态/速度/全关节/接触/交接/占据误差；真实入口诊断及后段准入误差；完整失败及负对照。按已知DEV决策选择RB seed0/update2275，事件读出seed1/update600；这是有限域候选，不是可靠部署认证或独立TEST泛化。

下一阶段P4必须补齐：地面模型记忆到可传递控制历史的接口，前段预测状态进入RB，再进入后段；G/I/C/Oracle及同候选强回溯；全链重新标定与共同裕量诊断。正式C分支只能起点读取真实历史，后续不能用实测未来。旧M1裕量下无计划的负结果不删除，禁止通过统一缩小裕量掩盖问题。
