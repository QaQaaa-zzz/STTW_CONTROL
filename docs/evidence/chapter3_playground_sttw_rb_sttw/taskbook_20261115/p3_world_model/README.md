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
