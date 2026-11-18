# P5 条件跳跃预测数据边界

648TRAIN与648已知DEV，共1296真实序列；0遗漏、0新物理、尚未拟合跳跃模型。原提取109100采样保留在原始目录，新handoff_only为97368采样；每条保留到第一个20ms网格上的phase2，恰有一个phase2样本。它可能比native交回晚不足20ms，不冒称精确事件时刻。

去掉交回后0.2秒驶离尾段，避免只接收跳前α的跳跃模型被要求拟合另一阶段α控制的后续运动。该截断只改变本轮预测数据边界，不改实际控制、频率、任务或旧训练集。所有qpos/qvel/537维history/contact/phase/context/time/action/action_valid与原提取对应前缀逐值一致，详见[prefix_audit.json](prefix_audit.json)。旧入口默认保持0.2秒；本轮显式使用`--stop-at-handoff`。

原始数据：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/taskbook_20261115_p5_alpha_jump_data/handoff_only`。TRAIN地图族/快照组和DEV无交叉，归一化仅TRAIN。当前shared225依然不是完整RB方法对照；源策略/完整增强策略须各自匹配数据与预测器。
