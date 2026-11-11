# 五档状态相关选档验证

12工况 × (25自然切换＋Zero＋2重复)=336条；每条10s、0训练。冻结协议见protocol.json。

先保留全部XY、每步/累计奖励、失败原因和轨迹；随后留一物理组检查状态选档。三个物理组均为已用DEV，不称新布局泛化。只在共同前段与切换时刻核对成立的组上作同入口选档分析；缺失或前段失败保留并单列。

路径：RESULTS.md（全覆盖XY）、ANALYSIS.md（完整任务与终止）、state_selection.md（选档诊断）、status.json。此处是执行计划，不是实验结果。

原始输出目录：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_state_alpha_scan`；队列状态：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_state_alpha_queue/status.json`。准备入口`learning/cli/prepare_state_alpha_scan.py`，分析入口`learning/cli/analyze_state_alpha_scan.py`。
