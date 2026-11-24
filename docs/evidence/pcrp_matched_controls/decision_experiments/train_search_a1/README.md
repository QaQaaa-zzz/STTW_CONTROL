# PCRP A1 四工况窗口搜索

|物理组 401|实际与参考 XY|观测误差前沿|批评估最优 V|同原点单槽 V|−1 / +1 cm 跑满 10 s|−1 / +1 cm 通过任务 C|
|---|---|---|---:|---:|---|---|
|平地|[XY](analysis/flat_original_v2.6_seed401/xy.png)|[前沿](analysis/flat_original_v2.6_seed401/frontier.png)|0.4660|0.4640|—|—|
|4.2 m 脊|[XY](analysis/position_s4.2_v2.6_seed401/xy.png)|[前沿](analysis/position_s4.2_v2.6_seed401/frontier.png)|0.7449|0.7649|1 / 0|0 / 0|
|4.5 m 脊|[XY](analysis/position_s4.5_v2.6_seed401/xy.png)|[前沿](analysis/position_s4.5_v2.6_seed401/frontier.png)|0.8353|17.3162|0 / 0|0 / 0|
|3.9 m 脊|[XY](analysis/position_s3.9_v2.6_seed401/xy.png)|[前沿](analysis/position_s3.9_v2.6_seed401/frontier.png)|0.8947|0.9126|1 / 0|1 / 0|

8 代在每个工况并行 32 条候选。共 1,024 条搜索轨迹、128 条零偏置及 10 条验证。平地纯搜索达到 V ≤ 0.676。4.5 m 在单槽回放第 2.6838 秒接触失败。4 个赢家都出现逐位差异，冻结判决为停止扩表。负偏移和正偏移各有 3 个脊位。对应的六条回放共有 2 条跑满、1 条通过任务 C。前沿由开发工况上实际测得的点构成。

[逐条轨迹索引](analysis/episodes.csv) · [实测决策](decision.json) · [单槽逐数组比较](single_comparisons.json) · [偏移复验](shift_results.json) · [赢家参数](winners.json)

原始轨迹与各工况完整图在运行主机的 `/media/qy/6A7E3DB07E3D75C3/STTW_runs/pcrp_decision_20261122/train_search_a1`。逐条索引记录每次评估的原始目录与物理槽位。实验输入记录在 [registration.json](registration.json)。
