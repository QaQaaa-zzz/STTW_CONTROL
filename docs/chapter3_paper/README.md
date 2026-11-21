# 第三章　跨技能规划如何影响跳台后的任务

[八页英文稿](MANUSCRIPT.pdf) · [LaTeX 源码](MANUSCRIPT.tex) · [同一起点在三种墙上的实际与参考 XY](figures/layout_detail.png) · [跳台与锁摆实录](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/native_platform/approach_alignment/replication/review_114/final_stages/realtime_15ms/verified_jump.mp4)

英文稿作为独立系统论文撰写。学位论文第三章可按以下逻辑接续前两章。

|学位论文衔接|第三章的操作与证据|对应英文稿|
|---|---|---|
|第一章给出锁摆地面控制及速度与路径偏好 α|地面控制器沿不同曲线进入同一跳台。先验从 30 个开发初态估计。在已公开两图左门任务上，将左弯的 α=0.5、1 加入候选后，R9 通关由 201 增至 405/720，撞墙从 6 增至 18。|V、VI-E|
|第二章给出 Source 与恢复策略|使用有明确权重身份的 R9。原生平台稳定恢复 31/32。第三章另行要求可验证腾空、上台承重、重新锁摆及限时到达指定门，逐步接触记录用于复核。|III、V、VII|
|第三章检验地面接近对后续任务的影响|同起点换左右门，先穷举三接近×三驶离，判定是否存在严格反转，再与固定规则和在线规划比较。冻结盲测中 R9 严格反转至少一档为 32/80 个地图–初态对。|III、VI-A|
|第三章检验预测与落地重选|用真实墙体与整车预测包络计算净空。15 ms 下，R9 从 C 到 C-R 少撞墙 69/1440，也少通关 18/1440；原路线实际能完成却被改道的 28 个样本均触发净空裕量。|IV、VI|
|跨布局检验|新取 30 个初态。左墙从固定规则的 81 次撞墙降至零；双墙上 C-R 少通关 64/540，撞墙多 8 次。|VI-D|

[15 ms 全方法表](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/native_platform/approach_alignment/replication/review_114/final_stages/safety_trigger/main_table.csv) · [三种墙的逐布局区间](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/native_platform/approach_alignment/replication/review_114/final_stages/wall_layouts/layout_metrics.csv) · [α 消融逐图结果](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/native_platform/approach_alignment/replication/review_114/final_stages/alpha_library/paired_metrics.csv) · [来源清单](source_manifest.json)

在工作树根目录运行 `python3 docs/chapter3_paper/build.py` 可从已提交的图表编译 PDF。方法与数据角色已按实验归档核对。文献 [18]、[19] 按本地未发表手稿著录；署名、单位及投稿审批由作者确认。

[新增零仿真归因、同批 C/Fixed 与 18 个撞墙合同](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/native_platform/approach_alignment/replication/review_114/final_stages/alpha_library/RESULTS_BREAKDOWN.md)。文献 [11] 已对照 arXiv 原始页面核验题名、作者顺序和日期，编号 2611.15280，2026 年 11 月 11 日。文献 [18]、[19] 沿用已核实的未发表稿题名，作者顺序待作者确认。
