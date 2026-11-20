# 第三章 · 跳台前的接近选择与落地避障

[论文 PDF](MANUSCRIPT.pdf) · [可编辑 LaTeX](MANUSCRIPT.tex) · [盲测报告与逐例数据](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/native_platform/approach_alignment/replication/review_114/final_stages/RESULTS.md) · [真跳通关回放](../evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/native_platform/approach_alignment/replication/review_114/final_stages/verified_test_jump.mp4)

论文连接第一章的锁摆 STTW 地面控制与第二章的 Source / Full RB 跳台控制。在同一高台上改变必经门，测量哪种接近还能完成后续任务，并检验落地时重选驶离带来的通关与碰撞变化。现在的主稿采用 +0.16 m 平台，以两张新图上的冻结盲测作为主证据。旧空沟和滚动规划稿归入历史版本。

|主图|依据|
|---|---|
|[1　真跳、包络与左右任务反转](figures/platform_reversal.png)|TEST A，Source 根 1002，左右任务均真实通关|
|[2　四张布局上的能力分布](figures/platform_layouts.png)|DEV 两图各 114 根，TEST 两图各 40 根|
|[3　交回状态与落地避障](figures/platform_state.png)|DEV 根 96 和 110，同接近同驶离，转角差 7.82°|
|[4　完成与撞墙的取舍](figures/platform_tradeoff.png)|左图同 DEV 物理库，右图为未见布局三档扰动及区间|
|[5　交回改道的两例配对回放](figures/platform_rescues.png)|两组独立 TEST 合同，C 撞墙、C-R 通关|

另附[原获准撞墙四例在两种时钟下的实际轨迹](figures/platform_approved_walls.png)、实速匹配表和逐轨迹去重表。四张原主表分别给出严格反转及上界、四规划器整程结果、准入与否决、分扰动配对技能差。全部计数来源及文件哈希见 [source_manifest.json](source_manifest.json)。90/85 ms 配对复验已纳入正文，R9 C-R 通关从 701 增至 757/1440，撞墙从 90 降至 18/1440。8 页主稿已编译，含 17 篇外部文献和两条章节自引占位，作者审核待完成。

从工作树根目录运行以下命令可从归档轨迹生成六幅图、四张自动生成的表、视频及 PDF。仅重新排版时使用 `python docs/chapter3_paper/build.py`。

```bash
wide=/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/platform_test0_20261120/approach_alignment/replication_wide
PYTHONPATH=/home/qy/DVGC/JIT/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python docs/chapter3_paper/build.py --platform-run "$wide"
```

图表与证据追溯工作使用本机 scientific-writing 和 scientific-visualization 工作流程。[Kassis 等，Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)
