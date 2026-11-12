# H75 精简研究证据包

[下载完整压缩包](h75_research_evidence.zip) · [大小、哈希与校验回执](h75_research_evidence.json) · [实验结论](../h75_theory_iteration/CLOSURE.md) · [论文PDF](../../paper/MANUSCRIPT.pdf)

覆盖当前论文E1–E9，以及S0–S4主线。旧失败、负结果、重复车道和未采用结论保留在相应数字表与报告中。

|内容|压缩包入口|
|---|---|
|正文、16个公式、两项证明、关键图|docs/paper/MANUSCRIPT.md、MANUSCRIPT.pdf、figures/|
|公式详细推导及数值测试|docs/evidence/h75_theory_iteration/S0_DERIVATION.md；learning/tests/test_paper_preference_theory.py、test_margin_theory.py|
|完整目标、η来源、偏好区间及弯道分配|docs/evidence/h75_theory_iteration/S1/；external/theory/S1/|
|两轮接触、角动量、恢复包络及跟踪域|docs/evidence/h75_theory_iteration/S2/；external/theory/S2/|
|S3全部2464回合与S4全部2112回合指标|external/theory/{mechanism_scan,layout_test}/analysis_summary.json|
|S3预测、S4冻结选档及逐条件得失|docs/evidence/h75_theory_iteration/{S3_mechanism,S4_layout}/|
|288条高度标定指标与配置|external/theory/calibration/|
|历史平地、通道消融、自然切换及H边界|external/E1/及docs/evidence/各h75专题|
|复算代码、源控制器、奖励、配置与测试|learning/{src,cli,configs,tests,scripts}/|
|文件来源、模型身份、路径与完整性|MANIFEST.json；各回合contract.json|

## 精简原则

保留所有数值表、公式推导和分析数组；报告逐工况结局，不只保留成功案例。正文保留关键图，重复PNG/PDF及运行日志省略，清单记录省略项。原始逐步仿真NPZ、高频接触、视频及模型权重仍留在原数据卷，合同保留模型哈希和路径；它们未装入本压缩包。这里交付的是研究证据与分析代码，不是包含Isaac/CUDA环境的完整重跑镜像。

解压后先读docs/evidence/h75_theory_iteration/CLOSURE.md。S4任务A/B/C机制选档64/64/75，各分母176；最佳固定68/71/70，简单规则51/71/66。新布局收益集中于单脊任务C，未采用全面优胜结论。所有当前和历史数字分协议报告。

## 复核

MANIFEST.json为每个文件提供原路径、字节数和SHA256；打包时已逐文件回读校验。公式测试可在具备项目依赖的Python环境运行：

```bash
PYTHONPATH=learning/src python -m unittest discover -s learning/tests -p test_paper_preference_theory.py
PYTHONPATH=learning/src python -m unittest discover -s learning/tests -p test_margin_theory.py
```

原始运行目录：`/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_theory_iteration/`。分析脚本中原始绝对路径须在另一台机器上按MANIFEST来源映射；未将本机路径假装成可迁移SDK环境。

打包入口：`python learning/cli/package_h75_research.py --output docs/evidence/h75_research_package/h75_research_evidence.zip`。

## 指定八工况原始轨迹补充包

10 cm单脊、2.6 m/s，layout 0/1 × seed 401–404，共八个工况。每个目录中的原始 `trajectory.npz` 完整保留，没有裁剪字段、车道或失败回合；解压后保留原工况目录名。配置不重复打包。

- [layout 0：四工况原始轨迹](ridge100_v2.6_layout0_trajectories.zip)
- [layout 1：四工况原始轨迹](ridge100_v2.6_layout1_trajectories.zip)
- [八文件清单、大小、SHA256及校验结果](ridge100_v2.6_trajectories.json)

来源：`layout_test/ridge100_v2.6_layout{0,1}_seed{401,402,403,404}/trajectory.npz`。已核对内外压缩文件CRC及八个解压文件与原文件的SHA256一致。

## 八工况视频

[视频目录与下载](videos/README.md)：layout0/1×seed401–404，每工况五档alpha及Zero同步六宫格，10秒、1920×940、20fps，八段合计7.71MB。包含真实单脊网格、原始车辆STL、速度/侧倾与真实终止标签。全部视频已逐帧解码核验200帧与10秒时长，首工况越脊及终止画面已人工检查。
