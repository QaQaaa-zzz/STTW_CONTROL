# STTW论文：偏好条件参考调节与越野恢复

第一章研究一个问题：底层平衡控制器 ECBC＋ESO＋Local300 冻结时，同一个中层如何随 α 联合调节名义速度和转向，进而改变入弯、入障状态及恢复结果？英文稿已写入同栈弯道、解析限速器和首个独立新训模型的六工况结果。[英文稿 PDF](english/PCRP.pdf) · [LaTeX 源文件](english/PCRP.tex)

| 论文问题 | 已有证据与表述范围 |
|---|---|
| α 改变了什么？ | 将 α 从偏路径调到偏速度，比较同初态下的实际与参考 XY、速度和曲率。完整端点目标的单调关系要求精确求解加权目标。对学得的 H75，20 个条件的 80 对相邻档位中，41 对呈现速度误差降低、路径误差增加。 |
| 为什么同一单脊换位置会换一个能通过的档位？ | 固定 2.6 m/s、10 cm 单脊和 H75，脊中心从 3.9 m 移至 4.5 m，低/高端点的十秒完成数从 4/4、0/4 变为 0/4、4/4。沿实际轨迹对齐入障状态和前轮越峰后首次复载，8 条失败端点在该段均无后轮承重，8 条完成端点均有承重重叠。名义侧倾包络解释可用转向余量，整车角动量预算核对接触冲量；跟踪域退出与侧倾越限分别计数。 |
| 据此能选到什么程度？ | 冻结的回合前规则在新布局的任务 C 通过 75/176，最佳固定档 70/176，配对净增益的描述性 95% 区间为 −3.41 至 +8.52 个百分点。任务 A、B 的固定档通过数更高。在线状态选档列入展望。 |

### 剩余实验各回答什么

1. **独立初始化。** 7101、7102、7103 按同一套 Isaac/Go2 协议和交互预算分别训练。各自的最终模型评测同一六工况，逐物理组、逐 α 报告十秒完成、末秒保持、A/B/C 和共同时窗误差，三组到齐再汇总种子间差异。历史 H75 单列为冻结对照。
2. **共享一个网络的代价。** 把同一初始化种子 7101 的共享模型与 α＝0、0.5、1 三个专用网络逐档配对，每个网络获得相同训练预算。每个专用档只有一个训练种子。若两者都未通过，保留失败分母，用连续误差描述差异，零通过不能证明两者等效。
3. **如何解释新训负结果。** 7101 已做完六工况，31/200 跑满、0/200 保持；三个脊位共 0/120 跑满。同条件 H75 为 118/200 跑满、107/200 保持。新旧协议同时改变了训练环境、初态和优化流程，这组数据限定了 H75 现象的复现范围。按已登记的六个模型和预算收尾；先回收正在运行的 7102，在同一评测矩阵下读数，再推进剩余配对。

[逐工况轨迹与解析基线](../evidence/pcrp_matched_controls/README.md) · [新训协议与预算](../evidence/pcrp_matched_controls/TRAIN.md) · [7101 与 H75 逐档结果](../evidence/pcrp_matched_controls/training/shared7101/README.md)

当前完整稿：[PDF](MANUSCRIPT.pdf) · [可编辑中文正文及英文摘要](MANUSCRIPT.md) · [HTML](MANUSCRIPT.html)。当前PDF由Markdown生成；MANUSCRIPT.tex及method_training.tex保留为历史平地英文材料，不是本版编辑源。

中文稿包含两项命题、16个公式、4张表和7张图。第一张图为实际／原参考XY。

S0–S4实验已完成后再整合本稿。S1支持协调方向，精确标量化最优假设未被采用；S2补齐接触角动量与低α失稳机制；S3共2464条细档验证；S4共2112条新布局对照。C选档75/176、固定最佳70/176、简单规则66/176，A/B仍由固定档取得更高通过数。正文同时给出布局分组范围与配对得失。

[逐项实验收尾](../evidence/h75_theory_iteration/CLOSURE.md) · [S4全部XY](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_theory_iteration/layout_test/RESULTS.md) · [S4精简结果](../evidence/h75_theory_iteration/S4_layout/results/README.md)。正文E1–E9与evidence_manifest.json提供来源。

## 构建

使用含Markdown、Matplotlib、WeasyPrint和Noto CJK字体的Python环境：

```bash
python docs/paper/build.py --pdf
```

`requirements.txt`记录本次使用的Python包版本。物理公式核对：

```bash
PYTHONPATH=learning/src python -m unittest discover -s learning/tests -p test_paper_preference_theory.py
PYTHONPATH=learning/src python -m unittest discover -s learning/tests -p test_margin_theory.py
```

本轮14项通过；排版和文章整合没有新增仿真或训练。PDF公式、结果页已检查，表4与机器汇总逐项一致。作者、单位、期刊、基金及最终作者审阅由作者补充；这是完整研究稿，尚未投稿。
