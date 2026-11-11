# STTW论文：偏好条件参考调节与越野恢复

当前完整稿：[PDF](MANUSCRIPT.pdf) · [可编辑中文正文及英文摘要](MANUSCRIPT.md) · [HTML](MANUSCRIPT.html)。当前PDF由Markdown生成；MANUSCRIPT.tex及method_training.tex保留为历史平地英文材料，不是本版编辑源。

主线：α改变速度—转向协调；接触状态决定转向恢复作用，侧倾与跟踪域分别限制任务；据此检验任务相关选档。正文包含两项命题、16个公式、4张表和7张图。第一张图为实际／原参考XY。

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
