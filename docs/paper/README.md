# Task-specified preference control article

The editable article source is [MANUSCRIPT.tex](MANUSCRIPT.tex), with its [PDF](MANUSCRIPT.pdf) compiled using Elsevier's `elsarticle` class in a provisional two-column RAS-like layout. The omitted simulation tables and explanations are in editable [SUPPLEMENT.tex](SUPPLEMENT.tex) and [SUPPLEMENT.pdf](SUPPLEMENT.pdf). The user-supplied `RAS车.pdf` guided the article structure and presentation; no publisher masthead, acceptance date, or journal endorsement is reproduced.

Generate the article figures from frozen local simulation artifacts, then compile:

```bash
PYTHONPATH=learning/src python docs/paper/analysis/make_article_figures.py
python docs/paper/build.py --pdf --engine tectonic
```

The build regenerates the supplement from frozen JSON, checks figure files, and records source hashes in [evidence_manifest.json](evidence_manifest.json) and [export_receipt.json](export_receipt.json). The five article figures and their source map are in [figures](figures/article_figure_sources.json). The [full task result package](../results/sttw-tolerance-20261101/task_preference_holdout/HOLDOUT.md) retains every route, prepared bank, alpha and frozen middle policy. The [matched geometric-rule comparison](../results/sttw-tolerance-20261101/traditional_rule/COMPARISON.md) contains its development selection and all eight new-combination results. The [signed-pulse extension](../results/sttw-tolerance-20261101/task_preference_disturbance/COMPARISON.md) retains all 48 matched case scores and 32 actual/reference XY panels. The [paired physical-domain task comparison](../results/sttw-tolerance-20261101/task_preference_physical_domain/COMPARISON.md) supplies 48 cases and 32 XY panels. The [90-degree left/right turn panel](../results/sttw-tolerance-20261101/task_right_angle/COMPARISON.md) supplies 48 cases and 16 XY panels. The supplement includes route-level tables for all four panels, individual outcomes for the positive disturbance subset, additional right-turn and force-sign figures, training diagnostics, and links to the full per-case data. The earlier Markdown stage report is in [history](history/20261101_task_preference_report_snapshot/MANUSCRIPT.md).

This is a scientific working draft. Model and score selection used development data; new route–speed–bank combinations were frozen before their evaluation. The geometric rule was constructed after the learned-policy new-combination results had been inspected, then selected on the earlier development routes. The current article reports simulation evidence. Author details, declarations, and human scientific approval remain pending.
