# Preference-conditioned single-track control article

[MANUSCRIPT.tex](MANUSCRIPT.tex) is the editable RAS-style main article and [MANUSCRIPT.pdf](MANUSCRIPT.pdf) is its compiled PDF. The paper has one integrated main text and no separately delivered supplement. The user-provided `RAS车.pdf` informed the problem presentation and figure style; its publisher identity was not copied.

The manuscript uses the frozen H75 policy as its main learned controller. U170 is a second trained middle-policy comparator. The geometric comparator follows the original path by pure pursuit with exactly zero learned middle correction; all three use the same frozen lower controller. The new route matrix covers left/right turns, left/right-leading S turns, exact unrounded left/right 90-degree polylines, low/medium/high requested speeds, two complete prepared initial states, and nominal/+36 N world-Y pulse conditions. Every destination lies 7 m beyond the last bend or vertex. The 504 declared physical trajectories and all failures are retained on the data volume under `task_matrix_v2_20261104`; the [result index](../results/sttw-task-matrix-20261104/README.md) gives exact paths and identities.

The five-alpha H75 nominal comparison and its development-selected curvature-speed pure-pursuit rule remain distinct from the new three-alpha matched matrix. Neither panel is an independent unseen-policy test. The [training methods](method_training.tex) include the middle-policy reward and PPO objective; H75 and U170 use the same `path_reward` and reference interface, while U170 continues H75 on an additional mixed low-speed training distribution. The nine resolved route-matrix configurations are in `learning/configs/task_matrix_v2_*.json` at the repository root.

Regenerate the matrix analysis and selected article figures from the saved raw run directories, then compile with:

```bash
PYTHONPATH=learning/src python docs/paper/analysis/task_matrix_v2.py --policy H75 --runs LOW_H75 LOW_PP MID_H75 MID_PP HIGH_H75 HIGH_PP --out ANALYSIS_H75
PYTHONPATH=learning/src python docs/paper/analysis/task_matrix_v2.py --policy U170 --runs LOW_U170 LOW_PP MID_U170 MID_PP HIGH_U170 HIGH_PP --out ANALYSIS_U170
python docs/paper/analysis/make_matrix_figures.py --analysis ANALYSIS_H75 --out docs/paper/figures
python docs/paper/analysis/make_matrix_table.py --h75 ANALYSIS_H75 --u170 ANALYSIS_U170 --out docs/paper/matrix_summary.tex
python docs/paper/build.py --pdf --engine tectonic
```

The data role is exploratory matched simulation. The current manuscript contains neither a trained tolerance predictor nor an online preference coordinator, and it makes no hardware claim. Author information and final human scientific approval remain to be supplied.
