"""Render the paper's matched XY and descriptive task figures from frozen runs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RESULTS = ROOT / "docs/results/sttw-tolerance-20261101/task_preference_holdout/holdout_comparison.json"
RULE_RESULTS = ROOT / "docs/results/sttw-tolerance-20261101/traditional_rule/comparison.json"
PULSE_RESULTS = ROOT / "docs/results/sttw-tolerance-20261101/task_preference_disturbance/comparison.json"
RIGHT_ANGLE_RESULTS = ROOT / "docs/results/sttw-tolerance-20261101/task_right_angle/comparison.json"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
data = json.loads(RESULTS.read_text())
rule_data = json.loads(RULE_RESULTS.read_text())
pulse_data = json.loads(PULSE_RESULTS.read_text())
angle_data = json.loads(RIGHT_ANGLE_RESULTS.read_text())
alphas = [0, .25, .5, .75, 1]
colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00"]
markers = ["o", "s", "^", "D", "P"]

with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42}):
    fig, ax = plt.subplots(figsize=(7.2, 2.25), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    blocks = [
        (0.1, 1.85, 1.6, .62, "Task: $P_0,v_0,g_0,\\alpha$", "#E7F2F8"),
        (2.15, 1.85, 1.55, .62, "U170 middle\n50 Hz", "#D9EEDB"),
        (4.1, 1.85, 1.55, .62, "Bounded reference\nfilter", "#F8EAD2"),
        (6.05, 1.85, 1.55, .62, "ECBC + ESO + U200\n200 Hz", "#E7F2F8"),
        (8.0, 1.85, 1.55, .62, "Single-track\nplant", "#ECE7F6"),
        (2.15, .45, 2.2, .62, "Causal path reconnection\n10 Hz", "#F8EAD2"),
        (6.4, .45, 2.4, .62, "Score on original $P_0,g_0$\nplus actual speed/contact", "#F5E4E4"),
    ]
    for x, y, w, h, label, color in blocks:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=color, ec="#374151", lw=.7))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=7.8)
    for start, end in [((1.72, 2.16), (2.12, 2.16)), ((3.72, 2.16), (4.07, 2.16)),
                       ((5.67, 2.16), (6.02, 2.16)), ((7.62, 2.16), (7.97, 2.16)),
                       ((8.76, 1.82), (8.76, 1.35)),
                       ((2.65, 1.82), (2.65, 1.10)), ((3.28, 1.35), (3.28, 1.82))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, lw=.85, color="#374151"))
    ax.plot([8.76, 3.28], [1.35, 1.35], color="#374151", lw=.85)
    ax.text(5.9, 1.40, "measured state", ha="center", fontsize=7)
    ax.add_patch(FancyArrowPatch((2.02, 2.0), (1.68, .78), arrowstyle="-|>", mutation_scale=9, lw=.85, color="#374151"))
    ax.add_patch(FancyArrowPatch((1.68, .78), (2.11, .78), arrowstyle="-|>", mutation_scale=9, lw=.85, color="#374151"))
    ax.add_patch(FancyArrowPatch((4.37, .78), (4.8, 1.82), arrowstyle="-|>", mutation_scale=9, lw=.85, color="#374151"))
    ax.add_patch(FancyArrowPatch((1.35, 1.82), (6.38, .78), arrowstyle="-|>", mutation_scale=9, lw=.8, color="#777777", linestyle="--"))
    ax.add_patch(FancyArrowPatch((9.15, 1.82), (8.25, 1.10), arrowstyle="-|>", mutation_scale=9, lw=.85, color="#374151"))
    fig.savefig(OUT / "task_preference_architecture.pdf")
    fig.savefig(OUT / "task_preference_architecture.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), layout="constrained")
    for ax, model, style in [(axes[0], "H75", "--"), (axes[0], "U170", "-")]:
        rows = sorted((r for r in data["summary"] if r["model"] == model), key=lambda r: r["alpha"])
        ax.plot([r["mean_pre_goal_max_cross_m"] for r in rows],
                [r["mean_turn_speed_m_s"] for r in rows], style, color="#333333", lw=.9, label=model)
        for row, color, marker in zip(rows, colors, markers):
            ax.scatter(row["mean_pre_goal_max_cross_m"], row["mean_turn_speed_m_s"],
                       color=color, marker=marker, s=27, edgecolor="#222222", linewidth=.25)
    axes[0].set(xlabel="Mean maximum original-path error (m)", ylabel="Mean turn speed (m/s)")
    axes[0].legend(frameon=False, loc="lower right")
    for model, style in [("H75", "--"), ("U170", "-")]:
        rows = sorted((r for r in data["summary"] if r["model"] == model), key=lambda r: r["alpha"])
        for task, color in [("narrow", "#0072B2"), ("wide", "#D55E00")]:
            axes[1].plot(alphas, [r[task] for r in rows], style, color=color,
                         marker="o" if model == "U170" else "s", lw=1.2,
                         label=f"{model} {task}")
    axes[1].set(xlabel="Fixed preference $\\alpha$", ylabel="Task successes / 8", xticks=alphas, ylim=(-.3, 8.5))
    axes[1].legend(frameon=False, ncol=2, loc="lower center")
    for ax, label in zip(axes, ("(a)", "(b)")):
        ax.text(.01, .99, label, transform=ax.transAxes, va="top", fontweight="bold")
        ax.grid(color="#dddddd", lw=.4)
    fig.savefig(OUT / "preference_task_tradeoff.pdf")
    fig.savefig(OUT / "preference_task_tradeoff.png", dpi=300)
    plt.close(fig)

    panels = [
        ("holdout_low_S_right_72_R1p55_v2p00", 5, "Low-speed right S turn"),
        ("holdout_high_S_left_70_R3p10_v3p10", 2, "High-speed left S turn"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
    for ax, (route, bank, title) in zip(axes, panels):
        model = "U170"
        run = Path(data["sources"][model]) / "evaluation"
        index = json.loads((run / "case_index.json").read_text())
        first = next(r for r in index if r["bank_index"] == bank and r["route"] == route)
        curves = []
        max_progress = 0.
        for alpha, color, marker in zip(alphas, colors, markers):
            row = next(r for r in index if r["bank_index"] == bank and r["route"] == route and r["alpha"] == alpha)
            with np.load(row["raw_file"]) as raw:
                active = raw["active_tick"].astype(bool)
                xy = raw["actual_xy"][active]
                max_progress = max(max_progress, float(raw["path_progress"][active].max()))
                curves.append((alpha, color, marker, xy))
        rule_run = Path(rule_data["rule_run"]) / "evaluation"
        rule_index = json.loads((rule_run / "case_index.json").read_text())
        rule_row = next(r for r in rule_index if r["bank_index"] == bank and r["route"] == route)
        with np.load(rule_row["raw_file"]) as raw:
            active = raw["active_tick"].astype(bool)
            rule_xy = raw["actual_xy"][active]
            max_progress = max(max_progress, float(raw["path_progress"][active].max()))
        with np.load(run / f"{first['path_key']}_path.npz") as path:
            xy = path["xy"][path["s"] <= max_progress + 2.]
            ax.plot(xy[:, 0], xy[:, 1], color="black", linestyle=":", lw=1.3, label="Original path")
        for alpha, color, marker, xy in curves:
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1, label=f"$\\alpha$={alpha:g}")
            ax.plot(xy[-1, 0], xy[-1, 1], marker=marker, color=color, ms=3)
        ax.plot(rule_xy[:, 0], rule_xy[:, 1], color="#333333", lw=1.05, linestyle="--", label="Rule")
        ax.plot(rule_xy[-1, 0], rule_xy[-1, 1], marker="x", color="#333333", ms=4)
        ax.set(title=title, xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")
        ax.grid(color="#dddddd", lw=.4)
    axes[1].legend(frameon=False, fontsize=6.7, loc="lower right", ncol=2)
    fig.savefig(OUT / "paired_xy_low_high_turn.pdf")
    fig.savefig(OUT / "paired_xy_low_high_turn.png", dpi=300)
    plt.close(fig)

    route = "holdout_low_S_right_72_R1p55_v2p00"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.35), layout="constrained")
    for column, bank in enumerate((2, 5)):
        ax, speed_ax = axes[:, column]
        curves = []
        path_file = None
        max_progress = 0.
        for label, root, alpha, color, style, marker in (
            ("U170 $\\alpha=0$", pulse_data["u170_run"], 0., "#0072B2", "-", "o"),
            ("U170 $\\alpha=1$", pulse_data["u170_run"], 1., "#D55E00", "-", "s"),
            ("Geometric rule", pulse_data["rule_run"], .5, "#333333", "--", "x"),
        ):
            run = Path(root) / "evaluation"
            index = json.loads((run / "case_index.json").read_text())
            row = next(r for r in index if r["route"] == route and r["bank_index"] == bank and r["event"] == "t2.8_F+36" and r["alpha"] == alpha)
            with np.load(row["raw_file"]) as raw:
                active = raw["active_tick"].astype(bool)
                xy = raw["actual_xy"][active]
                time = raw["time"][active]
                speed = raw["actual_forward_speed"][active]
                request = raw["v_user"][active]
                onset_xy = xy[np.searchsorted(raw["time"][active], 2.8)]
                max_progress = max(max_progress, float(raw["path_progress"][active].max()))
                curves.append((label, color, style, marker, xy, onset_xy, time, speed, request))
            if path_file is None:
                path_file = run / f"{row['path_key']}_path.npz"
        with np.load(path_file) as path:
            original = path["xy"][path["s"] <= max_progress + 2.]
        ax.plot(original[:, 0], original[:, 1], color="black", ls=":", lw=1.25, label="Original path")
        for label, color, style, marker, xy, onset_xy, time, speed, request in curves:
            ax.plot(xy[:, 0], xy[:, 1], color=color, ls=style, lw=1.1, label=label)
            ax.scatter(*onset_xy, color=color, s=11, zorder=3)
            ax.plot(xy[-1, 0], xy[-1, 1], marker=marker, color=color, ms=3.5)
            speed_ax.plot(time, speed, color=color, ls=style, lw=1, label=label)
        speed_ax.plot(curves[0][6], curves[0][8], color="black", ls=":", lw=1.1, label="Requested speed")
        speed_ax.axvspan(2.8, 2.95, color="#999999", alpha=.18, lw=0)
        ax.set(title=f"Low-speed S turn, bank {bank}", xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")
        ax.grid(color="#dddddd", lw=.4)
        speed_ax.set(xlabel="Task time (s)", ylabel="Forward speed (m/s)", xlim=(0, 10))
        speed_ax.grid(color="#dddddd", lw=.4)
    axes[0, 1].legend(frameon=False, fontsize=6.4, loc="lower right")
    axes[1, 1].legend(frameon=False, fontsize=6.4, loc="lower right")
    fig.savefig(OUT / "positive_pulse_xy.pdf")
    fig.savefig(OUT / "positive_pulse_xy.png", dpi=300)
    plt.close(fig)

    right_angle_panels = [
        ("right_angle_low_left_90_R1p6_v2p0", 2, "Low speed: left 90$^\\circ$"),
        ("right_angle_high_left_90_R3p0_v3p05", 2, "High speed: left 90$^\\circ$"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 4.0))
    fig.subplots_adjust(left=.09, right=.98, top=.90, bottom=.23, wspace=.34)
    for ax, (route, bank, title) in zip(axes, right_angle_panels):
        curves = []
        max_progress = 0.
        original_file = None
        for label, root, alpha, color, style, marker in (
            ("U170 $\\alpha=0$", angle_data["u170_run"], 0., "#0072B2", "-", "o"),
            ("U170 $\\alpha=1$", angle_data["u170_run"], 1., "#D55E00", "-", "s"),
            ("Geometric rule", angle_data["rule_run"], .5, "#333333", "--", "x"),
        ):
            run = Path(root) / "evaluation"
            index = json.loads((run / "case_index.json").read_text())
            row = next(r for r in index if r["route"] == route and r["bank_index"] == bank and r["alpha"] == alpha)
            with np.load(row["raw_file"]) as raw:
                active = raw["active_tick"].astype(bool)
                xy = raw["actual_xy"][active]
                max_progress = max(max_progress, float(raw["path_progress"][active].max()))
                curves.append((label, color, style, marker, xy))
            if original_file is None:
                original_file = run / f"{row['path_key']}_path.npz"
        with np.load(original_file) as path:
            original = path["xy"][path["s"] <= max_progress + 2.]
            goal = np.array([np.interp(path["goal"], path["s"], path["xy"][:, i]) for i in (0, 1)])
        ax.plot(original[:, 0], original[:, 1], color="black", ls=":", lw=1.25, label="Original path")
        ax.plot(*goal, marker="*", color="black", ms=5, linestyle="none", label="Original goal")
        for label, color, style, marker, xy in curves:
            ax.plot(xy[:, 0], xy[:, 1], color=color, ls=style, lw=1.05, label=label)
            ax.plot(xy[-1, 0], xy[-1, 1], marker=marker, color=color, ms=3.5)
        ax.set(title=title, xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")
        ax.grid(color="#dddddd", lw=.4)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.6, loc="lower center", bbox_to_anchor=(.5, .015), ncol=3)
    fig.savefig(OUT / "right_angle_xy.pdf", bbox_inches="tight")
    fig.savefig(OUT / "right_angle_xy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

(OUT / "article_figure_sources.json").write_text(json.dumps({
    "source_results": str(RESULTS),
    "raw_runs": data["sources"],
    "rule_run": rule_data["rule_run"],
    "pulse_runs": {"u170": pulse_data["u170_run"], "rule": pulse_data["rule_run"]},
    "positive_pulse_panel": {"route": route, "banks": [2, 5], "event": "t2.8_F+36"},
    "right_angle_result": str(RIGHT_ANGLE_RESULTS),
    "right_angle_panels": [{"route": route, "bank": bank} for route, bank, _ in right_angle_panels],
    "xy_panels": [{"route": route, "bank": bank} for route, bank, _ in panels],
    "transformations": "raw actual XY, forward speed and original path; task figure uses saved per-case summary without smoothing",
    "replicate_unit": "paired route and prepared-state bank, n=8 per model/alpha",
}, indent=2) + "\n")
