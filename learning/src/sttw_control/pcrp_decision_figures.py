"""Draw the PCRP phase experiment from measured 200 Hz trajectories and lane metrics."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@lru_cache(maxsize=8)
def _positions(source):
    with np.load(Path(source) / "trajectory.npz") as data:
        trace = data["lower_trace"]
        progress = data["progress"]
        lanes = []
        for lane in range(trace.shape[2]):
            samples = trace[:, :, lane].reshape(-1, trace.shape[-1])
            valid = samples[:, 6].astype(bool)
            lanes.append((samples[valid, :2], float(np.max(progress[valid, lane]))))
    return lanes


@lru_cache(maxsize=32)
def _ridge_xy(terrain):
    with np.load(Path(terrain) / "terrain.npz") as data:
        vertices = data["vertices"]
        return vertices[vertices[:, 2] > 0, :2]


def _finish(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _xy_grid(group_rows, route, shifts, alphas, shift_key, units, title, out):
    nrows, ncols = len(shifts), len(alphas)
    points = [(_positions(r["source"])[r["lane"]], r) for r in group_rows]
    progress = max(end_s for (_, end_s), _ in points)
    ref = route["xy"][route["s"] <= progress + 0.25]
    xmin, ymin = np.minimum(ref.min(axis=0), np.min([xy.min(axis=0) for (xy, _), _ in points], axis=0)) - 0.3
    xmax, ymax = np.maximum(ref.max(axis=0), np.max([xy.max(axis=0) for (xy, _), _ in points], axis=0)) + 0.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.10*ncols, 2.75*nrows+0.9),
                             squeeze=False, layout="constrained")
    by_cell = {(r[shift_key], r["alpha"]): (xy, r) for (xy, _), r in points}
    for i, shift in enumerate(shifts):
        for j, alpha in enumerate(alphas):
            ax = axes[i, j]
            ax.set(aspect="equal", xlim=(xmin, xmax), ylim=(ymin, ymax))
            ax.locator_params(axis="x", nbins=2, integer=True)
            if j == 0:
                ax.set_ylabel(f"{shift:+g} {units}\nWorld y (m)")
            if i == nrows-1:
                ax.set_xlabel("World x (m)")
            if (shift, alpha) not in by_cell:
                ax.text(.5, .5, "Missing", ha="center", transform=ax.transAxes)
                continue
            xy, row = by_cell[shift, alpha]
            ax.plot(*ref.T, "--", color="#626262", lw=1.2, zorder=3)
            ax.plot(*xy.T, color="#0072b2", lw=2.1)
            if row["center_m"] is not None:
                ax.scatter(*_ridge_xy(row["terrain"]).T, color="#c69d53", s=2, linewidths=0, zorder=1)
            marker = "o" if row["full"] else "x"
            ax.plot(*xy[-1], marker=marker, color="k", ms=5, fillstyle="none", zorder=4)
            term = row["terminal"].replace("_failure", "").replace("full_window", "complete").replace("_", " ")
            top = f"α = {alpha:g}\n" if i == 0 else ""
            ax.set_title(f'{top}{term} · {row["time_s"]:.3f} s', fontsize=9)
    caption = "Blue: actual · dashed: original route · ○ full 10 s · × early endpoint"
    if group_rows[0]["center_m"] is not None:
        caption += " · ochre: raised mesh vertices"
    fig.suptitle(f"{title}\n{caption}", fontsize=12)
    return _finish(fig, out)


def _completion_maps(rows, groups, shifts, alphas, shift_key, units, title, out):
    cells = defaultdict(list)
    for r in rows:
        cells[r["group"], r[shift_key], r["alpha"]].append(r["full"])
    panels = groups + [None]
    cols = min(3, len(panels))
    height = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(height, cols, squeeze=False, figsize=(3.8*cols, 0.46*len(shifts)*height+1.4*height),
                             layout="constrained")
    for ax, group in zip(axes.flat, panels):
        block = groups if group is None else [group]
        data = [[sum((cells[g, shift, alpha] for g in block), []) for alpha in alphas] for shift in shifts]
        rate = np.array([[np.mean(v) if v else np.nan for v in line] for line in data])
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set(xticks=range(len(alphas)), xticklabels=[f"{a:g}" for a in alphas],
               yticks=range(len(shifts)), yticklabels=[f"{s:+g}" for s in shifts],
               xlabel="H75 preference α", ylabel=f"Offset ({units})",
               title="All groups" if group is None else f"Group {group}")
        for i, line in enumerate(data):
            for j, v in enumerate(line):
                ax.text(j, i, f"{sum(v)}/{len(v)}" if v else "—", ha="center", va="center",
                        color="white" if rate[i, j] >= .55 else "black", fontsize=8)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"{title}\nFull 10 s / observed runs", fontsize=13)
    return _finish(fig, out)


def _entry_curves(rows, shifts, alphas, shift_key, xlabel, title, out):
    fields = [("entry_v_m_s", "Entry speed (m/s)"), ("entry_phi_rad", "Entry roll (rad)"),
              ("entry_omega_rad_s", "Entry roll rate (rad/s)"), ("entry_delta_rad", "Entry steer angle (rad)"),
              ("entry_H_rad", "Entry nominal H (rad)"), ("peak_roll_rad", "Peak |roll| over observed run (rad)")]
    cells = {(r["alpha"], r[shift_key]): r for r in rows}
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.5), layout="constrained")
    colors = ["#0072b2", "#d55e00", "#008060", "#ae588a", "#665a00"]
    markers = ["o", "s", "^", "D", "v"]
    for (key, ylabel), ax in zip(fields, axes.flat):
        for color, marker, alpha in zip(colors, markers, alphas):
            values = [cells.get((alpha, s), {}).get(key, np.nan) for s in shifts]
            ax.plot(shifts, np.asarray(values, dtype=float), color=color, marker=marker, ms=4, label=f"α {alpha:g}")
        ax.set(xlabel=xlabel, ylabel=ylabel, xticks=shifts)
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8, ncols=2)
    fig.suptitle(f"{title}\nFront-wheel upstream crossing · gaps indicate missing measurements", fontsize=12)
    return _finish(fig, out)


def write_e2_figures(rows, output):
    """Return the figures indexed by the E2 analysis; rows come from the lane analysis."""
    out = Path(output) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    layouts = defaultdict(list)
    for r in rows:
        layouts[r["experiment"], r["center_m"]].append(r)
    paths = []
    for (experiment, center), layout in sorted(layouts.items()):
        geometry = experiment == "E2a"
        key = "ds_cm" if geometry else "dv_m_s"
        units = "cm" if geometry else "m/s"
        xlabel = f'{"Ridge-center" if geometry else "Initial-speed"} offset ({units})'
        prefix = experiment + ("_flat" if center is None else f"_s{center:g}")
        name = f'{experiment} · {"Flat ground" if center is None else f"Ridge center {center:g} m"} · H75 + Local300'
        groups = sorted({r["group"] for r in layout})
        shifts = sorted({r[key] for r in layout})
        alphas = sorted({r["alpha"] for r in layout})
        with np.load(layout[0]["route"]) as data:
            route = {key: data[key] for key in ("s", "xy")}
        for group in groups:
            group_rows = [r for r in layout if r["group"] == group]
            title = f"{name} · physical group {group}"
            paths.append(_xy_grid(group_rows, route, shifts, alphas, key, units, title,
                                  out / f"{prefix}_g{group}_xy.png"))
            if center is not None:
                paths.append(_entry_curves(group_rows, shifts, alphas, key, xlabel, title,
                                           out / f"{prefix}_g{group}_state.png"))
        paths.append(_completion_maps(layout, groups, shifts, alphas, key, units, name,
                                      out / f"{prefix}_completion.png"))
    _positions.cache_clear()
    _ridge_xy.cache_clear()
    return paths
