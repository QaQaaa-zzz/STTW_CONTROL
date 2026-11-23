"""Build the static paired examples and a local XY viewer for every audited task."""
import argparse
import base64
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


COLOURS = ("#176388", "#ac470f")
ARMS = ("fixed", "selected")
SECTION_NAMES = ("near", "mid", "far")


def sample_mm(points, step, extra=()):
    indices = np.unique(np.r_[np.arange(0, len(points), step), len(points) - 1, extra]).astype(int)
    return np.rint(points[indices] * 1000).astype(int).tolist()


def outcome(row, arm):
    if row[arm + "_success"]:
        return "done"
    if row[arm + "_wall"]:
        return "wall"
    return str(row[arm + "_endpoint"])


def plot_examples(cases, maps, out):
    selected = []
    ordered = cases.sort_values(["root", "force", "section"])
    for layout in sorted(ordered.layout.unique()):
        group = ordered.loc[(ordered.layout == layout) & (ordered.side == -1)]
        for fixed, new in [(1, 0), (0, 1)]:
            rows = group.loc[(group.fixed_success == fixed) & (group.selected_success == new)]
            if len(rows):
                selected.append(rows.iloc[0])
    fig, axes = plt.subplots(len(selected), 1, figsize=(12.2, 2.45 * len(selected)),
                             layout="constrained", squeeze=False)
    for ax, row in zip(axes[:, 0], selected):
        geo = maps[row.layout]
        g = geo["goal"]
        ax.add_patch(Rectangle((g["x_min_m"], -g["abs_y_max_m"]), g["x_max_m"]-g["x_min_m"],
                               2*g["abs_y_max_m"], color="#d2e7db", zorder=0))
        for x0, x1, y0, y1 in geo["walls"]:
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, color="#333333", zorder=6))
        gate_x = geo["section"][row.section]
        ax.vlines(gate_x, -3.9, -.06, color="#8a6a16", linewidth=1)
        ax.axvline(geo["edge"], color="#949494", linestyle=":", linewidth=1)
        ybound = 1.15
        for arm, colour, marker in zip(ARMS, COLOURS, ("o", "s")):
            with np.load(row[arm + "_trajectory"]) as data:
                xy = data["qpos"][:, :2]
                ref = data["published_reference_xy"]
                tr = data["trace"]
            ax.plot(*ref.T, color=colour, linestyle=":", linewidth=1.2)
            ax.plot(*xy.T, color=colour, linewidth=1.7, alpha=.92)
            ax.plot(*xy[np.isfinite(xy).all(axis=1)][-1], marker=marker, markerfacecolor="white" if row[arm+"_success"] else colour,
                    color=colour, markersize=6)
            hb = row[arm+"_handoff_s"]
            if pd.notna(hb) and hb >= 0:
                idx = int(np.argmin(np.abs(tr[:, 0] - hb)))
                ax.plot(*xy[idx], marker="|", color=colour, markersize=13)
            ybound = max(ybound, np.nanmax(np.abs(xy[:,1]))+.25)
        ax.set(xlim=(0.7, 19.5), ylim=(-ybound, ybound), aspect="equal",
               ylabel="World y (m)", xlabel="World x (m)")
        ax.set_title(f'{row.layout}   |   right {row.section}   |   start {row.root}   |   push {row.force:+g} N'
                     f'   |   Fixed: {outcome(row, "fixed")}    {row.selected_planner}: {outcome(row, "selected")}',
                     loc="left", fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("First paired loss and rescue (when present) in each layout — right gate",
                 fontsize=14)
    axes[0,0].legend(handles=[Line2D([],[],color=COLOURS[0],marker="o",label="Fixed"),
                              Line2D([],[],color=COLOURS[1],marker="s",label="Evaluated planner"),
                              Line2D([],[],color="#444",linestyle=":",label="Published reference"),
                              Line2D([],[],color="#444",marker="|",linestyle="",label="Handback")],
                     frameon=False, ncol=4, fontsize=8, loc="upper left")
    fig.savefig(out/"paired_xy.png",dpi=150)
    plt.close(fig)
    pd.DataFrame(selected).to_csv(out/"paired_xy_examples.csv",index=False)


def write_viewer(cases, out):
    maps = {}
    paths = {}
    references = {}
    trajectories = []
    rows = []
    for _, r in cases.iterrows():
        arms = []
        for arm in ARMS:
            path = str(r[arm+"_trajectory"])
            if path not in paths:
                p = Path(path)
                if r.layout not in maps:
                    cfgdir = next(parent for parent in p.parents if (parent/"resolved_config.json").is_file())
                    cfg = json.loads((cfgdir/"resolved_config.json").read_text())
                    scene = cfg["world_map_spec"]
                    manifest_path = next(parent/"test_freeze_manifest.json" for parent in p.parents
                                         if (parent/"test_freeze_manifest.json").is_file())
                    sections = json.loads(manifest_path.read_text())["section_x"]
                    maps[r.layout] = dict(walls=scene["terrain"]["forbidden_rectangles"],
                                          edge=scene["terrain"]["front_x_m"], goal=scene["goal"],
                                          section=dict(zip(SECTION_NAMES, sections)))
                with np.load(p) as archive:
                    xy = archive["qpos"][:, :2]
                    t = archive["trace"][:, 0]
                    ref = archive["published_reference_xy"]
                finite = np.isfinite(xy).all(axis=1)
                if not finite.any() or not np.isfinite(ref).all():
                    raise ValueError(f"Missing entire XY or incomplete reference: {path}")
                finite_indices = np.flatnonzero(finite)
                segments = np.split(finite_indices, np.flatnonzero(np.diff(finite_indices) != 1) + 1)
                ref_key = ref.tobytes()
                if ref_key not in references:
                    references[ref_key] = (len(references), sample_mm(ref, 6))
                paths[path] = len(trajectories)
                hb = r[arm+"_handoff_s"]
                marker = None
                extra = []
                if pd.notna(hb) and hb >= 0:
                    hi = int(np.argmin(np.abs(t - hb)))
                    if finite[hi]:
                        marker = np.rint(1000 * xy[hi]).astype(int).tolist()
                        extra = [hi]
                parts = []
                for segment in segments:
                    first, last = int(segment[0]), int(segment[-1])
                    markers = [i-first for i in extra if first <= i <= last]
                    parts.append(sample_mm(xy[first:last+1],10,markers))
                trajectories.append(dict(parts=parts, ref=references[ref_key][0], hb=marker,
                                         t=round(float(t[-1]),4), last_xy_t=round(float(t[finite_indices[-1]]),4),
                                         missing_frames=int((~finite).sum())))
            arms.append([paths[path], outcome(r, arm)])
        rows.append([r.layout, int(r.root), int(r.force), r.section, int(r.side), str(r.selected_planner), *arms])
    payload = dict(maps=maps, cases=rows, trajectories=trajectories,
                   references=[v[1] for v in references.values()])
    encoded = base64.b64encode(gzip.compress(json.dumps(payload,separators=(",",":"),
                                                            ensure_ascii=False).encode(),mtime=0)).decode()
    template = Path(__file__).with_name("chapter3_p0_xy_viewer.html").read_text()
    (out/"all_cases_xy.html").write_text(template.replace("{{PAYLOAD}}",encoded))
    plot_examples(cases,maps,out)
    provenance = dict(tasks=len(rows), unique_files=len(paths), unique_references=len(references),
                      files_with_missing_xy=sum(t["missing_frames"] > 0 for t in trajectories),
                      layout_counts=cases.groupby("layout").size().to_dict(), actual_sample_ms=50,
                      reference="archived final published reference, drawn as dotted geometric curve",
                      units="metres; display rounded to 0.001 m. Endpoints and handbacks retained.",
                      raw_index="contracts.csv",selection="Static examples: lowest root, force, then section label per layout and loss/rescue.")
    (out/"xy_manifest.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    out=parser.parse_args().folder
    frame=pd.read_csv(out/"contracts.csv")
    write_viewer(frame,out)
