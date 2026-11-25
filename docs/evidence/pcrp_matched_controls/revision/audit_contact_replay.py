"""Compare completed T7 nominal trajectories with their archived E2 counterparts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_trace(path):
    with np.load(path / "trajectory.npz") as data:
        trace = data["lower_trace"].reshape(-1, 11)
        active = np.flatnonzero(trace[:, 6])
        last = active[-1]
        return trace[active, :2], int(last * 25 + trace[last, 7]), not bool(data["failure"].any())


def main(root):
    read = lambda filename: json.loads((root / filename).read_text())
    folder = root / "nominal_replay_audit"
    folder.mkdir(exist_ok=True)
    with np.load(Path(read("protocol.json")["bundle"]) / "route.npz") as data:
        route = data["xy"][data["s"] <= 29]
    lanes = [lane for lane in read("lanes.json") if lane["setting"] == "nominal"]
    rows = []
    for offset, alpha in sorted({(lane["ds_cm"], lane["alpha"]) for lane in lanes}):
        group = [lane for lane in lanes if (lane["ds_cm"], lane["alpha"]) == (offset, alpha)]
        if not all((Path(lane["output"]) / "status.json").is_file() for lane in group):
            continue
        fig, axes = plt.subplots(3, 4, figsize=(11, 9), constrained_layout=True)
        for ax, lane in zip(axes.flat, group, strict=True):
            paths = [Path(lane[k]) for k in ("output", "original_e2_source")]
            (new, t, full), (old, u, prior) = map(read_trace, paths)
            n = min(len(new), len(old))
            shared_xy = bool(np.allclose(new[:n], old[:n], atol=1e-8, rtol=1e-5))
            delta = np.linalg.norm(new[:n]-old[:n], axis=1)
            rows.append(dict(ds_cm=offset, alpha=alpha, center_m=lane["center_m"], seed=lane["group"],
                             c10_new=int(full), c10_e2=int(prior), end_tick_new=t, end_tick_e2=u,
                             common_ticks_200hz=n, max_common_xy_distance_m=float(delta.max()),
                             common_xy_allclose=shared_xy, full_trace_match=shared_xy and t==u and full==prior,
                             new_path=str(paths[0]), e2_path=str(paths[1])))
            # Restrict the displayed reference to the part of the original route reached by either trajectory.
            last_s = max(np.linalg.norm(route-xy[-1], axis=1).argmin() for xy in (new, old))
            ax.plot(*route[:last_s+4].T, color="0.5", lw=2, label="Original route")
            for xy, ok, name, color, style in [(old,prior,"E2 (24 slots)","#af5e16","--"),
                                               (new,full,"Review (12 slots)","#196582","-")]:
                ax.plot(*xy.T, color=color, ls=style, lw=1.4, label=name)
                ax.scatter(*xy[-1], color=color, marker="o" if ok else "x", s=26)
            ax.set(aspect="equal", title=f"{lane['center_m']:.1f} m, group {lane['group']}", xlabel="x (m)", ylabel="y (m)")
            ax.grid(alpha=.2)
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3)
        fig.suptitle(f"H75 + Local300, nominal contact, shift {offset:+d} cm, α={alpha:g}\n"
                     "DEV  •  circles: 10 s  •  crosses: actual failure  •  10 cm ridge at 2.6 m/s")
        fig.savefig(folder / f"xy_ds{offset:+d}cm_a{alpha:g}.png", dpi=145)
        plt.close(fig)
    if rows:
        with (folder/"paired.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)}/{len(lanes)} pairs available, {sum(r['c10_new'] != r['c10_e2'] for r in rows)} changes in C10")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run)
