"""Reproduce the Chapter 3 T3 and clock audits from archived physical traces."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


KEYS = ["layout", "skill", "root", "force", "section", "side"]
ARMS = ["alpha0", "alpha_enabled"]
SECTIONS = ["near", "mid", "far"]
CLOCKS = {
    "305_300": "blind_test",
    "90_85": "post_review/latency_ablation",
    "15_15": "post_review/realtime_15ms",
}
STATE = ["x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"]


def oracle_options(raw):
    oracle = pd.read_csv(raw / "oracle_cases.csv")
    assert (oracle.groupby(KEYS).size() == 15).all()
    assert (oracle[oracle.approach_alpha == 0].groupby(KEYS).size() == 9).all()
    assert not oracle.duplicated(KEYS + ["approach", "approach_alpha", "planned_dep"]).any()
    return oracle, pd.concat(
        [
            oracle[oracle.approach_alpha == 0].groupby(KEYS).success.any().rename("oracle_9"),
            oracle.groupby(KEYS).success.any().rename("oracle_15"),
        ],
        axis=1,
    )


def alpha_audit(raw, output, layout_cases):
    oracle, options = oracle_options(raw)
    options.reset_index().groupby(["layout", "skill", "side"]).agg(
        contracts=("root", "size"), oracle_9=("oracle_9", "sum"), oracle_15=("oracle_15", "sum")
    ).to_csv(output / "oracle_ceiling.csv")
    cases = pd.read_csv(raw / "online_cases.csv")
    archived = pd.read_csv(output / "assignments.csv.gz")
    pd.testing.assert_frame_equal(cases[archived.columns], archived)
    logged_oracle = cases.set_index(KEYS).oracle_feasible
    assert logged_oracle.eq(options.oracle_15.reindex(logged_oracle.index)).all()
    cases["queried"] = cases.query_batch_ms > 0
    cases["admitted_done"] = cases.query_feasible & cases.success
    cases["admitted_wall"] = cases.query_feasible & cases.collision
    cases["vetoed_wall"] = cases.queried & ~cases.query_feasible & cases.collision
    cases["unqueried_wall"] = ~cases.queried & cases.collision
    columns = ["success", "collision", "queried", "query_feasible", "admitted_done",
               "admitted_wall", "vetoed_wall", "unqueried_wall"]
    cases.groupby(["layout", "skill", "planner"])[columns].sum().to_csv(output / "admission_funnel.csv")
    walls = cases[(cases.skill == "rb_run1") & (cases.planner == "alpha_enabled") & cases.collision]
    walls[KEYS + ["approach", "approach_alpha", "planned_dep", "chosen_dep",
                  "queried", "query_feasible", "handoff_s", "termination"]].to_csv(
        output / "expanded_wall_contracts.csv", index=False
    )
    left = oracle[oracle.side == 1]
    branch = KEYS + ["approach", "approach_alpha"]
    for mode in ("replay", "commit_only", "fixed_lookup"):
        inputs = cases.copy()
        dep = "chosen_dep" if mode == "replay" else "planned_dep"
        if mode == "fixed_lookup":
            inputs = inputs[inputs.planner == "alpha0"].assign(approach=1, approach_alpha=0., planned_dep=0)
        matched = inputs.merge(left.rename(columns={"planned_dep":"branch_dep"}),
                               left_on=branch+[dep], right_on=branch+["branch_dep"],
                               suffixes=("", "_o"), validate="many_to_one")
        assert len(inputs) == len(matched)
        if mode != "fixed_lookup":
            assert matched.prefix.eq(matched.prefix_o).all()
        if mode == "replay":
            assert matched.success.eq(matched.success_o).all() and matched.collision.eq(matched.collision_o).all()
        else:
            matched.groupby(["layout","skill","planner"])[["success_o","collision_o"]].sum().to_csv(output/(mode+".csv"))
    other = cases.query("layout=='test_b' and skill=='rb_run1' and planner=='alpha_enabled'")
    fields = ["root","force","section","approach","approach_alpha","chosen_dep","termination","success","query_feasible"]
    walls[fields].merge(other[fields],on=["root","force","section"],suffixes=("_a","_b"),
                        validate="one_to_one").to_csv(output/"matched_wall_outcomes.csv",index=False)
    layout_cases["queried"] = layout_cases.query_batch_ms > 0
    layout_cases["admitted_done"] = layout_cases.query_feasible & layout_cases.success
    layout_cases["admitted_wall"] = layout_cases.query_feasible & layout_cases.collision
    layout_cases["vetoed"] = layout_cases.queried & ~layout_cases.query_feasible
    layout_cases["wall_after_veto"] = layout_cases.vetoed & layout_cases.collision
    layout_cases["feasible_vetoed"] = layout_cases.vetoed & layout_cases.oracle_feasible
    layout_cases.groupby(["layout", "skill", "planner"]).agg(
        contracts=("root", "size"), success=("success", "sum"), collision=("collision", "sum"),
        verified=("verified", "sum"), queried=("queried", "sum"), admitted=("query_feasible", "sum"),
        admitted_done=("admitted_done", "sum"), admitted_wall=("admitted_wall", "sum"),
        vetoed=("vetoed", "sum"), wall_after_veto=("wall_after_veto", "sum"),
        feasible_vetoed=("feasible_vetoed", "sum"),
    ).to_csv(output.parent / "wall_layouts/admission_funnel.csv")
    alpha_figure(raw, output, cases)


def alpha_figure(raw, output, cases):
    base, expanded = (cases[cases.planner == arm].set_index(KEYS) for arm in ARMS)
    paired = pd.DataFrame({
        "rescued": ~base.success & expanded.success,
        "lost": base.success & ~expanded.success,
        "changed": base.chosen_dep != expanded.chosen_dep,
    })
    plan = json.loads((raw / "plan.json").read_text())
    styles = [("#005a8d", "-"), ("#a64112", "--")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 4.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.99, bottom=.13, top=.80, hspace=.42, wspace=.12)
    selected = []
    for (layout, skill), group in paired.reset_index().groupby(["layout", "skill"]):
        row = group.sort_values(["rescued", "lost", "changed"], ascending=False).iloc[0]
        key = tuple(row[k] for k in KEYS)
        ax = axes[int(layout == "test_a"), int(skill == "rb_run1")]
        base_path = Path(base.loc[key].trajectory)
        terrain = json.loads((base_path.parents[1] / "resolved_config.json").read_text())["ditch_terrain"]
        for x0, x1, y0, y1 in terrain["forbidden_rectangles"]:
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fc=".38", ec=".20", zorder=3))
        gate_x = plan["section_x"][SECTIONS.index(row.section)]
        ax.plot([gate_x, gate_x], [.06, 3.9], color="#28614d", lw=1.2, alpha=.8)
        for arm, df, (color, ls) in zip(ARMS, (base, expanded), styles):
            record = df.loc[key]
            path = Path(record.trajectory)
            with np.load(path) as data:
                trace = data["trace"][:, 1:3]
                ref = data["published_reference_xy"]
            ax.plot(*ref.T, ":", color=color, lw=1, alpha=.8)
            ax.plot(*trace.T, color=color, ls=ls, lw=1.6, zorder=4)
            ax.scatter(*trace[-1], marker="o" if record.success else "x", color=color,
                       s=32, lw=1.4, zorder=5)
            selected.append(dict(zip(KEYS, key), arm=arm, alpha=record.approach_alpha,
                                 success=bool(record.success), termination=record.termination,
                                 trajectory=str(path.relative_to(raw))))
        skill_label = "R9" if skill == "rb_run1" else "Source"
        ax.set_title(f"TEST {layout[-1].upper()} / {skill_label}, root {row.root}, "
                     f"{row.force:+g} N, {row.section}", fontsize=11, pad=5)
        ax.set(xlim=(9, 22), ylim=(-1.55, 1.55), aspect="equal", yticks=(-1, 0, 1))
        ax.grid(alpha=.2, lw=.5)
        ax.tick_params(labelsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("World y (m)", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("World x (m)", fontsize=10)
    handles = [
        Line2D([], [], color=color, ls=ls, lw=1.6, label=label)
        for (color, ls), label in zip(styles, ("Fixed preference", "Selectable preference"))
    ] + [
        Line2D([], [], color=".35", ls=":", label="Published path"),
        Line2D([], [], color="#28614d", label="Assigned gate"),
        Patch(facecolor=".38", edgecolor=".20", label="Wall"),
        Line2D([], [], marker="x", color=".25", ls="none", label="Failed contract endpoint"),
    ]
    fig.legend(handles=handles, ncol=3, loc="upper center", frameon=False, fontsize=10)
    fig.savefig(output / "online_xy.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(selected).to_json(output / "online_xy_cases.json", orient="records", indent=2)


@lru_cache(maxsize=32)
def read_batch(folder):
    import mujoco

    model = mujoco.MjModel.from_binary_path(str(folder / "terrain.mjb"))
    terrain = json.loads((folder / "resolved_config.json").read_text())["ditch_terrain"]
    rows = json.loads((folder / "status.json").read_text())["results"]
    return model, terrain, {row["case_dir"]: row for row in rows}


def flight_events(z, model, terrain, rb_entry):
    from report_platform_controls import flight

    if rb_entry is None:
        return None, None, None
    loads = z["contact_substep"]
    support = z["wheel_contact_flags"][:, 2] & (loads[:, -1, :2] > 1e-5).all(1)
    edge = terrain["front_x_m"]
    return flight(model, z["qpos"], loads, support, z["trace"][:, 0], edge, edge,
                  terrain["far_bank_height_m"], rb_entry)


def clock_audit(raw, output):
    assignments = {}
    for clock, folder in CLOCKS.items():
        cases = pd.read_csv(raw / folder / "online_cases.csv")
        assignments[clock] = cases.query("skill == 'rb_run1' and planner == 'C-R'").set_index(KEYS)
    slow, fast = assignments["90_85"], assignments["15_15"]
    assert all(cases.index.equals(slow.index) for cases in assignments.values())
    worse = (slow.success & ~fast.success) | (~slow.collision & fast.collision)
    rows = []
    for clock, cases in assignments.items():
        for key, record in cases[worse].iterrows():
            path = Path(record.trajectory)
            model, terrain, status = read_batch(path.parents[1])
            physical = status[path.parent.name]
            with np.load(path) as z:
                trace = pd.DataFrame(z["trace"], columns=z["columns"])
                takeoff, _, _ = flight_events(z, model, terrain, physical["rb_entry_s"])
            row = dict(zip(KEYS, key), clock=clock, approach=record.approach,
                       planned_dep=record.planned_dep, chosen_dep=record.chosen_dep,
                       success=record.success, collision=record.collision, termination=record.termination,
                       admitted=record.query_feasible, release_tick=physical.get("initial_release_tick"),
                       rb_entry_s=physical["rb_entry_s"], handback_s=record.handoff_s,
                       departure_due_tick=physical.get("departure_due_tick"),
                       departure_applied_tick=record.departure_applied_tick, takeoff_s=takeoff,
                       trajectory=str(path.relative_to(raw)))
            for event, t in [("takeoff", takeoff), ("handback", record.handoff_s)]:
                if pd.notna(t):
                    sample = trace.iloc[(trace.time - t).abs().argmin()]
                    row.update({f"{event}_{signal}": sample[signal] for signal in STATE})
            rows.append(row)
    pd.DataFrame(rows).sort_values(KEYS + ["clock"]).to_csv(output / "clock_state_pairs.csv", index=False)


def transfer_audit(raw, output):
    from plot_platform_layout_detail import plot

    output.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(raw / "online_cases.csv")
    oracle, options = oracle_options(raw)
    branch = KEYS + ["approach", "approach_alpha", "chosen_dep"]
    executed = oracle.rename(columns={"planned_dep": "branch_exit"})
    matched = cases.merge(executed, left_on=branch, right_on=branch[:-1] + ["branch_exit"],
                          suffixes=("", "_oracle"), validate="many_to_one")
    assert len(matched) == len(cases)
    for label in ("prefix", "success", "collision"):
        assert matched[label].eq(matched[label + "_oracle"]).all(), label
    cases = cases.join(options, on=KEYS)
    cases["solvable"] = np.where(cases.planner == "alpha0", cases.oracle_9, cases.oracle_15)
    cases["veto"] = (cases.query_batch_ms > 0) & ~cases.query_feasible
    cases["solvable_veto"] = cases.veto & cases.solvable
    cases["newly_solved"] = cases.success & ~cases.oracle_9
    cases["nonzero_alpha"] = cases.approach_alpha > 0
    cases["trajectory"] = cases.trajectory.map(lambda path: str(Path(path).relative_to(raw)))
    cols = KEYS + ["planner", "approach", "approach_alpha", "chosen_dep", "verified",
                   "solvable", "oracle_9", "oracle_15", "query_feasible", "veto", "success",
                   "collision", "termination", "trajectory"]
    cases[cols].to_csv(output / "assignments.csv.gz", index=False)
    cases.groupby(["layout", "section", "force", "planner"]).agg(
        contracts=("root", "size"), completed=("success", "sum"), wall=("collision", "sum"),
        relocked=("verified", "sum"), admitted=("query_feasible", "sum"),
    ).to_csv(output / "by_section_force.csv")
    counts = cases.groupby(["layout", "planner"]).agg(
        contracts=("root", "size"), verified=("verified", "sum"), solvable=("solvable", "sum"),
        newly_solved=("newly_solved", "sum"), solvable_veto=("solvable_veto", "sum"),
        nonzero_alpha=("nonzero_alpha", "sum"))
    counts.join(pd.read_csv(raw / "admission_funnel.csv").set_index(["layout", "planner"])
                .drop(columns="contracts")).to_csv(output / "layout_outcomes.csv")
    pd.read_csv(raw / "paired_metrics.csv").to_csv(output / "paired_metrics.csv", index=False)
    examples = cases[(cases.root == cases.root.min()) & (cases.force == 0) & (cases.section == "near")]
    events = []
    for case in examples.itertuples():
        path = raw / case.trajectory
        model, terrain, rows = read_batch(path.parents[1])
        with np.load(path) as z:
            times = flight_events(z, model, terrain, rows[path.parent.name]["rb_entry_s"])
        events.append((case.layout, case.root, case.planner, case.approach_alpha, *times,
                       case.handoff_s, case.success, case.collision, case.trajectory))
    pd.DataFrame(events, columns=["layout", "root", "arm", "alpha", "takeoff_s", "airborne_end_s",
                 "loaded_landing_s", "relock_s", "success", "collision", "trajectory"]).to_csv(
        output / "illustrated_events.csv", index=False)
    plot(raw, output / "online_xy.png", arms=ARMS)
    plot(raw, output / "selected_paths.png", arms=ARMS, selected=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter_mainline", type=Path)
    parser.add_argument("final_stages", type=Path)
    parser.add_argument("--transfer", action="store_true", help="Report the frozen library on the three T2 wall maps")
    args = parser.parse_args()
    if args.transfer:
        transfer_audit(args.chapter_mainline / "post_review/alpha_transfer", args.final_stages / "alpha_transfer")
        parser.exit()
    wall = pd.read_csv(args.chapter_mainline / "post_review/wall_layouts/online_cases.csv")
    alpha_audit(args.chapter_mainline / "post_review/alpha_library", args.final_stages / "alpha_library", wall)
    clock_audit(args.chapter_mainline, args.final_stages / "realtime_15ms")
