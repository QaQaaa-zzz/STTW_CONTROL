"""Report the registered E1 CEM archive using the unmodified reference and task gates."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from analyze_pcrp_e2 import entry_index
from analyze_roll_contacts import roll_state
from sttw_control.isaac_path_reporting import replay_rewards, write_zero_scores

ALPHAS = (0., .25, .5, .75, 1.)
COLORS = ("#0072b2", "#d55e00", "#008060", "#ae588a", "#665a00", "#606060")
GATES = {"A": (.30, .20), "B": (.60, .10), "C": (.40, .15)}


def table(path, rows, columns=None):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finish(fig, filename):
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def physical_lane(row, route, terrain):
    raw, lane = Path(row["raw"]), int(row["lane"])
    contract = json.loads((raw / "contract.json").read_text())
    ch, front, rear = [contract["bodies"].index(b) for b in ("chasis", "frontwheel", "rearwheel")]
    with np.load(raw / "trajectory.npz") as z:
        trace = z["lower_trace"][:, :, lane].reshape(-1, 11)
        ids = np.flatnonzero(trace[:, 6])
        x = trace[ids]
        t = ids*contract["lower_dt"] + x[:, 7]*contract["physics_dt"]
        body = z["body_state"][ids, lane]
        roll, _ = roll_state(body[:, ch])
        force = np.linalg.norm(z["contact_force"][ids, lane], axis=-1)
        data = dict(time_s=t, world_x_m=x[:, 0], world_y_m=x[:, 1], speed_m_s=x[:, 2],
                    path_error_m=x[:, 3], delivered_speed_m_s=x[:, 8], delivered_steer_rad=x[:, 9],
                    steer_rad=z["measurement"][ids, lane, 2], roll_rad=roll,
                    progress_m=z["progress"][ids, lane], front_normal_resultant_N=force[:, front],
                    rear_normal_resultant_N=force[:, rear])
    upper_dt = contract["policy_dt"]
    action_t = (ids // 4) * upper_dt
    offsets = np.zeros((len(t), 2))
    if int(row["generation"]) > 0:
        with np.load(row["knots_npz"]) as knots:
            values = knots["offsets"][int(row["knots_lane"])]
            for j in range(2):
                offsets[:, j] = np.interp(action_t, knots["times_s"], values[:, j])
    data.update(bias_time_s=action_t, bias_v_m_s=offsets[:, 0], bias_steer_rad=offsets[:, 1])
    active = np.flatnonzero((offsets != 0).any(axis=1))
    first = float(action_t[active[0]]) if len(active) else None
    entry = None
    if "oblique_s_per_lateral_m" in terrain and terrain.get("height_m", 0) > 0:
        wheel = body[:, front, :2] - np.asarray(contract["origin"][lane][:2])
        crossing = entry_index(wheel, route, terrain)
        if crossing is not None:
            entry = float(t[crossing])
    return data, dict(method=row["method"], entry_s=entry, first_nonzero_bias_s=first,
                      lead_before_entry_s=entry-first if entry is not None and first is not None else None)


def policy_rewards(row):
    lane, raw = int(row["lane"]), Path(row["raw"])
    c = json.loads((raw / "contract.json").read_text())
    config = json.loads((Path(c["bundle"]) / "config.json").read_text())
    with np.load(raw / "trajectory.npz") as z:
        data = {key:z[key][:, lane:lane+1] for key in z.files
                if key.startswith("effective_") or key in ("reward", "valid", "failure", "offsets")}
        data["lower_trace"] = z["lower_trace"][:, :, lane:lane+1]
    scores, parts, closure = replay_rewards(data, config)
    valid = data["valid"][:, 0].astype(bool)
    steps = np.flatnonzero(valid)
    arrays = {key:val.reshape(-1, 4).sum(axis=1)[valid] for key,val in parts.items()}
    arrays.update(policy_step=steps+1, total=scores.reshape(-1, 4).sum(axis=1)[valid])
    arrays["cumulative"] = np.cumsum(arrays["total"])
    return arrays, closure


def zero_rewards(row, alpha):
    raw, lane = Path(row["raw"]), int(row["lane"])
    stem = ("alpha0", "alpha025", "alpha05", "alpha075", "alpha1")[ALPHAS.index(alpha)]
    file = raw / (stem + "_zero_rescored.npz")
    with TemporaryDirectory() as tmp:
        if not file.is_file():
            # The shared rescorer saves beside its inputs. Point it at an isolated
            # directory so that no evaluation record is edited.
            for name in ("trajectory.npz", "contract.json"):
                (Path(tmp) / name).symlink_to((raw / name).resolve())
            os.environ["JAX_PLATFORMS"] = "cpu"
            write_zero_scores(Path(tmp))
            file = Path(tmp) / file.name
        with np.load(file) as saved:
            t, reward = saved["time"], saved["reward"]
    with np.load(raw / "trajectory.npz") as z:
        x = z["lower_trace"][:, :, lane].reshape(-1, 11)
        lower_ids = np.flatnonzero(x[:, 6])
        np.testing.assert_allclose(t, lower_ids*.005 + x[lower_ids, 7]*.0002, atol=1e-7)
        valid = z["valid"][:, lane].astype(bool)
        steps = np.flatnonzero(valid)
        total = np.bincount(lower_ids//4, weights=reward, minlength=len(valid))
        failure = z["failure"][:, lane].astype(bool)
        # A pre-step failure can have zero active 200 Hz samples in its last 50 Hz
        # interval. The policy-rate log always carries the alpha-independent penalty.
        total[failure] = z["reward"][failure, lane]
    return dict(policy_step=steps+1, total=total[valid], cumulative=np.cumsum(total[valid]))


def frontier(rows):
    eligible = [r for r in rows if int(r["full"]) and int(r["hold"])]
    points = np.array([(float(r["peak_path_m"]), float(r["speed_rmse_m_s"])) for r in eligible])
    front = [r for r, p in zip(eligible, points)
             if not np.any(np.all(points <= p, axis=1) & np.any(points < p, axis=1))]
    return eligible, front


def archive_lanes(rows, output):
    """One compact file pair per rollout, retaining the direct link to its full raw lane."""
    output.mkdir(exist_ok=True)
    sources = defaultdict(list)
    for row in rows:
        sources[row["raw"]].append(row)
    index = []
    for raw, batch in sources.items():
        raw = Path(raw)
        c = json.loads((raw / "contract.json").read_text())
        cfg = json.loads((Path(c["bundle"]) / "config.json").read_text())
        with np.load(raw/"trajectory.npz") as z:
            archive = {k:z[k] for k in z.files if k.startswith("effective_") or
                       k in ("valid","failure","lower_trace","reward","offsets")}
            components, _, _ = replay_rewards(archive, cfg)
            policy_scores = components.reshape(-1, 4, len(c["settings"]["alphas"])).sum(axis=1)
            for r in batch:
                lane = int(r["lane"])
                trace = archive["lower_trace"][:, :, lane].reshape(-1, 11)
                ids = np.flatnonzero(trace[:, 6])
                t = ids * c["lower_dt"] + trace[ids, 7] * c["physics_dt"]
                steps = np.flatnonzero(archive["valid"][:, lane])
                scores = policy_scores[steps, lane]
                cumulative = np.cumsum(scores)
                stem = f'g{int(r["generation"]):02d}_c{int(r["candidate"]):02d}'
                np.savez_compressed(output/(stem+".npz"), time_s=t, xy_m=trace[ids,:2],
                                    true_v_m_s=trace[ids,2], cross_track_m=trace[ids,3],
                                    delivered_speed_steer=trace[ids,8:10], policy_step=steps+1,
                                    total_reward=scores, cumulative_reward=cumulative)
                np.savetxt(output/(stem+".csv"), np.column_stack((steps+1,scores,cumulative)),
                           delimiter=",", header="policy_step,total_reward,cumulative_reward",
                           comments="", fmt=("%d","%.9g","%.9g"))
                index.append(dict(**r, logged_alpha=c["settings"]["alphas"][lane], npz=stem+".npz", csv=stem+".csv"))
    table(output/"index.csv",index)


def report_case(name, rows, out):
    out.mkdir(parents=True, exist_ok=True)
    if "physical_slot" in rows[0] and len({r["physical_slot"] for r in rows})!=1:
        raise ValueError(f"{name}: warm and CEM trajectories occupy different physical slots")
    archive_lanes(rows, out/"rollouts")
    warm = [r for r in rows if int(r["generation"]) == 0]
    zero, = (r for r in warm if r["method"].lower() == "zero")
    fixed = sorted((r for r in warm if r is not zero), key=lambda r: float(r["method"].removeprefix("fixed")))
    fixed_by_alpha = {float(r["method"].removeprefix("fixed")):r for r in fixed}
    if sorted(fixed_by_alpha) != list(ALPHAS):
        raise ValueError(f"{name} requires a generation-zero result for each frozen preference")
    best = min(rows, key=lambda r: float(r["V"]))
    best_fixed = min(fixed, key=lambda r: float(r["V"]))
    alpha = float((best_fixed if best is zero else best)["alpha_seed"])
    paired = fixed_by_alpha[alpha]
    contract = json.loads((Path(zero["raw"]) / "contract.json").read_text())
    with np.load(Path(contract["bundle"]) / "route.npz") as z:
        route = {k:z[k] for k in ("xy", "s", "turn_end")}
    terrain_path = Path(next(iter(contract["settings"]["terrains"].values())))
    terrain = contract["terrain"][int(zero["lane"])]
    data = {}
    events = {}
    for row in warm+[best]:
        if id(row) not in data:
            data[id(row)], events[id(row)] = physical_lane(row, route, terrain)
    fig, ax = plt.subplots(figsize=(7.4, 6), layout="constrained")
    if terrain.get("height_m", 0) > 0:
        with np.load(terrain_path / "terrain.npz") as z:
            vertices = z["vertices"]
        raised = vertices[vertices[:, 2] > 0]
        ax.scatter(*raised[:, :2].T, s=3, color="#c69d53", linewidths=0, label="Raised mesh vertices")
    end = max(d["progress_m"].max() for d in data.values())
    ax.plot(*route["xy"][route["s"]<=end+.25].T, "--", color="0.35", lw=1.5, label="Original route")
    for row, color in zip(fixed+[zero,best], COLORS+("black",)):
        d = data[id(row)]
        # Draw the oracle separately even when a zero-bias warm start is retained.
        oracle = color == "black"
        label = (f'Oracle incumbent (gen {row["generation"]})' if oracle else row["method"])
        ax.plot(d["world_x_m"], d["world_y_m"], color=color, lw=1.5, ls=":" if oracle else "-", label=label)
        ax.plot(d["world_x_m"][-1], d["world_y_m"][-1], marker="o" if int(row["full"]) else "x",
                color=color, ms=5, fillstyle="none")
    ax.set(aspect="equal", xlabel="World x (m)", ylabel="World y (m)", title=name+" · H75 + Local300")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01,1), loc="upper left", title="○ full 10 s   × early endpoint")
    finish(fig, out/"xy.png")

    eligible, pareto = frontier(rows)
    columns = list(rows[0])
    table(out/"frontier.csv", pareto, columns)
    fig, ax = plt.subplots(figsize=(8.2, 5), layout="constrained")
    for task, color in zip("ABC", ("#0072b2", "#966100", "#008060")):
        w, h = GATES[task]
        ax.add_patch(Rectangle((0,0), w,h,fill=False, edgecolor=color, linestyle="--", lw=1.4))
        ax.annotate(task,(w,h),ha="right", va="bottom",color=color)
    def points(records, **kwargs):
        ax.scatter([float(r["peak_path_m"]) for r in records],
                   [float(r["speed_rmse_m_s"]) for r in records], **kwargs)
    points(eligible, color="0.75", s=18, label=f"Full + hold ({len(eligible)})")
    points(pareto, marker="D", facecolors="none", edgecolors="#b31b4b", s=62,
           label=f"Nondominated ({len(pareto)})")
    for row, color in zip(fixed, COLORS):
        good = int(row["full"]) and int(row["hold"])
        points([row], marker="o" if good else "x", color=color, s=32,
               label=f'H75 α={row["method"].removeprefix("fixed")} · {row["terminal"]}'+("" if good else " / ineligible"))
    ax.set(xlim=(0,None), ylim=(0,None), xlabel="Peak |cross-track error| (m)", ylabel="Speed RMSE (m/s)",
           title=name+" · reachable archive\nBoxes are conditional on full 10 s and final hold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01,1), loc="upper left")
    finish(fig, out/"frontier.png")

    fig, axes = plt.subplots(4, 2, figsize=(12,10), layout="constrained", sharex=True)
    fields = [("bias_v_m_s","Δv (m/s)"),("bias_steer_rad","Δδ (rad)"),
              ("delivered_speed_m_s","Delivered speed target (m/s)"),("delivered_steer_rad","Delivered steer target (rad)"),
              ("speed_m_s","Actual speed (m/s)"),("steer_rad","Actual steer angle (rad)"),
              ("roll_rad","Actual roll (rad)")]
    record = {}
    for row,label,color,ls in [(best_fixed,"best_fixed","#0072b2","-"),(best,"oracle","#ab3c00","--")]:
        d,e = data[id(row)], events[id(row)]
        for (key,ylabel),ax in zip(fields,axes.flat):
            clock = d["bias_time_s"] if key.startswith("bias") else d["time_s"]
            ax.plot(clock, d[key], ls, color=color, lw=1.1, label=label,
                    drawstyle="steps-post" if key.startswith("bias") else "default")
            ax.set_ylabel(ylabel)
        ax = axes[-1,-1]
        for key,marker in [("front_normal_resultant_N","o"),("rear_normal_resultant_N","x")]:
            ax.plot(d["time_s"],d[key],ls,color=color,label=label+" "+key.split("_")[0],
                    marker=marker,ms=3,markevery=max(len(d[key])//14,1),lw=.9)
        ax.set_ylabel("Wheel normal resultant norm (N)")
        if e["first_nonzero_bias_s"] is not None:
            for sub in axes[0]:
                sub.axvline(e["first_nonzero_bias_s"], color=color, lw=1.1, ls="-.")
        for sub in axes.flat:
            if e["entry_s"] is not None:
                sub.axvline(e["entry_s"], color=color, lw=.8, ls=":")
        for key,arr in d.items():
            record[label+"_"+key]=arr
    np.savez_compressed(out/"selected_traces.npz", **record)
    table(out/"entry_events.csv",[dict(series=s,**events[id(r)]) for s,r in (("best_fixed",best_fixed),("oracle",best))])
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    axes[0,0].legend(fontsize=8)
    axes[-1,-1].legend(fontsize=7,ncol=2)
    onset = events[id(best)]["first_nonzero_bias_s"]
    lead = events[id(best)]["lead_before_entry_s"]
    if onset is None:
        note = "Incumbent retains zero bias throughout"
    else:
        note = f"First nonzero bias at {onset:.3f} s"
        if lead is not None:
            note += f" ({abs(lead):.3f} s {'before' if lead >= 0 else 'after'} measured entry)"
    fig.suptitle(name+" · best fixed H75 and oracle incumbent\n"+note+" · dotted: front-wheel entry")

    finish(fig,out/"timeline.png")

    scores, closure = policy_rewards(best)
    paired_scores, paired_closure = policy_rewards(paired)
    zero_scores = zero_rewards(zero, alpha)
    if best is zero:
        scores = zero_scores
    curves = {"H75":paired_scores, "Zero":zero_scores, "Oracle":scores}
    csv_rows = []
    fig,axes=plt.subplots(2,1,figsize=(8.6,6),layout="constrained",sharex=True)
    for (label,series),color,ls in zip(curves.items(),("#0072b2","#696969","#b93f00"),("-","--",":")):
        for ax,key in zip(axes,("total","cumulative")):
            ax.plot(series["policy_step"],series[key],ls,color=color,lw=1.3,label=label)
        for step,total,cumulative in zip(*(series[k] for k in ("policy_step","total","cumulative"))):
            csv_rows.append(dict(series=label,alpha=alpha,policy_step=step,total=total,cumulative=cumulative))
    table(out/"reward.csv",csv_rows)
    np.savez_compressed(out/"reward_parts.npz", **{k+"_"+s:a for k,v in curves.items() for s,a in v.items()})
    axes[0].set_ylabel("Reward per 50 Hz step")
    axes[1].set(xlabel="Policy step",ylabel="Cumulative reward")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{name} · same-α scoring, α={alpha:g} · terminal penalties included")
    finish(fig,out/"reward.png")
    return dict(case=name,candidates=len(rows),eligible=len(eligible),pareto=len(pareto),method=best["method"],
                generation=int(best["generation"]),candidate=best["candidate"],alpha=alpha,V=float(best["V"]),
                A=int(best["A"]),B=int(best["B"]),C=int(best["C"]),fixed_V=float(best_fixed["V"]),
                fixed_C=int(best_fixed["C"]),raw=best["raw"],lane=int(best["lane"]),physical_slot=best.get("physical_slot",""),
                reward_closure_max=max(closure,paired_closure),**{k:v for k,v in events[id(best)].items() if k!="method"})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",type=Path,help="E1a or E1b directory containing candidates.csv")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    stage=args.stage.resolve()
    out=(args.output or stage/"analysis").resolve()
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    groups=defaultdict(list)
    with (stage/"candidates.csv").open() as f:
        for row in csv.DictReader(f):
            groups[row["case"]].append(row)
    result=[report_case(case,rows,out/case) for case,rows in groups.items()]
    table(out/"cases.csv",result)
    batches = []
    timing_files = sorted([*stage.glob("warm/w*/timing.json"),*stage.glob("g[0-9][0-9]/c*/timing.json")])
    for path in timing_files:
        values = json.loads(path.read_text())
        values["queue_setup_and_process_overhead_s"] = values["wall_s"]-values["sampling_s"]-values["export_s"]
        batches.append(dict(batch=str(path.parent.relative_to(stage)), **values))
    lines=["# E1 可行性搜索档案","","每一行对应冻结 H75、Local300 和原始参考下的一个布局/物理组。XY 包含五档 H75、Zero 与搜索返回的最佳候选。","",
           "|案例|实际/参考 XY|可达前沿|入障时序|同 α 奖励|候选数|满窗且保持|非支配|最佳代 / V|固定档 C → 最佳 C|",
           "|---|---|---|---|---|---:|---:|---:|---|---|"]
    for r in result:
        case=r["case"]
        lines.append(f'|{case}|[XY]({case}/xy.png)|[Ep/Ev]({case}/frontier.png) · [CSV]({case}/frontier.csv)|[时间轴]({case}/timeline.png) · [入障提前量]({case}/entry_events.csv)|[每步与累计]({case}/reward.png) · [CSV]({case}/reward.csv)|[{r["candidates"]} 条索引]({case}/rollouts/index.csv)|{r["eligible"]}|{r["pareto"]}|{r["generation"]} / {r["V"]:.4f}|{r["fixed_C"]} → {r["C"]}|')
    lines+=["","本阶段每个案例固定一个物理槽。各匹配热启动和 CEM 候选由相同的案例次序、槽位、地形集合与世界坐标组成的独立批次产生。每代 32 个候选全部完成后才更新分布。前沿仅在满 10 秒且末秒保持的实际候选中计算，图中提前终止的 H75 点仅描述其观察窗口。A/B 使用 C 搜索档案事后评分。第 0 代代表固定档或 Zero 热启动。", "",
            "时间轴的前后轮接触量为各轮法向接触力向量之和的欧氏范数，单位 N。入障提前量取各自的前轮上游几何边界越过时刻减首次施加非零偏置时刻。平地、碎石、未越界或全零偏置的对应字段留空。", "",
            "每条候选均有单独的压缩 NPZ（200 Hz 轨迹、真实速度、交付参考与 50 Hz 奖励）及逐步奖励 CSV。各自按生成时实际 α 计分。索引的 raw、lane、knots_npz、knots_lane 定位完整原始轨迹与对应偏置。各案例的 selected_traces.npz 保存最佳候选与最佳固定档的时间序列，reward_parts.npz 保存重建奖励。", "",
            f'[候选索引]({os.path.relpath(stage/"candidates.csv",out)}) · [逐案例指标](cases.csv) · [分析与绘图墙钟](timing.json)。']
    if stage.name == "E1b":
        pilot = stage.parent / "E1a"
        pilot_report = pilot / "analysis"
        if not (pilot_report / "cases.csv").is_file():
            pilot_groups = defaultdict(list)
            with (pilot / "candidates.csv").open() as f:
                for row in csv.DictReader(f):
                    pilot_groups[row["case"]].append(row)
            pilot_report.mkdir(parents=True, exist_ok=True)
            pilot_rows = [report_case(k, v, pilot_report/k) for k,v in pilot_groups.items()]
            table(pilot_report/"cases.csv", pilot_rows)
        with (pilot_report / "cases.csv").open() as f:
            combined = [dict(row, stage="E1a", case_dir=os.path.relpath(pilot_report/row["case"],out))
                        for row in csv.DictReader(f)]
        combined += [dict(row,stage="E1b",case_dir=row["case"]) for row in result]
        if len({r["case"] for r in combined}) != 24 or len(combined) != 24:
            raise ValueError("Final comparison requires the registered 24 distinct layout/group cases")
        table(out/"all_cases.csv",combined)
        rough = [r for r in combined if not r["case"].startswith("flat_original")]
        fixed_passes = sum(int(r["fixed_C"]) for r in rough)
        any_passes = sum(int(r["C"]) for r in rough)
        lines += ["", "## E1a 与 E1b 全部 24 例", "",
                  f"20 个非平地案例中，固定 H75 的最小 V 候选通过 {fixed_passes}/20，包含热启动的完整档案通过 {any_passes}/20。"
                  "偏移稳健性由 E1c 检验。", "",
                  "|案例|来源|实际/参考 XY|可达前沿|固定档 C|档案 C|最佳代|",
                  "|---|---|---|---|---:|---:|---:|"]
        for r in combined:
            d=r["case_dir"]
            lines.append(f'|{r["case"]}|{r["stage"]}|[XY]({d}/xy.png)|[Ep/Ev]({d}/frontier.png)|{r["fixed_C"]}|{r["C"]}|{r["generation"]}|')
        lines += ["", "[24 例汇总及证据路径](all_cases.csv)"]
    (out/"README.md").write_text("\n".join(lines)+"\n")
    (out/"timing.json").write_text(json.dumps(dict(analysis_and_plot_s=time.monotonic()-start,
                                                   batches=batches),indent=2)+"\n")


if __name__=="__main__":
    main()
