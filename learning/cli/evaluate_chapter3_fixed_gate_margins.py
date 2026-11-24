"""Prepare, run and score the fixed-entry three-exit Chapter 3 margin comparison."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

ARMS = ("dynamic", "kinematic")
ROLES = ("near", "mid", "far")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def notice(out, phase, message):
    script = Path(__file__).resolve().parents[1] / "scripts/notify_chapter3_job.py"
    subprocess.run(["/usr/bin/python3", str(script), "--message", message, "--title", "第三章 B2 终审",
                    "--receipt", str(out / f"{phase}_bark.json")], check=False)


def freeze(out, p1, margin_parent):
    """Freeze the full pairing independently of the TEST geometry and its outcomes."""
    out.mkdir(parents=True, exist_ok=True)
    split_path = p1 / "split.csv"
    registered = read(p1 / "split_receipt.json")
    with split_path.open() as stream:
        split = [row for row in csv.DictReader(stream) if row["role"] == "sealed_test"]
    root0, root1 = registered["sealed_test_roots"]
    if len(split) != 64 or {int(r["bank_row"]) for r in split} != set(range(root0, root1 + 1)):
        raise ValueError("Incomplete sealed initial-state reservation")
    protocol = read(margin_parent / "protocol.json")
    original = read(protocol["parent_manifest"])
    direct = protocol["margin_comparison"]
    actions = read(p1 / "actions/receipt.json")["old_actions"]
    maps = read(p1 / "maps/manifest.json")["records"]
    if actions[4] != [1, 1.] or actions[0] != [0, 0.]:
        raise ValueError("Fixed rule action slots differ from the registered action bank")
    initial = p1 / "initial_bank.npz"
    if digest(initial) != registered["prepared_bank_sha256"]:
        raise ValueError("Initial-state bank digest differs from its registration")
    tasks = []
    for reset, force, (gate, section), side in product(split, original["forces_N"],
                                                       zip(ROLES, original["section_x"], strict=True), (1, -1)):
        slot = 4 if side == 1 else 0
        geometry, alpha = actions[slot]
        tasks.append(dict(map_slot=int(reset["map_slot"]), root=int(reset["bank_row"]),
                          side=side, gate=gate, gate_x=section, force=force,
                          action_slot=slot, geometry=geometry, alpha=alpha))
    if len(tasks) != 1152 or any(sum(r["map_slot"] == i for r in tasks) != 288 for i in range(4)):
        raise ValueError("Expected four TEST slots with 288 paired gate tasks each")
    contract = dict(stage="B2", decision="Fixed entry and three original closed-loop exits",
                    map_slots=4, roots_per_map=16, tasks_per_arm=1152, physical_rollouts=2304,
                    gate_sections_m=original["section_x"], hidden_forces_N=original["forces_N"],
                    margin_m=dict(dynamic=direct["tested_threshold_m"], kinematic=direct["parent_threshold_m"]),
                    fallback="straight if none of three passes, 15 ms at entry and after measured relock",
                    search_horizon_cap_s=5.1, expansion=False, training=0,
                    p1=str(p1), margin_parent=str(margin_parent), bank=str(initial),
                    bank_sha256=digest(initial), split_sha256=digest(split_path),
                    source_config=str(margin_parent / "right_offset/part_0/run.json"),
                    source_config_sha256=digest(margin_parent / "right_offset/part_0/run.json"),
                    known_non_test_worlds=[x["sha256"]["world"] for x in maps],
                    comparison="whole-run physical crossing, relock, no forbidden contact, designated gate and goal hold",
                    clearance="secondary post-relock sampled body/obstacle envelope at a common 5.33 cm cutoff",
                    single_use="one registered evaluation of each map, force, root, gate and margin")
    if (out / "protocol.json").exists():
        raise FileExistsError(out / "protocol.json")
    table(out / "tasks.csv", tasks)
    contract["tasks_sha256"] = digest(out / "tasks.csv")
    save(out / "protocol.json", contract)


def prepare(out, manifest):
    """Bind only the four registered TEST geometries to the already frozen root slots."""
    p = read(out / "protocol.json")
    source = read(p["source_config"])
    if digest(p["source_config"]) != p["source_config_sha256"] or digest(out / "tasks.csv") != p["tasks_sha256"]:
        raise ValueError("B2 registration or the frozen native source configuration changed")
    records = read(manifest)["records"]
    records = sorted((r for r in records if r["role"] == "sealed_test"), key=lambda r: r["map_slot"])
    if [m["map_slot"] for m in records] != [0, 1, 2, 3]:
        raise ValueError("P2 requires four sealed_test map records in slots 0 through 3")
    if (out / "release.json").exists():
        raise FileExistsError(out / "release.json")
    with (out / "tasks.csv").open() as stream:
        tasks = list(csv.DictReader(stream))
    batches = []
    previous = read(Path(p["p1"]) / "maps/manifest.json")["records"]
    excluded = read(Path(p["p1"]) / "maps/protocol.json")["exclude_worlds"].values()
    used_terrain = {
        json.dumps(read(path)["terrain"], sort_keys=True)
        for path in [*(m["world"] for m in previous), *(m["path"] for m in excluded)]
    }
    for rec in records:
        slot = rec["map_slot"]
        subset = [t for t in tasks if int(t["map_slot"]) == slot]
        if set(map(int, rec["roots"])) != {int(t["root"]) for t in subset}:
            raise ValueError(f"Map {slot} carries different roots")
        for key in ("world", "mjb", "raster"):
            if digest(rec[key]) != rec["sha256"][key]:
                raise ValueError(f"Map {slot}, {key} differs from its freeze")
        terrain = json.dumps(read(rec["world"])["terrain"], sort_keys=True)
        if terrain in used_terrain:
            raise ValueError(f"Map {slot} reuses existing terrain geometry")
        used_terrain.add(terrain)
        for arm in ARMS:
            cfg = dict(source)
            cfg.pop("p0_branches")
            hand = dict(source["ground_handoff"])
            variants, lanes = [], []
            for t in subset:
                g, root = int(t["geometry"]), int(t["root"])
                variants.append(dict(hand["handoff_variants"][0], root_variant=root, bank_row=root,
                                data_split="sealed_test", planner="C-R", geometry=g, arm=arm, bin=t["gate"],
                                side=int(t["side"]), section_x=float(t["gate_x"]), force_y_N=float(t["force"]),
                                planned_departure=1, departure="straight", approach_alpha=float(t["alpha"])))
                lanes.append([g, "h75", len(lanes)])
            hand["handoff_variants"] = variants
            online = dict(source["platform_online"], margin_m=p["margin_m"][arm])
            online["relative"] = dict(online["relative"], raster=rec["raster"])
            cfg.update(role="Frozen B2 fixed entry / three-exit filter", world_map=rec["world"],
                       approach_initial_bank=p["bank"], ccd_capacity=8*len(lanes), evaluation_lanes=lanes,
                       platform_online=online, ground_handoff=hand,
                       paired_departures=dict(source["paired_departures"], direction_ids=[1]*len(lanes)))
            loc = out / arm / f"slot_{slot}"
            loc.mkdir(parents=True, exist_ok=True)
            save(loc / "config.json", cfg)
            batches.append(dict(arm=arm, slot=slot, config=str(loc/"config.json"),
                                sha256=digest(loc/"config.json"), mjb_sha256=rec["sha256"]["mjb"]))
    src = Path(__file__).resolve().parents[1]
    native_source = [
        src/"cli/evaluate_playground_handoff.py", src/"src/sttw_control/playground_handoff_evaluation.py",
        src/"src/sttw_control/platform_online.py", src/"src/sttw_control/playground_prediction.py",
        src/"cli/analyze_chapter3_continuous_p0_branches.py", src/"cli/report_platform_controls.py",
        src/"cli/run_playground_task_reversal.py",
    ]
    plan_files = [p["bank"], source["ground_handoff"]["reference_path"], source["paired_departures"]["route_bank"],
                  source["paired_departures"]["approach_file"], source["platform_online"]["relative"]["initializer"]]
    save(out / "release.json", dict(geometry_manifest=str(manifest), sha256=digest(manifest),
                                    source_sha256=digest(__file__), batches=batches,
                                    assets={str(f): digest(f) for f in [*native_source, *plan_files]}))


def worker(out, arm, slot):
    """Use the native query at true relock and the old filter's 5.1 s horizon and veto rule."""
    from sttw_control import platform_online, playground_handoff_evaluation as simulator
    import evaluate_playground_handoff as entry

    release = read(out / "release.json")
    b = next(b for b in release["batches"] if (b["arm"], b["slot"]) == (arm, slot))
    loc = Path(b["config"]).parent
    if digest(b["config"]) != b["sha256"] or digest(__file__) != release["source_sha256"]:
        raise ValueError("Released B2 worker or arm configuration changed")

    class FrozenThreeExits(platform_online.PlatformDepartureQuery):
        def choose(self, q, body_speed, request, route, branch_ids, section, side, old,
                   seconds, velocity=None, previous=None):
            choice, admitted, incumbent = super().choose(q, body_speed, request, route, branch_ids,
                section, side, old, min(5.1, seconds), velocity, previous)
            return choice.where(admitted, old), admitted, incumbent

    platform_online.PlatformDepartureQuery = FrozenThreeExits
    native = simulator.evaluate_handoff

    def checked(model, *args):
        if digest(loc/"physical/terrain.mjb") != b["mjb_sha256"]:
            raise ValueError("Compiled physical terrain differs from its pre-outcome registration")
        return native(model, *args)

    simulator.evaluate_handoff = checked
    sys.argv = [entry.__file__, "--config", b["config"], "--output", str(loc/"physical"), "--skills",
                "rb_run1", "--modes", "h75", "--routes", "0", "1", "--physics-substeps", "5", "--seconds", "10"]
    entry.main()


def run(out):
    """One 1,152-environment margin arm at a time, under the project GPU lease."""
    release = read(out/"release.json")
    if (out/"started.json").exists() or any((Path(b["config"]).parent/"physical").exists() for b in release["batches"]):
        raise FileExistsError("A sealed evaluation already started; inspect its receipt and outputs")
    worktree=Path(__file__).resolve().parents[2]
    environment=dict(os.environ, CUDA_VISIBLE_DEVICES="0", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                     OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1",
                     PYTHONPATH=os.pathsep.join(map(str,[worktree/"learning/src",worktree/"learning/cli",
                                                        "/home/qy/DVGC/JIT/src"])))
    environment.pop("JAX_PLATFORMS",None)
    import fcntl
    lock_path=Path("/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock")
    with lock_path.open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if any(digest(path)!=expected for path,expected in release["assets"].items()):
            raise ValueError("A frozen input or native runner changed before B2 launch")
        if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip():
            raise RuntimeError("Another GPU compute process is active")
        with (out/"started.json").open("x") as f:
            json.dump(dict(release_sha256=digest(out/"release.json"), pid=os.getpid()), f)
        def launch(batch):
            loc=Path(batch["config"]).parent
            start=time.time_ns()
            with (loc/"worker.log").open("w") as log:
                subprocess.run([sys.executable,__file__,"worker",str(out),batch["arm"],str(batch["slot"])],
                               check=True,stdout=log,stderr=subprocess.STDOUT,cwd=worktree,env=environment)
            save(loc/"wall_clock.json",dict(start_ns=start,end_ns=time.time_ns(),worlds=288))
        for arm in ARMS:
            notice(out, arm+"_start", f"{arm} 裕量四图批次启动，1,152 车。")
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(launch, [b for b in release["batches"] if b["arm"]==arm]))
            notice(out, arm+"_done", f"{arm} 裕量四图物理批次完成。")
    subprocess.run([sys.executable,__file__,"score",str(out)],check=True,cwd=worktree,
                   env=dict(environment,CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu"))


def score(out):
    import mujoco
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sttw_control.playground_prediction import BodyEnvelope
    from analyze_chapter3_continuous_p0_branches import prefix_flight_force
    from run_playground_task_reversal import crossings_by_section

    torch.set_num_threads(1)
    plan=read(out/"protocol.json"); result=[]; paths={}
    for b in read(out/"release.json")["batches"]:
        loc=Path(b["config"]).parent/"physical"; status=read(loc/"status.json"); cfg=read(b["config"])
        if status["stage"]!="complete" or len(status["results"])!=288:
            raise ValueError(f"Missing physical outcomes at {loc}")
        terrain=read(cfg["world_map"])["terrain"]; model=mujoco.MjModel.from_binary_path(str(loc/"terrain.mjb"))
        env=BodyEnvelope(model,dimensions=3,device="cpu",dtype=torch.float32)
        wall=[j for j in range(model.ngeom) if model.geom(j).name.startswith("planning_obstacle_")]
        center=torch.tensor(model.geom_pos[wall],dtype=torch.float32); half=torch.tensor(model.geom_size[wall],dtype=torch.float32)
        for lane,row in enumerate(status["results"]):
            v=row["variant"]; key=(b["slot"],v["bank_row"],v["force_y_N"],v["bin"],v["side"])
            raw=loc/row["case_dir"]/"trajectory.npz"; hand=row["handoff_s"]; minimum=None
            jump=prefix_flight_force(row,raw,model,terrain)["verified"]; coll=bool(row["forbidden_contact_steps"])
            with np.load(raw) as z:
                trace=z["trace"]; gate=next(c for c in crossings_by_section(trace,hand if hand is not None else np.inf,
                                            plan["gate_sections_m"]) if c[0]==v["bin"])
                passed=gate[3] is not None and .06<=v["side"]*gate[3]<=3.9
                prefix_end=len(trace) if hand is None else round(hand/.005)+2
                prefix=hashlib.sha256()
                for field in ("qpos","qvel"):
                    prefix.update(z[field][:prefix_end].tobytes())
                if hand is not None:
                    q=z["qpos"][round(hand/.005)+1:]; pose=torch.tensor(q[:,[0,1,2,3,4,5,6,8,9,7,10,11]],dtype=torch.float32)
                    if len(pose):
                        lo,hi=env(pose)
                        gap=torch.maximum(lo[:,:,None]-(center+half),(center-half)-hi[:,:,None]).amax(-1)
                        minimum=float(gap.amin())
                success=bool(jump and passed and row["goal"] and not coll)
                paths[key,b["arm"]]=(trace[:,1:3].copy(),z["reference_xy"].copy(),z["published_reference_xy"].copy())
            result.append(dict(slot=key[0],root=key[1],force=key[2],gate=key[3],side=key[4],arm=b["arm"],
                  lane=lane,verified_jump=jump,cross_y=gate[3],goal=bool(row["goal"]),wall=coll,
                  eligible=success,strict=bool(success and minimum is not None and minimum>plan["margin_m"]["dynamic"]),
                  observed_clearance_m=minimum,prefix_sha256=prefix.hexdigest(),terminal=row["termination"],end_s=row["duration_s"],
                  hold_s=row["goal_hold_s"],actual=str(raw),commanded_departure=row["selected_departure"]))
    df=pd.DataFrame(result);df.to_csv(out/"episodes.csv",index=False)
    ids=["slot","root","force","gate","side"]; first=df[df.arm.eq("dynamic")].set_index(ids)
    baseline=df[df.arm.eq("kinematic")].set_index(ids).loc[first.index]
    if not first.index.is_unique or len(first)!=1152 or len(baseline)!=1152:
        raise ValueError("Incomplete paired tasks")
    paired=pd.DataFrame({k+"_"+a:frame[k] for a,frame in [("dynamic",first),("kinematic",baseline)]
                       for k in ["eligible","strict","wall","terminal","actual"]})
    paired["rescued"]=paired.eligible_dynamic & ~paired.eligible_kinematic
    paired["lost"]=~paired.eligible_dynamic & paired.eligible_kinematic
    paired["matched_prefix"]=first.prefix_sha256.eq(baseline.prefix_sha256)
    paired.to_csv(out/"pairs.csv")
    cols=["eligible_dynamic","eligible_kinematic","rescued","lost","wall_dynamic","wall_kinematic","strict_dynamic","strict_kinematic"]
    cases=paired.reset_index()
    group=cases.groupby(["slot","gate","side"])
    totals=group[cols].sum();totals["n"]=group.size();totals.to_csv(out/"per_map_gate.csv")
    parent=read(Path(plan["margin_parent"])/"protocol.json")
    seed=read(parent["parent_manifest"])["root_bootstrap_seed"]
    draws=np.random.default_rng(seed).integers(0,16,size=(10000,4,16))
    intervals={}
    for label, measure, side in (("success","eligible",None),("left_success","eligible",1),
                                 ("right_success","eligible",-1),("wall","wall",None)):
        part=cases if side is None else cases[cases.side.eq(side)]
        dif=part[f"{measure}_dynamic"].astype(int)-part[f"{measure}_kinematic"].astype(int)
        roots=dif.groupby([part.slot,part.root],sort=True).mean().to_numpy().reshape(4,16)
        sampled=roots[np.arange(4)[None,:,None],draws].mean((1,2))
        intervals[label]=dict(difference=float(dif.mean()),paired_95=np.quantile(sampled,[.025,.975]).tolist(),
                              lower_onesided_95=float(np.quantile(sampled,.05)),
                              upper_onesided_95=float(np.quantile(sampled,.95)))
    gallery=out/"xy";gallery.mkdir(exist_ok=True); index=[]
    for (slot,root,gate),group in paired.reset_index().groupby(["slot","root","gate"]):
        fig,axes=plt.subplots(2,3,figsize=(11,6),sharex=True,sharey=True,layout="constrained")
        for ax,(_,r) in zip(axes.flat,group.sort_values(["side","force"]).iterrows(),strict=True):
            key=(slot,root,r.force,gate,r.side)
            for arm,color in zip(ARMS,["#0868ac","#bb4411"]):
                xy,ref,published=paths[key,arm]
                ax.plot(*xy.T,color=color,label=arm);ax.plot(*published.T,color=color,ls=":",lw=.7)
                ax.scatter(*xy[-1],c=color,marker="x")
            ax.plot(*ref.T,color=".4",ls="--",lw=.7,label="approach");ax.set_aspect("equal",adjustable="box")
            ax.set_title(f"{'left' if r.side==1 else 'right'}, {r.force:+g} N");ax.set_xlabel("x (m)");ax.set_ylabel("y (m)")
        fig.suptitle(f"Map {slot}, initial state {root}, {gate} gate");axes[0,0].legend(fontsize=7)
        file=gallery/f"slot{slot}_root{root}_{gate}.png";fig.savefig(file,dpi=120);plt.close(fig)
        index.append(f"- [{slot} / {root} / {gate}](xy/{file.name})")
    (out/"XY.md").write_text("# Actual and reference trajectories\n\nSolid curves show measured motion. Dotted curves show the last published references. Crosses mark measured endpoints.\n\n"+"\n".join(index)+"\n")
    save(out/"summary.json",dict(pairs=len(first),**{k:int(paired[k].sum()) for k in cols},
                                 bitwise_matched_prefixes=int(paired.matched_prefix.sum()),
                                 matched_prefix_counts={k:int(paired.loc[paired.matched_prefix,k].sum()) for k in cols},
                                 bootstrap=dict(draws=10000,seed=seed,clusters="16 roots within each of the four fixed maps",
                                                dynamic_minus_kinematic=intervals)))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    for name in ["freeze","prepare","run","worker","score"]:
        s=sub.add_parser(name);s.add_argument("output",type=Path)
        if name=="freeze":
            s.add_argument("p1",type=Path);s.add_argument("margin_parent",type=Path)
        if name=="prepare":s.add_argument("geometry_manifest",type=Path)
        if name=="worker":
            s.add_argument("arm",choices=ARMS);s.add_argument("slot",type=int)
    a=p.parse_args();out=a.output.resolve()
    if a.command=="freeze":freeze(out,a.p1.resolve(),a.margin_parent.resolve())
    elif a.command=="prepare":prepare(out,a.geometry_manifest.resolve())
    elif a.command=="worker":worker(out,a.arm,a.slot)
    elif a.command=="run":
        try:
            run(out)
        except Exception as exc:
            notice(out,"error",str(exc))
            raise
        notice(out,"complete","两种裕量的物理和配对分析完成。")
    else:score(out)


if __name__=="__main__":
    main()
