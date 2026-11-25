"""Pair a velocity-servo gain perturbation with the registered PCRP sensor experiment."""
import argparse
import csv
import os
import select
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from pcrp_contact_sensitivity import notice
from pcrp_corridor import digest
from pcrp_review_batches import LAB, load, write
from pcrp_sensor_feedback import ARMS, analyze as analyze_feedback

GAIN = 0.85
OBSERVATIONS = ("native", "loc10_delay30")


def prepare(root, parent):
    original = load(parent / "protocol.json")
    assert original["budget"] == 96 and original["physical_slots"] == 32
    root.mkdir(parents=True, exist_ok=False)
    inputs = dict(original["identities"])
    inputs[str(parent / "protocol.json")] = digest(parent / "protocol.json")
    jobs = []
    for level in OBSERVATIONS:
        dest = root / (level + "_gain85")
        dest.mkdir()
        cases = load(parent / level / "cases.json")
        for case in cases:
            gain_range = case["randomization"]["servo_gain_scale"]
            assert len(set(gain_range)) == 1
            case["randomization"]["servo_gain_scale"] = [gain_range[0] * GAIN] * 2
            case["output"] = str(dest / Path(case["output"]).name)
        write(dest / "cases.json", cases)
        inputs[str(dest / "cases.json")] = digest(dest / "cases.json")
        jobs.append(dict(config=str(parent / level / "settings.json"), terrain=cases[0]["terrain_path"],
                         case_manifest=str(dest / "cases.json"), output=str(dest / "batch")))
    write(root / "jobs.json", jobs)
    inputs[str(root / "jobs.json")] = digest(root / "jobs.json")
    write(root / "protocol.json", dict(
        parent=str(parent), bundle=original["bundle"], identities=inputs, factor=GAIN,
        code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        role="Paired simulation DEV. Two feedback settings crossed with group gains or 0.85 times both group gains.",
        cases=original["cases"], groups=original["groups"], labels=original["labels"],
        clocks=original["dt_s"], episode_s=10, new_episodes=64, reused_episodes=64, optimizer_updates=0,
        actuator="Scale the physical steering and rear explicit velocity-servo gains together. "
                 "Keep command signs, saturations, torque ceilings, delay and the frozen control software fixed. "
                 "This is a specified sensitivity test without hardware calibration.",
        reads="Compare actual final runtime gains from all four resident batches, torque and command limits, "
              "per-slot origins, terrain, initial state, sampled-feedback configuration and all other group parameters.",
        endpoints="Full ten seconds and final hold separately, gains and losses by condition, true roll stop, A/B/C, "
                  "true path and speed errors over common physical observation time, warning and per-step reward.",
        stop="Two 32-slot batches once, then four-cell report. Stop at first failure."))
    write(root / "status.json", dict(stage="prepared", new_episodes=64, reused_episodes=64))


def check_physics(level, contract, nominal):
    from analyze_matched_controls import check_pair

    stress = level.endswith("_gain85")
    scale = GAIN if stress else 1.
    settings = contract["settings"]
    nominal_random = nominal["settings"]["randomization"]
    expected = {**nominal_random, "servo_gain_scale":[x * scale for x in nominal_random["servo_gain_scale"]]}
    if settings["randomization"] != expected:
        raise ValueError(f"Group configuration differs outside the declared gain change in {level}")
    normalized = {**settings, "randomization":nominal_random}
    check_pair({**contract, "settings":normalized}, nominal)
    for key in ("origin", "delay_steps", "randomization_readback"):
        if contract[key] != nominal[key]:
            raise ValueError(f"Measured {key} differs in {level}")
    for key in ("lane_start", "lane_stop"):
        assert contract["batch_source"][key] == nominal["batch_source"][key]
    assert contract["lab_physics"]["terrain_layout"]["clone_origins"] == nominal["lab_physics"]["terrain_layout"]["clone_origins"]


def analyze(root):
    spec = load(root / "protocol.json")
    paper = str(Path(__file__).resolve().parents[2] / "docs/paper/english")
    sys.path.insert(0, paper)
    parent = Path(spec["parent"])
    levels = {}
    sources = {}
    for obs in OBSERVATIONS:
        levels[obs] = levels[obs + "_gain85"] = ARMS[obs]
        sources[obs] = parent / obs
        sources[obs + "_gain85"] = root / (obs + "_gain85")
        original = load(sources[obs] / "batch/resident_timing.json")["velocity_servos"]
        changed = load(sources[obs + "_gain85"] / "batch/resident_timing.json")["velocity_servos"]
        for key in ("order", "torque_limits_Nm", "velocity_limits_rad_s"):
            assert original[key] == changed[key], (obs, key)
        base, weak = (np.asarray(a["gain_Nm_per_rad_s"]) for a in (original, changed))
        assert weak.shape == base.shape == (32, 2) and np.all(base > 0)
        np.testing.assert_allclose(weak, GAIN * base, rtol=2e-7, atol=0)
        settings = load(parent / obs / "settings.json")
        for level in (obs, obs + "_gain85"):
            for case in load(sources[level] / "cases.json"):
                c = load(Path(case["output"]) / "contract.json")
                assert c["settings"].get("sampled_feedback") == settings.get("sampled_feedback")
                assert c["settings"]["case_labels"] == spec["labels"]
                assert c["settings"]["alphas"] == case["alphas"] and c["settings"]["zero_middle"] == case["zero_middle"]
    analyze_feedback(root, arms=levels, sources=sources, check=check_physics,
                     title="Matched feedback and velocity-servo gain perturbations")

    out = root / "analysis"
    with (out / "episodes.csv").open() as stream:
        records = {(r["level"], r["condition"], r["method"]):r for r in csv.DictReader(stream)}
    pairs = []
    for (level, condition, method), original in records.items():
        if level not in OBSERVATIONS:
            continue
        changed = records[level + "_gain85", condition, method]
        i = spec["labels"].index(method)
        source = Path(original["raw"])
        other = Path(changed["raw"])
        with np.load(source / "trajectory.npz") as a, np.load(other / "trajectory.npz") as b:
            nominal_trace, changed_trace = (z["lower_trace"][:, :, i].reshape(-1, 11) for z in (a, b))
            # End-of-tick speed and path error can only be compared at equal elapsed physical time.
            common = ((nominal_trace[:, 6] > 0) & (changed_trace[:, 6] > 0)
                      & (nominal_trace[:, 7] == changed_trace[:, 7]))
            errors = {}
            for k, target, col in (("path", 0., 3), ("speed", 2.6, 2)):
                for name, trace in (("group", nominal_trace), ("weakened", changed_trace)):
                    v = trace[common, col].astype(float) - target
                    errors[f"{name}_{k}_rmse"] = float(np.sqrt(np.mean(v*v))) if len(v) else None
        row = dict(feedback=level, condition=condition, method=method, group=original["group"],
                   common_ticks=int(common.sum()), group_end_s=original["time_s"], weakened_end_s=changed["time_s"],
                   group_terminal=original["terminal"], weakened_terminal=changed["terminal"],
                   group_trace=str(source), weakened_trace=str(other), **errors)
        for key in ("full", "hold", "A", "B", "C"):
            row["group_" + key] = int(original[key])
            row["weakened_" + key] = int(changed[key])
        pairs.append(row)
    assert len(pairs) == 64
    with (out / "gain_pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pairs[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(pairs)
    with (out / "README.md").open("a") as stream:
        stream.write("\n## Same-slot changes in physical outcomes\n\n"
                     "[64 matched pairs, actual endpoints and common-window errors](gain_pairs.csv)\n\n"
                     "|Feedback|Terrain|Method|Pairs|Full gained / lost|Hold gained / lost|\n|---|---|---|---:|---:|---:|\n")
        for obs in OBSERVATIONS:
            for terrain in spec["cases"]:
                for method in spec["labels"]:
                    group = [r for r in pairs if r["feedback"] == obs and r["method"] == method
                             and r["condition"].startswith(terrain)]
                    assert len(group) == 4
                    counts = []
                    for key in ("full", "hold"):
                        delta = [r["weakened_"+key] - r["group_"+key] for r in group]
                        counts.append(f"{delta.count(1)} / {delta.count(-1)}")
                    stream.write("| " + " | ".join([obs,terrain,method,str(len(group)),*counts]) + " |\n")


def audit_torques(root):
    """Replay the first physics substep per lower tick using the previous recorded joint velocity."""
    import torch
    from sttw_control.torch_controller import TorchVelocityServos

    spec = load(root / "protocol.json")
    parent = Path(spec["parent"])
    manifest = load(Path(load(parent / "protocol.json")["source_cases"]) / "source_cases.json")
    cases = load(parent / "native/cases.json")
    records = []
    for case in cases:
        condition = Path(case["output"]).name
        path = Path(manifest[condition])
        c = load(path / "contract.json")
        vehicle = load(Path(c["settings"]["asset"]).with_name("contract.json"))
        joints = [j["name"] for j in vehicle["joints"]]
        servo = TorchVelocityServos(vehicle, joints)
        base_gain = servo.gain * case["randomization"]["servo_gain_scale"][0]
        assert c["settings"]["randomization"] == case["randomization"]
        assert c["settings"]["case_labels"][:6] == ["fixed0","fixed0.25","fixed0.5","fixed0.75","fixed1","Zero"]
        assert not any(c["delay_steps"]["command"][:6])
        with np.load(path / "trajectory.npz") as z:
            tr = z["lower_trace"].reshape(-1, len(c["settings"]["alphas"]), 11)
            measurement = z["measurement"]
            commands = z["final_command"]
            for label, i in zip(spec["labels"], (0,2,4,5), strict=True):
                active = (tr[1:,i,6] > 0) & (tr[:-1,i,6] > 0)
                cmd = torch.from_numpy(commands[1:,i][active])
                velocity = torch.zeros((len(cmd), len(joints)), dtype=cmd.dtype)
                # The recorded motor observer negates the rear joint velocity once.
                m = measurement[:-1,i][active]
                velocity[:,servo.indices] = torch.from_numpy(m[:,[3,5]] * np.array([1,-1],dtype=m.dtype))
                servo.gain = base_gain
                reference = servo.torque(cmd, velocity)[:,servo.indices].clone()
                servo.gain = base_gain * GAIN
                stress = servo.torque(cmd, velocity)[:,servo.indices].clone()
                target = np.clip(cmd.numpy() * servo.sign.numpy(), *servo.control.numpy().T)
                for scale, actual in ((1.,reference), (GAIN,stress)):
                    pred = (target - velocity[:,servo.indices].numpy()) * (base_gain.numpy() * scale)
                    np.testing.assert_allclose(np.clip(pred, *servo.force.numpy().T), actual.numpy(), atol=1e-6, rtol=0.)
                delta = (stress-reference).numpy()
                row = dict(condition=condition, group=case["seed"], method=label, lower_ticks=len(cmd), source=str(path))
                for j, name in enumerate(("steering","rear")):
                    row[name+"_changed_ticks"] = int(np.count_nonzero(delta[:,j]))
                    row[name+"_rms_Nm"] = float(np.sqrt(np.mean(delta[:,j].astype(float)**2)))
                records.append(row)
    assert len(records) == 32
    with (root / "same_state_torque.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def queue_after(root):
    registration = load(root / "registration.json")
    previous = Path(registration["predecessor"])
    pid = registration["pid"]
    fd = os.pidfd_open(pid)
    try:
        live = subprocess.check_output(["systemctl", "--user", "show", registration["unit"],
                                        "-p", "MainPID", "--value"], text=True).strip()
        if int(live) != pid:
            raise RuntimeError("The registered predecessor process changed")
        write(root / "dispatch_status.json", dict(stage="waiting_for_predecessor", pid=pid))
        select.select([fd], [], [])
    finally:
        os.close(fd)
    receipt = load(previous / "dispatch_status.json")
    if receipt["stage"] != "complete" or not Path(receipt["report"]).is_file():
        raise RuntimeError("The preceding architecture comparison did not complete")
    for f, expected in registration["source_files"].items():
        if digest(Path(f)) != expected:
            raise ValueError(f"Registered actuator experiment changed: {f}")
    write(root / "dispatch_status.json", dict(stage="waiting_for_gpu", predecessor_pid=pid))
    subprocess.run(["learning/scripts/gpu_queue.sh", LAB, "-u", __file__, "run", str(root)], check=True)
    write(root / "dispatch_status.json", dict(stage="complete", report=str(root / "analysis/README.md")))


def run(root):
    p = load(root / "protocol.json")
    for path, sha in p["identities"].items():
        if digest(Path(path)) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    if load(Path(p["parent"]) / "status.json")["stage"] != "complete":
        raise ValueError("The registered sensor batch must finish before its two controls can be reused")
    while subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip():
        time.sleep(30)
    write(root / "status.json", dict(stage="running", new_episodes=64))
    notice(root,"start","PCRP 执行器增益减至组内原值的85%，两个闭环反馈条件各32槽。")
    with (root / "simulation.log").open("x") as log:
        subprocess.run([LAB, "-u","learning/cli/evaluate_isaac_gpu.py","--backend","lab","--bundle",p["bundle"],
                        "--seconds","10","--resident-jobs",str(root/"jobs.json")],stdout=log,stderr=subprocess.STDOUT,check=True)
    for obs in OBSERVATIONS:
        status = load(root / (obs + "_gain85") / "batch/status.json")
        assert (status["stage"],status["policy_steps"],status["physics_steps"]) == ("complete",500,1_600_000)
    write(root / "status.json", dict(stage="physics_complete", new_episodes=64))
    analyze(root)
    write(root / "status.json", dict(stage="complete", new_episodes=64, reused_episodes=64))
    notice(root,"complete",f"PCRP 执行器与反馈配对完成，逐组轨迹见 {root}/analysis/README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare","queue-after","run","analyze","audit-torques"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--sensor-run", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.stage == "prepare":
        if args.sensor_run is None:
            parser.error("prepare requires --sensor-run")
        prepare(root,args.sensor_run.resolve())
    elif args.stage == "audit-torques":
        audit_torques(root)
    elif args.stage == "analyze":
        analyze(root)
    else:
        try:
            if args.stage == "queue-after":
                queue_after(root)
            else:
                run(root)
        except Exception as exc:
            state = "dispatch_status" if args.stage == "queue-after" else "status"
            write(root / (state + ".json"),dict(stage="error",reason=str(exc)))
            if state == "status" or load(root / "status.json")["stage"] != "error":
                notice(root,"error",f"PCRP 执行器与反馈试验停止，见 {root}/{state}.json")
            raise
