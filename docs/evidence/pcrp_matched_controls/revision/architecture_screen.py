"""Single-seed 7101 component screen. Prepare/verify/report are CPU-only."""
import argparse
from collections import defaultdict
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from dedicated_budget import ANCHOR, DATA, EARLIER, INITIAL, ORIGINAL, PY, ROOT, load, notify, run, save, sha

sys.path[:0] = [str(ROOT / "learning/src"), str(ROOT / "learning/cli")]
from sttw_control.isaac_evaluation_batch import batch_settings

EVAL = EARLIER / "shared7101_eval"
REF = EARLIER / "shared7101_final_bundle"
ZERO = DATA / "pcrp_curvature_governor_20261121/source_cases.json"
EXCLUDED = ["actor." + name for name in ("gain_2", "gain_3", "shift_2", "shift_3")]
ARMS = {"concat": {"actor_architecture": "legacy", "initial_weight_reference_excluded": EXCLUDED},
        "speed_only": {"reference_action_channel": "speed"}, "steer_only": {"reference_action_channel": "steer"}}
ALPHAS = [0, .25, .5, .75, 1]
BUDGET = 4096 * 24 * 150
PORT = 6011


def same(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Mismatch: {label}")


def same_slots(current, reference, start):
    for key in ("backend", "origin", "terrain", "randomization_readback", "delay_steps", "physics_dt", "lower_dt", "policy_dt", "bodies"):
        same(current[key], reference[key], key)
    for contract in (current, reference):
        same([contract["batch_source"][key] for key in ("lane_start", "lane_stop")], [start, start + 5], "batch slot")
        same(contract["settings"]["alphas"], ALPHAS, "slot preference")
    for key in ("initial_state", "asset", "lower_actor", "lower_sha256", "controller_config", "randomization",
                "selected_lower", "user_speed_m_s", "episode_s", "zero_middle", "seed", "terrains", "auto_reset_horizon"):
        same(current["settings"][key], reference["settings"][key], key)
    for key in ("env_origins", "solver_iterations", "compliant_wheels", "physics_dt_s", "wheel_collision", "materials", "masses"):
        same(current["lab_physics"][key], reference["lab_physics"][key], key)
    same(current["lab_physics"]["terrain_layout"]["clone_origins"],
         reference["lab_physics"]["terrain_layout"]["clone_origins"], "actual clone coordinates")
    same(current["identity"]["files"]["route.npz"], reference["identity"]["files"]["route.npz"], "original geometry")


def training_inputs(cfg):
    run_cfg = load(ANCHOR / "config.json")
    input_files = [ORIGINAL, INITIAL, ANCHOR / "training/model_150.pt", ANCHOR / "config.json", ANCHOR / "status.json",
                   EVAL / "settings.json", EVAL / "cases.json", EVAL / "batch/contract.json", EVAL / "batch/status.json", ZERO,
                   Path(run_cfg["upstream"]["source"]), Path(cfg["initial_state"]), Path(cfg["controller_config"]),
                   Path(cfg["lower_actor"]), Path(cfg["asset"]), Path(load(EVAL / "settings.json")["selected_lower"])]
    asset = Path(cfg["asset"])
    input_files += [asset.parent / mesh.get("filename") for mesh in ET.parse(asset).iter("mesh")]
    for directory in [REF, Path(cfg["bundle"])]:
        input_files += [directory / name for name in ("identity.json", "route.npz", "actor.npz", "config.json")]
    input_files += [Path(cfg["training_bank"]) / name for name in ("states.npz", "routes.npz", "identity.json")]
    old_sources = sorted((ANCHOR / "source").glob("*.py"))
    input_files += old_sources
    rsl = Path(cfg["go2_checkout"]) / "rsl_rl/rsl_rl"
    runner = rsl / "runners/on_policy_runner.py"
    same(sha(runner), sha(ANCHOR/"source/on_policy_runner.py"), "archived upstream runner")
    input_files += [rsl/"algorithms/ppo.py", rsl/"storage/rollout_storage.py", runner]
    input_files += [ROOT/"learning/src/sttw_control"/p.name for p in old_sources
                    if (ROOT/"learning/src/sttw_control"/p.name).is_file()]
    input_files += [ROOT/"learning/cli"/name for name in
                    ("train_go2_path.py","export_go2_local_actor.py","evaluate_isaac_gpu.py","pcrp_frontier_comparison.py")]
    input_files += [ROOT/"learning/src/sttw_control"/name for name in
                    ("reference_action_channel.py","isaac_evaluation_batch.py","isaac_path_reporting.py","go2_training_config.py")]
    for directory in {*cfg["terrains"].values(), *(c["terrain_path"] for c in load(EVAL / "cases.json"))}:
        input_files.extend(Path(directory) / name for name in ("identity.json", "terrain.npz"))
    return input_files


def prepare(out):
    import torch
    from sttw_control.go2_training_config import spring_jump_recipe
    from sttw_control.torch_path_policy import make_torch_path_network
    cfg = load(ORIGINAL)
    sys.path.insert(0, str(Path(cfg["go2_checkout"]) / "rsl_rl"))
    from sttw_control.go2_ppo_adapter import H75ActorCritic, verify_initial_weights
    stored = load(ANCHOR / "config.json")
    same({k: stored["task"][k] for k in cfg}, cfg, "archive training task")
    same([cfg[k] for k in ("seed", "num_envs", "rollout_steps", "updates", "alphas")],
         [7101, 4096, 24, 150, ALPHAS], "7101 protocol")
    recipe = spring_jump_recipe(cfg["go2_checkout"])
    recipe["train"]["algorithm"].update(cfg["ppo_overrides"])
    recipe["train"]["runner"].update(policy_class_name="STTWPathActorCritic", algorithm_class_name="PPO",
                                     num_steps_per_env=24, save_interval=cfg["save_interval"],
                                     max_iterations=150, experiment_name="sttw_h75_offroad")
    same(recipe, stored["upstream"], "resolved Go2 settings")
    input_files = training_inputs(cfg)
    policy = load(Path(cfg["bundle"]) / "config.json")
    initial = torch.load(INITIAL, map_location="cpu", weights_only=False)["model_state_dict"]
    settings, cases, batch = load(EVAL / "settings.json"), load(EVAL / "cases.json"), load(EVAL / "batch/contract.json")
    packed = batch_settings(settings, deepcopy(cases))
    same({k: batch["settings"][k] for k in packed}, packed, "native case packer against archived batch")
    origins = batch["lab_physics"]["terrain_layout"]["clone_origins"]
    same((len(cases), len(origins), len(set(map(tuple, origins)))), (40, 200, 200), "physical groups and slots")
    from pcrp_frontier_comparison import check_pair
    zeros = load(ZERO)
    for i, case in enumerate(cases):
        src = Path(case["output"])
        ref = load(src / "contract.json")
        same_slots(ref, ref, 5 * i)
        same(origins[5*i:5*i+5], ref["origin"], src.name)
        zero = Path(zeros[src.name])
        z = load(zero / "contract.json")
        zi = z["settings"]["zero_middle"].index(True)
        check_pair(dict(raw=str(zero), h75_reference=str(src), h75_lane=0, cohort="zero"), z, zi)
        for directory in (src, zero):
            input_files.extend(directory / name for name in ("contract.json", "trajectory.npz"))
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "source_files.json"
    save(inputs, {str(p): sha(p) for p in sorted({p.resolve() for p in input_files})})
    frozen = {inputs.name: sha(inputs)}
    audits = {}
    for tag, change in ARMS.items():
        dest = out / (tag + "_eval")
        dest.mkdir()
        task = dict(cfg, **change, initial_weight_reference=str(INITIAL),
                    role=f"TRAIN {tag} matched to seed 7101, completed update 150",
                    selection_note="Predeclared final update 150 only; all interim and best checkpoints excluded from DEV selection.")
        actor_cfg = deepcopy(policy)
        actor_cfg["network"]["architecture"] = task.get("actor_architecture", policy["network"]["architecture"])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(task["seed"])
            model = H75ActorCritic(make_torch_path_network(351, 2, actor_cfg), make_torch_path_network(352, 1, policy),
                                   torch.tensor(policy["training_future_phase_C"]["std_initial"]).log(),
                                   active_channel=task.get("reference_action_channel"))
        audits[tag] = verify_initial_weights(model.state_dict(), initial, change.get("initial_weight_reference_excluded", ()))
        s = dict(settings, role=f"{tag}, single seed 7101 DEV",
                 evaluation_role="40 registered seed-7101 DEV groups with five ordered preferences", plant_options=dict(settings["plant_options"], replay_clone_origins=origins))
        output_cases = [dict(c, output=str(dest / Path(c["output"]).name)) for c in cases]
        same(batch_settings(s, deepcopy(output_cases))["fixed_randomization_groups"], packed["fixed_randomization_groups"], tag)
        for path, obj in ((out/(tag+".json"), task), (dest/"settings.json", s), (dest/"cases.json", output_cases)):
            save(path, obj)
            frozen[str(path.relative_to(out))] = sha(path)
    save(out/"protocol.json", dict(
        role="Single-seed matched development screen of three middle-policy controls; independent seeds and full H75 lineages absent",
        archival_eval_revision=batch["code_revision"], archive=str(ANCHOR), archive_eval=str(EVAL), source_inputs=str(inputs),
        budget_per_arm=BUDGET, total_new_transitions=3*BUDGET, selection="training/model_150.pt exclusively",
        new_episodes=600, historical_7101_episodes=200, physical_groups=40, alphas=ALPHAS,
        historical_zero="40 contextual trajectories; group-matched but different placement",
        stages="run each arm fresh, export update 150, evaluate one 200-slot scene per arm for 10 seconds, compare to archived 7101",
        failure_rule="First budget, identity, slot or permission violation stops that stage. Never replace missing, failed or interrupted samples.",
        initial_tensor_checks=audits, frozen=frozen))


def verify(out):
    p = load(out/"protocol.json")
    registered = Path(__file__).with_name("architecture_screen_protocol.json")
    if registered.exists() and str(out/"source_files.json")==load(registered)["source_inputs"]:
        same(sha(out/"protocol.json"), sha(registered), "versioned protocol")
    for name, expected in p["frozen"].items():
        same(sha(out/name), expected, name)
    for name, expected in load(out/"source_files.json").items():
        same(sha(Path(name)), expected, name)


def train(out, tag):
    import json
    dest = out / tag
    if dest.exists():
        raise FileExistsError(dest)
    if subprocess.run(["systemctl","--user","is-active","--quiet","pcrp-architecture-screen-tb"]).returncode:
        subprocess.run(["systemd-run","--user","--unit=pcrp-architecture-screen-tb","--collect",
                        str(PY.with_name("tensorboard")), "--host","127.0.0.1","--port",str(PORT),"--logdir",str(out),
                        "--reload_interval","5"],check=True)
    endpoint = f"http://127.0.0.1:{PORT}/data/plugin/scalars/tags"
    notify(out,tag+"_start",f"PCRP {tag} 7101训练开始。4096×24×150；TensorBoard http://127.0.0.1:{PORT}")
    cmd = [PY, "-u", "learning/cli/train_go2_path.py", "--config", out/(tag+".json"), "--output", dest]
    with (out/(tag+".log")).open("w") as stream, subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT) as proc:
        while proc.poll() is None:
            try:
                with urlopen(endpoint, timeout=3) as response:
                    tags = json.load(response).get(tag+"/training", {})
            except (URLError, TimeoutError):
                tags = {}
            if tags:
                save(out/(tag+"_tensorboard.json"), dict(url=endpoint, run=tag+"/training", tags=sorted(tags)))
                break
            time.sleep(10)
        if proc.wait():
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    s = load(dest/"status.json")
    same([s["stage"], s["completed_updates"], s["run_sampled_transitions"], s["final_checkpoint"]],
         ["training_complete", 150, BUDGET, "training/model_150.pt"], "actual training budget and selection")
    same(load(dest/"initial_weight_check.json"), dict(reference=str(INITIAL), **load(out/"protocol.json")["initial_tensor_checks"][tag]), "actual initialized network")
    if tag != "concat":
        audits = [json.loads(line) for line in (dest/"reference_action_audit.jsonl").read_text().splitlines()]
        same([r["completed_updates"] for r in audits], list(range(1,151)), "single-channel per-update audit")
        for r in audits:
            same([r[k] for k in ("ppo_action_dim", "disabled_target_nonzero", "disabled_filter_nonzero", "disabled_offset_nonzero")], [1,0,0,0], "disabled channel")
    notify(out,tag+"_done",f"PCRP {tag} 第150轮完成。实际 {BUDGET} 转移。")
    run([PY, "learning/cli/export_go2_local_actor.py", "--training-run", dest, "--checkpoint", dest/s["final_checkpoint"],
         "--h75-reference-bundle",load(ORIGINAL)["bundle"],"--output",out/(tag+"_bundle")],out/(tag+"_export.log"))


def evaluate(out, tag):
    bundle = out/(tag+"_bundle")
    identity = load(bundle/"identity.json")
    native = out/tag/"training/model_150.pt"
    same([identity["native_checkpoint"][k] for k in ("completed_updates","sha256")], [150,sha(native)], "selected native actor")
    for k,v in identity["files"].items():
        same(sha(bundle/k), v, k)
    cfg = load(REF/"config.json")
    if tag == "concat":
        cfg["network"]["architecture"] = "legacy"
    else:
        cfg["reference_action_channel"] = ARMS[tag]["reference_action_channel"]
    same(load(bundle/"config.json"), cfg, "exported task/actor configuration")
    dest = out/(tag+"_eval")
    if (dest/"batch").exists():
        raise FileExistsError(dest/"batch")
    cases = load(dest/"cases.json")
    notify(out,tag+"_dev_start",f"PCRP {tag} 200 槽同批 DEV 开始。")
    run([PY,"-u","learning/cli/evaluate_isaac_gpu.py","--backend","lab","--bundle",bundle,"--config",dest/"settings.json",
         "--case-manifest",dest/"cases.json","--terrain",cases[0]["terrain_path"],"--output",dest/"batch","--seconds","10"],dest/"simulation.log")
    audit_evaluation(out, tag)
    notify(out,tag+"_dev_done",f"PCRP {tag} 200 槽 DEV 完成并核对槽位。")


def audit_evaluation(out, tag):
    import numpy as np
    dest = out/(tag+"_eval")
    batch = load(dest/"batch/status.json")
    same([batch[k] for k in ("stage","policy_steps","physics_steps")], ["complete",500,10_000_000], "actual evaluation budget")
    for i,c in enumerate(load(dest/"cases.json")):
        path = Path(c["output"])
        current = load(path/"contract.json")
        same_slots(current, load(EVAL/path.name/"contract.json"), 5*i)
        same(current["identity"]["files"], load(out/(tag+"_bundle")/"identity.json")["files"], "executed actor/route/task")
        with np.load(path/"trajectory.npz") as trace:
            same(trace["valid"].shape, (500,5), "captured evaluation window")
            if tag!="concat":
                disabled = 1 if tag=="speed_only" else 0
                same(int(np.count_nonzero(trace["offsets"][:,:,disabled])), 0, "exported disabled offsets")
    save(dest/"matched_slots.json", dict(actual_matched_groups=40, actual_matched_slots=200, reference=str(EVAL), policy_steps=500))


def report(out):
    from pcrp_frontier_comparison import RETRAINED, common_window, csv_write, diagnostics, read_traces
    for tag in ARMS:
        audit_evaluation(out, tag)
    dest = out/"comparison"
    dest.mkdir(exist_ok=True)
    all_rows, links = [], []
    zeros = load(ZERO)
    for case in load(EVAL/"cases.json"):
        source = Path(case["output"])
        name, seed = source.name, case["seed"]
        condition = name.rsplit("_seed", 1)[0]
        rows = []
        for tag in ["shared7101", *ARMS]:
            raw = source if tag=="shared7101" else out/(tag+"_eval")/name
            rows.extend(dict(cohort=tag, raw=str(raw), lane=i, condition=condition, group=seed, alpha=a) for i,a in enumerate(ALPHAS))
        zero = Path(zeros[name])
        zi = load(zero/"contract.json")["settings"]["zero_middle"].index(True)
        rows.append(dict(cohort="zero",raw=str(zero),lane=zi,condition=condition,group=seed,alpha=0))
        case_output = dest/name
        data = case_output/"traces"
        data.mkdir(parents=True,exist_ok=True)
        traces = read_traces(rows, data)
        common_window(traces[:-1])
        links.append(diagnostics((condition,seed),traces,case_output,("Shared Go2",*RETRAINED.values()))[0])
        for d in traces:
            row = d["row"]
            row["diagnostic_npz"] = str(Path(name)/row["diagnostic_npz"])
            all_rows.append(row)
    csv_write(dest/"episodes.csv", all_rows)
    anchors = {(r["condition"],r["group"],r["alpha"]):r for r in all_rows if r["cohort"]=="shared7101"}
    cells = defaultdict(list)
    pairs = []
    for row in all_rows:
        if row["cohort"]=="zero":
            continue
        cells[(row["condition"],row["cohort"],row["alpha"])].append(row)
        if row["cohort"] in ARMS:
            other = anchors[(row["condition"],row["group"],row["alpha"])]
            pairs.append({**{k:row[k] for k in ("condition","group","cohort","alpha","terminal","lane")},
                          "reference_terminal":other["terminal"], "common_s":row["common_s"],
                          **{"delta_"+k:row[k]-other[k] for k in
                             ("full","hold","A","B","C","end_s","common_path_rmse_m","common_speed_rmse_m_s")}})
    same((len(pairs),len(anchors)),(600,200),"paired outcomes")
    csv_write(dest/"differences_from_7101.csv",pairs)
    csv_write(dest/"condition_counts.csv",[dict(condition=condition, cohort=tag, alpha=a, n=len(rows),
              **{k:sum(r[k] for r in rows) for k in ("full","hold","A","B","C")})
              for (condition,tag,a),rows in cells.items()])
    cases = load(EVAL/"cases.json")
    (dest/"README.md").write_text("# Seed 7101 development screen\n\n"
        + "\n".join(f"- [{Path(c['output']).name} actual/reference XY]({Path(c['output']).name}/{f})" for c,f in zip(cases,links))
        + "\n\n[Per-lane outcomes, common-window errors, gate flags and reward reconstruction](episodes.csv). "
          "[Paired changes](differences_from_7101.csv) and [condition denominators](condition_counts.csv). Each case directory contains per-arm five-preference diagnostics. Zero is a contextual, different-slot trace.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare","verify","train","evaluate","report"))
    p.add_argument("out", type=Path)
    p.add_argument("--arm", choices=ARMS)
    a = p.parse_args()
    out = a.out.resolve()
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = str(ROOT/"learning/src") + os.pathsep + os.environ.get("PYTHONPATH","")
    if a.stage=="prepare":
        prepare(out)
    else:
        verify(out)
        if a.stage in ("train","evaluate"):
            if not a.arm:
                p.error("--arm is required for this stage")
            try:
                (train if a.stage=="train" else evaluate)(out, a.arm)
            except Exception as exc:
                notify(out,a.arm+"_"+a.stage+"_error",f"PCRP {a.arm} {a.stage} 出错，阶段停止：{exc}")
                raise
        elif a.stage=="report":
            report(out)
