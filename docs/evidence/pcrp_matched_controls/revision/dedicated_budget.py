"""Prepare and execute the three budget-matched fixed-preference Go2 runs.

From the gravel tree, prepare on CPU with 'prepare'. Execute 'train' or 'evaluate'
under learning/scripts/gpu_queue.sh as stages of the bounded review queue.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA = Path("/media/qy/6A7E3DB07E3D75C3/STTW_runs")
EARLIER = DATA / "pcrp_train_20261121"
ANCHOR = EARLIER / "shared7101"
INITIAL = ANCHOR / "training/model_initial.pt"
ORIGINAL = ROOT / "learning/configs/pcrp_shared_train.json"
PY = Path("/home/qy/miniconda3/envs/env_isaaclab/bin/python")
SETTINGS = {"fixed0":0., "fixed05":.5, "fixed1":1.}
ALPHAS = list(SETTINGS.values())
PORT = 6008


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, log):
    with log.open("w") as f:
        subprocess.run([str(a) for a in args], stdout=f, stderr=subprocess.STDOUT, check=True)


def notify(out, key, message):
    receipt = out / (key+"_bark.json")
    result = subprocess.run(["/usr/bin/python3", "learning/scripts/notify_sttw_job.py",
                             "--message", message, "--receipt", str(receipt)])
    if result.returncode:
        save(receipt, {"status":"failed","returncode":result.returncode,"message":message})


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    cfg = load(ORIGINAL)
    old = load(ANCHOR/"config.json")["task"]
    for key, value in cfg.items():
        assert old[key] == value, key
    assert cfg["updates"]==150 and cfg["num_envs"]==4096 and cfg["rollout_steps"]==24
    for tag, alpha in SETTINGS.items():
        s = dict(cfg, alphas=[alpha], updates=cfg["updates"]//3, initial_weight_reference=str(INITIAL),
                 task_note="Fixed preference for the complete Go2 training trajectory, sampled by the common-start reset.",
                 selection_note="Use the final model at update 50, independently of its development score.",
                 role=f"TRAIN dedicated alpha {alpha}, matched pooled transition budget to shared seed 7101.")
        save(out/(tag+".json"),s)
    specs = []
    bend_root = DATA/"pcrp_local300_p0_20261121/full_ground/both_bends"
    for row in load(bend_root/"cases.json"):
        source = Path(row["output"])
        specs.append((source.parent.name,row["seed"],source))
    for terrain in ("flat","oblique_2cm","oblique_4cm"):
        for seed in range(101,106):
            specs.append((terrain,seed,(ROOT/f"runs/h75_tradeoff_repeats/{terrain}_v2p6_seed{seed}/evaluation").resolve()))
    assert len(specs)==len({(cond,seed) for cond,seed,_ in specs})==25
    index = []
    jobs = []
    for tag in [*SETTINGS, "shared7101"]:
        levels = ALPHAS if tag=="shared7101" else [SETTINGS[tag]]
        for route, conditions in [("s_bend_pulse",["s_bend_pulse"]),("right_bend",["right_bend"]),
                                  ("original",["flat","oblique_2cm","oblique_4cm"])]:
            dest = out/"evaluations"/tag/route
            dest.mkdir(parents=True)
            cases, origins = [], []
            for condition, seed, source in specs:
                if condition not in conditions:continue
                contract = load(source/"contract.json")
                s = contract["settings"]
                route_bundle = Path(contract["bundle"])
                terrain_path = next(iter(s["terrains"].values()))
                target = dest/f"{condition}_g{seed}"
                case = dict(user_speed_m_s=s["user_speed_m_s"], seed=seed, randomization=s["randomization"],
                            alphas=levels,zero_middle=[False]*len(levels),case_labels=[f"{tag}_a{a:g}" for a in levels],
                            terrain_path=str((ROOT/terrain_path).resolve()),output=str(target))
                if "evaluation_pulse" in s:case["evaluation_pulse"]=s["evaluation_pulse"]
                for lane,a in enumerate(levels):
                    orig_lane = s["alphas"].index(a)
                    origins.append(contract["lab_physics"]["terrain_layout"]["clone_origins"][orig_lane])
                    index.append(dict(cohort=tag,condition=condition,group=seed,alpha=a,lane=lane,
                                      raw=str(target),h75_reference=str(source),h75_lane=orig_lane))
                cases.append(case)
            s = dict(s,plant_options=dict(s["plant_options"],replay_clone_origins=origins),
                     role="Matched review DEV replay at frozen physical groups",
                     evaluation_role=f"{tag}, fixed lower controller and H75 archived origins")
            for key in ("lane_alphas","lane_speeds","lane_terrain_ids","fixed_randomization_groups",
                        "alpha_switch","traditional_rules","evaluation_pulse"):
                s.pop(key,None)
            save(dest/"settings.json",s)
            save(dest/"cases.json",cases)
            jobs.append(dict(tag=tag,route=route,job=str(dest),reference_bundle=str(route_bundle),
                             lanes=sum(len(c["alphas"]) for c in cases)))
    assert len(index)==150
    with (out/"episodes_index.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
    save(out/"eval_jobs.json",jobs)
    save(out/"protocol.json",dict(
        phase="bounded TRAIN then matched DEV",
        training_configs={tag:sha(out/(tag+".json")) for tag in SETTINGS},
        base_sha256=sha(ORIGINAL), shared_completed_run=str(ANCHOR),
        original_initial_checkpoint_sha256=sha(INITIAL),
        original_final_checkpoint_sha256=sha(ANCHOR/"training/model_150.pt"),
        shared_completed_interactions=14745600,each_training_budget=4915200,
        total_new_training_interactions=14745600,training_initialization="7101 fresh shared weights; exact tensor check before first sample",
        reference_policy="Archived independently trained H75 serves as a historical comparison",
        eval_index_sha256=sha(out/"episodes_index.csv"),new_eval_episodes=150,
        terminal_selection="Three dedicated update 50 models and the recorded shared update 150; stop on contract violation or first failure",
        tensorboard=f"http://127.0.0.1:{PORT}",tensorboard_logdir=str(out)))
    save(out/"status.json",{"stage":"prepared","eval_episodes":150})


def train(out):
    # This service stays available after the training-stage exit for inspection.
    subprocess.run(["systemd-run","--user","--unit=pcrp-t3b-tensorboard","--collect",
                    str(PY.with_name("tensorboard")), "--host","127.0.0.1","--port",str(PORT),
                    "--logdir",str(out),"--reload_interval","5"],check=True)
    for tag in SETTINGS:
        cfg=out/(tag+".json")
        if sha(cfg)!=load(out/"protocol.json")["training_configs"][tag]:
            raise ValueError("The training config differs from the frozen protocol")
        if (out/tag).exists():
            raise RuntimeError(f"{out/tag} already exists; inspect completed or partial work before resuming")
        save(out/"status.json",dict(stage="training",model=tag))
        notify(out,tag+"_start",f"PCRP 第一章固定α={SETTINGS[tag]:g}训练启动，4096环境×24步×50轮。TensorBoard http://127.0.0.1:{PORT}")
        command = [PY,"-u","learning/cli/train_go2_path.py","--config",cfg,"--output",out/tag]
        with (out/(tag+".log")).open("w") as log, subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT) as training:
            endpoint = f"http://127.0.0.1:{PORT}/data/plugin/scalars/tags"
            while training.poll() is None:
                try:
                    with urlopen(endpoint, timeout=3) as reply:
                        tags = json.load(reply).get(tag+"/training", {})
                except (URLError, TimeoutError):
                    tags = {}
                if tags:
                    save(out/(tag+"_tensorboard.json"),dict(run=tag+"/training",url=endpoint,http=200,tags=sorted(tags),time=time.time()))
                    break
                time.sleep(10)
            if training.wait():
                raise subprocess.CalledProcessError(training.returncode,command)

        status=load(out/tag/"status.json")
        assert status["stage"]=="training_complete" and status["completed_updates"]==50
        assert status["valid_transitions"]==4915200 and status["wrong_preference_transitions"]==0
        assert load(out/tag/"initial_weight_check.json")["identical"]
        checkpoint=out/tag/status["final_checkpoint"]
        assert checkpoint.is_file()
        notify(out,tag+"_done",f"PCRP 固定α={SETTINGS[tag]:g}训练完成。50轮，末模型 {checkpoint}")
        run([PY,"learning/cli/export_go2_local_actor.py","--training-run",out/tag,
             "--checkpoint",checkpoint,"--h75-reference-bundle","runs/h75_selected_local300_flat/bundle",
             "--output",out/(tag+"_bundle")],out/(tag+"_export.log"))
    save(out/"status.json",dict(stage="training_complete",completed_interactions=14745600))


def evaluate(out):
    assert load(out/"status.json")["stage"]=="training_complete"
    for job in load(out/"eval_jobs.json"):
        tag = job["tag"]
        source = EARLIER/"shared7101_final_bundle" if tag=="shared7101" else out/(tag+"_bundle")
        dest = Path(job["job"]); route = Path(job["reference_bundle"])
        bundle = dest/"bundle";bundle.mkdir()
        for key in ("actor.npz","config.json","route.npz"):
            shutil.copyfile((source if key=="actor.npz" else route)/key,bundle/key)
        identity = load(route/"identity.json")
        actor = load(source/"identity.json")
        identity.update(source_actor=actor["source_actor"],native_checkpoint=actor["native_checkpoint"],
                        role="Frozen specialist/Go2 evaluation on matched original route")
        identity["files"]={k:sha(bundle/k) for k in ("actor.npz","config.json","route.npz")}
        assert identity["files"]["config.json"]==actor["files"]["config.json"]
        save(bundle/"identity.json",identity)
        cases=load(dest/"cases.json"); batch=dest/"batch";key=tag+"_"+job["route"]
        save(out/"status.json",dict(stage="evaluation",job=key,lanes=job["lanes"]))
        notify(out,key+"_start",f"PCRP 专用策略比较 {key}，{job['lanes']} 条配对开发轨迹开始。")
        run([PY,"-u","learning/cli/evaluate_isaac_gpu.py","--backend","lab","--bundle",bundle,
             "--config",dest/"settings.json","--case-manifest",dest/"cases.json","--terrain",cases[0]["terrain_path"],
             "--output",batch,"--seconds","10"],dest/"simulation.log")
        assert load(batch/"status.json")["policy_steps"]==500
        assert all((Path(c['output'])/"trajectory.npz").is_file() for c in cases)
        notify(out,key+"_done",f"PCRP 专用策略比较 {key} 完成，逐轨文件 {dest}")
    save(out/"status.json",dict(stage="evaluation_complete",new_episodes=150))
    run([PY, "learning/cli/pcrp_frontier_comparison.py", "specialist-comparison", "--output", out],
        out/"comparison.log")
    notify(out,"all_done",f"PCRP 固定α三模型与共享7101的匹配评测完成，逐组XY与结果 {out/'comparison/README.md'}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","train","evaluate"))
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    os.chdir(ROOT)
    os.environ["PYTHONPATH"]=str(ROOT/"learning/src")+os.pathsep+os.environ.get("PYTHONPATH","")
    if args.stage=="prepare":
        prepare(args.out)
    else:
        try:
            (train if args.stage=="train" else evaluate)(args.out)
        except Exception as exc:
            last=load(args.out/"status.json")
            save(args.out/"status.json",dict(stage="error",previous=last,error=str(exc)))
            notify(args.out,"error",f"PCRP 第一章固定α作业报错，{last}，{exc}；已停止。")
            raise
