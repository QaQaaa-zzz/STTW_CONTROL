"""Post-review dispatcher for the registered seed-7101 architecture screen (system Python)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ENTRY = HERE / "architecture_screen.py"
PY = "/home/qy/miniconda3/envs/env_isaaclab/bin/python"
UNIT = "pcrp-ch1-review-continuation-20261125.service"
STAGES = [(arm,stage) for arm in ("concat","speed_only","steer_only") for stage in ("train","evaluate")] + [(None,"report")]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def state(out, name, **fields):
    temp = out/"dispatch_status.tmp"
    temp.write_text(json.dumps(dict(stage=name,utc=stamp(),**fields),indent=2)+"\n")
    temp.replace(out/"dispatch_status.json")


def properties():
    names = "MainPID InvocationID ActiveState SubState Result ExecMainCode ExecMainStatus".split()
    args = ["systemctl","--user","show",UNIT,*(f"--property={n}" for n in names)]
    return dict(line.split("=",1) for line in subprocess.check_output(args,text=True).splitlines())


def verify_predecessor(plan, unit):
    if not (unit["InvocationID"]==plan["prior_invocation"] and unit["ActiveState"]=="inactive" and
            unit["Result"]=="success" and unit["ExecMainCode"]=="1" and unit["ExecMainStatus"]=="0"):
        raise RuntimeError(f"Predecessor service did not complete successfully: {unit}")
    prior = Path(plan["prior_run"])
    if sha(prior/"protocol.json")!=plan["prior_protocol_sha256"]:
        raise RuntimeError("Predecessor protocol changed")
    if load(prior/"status.json").get("stage")!="complete":
        raise RuntimeError("Predecessor has no complete queue status")
    if not Path(plan["prior_final_report"]).is_file() or load(Path(plan["prior_final_receipt"])).get("specialists")!=plan["prior_specialist_index"]:
        raise RuntimeError("Predecessor has no verified combined frontier report")


def prepare(out):
    unit = properties()
    if unit["ActiveState"]!="active" or int(unit["MainPID"])<=0:
        raise RuntimeError(f"Predecessor not active: {unit}")
    prior = out.parent/"pcrp_review_queue_20261125"
    steps = load(prior/"protocol.json")["stages"]
    last = steps[-1]
    if last["name"]!="combined_frontier":
        raise ValueError("Review protocol does not end with combined frontier analysis")
    idx = last["command"].index("--specialists")
    analysis = Path(last["run"])/"analysis"
    stages = []
    for arm, stage in STAGES:
        command = [PY,"-u",str(ENTRY),stage,str(out)] + (["--arm",arm] if arm else [])
        stages.append(dict(name=f"{arm}_{stage}" if arm else stage, resource="gpu" if arm else "cpu", command=command))
    plan = dict(screen_protocol_sha256=sha(out/"protocol.json"), dispatcher_sha256=sha(Path(__file__)),
                entry_sha256=sha(ENTRY), prepared_utc=stamp(), prior_unit=UNIT, prior_pid=int(unit["MainPID"]),
                prior_invocation=unit["InvocationID"], prior_run=str(prior), prior_protocol_sha256=sha(prior/"protocol.json"),
                prior_final_report=str(analysis/"README.md"), prior_final_receipt=str(analysis/"receipt.json"),
                prior_specialist_index=last["command"][idx+1], stages=stages, updates_per_arm=150, total_new_evaluations=600,
                total_new_sampled_transitions=44236800, stop="First failed gate or process exits the dispatcher; no retries, restart or extra jobs.")
    with (out/"dispatch.json").open("x") as stream:
        stream.write(json.dumps(plan,indent=2)+"\n")
    state(out,"prepared",prior_unit=UNIT,previous_main_pid=plan["prior_pid"])


def wait_for_predecessor(out, plan):
    import dbus
    manager = dbus.Interface(dbus.SessionBus().get_object("org.freedesktop.systemd1","/org/freedesktop/systemd1"),
                             "org.freedesktop.systemd1.Manager")
    # Hold the transient unit's result in the systemd manager until exit status is inspected.
    manager.RefUnit(UNIT)
    try:
        unit = properties()
        if unit["InvocationID"]!=plan["prior_invocation"]:
            raise RuntimeError("Predecessor service invocation changed")
        if unit["ActiveState"]=="active":
            if int(unit["MainPID"])!=plan["prior_pid"]:
                raise RuntimeError("Predecessor main PID changed")
            fd = os.pidfd_open(plan["prior_pid"])
            try:
                state(out,"waiting_for_predecessor",unit=UNIT,pid=plan["prior_pid"],invocation=plan["prior_invocation"])
                select.select([fd],[],[])
            finally:
                os.close(fd)
        for _ in range(100):
            unit = properties()
            if unit["ActiveState"] in ("inactive","failed"):
                break
            time.sleep(.1)
        verify_predecessor(plan,unit)
        (out/"predecessor_exit.json").write_text(json.dumps(unit,indent=2)+"\n")
    finally:
        manager.UnrefUnit(UNIT)


def dispatch(out, plan):
    wait_for_predecessor(out,plan)
    for index, stage in enumerate(plan["stages"]):
        name = stage["name"]
        command = stage["command"]
        if stage["resource"]=="gpu":
            state(out,"waiting_for_gpu_lock",job=name)
            command = [str(ROOT/"learning/scripts/gpu_queue.sh"),"/usr/bin/python3","-u",__file__,"under-lock",str(out),
                       "--plan-sha256",sha(out/"dispatch.json"),"--index",str(index)]
        else:
            state(out,"reporting",job=name)
        with (out/("dispatch_"+name+".log")).open("x") as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    if not (out/"comparison/README.md").is_file():
        raise RuntimeError("Final architecture report missing")
    state(out,"complete",new_sampled_transitions=44236800,new_evaluations=600,report=str(out/"comparison/README.md"))
    notify(out,"dispatch_complete","PCRP 7101三臂训练与同槽开发筛查已完成："+str(out/"comparison/README.md"))


def notify(out, tag, text):
    subprocess.run(["/usr/bin/python3","learning/scripts/notify_sttw_job.py","--message",text,"--receipt",str(out/(tag+".json"))])


def owns_gpu_lock(path):
    stat = path.stat()
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if fields[1:5]==["FLOCK","ADVISORY","WRITE",str(os.getpid())]:
            major,minor,inode=fields[5].split(":")
            if (int(major,16),int(minor,16),int(inode))==(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino):
                return True
    return False


def run_under_lock(out,plan,stage):
    verify_predecessor(plan,load(out/"predecessor_exit.json"))
    if stage["resource"]!="gpu" or not owns_gpu_lock(Path("/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock")):
        raise RuntimeError("Project GPU lease absent")

    # Other projects may use this GPU without taking STTW's flock.
    while True:
        pids = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
        active = [int(s) for s in pids.splitlines() if s.strip()]
        if not active:
            break
        state(out,"waiting_for_foreign_gpu",job=stage["name"],active_compute_pids=active)
        time.sleep(30)
    state(out,"running",job=stage["name"],gpu_compute_pids_before=[])
    os.execv(stage["command"][0],stage["command"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","dispatch","under-lock"))
    parser.add_argument("out",type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--index",type=int)
    args=parser.parse_args()
    out=args.out.resolve()
    os.chdir(ROOT)
    if args.mode=="prepare":
        prepare(out)
    else:
        try:
            plan=load(out/"dispatch.json")
            if (sha(out/"dispatch.json")!=args.plan_sha256 or sha(out/"protocol.json")!=plan["screen_protocol_sha256"] or
                    sha(Path(__file__))!=plan["dispatcher_sha256"] or sha(ENTRY)!=plan["entry_sha256"]):
                raise ValueError("Registered dispatcher or screen protocol hash mismatch")
            if args.mode=="dispatch":
                dispatch(out,plan)
            else:
                run_under_lock(out,plan,plan["stages"][args.index])
        except Exception as e:
            if args.mode=="dispatch":
                previous=load(out/"dispatch_status.json")
                state(out,"error",previous=previous,reason=str(e))
                notify(out,"dispatch_error","PCRP 7101三臂队列已停："+str(e))
            raise
