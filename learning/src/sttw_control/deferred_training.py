"""Read-only dependency watcher for one frozen STTW training pipeline."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def verify_files(files):
    for path,digest in files.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen input changed: '+path)


def gate(status,expected_identity,processes,gpu):
    if status.get('plan_sha256')!=expected_identity:
        raise ValueError('dependency plan identity changed')
    phase=status.get('phase')
    if phase in ('error','failed','cancelled','aborted','timeout','blocked'):
        raise ValueError('dependency ended unsuccessfully: '+phase)
    if phase!='complete':return 'waiting_dependency'
    if processes or gpu:return 'waiting_processes'
    return 'ready'


def dependency_processes(root):
    matches=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            argv=(proc/'cmdline').read_bytes().decode().split('\0')
            if not argv or 'python' not in Path(argv[0]).name:continue
            cwd=str((proc/'cwd').resolve())
            if cwd==root or cwd.startswith(root+'/') or any(root in x for x in argv):matches.append(int(proc.name))
        except (OSError,UnicodeError):continue
    return matches


def gpu_processes(allowed):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    return [{'pid':int(row[0].strip()),'name':row[1].strip()} for row in csv.reader(result.stdout.splitlines()) if row and row[1].strip() not in allowed]


def watch(plan_path):
    plan_path=Path(plan_path).resolve();raw=plan_path.read_bytes();plan=json.loads(raw)
    folder=plan_path.parent
    def status(phase,**extra):
        dest=folder/'status.json';tmp=folder/'status.tmp'
        tmp.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),'watcher_pid':os.getpid(),**extra},indent=2)+'\n');tmp.replace(dest)
    lock=(folder/'watch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    claim=folder/'launch_claim.json'
    if claim.exists():raise ValueError('launch already claimed; refusing another launch')
    deadline=plan['created_unix']+plan['max_wait_seconds'];ready=0
    try:
        while True:
            if plan_path.read_bytes()!=raw:raise ValueError('watch plan changed')
            verify_files(plan['input_sha256'])
            if Path(plan['output']).exists():raise ValueError('output already exists')
            if time.time()>deadline:raise TimeoutError('dependency wait expired')
            try:
                dependency=json.loads(Path(plan['dependency_status']).read_text())
            except (OSError,json.JSONDecodeError) as exc:
                ready=0;status('waiting_status',reason=str(exc));time.sleep(plan['poll_seconds']);continue
            # Failure/identity checks apply even if GPU probing fails.
            gate(dependency,plan['dependency_plan_sha256'],[],[])
            processes=dependency_processes(plan['dependency_repository'])
            try:gpu=gpu_processes(plan['allowed_gpu_process_names'])
            except (OSError,subprocess.SubprocessError,ValueError,IndexError) as exc:
                ready=0;status('waiting_gpu_query',reason=str(exc));time.sleep(plan['poll_seconds']);continue
            phase=gate(dependency,plan['dependency_plan_sha256'],processes,gpu)
            ready=ready+1 if phase=='ready' else 0
            status(phase,dependency_phase=dependency.get('phase'),dependency_processes=processes,gpu_processes=gpu,ready_checks=ready)
            if ready>=2:break
            time.sleep(plan['poll_seconds'])
        # Exclusive durable claim prevents relaunch after a watcher crash.
        with claim.open('x') as stream:json.dump({'claimed_unix':time.time(),'plan_sha256':hashlib.sha256(raw).hexdigest()},stream)
        env=os.environ.copy();env.pop('JAX_PLATFORMS',None)
        env.update(PYTHONPATH=str(Path(plan['repository'])/'learning/src'),MUJOCO_GL='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1')
        command=[sys.executable,'learning/cli/recovery_pipeline.py','--task',plan['task'],'--training',plan['training'],'--panel',plan['panel'],'--output',plan['output']]
        with (folder/'pipeline.log').open('x') as log:
            child=subprocess.Popen(command,cwd=plan['repository'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status('training_pipeline_running',pid=child.pid,command=command,output=plan['output'])
            code=child.wait()
        if code:raise RuntimeError('training pipeline exited '+str(code))
        result=json.loads((Path(plan['output'])/'pipeline_status.json').read_text())
        if result.get('phase')!='complete':raise RuntimeError('pipeline did not report complete')
        status('complete',pid=child.pid,output=plan['output'])
    except Exception as exc:
        status('error',error=str(exc));raise
