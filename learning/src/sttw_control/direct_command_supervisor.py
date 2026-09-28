"""External deadline enforcement for only the owned direct-command worker group."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def supervise(command, output, spec):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    path=root/'budget.json'
    prior=json.loads(path.read_text()) if path.exists() else {'seconds':{}}
    used=sum(prior['seconds'].values());start=time.time()
    with (root/'execution.log').open('a') as log:
        worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (root/'launch.json').write_text(json.dumps(dict(pid=worker.pid,supervisor_pid=os.getpid(),
            started_epoch=start,command=command),indent=2))
        reason=None
        while worker.poll() is None:
            time.sleep(.05)
            try:
                d=json.loads(path.read_text());active=d.get('active')
                if time.time()-start+used>=spec['budget']['additional_compute_wall_seconds']:
                    reason='external total compute cutoff'
                if active and time.time()>=active['started_epoch']+active['reserved_s']:
                    reason=f"external {active['stage']} stage cutoff"
                if reason:
                    os.killpg(worker.pid,signal.SIGTERM)
                    try:worker.wait(timeout=.5)
                    except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL)
                    break
            except (json.JSONDecodeError,FileNotFoundError):
                pass
        code=worker.wait()
    if not path.exists():return code
    d=json.loads(path.read_text());elapsed=time.time()-start
    charged=sum(d['seconds'].values())-used
    if d.get('active'):
        a=d['active'];delta=max(0.,time.time()-a['started_epoch'])
        d['seconds'][a['stage']]=d['seconds'].get(a['stage'],0)+delta
        d['events'].append(dict(kind='external_interrupted',**a,seconds=delta))
        d['active']=None;charged+=delta
    d['seconds']['process_overhead']=d['seconds'].get('process_overhead',0)+max(0.,elapsed-charged)
    d['total_compute_seconds']=sum(d['seconds'].values());d['external_returncode']=code;d['external_stop_reason']=reason
    path.write_text(json.dumps(d,indent=2)+'\n')
    if reason:
        status_path=root/'status.json';status=json.loads(status_path.read_text())
        status.update(state='budget_stopped',reason=reason,last_update_epoch=time.time())
        status_path.write_text(json.dumps(status,indent=2)+'\n')
    return code
