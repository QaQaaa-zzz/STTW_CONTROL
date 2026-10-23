"""One declared learner continuation, queued after the existing GPU evaluation."""
from pathlib import Path
import os,sys,json,time,subprocess
R=Path(__file__).resolve().parent;REPO=R.parents[1]
sys.path[:0]=[str(REPO/'learning/src')]
from sttw_control.preference_best import atomic_json,digest
os.chdir(REPO)
status=dict(phase='running',pid=os.getpid(),stage='waiting_for_dual_evaluation',training_launched=False)
def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(R/'status.json',status)
update()
try:
 while True:
  s=json.loads((REPO/'runs/dual_endpoint_tolerance_revisit_v1/status.json').read_text())
  if s['phase']=='completed':break
  if s['phase'] in ('error','failed'):raise RuntimeError('dependency failed; inspect before starting continuation')
  os.kill(s['pid'],0);time.sleep(60)
 launch=json.loads((R/'launch.json').read_text())
 if digest(R/'uniform/plan.json')!=launch['plan_sha256']:raise ValueError('queued plan changed')
 with (R/'training.log').open('x') as f:
  child=subprocess.Popen([sys.executable,'-u','learning/cli/train_tolerance_shaping.py','--plan',str(R/'uniform/plan.json'),'--output',str(R/'uniform')],stdout=f,stderr=subprocess.STDOUT)
 update(stage='training',training_launched=True,child_pid=child.pid)
 while child.poll() is None:time.sleep(60)
 if child.returncode:raise RuntimeError('training child exited '+str(child.returncode))
 s=json.loads((R/'uniform/status.json').read_text())
 if s['phase']!='completed' or s['completed_updates']!=77:raise RuntimeError('training missing declared final boundary')
 update(phase='completed',stage='learner_ready_evaluation_pending',executed_policy_transitions=s['executed_policy_transitions'])
except BaseException as e:update(phase='error',error=repr(e));raise
