"""Finite sequential campaign controller; never restarts failed jobs."""
import sys,os,json,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.preference_best import atomic_json
root=Path(sys.argv[1]).resolve();plan=json.loads((root/'campaign.json').read_text());child=None

def status(phase,**kw):
 states=[json.loads((root/name/'status.json').read_text()) for name in plan['jobs']]
 atomic_json(root/'status.json',dict(phase=phase,state=phase,pid=os.getpid(),checked_unix=time.time(),completed_jobs=sum(x.get('phase')=='completed' for x in states),declared_jobs=len(states),policy_transitions_total=sum(x.get('policy_transitions_total',0) for x in states),declared_policy_transitions=19660800,training_control_ticks_total=sum(x.get('training_control_ticks_total',0) for x in states),**kw))
try:
 for name in plan['jobs']:
  job=root/name;old=json.loads((job/'status.json').read_text())
  if old['phase']!='planned':raise RuntimeError('Refuse implicit restart of '+name)
  with (job/'execution.log').open('w') as stream:
   child=subprocess.Popen([sys.executable,'learning/cli/train_tolerance_shaping.py','--plan',str(job/'plan.json'),'--output',str(job)],stdout=stream,stderr=subprocess.STDOUT)
   while child.poll() is None:
    status('running',active_job=name,child_pid=child.pid);time.sleep(10)
  if child.returncode!=0:raise RuntimeError(f'{name} exited {child.returncode}; inspect child status/log')
  end=json.loads((job/'status.json').read_text())
  if end['phase']!='completed' or end['completed_updates']!=50:raise RuntimeError('Incomplete training receipt '+name)
 status('completed',stage='training_complete',evaluation_status='not_started')
except BaseException as exc:
 if child is not None and child.poll() is None:child.terminate();child.wait()
 status('error',error=f'{type(exc).__name__}: {exc}');raise
