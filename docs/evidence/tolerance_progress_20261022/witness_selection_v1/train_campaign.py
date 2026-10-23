"""Sequential CPU teaching with authenticated frozen declarations and live fit logs."""
from pathlib import Path
import os,sys,json,time,subprocess,urllib.request,urllib.parse
R=Path(__file__).resolve().parent;REPO=R.parents[1];sys.path.insert(0,str(REPO/'learning/src'))
from sttw_control.preference_best import atomic_json,digest
os.chdir(REPO);plan=json.loads((R/'plan.json').read_text());launch=json.loads((R/'launch.json').read_text())
os.sched_setaffinity(0,set(launch['cpu_affinity']))
s=dict(phase='running',pid=os.getpid(),stage='teaching',completed_jobs=0,declared_jobs=6,completed_updates=0,declared_updates=2400,physics_ticks=0)
def update(**kw):s.update(kw,checked_unix=time.time());atomic_json(R/'status.json',s)
update()
try:
 for job in plan['jobs']:
  folder=R/job['name']
  if digest(folder/'declaration.json')!=job['declaration_sha256']:raise ValueError('frozen teaching declaration changed')
  if (folder/'training/manifest.json').exists():
   prior=json.loads((folder/'training/status.json').read_text())
   manifest=json.loads((folder/'training/manifest.json').read_text())
   if prior['phase']!='completed' or prior['completed_updates']!=400 or manifest['declaration']!=json.loads((folder/'declaration.json').read_text()):raise RuntimeError('existing incomplete or changed teaching job; no implicit rerun')
   rec=json.loads((folder/'training/candidate_record.json').read_text())
   for k in ['actor','checkpoint']:
    if digest(rec[k])!=rec[k+'_sha256']:raise RuntimeError('completed candidate changed')
   update(job=job['name'],reused_completed=True)
  else:
   with (folder/'training.log').open('x') as f:p=subprocess.Popen([sys.executable,'-u','learning/cli/train_recovery_teaching.py','--declaration',str(folder/'declaration.json'),'--output',str(folder/'training')],stdout=f,stderr=subprocess.STDOUT)
   update(job=job['name'],child_pid=p.pid,reused_completed=False)
   while p.poll() is None:time.sleep(5)
   if p.returncode:raise RuntimeError(job['name']+' training failed '+str(p.returncode))
  state=json.loads((folder/'training/status.json').read_text())
  if state['phase']!='completed' or state['completed_updates']!=400:raise ValueError('teaching incomplete')
  actual_run=job['name']+'/.'
  query=urllib.parse.urlencode(dict(run=actual_run,tag='fit/loss'));values=[]
  for _ in range(20):
   try:
    with urllib.request.urlopen(launch['tensorboard_url']+'/data/plugin/scalars/scalars?'+query,timeout=3) as f:values=json.load(f)
    if values and values[-1][1]==400:break
   except Exception:pass
   time.sleep(1)
  if not values or values[-1][1]!=400:raise RuntimeError('TensorBoard actual fit scalar unavailable')
  atomic_json(folder/'fit_http_verified.json',dict(run=actual_run,url=launch['tensorboard_url'],tag='fit/loss',first=values[0],last=values[-1],checked_unix=time.time()))
  update(completed_jobs=s['completed_jobs']+1,completed_updates=s['completed_updates']+400)
 update(phase='completed',stage='six_teaching_candidates_ready')
except BaseException as e:update(phase='error',error=repr(e));raise
