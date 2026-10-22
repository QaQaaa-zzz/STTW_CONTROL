"""Adopt a live worker and schedule the same finite campaign concurrently."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.preference_best import atomic_json


def alive(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return fields[0] not in ('Z','X')
 except FileNotFoundError:return False

def main():
 p=argparse.ArgumentParser();p.add_argument('campaign');p.add_argument('--adopt-job',required=True);p.add_argument('--adopt-pid',type=int,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
 root=Path(a.campaign).resolve();definition=json.loads((root/'campaign.json').read_text());jobs=definition['jobs']
 if not 1<=a.workers<=2:raise ValueError('declared maximum two workers')
 cmd=Path(f'/proc/{a.adopt_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 if str(root/a.adopt_job) not in cmd or 'train_tolerance_shaping.py' not in cmd:raise ValueError('adopted process identity mismatch')
 running={a.adopt_job:(a.adopt_pid,None)};finished=set();order=[a.adopt_job,'dynamic_alpha0']+[x for x in jobs if x not in (a.adopt_job,'dynamic_alpha0')]
 def states():return {name:json.loads((root/name/'status.json').read_text()) for name in jobs}
 def update(phase='running',**kw):
  s=states();atomic_json(root/'status.json',dict(phase=phase,state=phase,pid=os.getpid(),scheduler='parallel_v1',max_workers=a.workers,checked_unix=time.time(),completed_jobs=sum(x.get('phase')=='completed' for x in s.values()),declared_jobs=len(jobs),policy_transitions_total=sum(x.get('policy_transitions_total',0) for x in s.values()),declared_policy_transitions=definition['total_policy_transitions'],training_control_ticks_total=sum(x.get('training_control_ticks_total',0) for x in s.values()),active_jobs=list(running),active_job=next(iter(running),None),child_pids={k:v[0] for k,v in running.items()},**kw))
 try:
  update(stage='adopted_existing_worker')
  while len(finished)<len(jobs):
   s=states()
   for name,(pid,process) in list(running.items()):
    live=process.poll() is None if process is not None else alive(pid)
    if not live:
     if s[name].get('phase')!='completed' or s[name].get('completed_updates')!=50:raise RuntimeError(f'{name} ended without complete receipt')
     if process is not None and process.returncode:raise RuntimeError(f'{name} exited {process.returncode}')
     receipt=json.loads((root/name/'last_completed.json').read_text())
     if receipt['update']!=50:raise RuntimeError(f'{name} final checkpoint missing')
     finished.add(name);del running[name]
    elif s[name].get('phase') in ('error','failed'):raise RuntimeError(f'{name} reports failure')
   for name in order:
    if len(running)>=a.workers:break
    if name in running or name in finished:continue
    if s[name]['phase']!='planned':raise RuntimeError('Refuse implicit restart: '+name)
    job=root/name
    with (job/'execution.log').open('x') as stream:
     process=subprocess.Popen([sys.executable,'learning/cli/train_tolerance_shaping.py','--plan',str(job/'plan.json'),'--output',str(job)],stdout=stream,stderr=subprocess.STDOUT)
    running[name]=(process.pid,process)
   update(stage='training');time.sleep(10)
  update('completed',stage='training_complete',evaluation_status='not_started')
 except BaseException as exc:
  update('error',error=f'{type(exc).__name__}: {exc}',running_workers_preserved=True);raise
if __name__=='__main__':main()
