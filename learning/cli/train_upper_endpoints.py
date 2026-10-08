#!/usr/bin/env python3
"""Bounded serial four-endpoint experiment, no automatic retries or extensions."""
import os,sys,argparse,json,time,subprocess,signal
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OMP_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--worker-config');p.add_argument('--review-alias');a=p.parse_args()
if a.worker_config:
 from sttw_control.upper_endpoint_training import run_endpoint
 run_endpoint(a.worker_config,a.output)
elif a.review_alias:
 from sttw_control.upper_endpoint_review import review_pair
 review_pair(a.output,a.review_alias)
else:
 from sttw_control.direct_command_training import write,notify
 root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
 if (root/'queue_status.json').exists():raise FileExistsError('queue identity exists; no automatic resume')
 start=time.monotonic();tasks=[]
 for alias in ['STTW_R196_ALPHA1','STTW_R244_ALPHA1']:
  short=alias.split('_')[1]
  for alpha in [0,1]:tasks.append(dict(name=f'{short}_upper{alpha}',kind='training',alias=alias,alpha=alpha,budget=16800))
  tasks.append(dict(name=f'{short}_comparison',kind='review',alias=alias,budget=1200))
 status=dict(state='running',declared_groups=4,updates_per_group=250,completed_groups=0,completed_tasks=[],current=None)
 write(root/'queue_status.json',status)
 try:
  for task in tasks:
   status.update(current=task,updated_epoch=time.time());write(root/'queue_status.json',status)
   if task['kind']=='training':
    cfg=Path(__file__).resolve().parents[1]/'configs'/f"upper_{task['alias']}_alpha{task['alpha']}.json"
    cmd=[sys.executable,__file__,'--output',str(root/task['name']),'--worker-config',str(cfg)]
   else:cmd=[sys.executable,__file__,'--output',str(root),'--review-alias',task['alias']]
   job=root/task['name'];job.mkdir(parents=True,exist_ok=True)
   with (job/'execution.log').open('w') as log:
    worker=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(job/'launch.json',dict(pid=worker.pid,started_epoch=time.time(),command=cmd))
    status['worker_pid']=worker.pid;write(root/'queue_status.json',status)
    try:code=worker.wait(timeout=min(task['budget'],72000-(time.monotonic()-start)))
    except subprocess.TimeoutExpired:
     os.killpg(worker.pid,signal.SIGTERM)
     try:worker.wait(timeout=5)
     except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
     raise TimeoutError(task['name']+' external budget stop')
   if code:raise RuntimeError(f"{task['name']} exited{code}; stopped queue without retry; see its execution.log")
   status['completed_tasks'].append(task['name'])
   if task['kind']=='training':status['completed_groups']+=1
   write(root/'queue_status.json',status)
  status.update(state='complete',current=None,elapsed_s=time.monotonic()-start);write(root/'queue_status.json',status);notify('STTW4组实验完成','四组250轮与两套最终配对评价已保存')
 except BaseException as e:
  status.update(state='error',reason=str(e),elapsed_s=time.monotonic()-start);write(root/'queue_status.json',status);notify('STTW4组队列停止',str(e));raise
