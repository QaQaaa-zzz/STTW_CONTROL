"""Stage a finite real-prefix recovery campaign; never restart failed jobs."""
import argparse,json,os,sys,time,subprocess,socket,fcntl,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.preference_best import atomic_json
from sttw_control.recovery_bank import load_parent
from sttw_control.tolerance_training import short_training_config


def register(path,name,status):
 atomic_json(path,dict(name=name,execution=str(status),lineage=str(status),bark_notify_completion=True,automatic_repair=False))
 watch=Path('/home/qy/DVGC/runs/queue_control/jit_optimized_ablations_20261020/email-watchlist.json')
 with watch.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);data=json.loads(watch.read_text())
  if str(path) not in data['active_runs']:data['active_runs'].append(str(path))
  atomic_json(watch,data)


def main():
 p=argparse.ArgumentParser();p.add_argument('--source-campaign',required=True);p.add_argument('--output',required=True);p.add_argument('--post-output',required=True);p.add_argument('--smoke-status',required=True);a=p.parse_args()
 source=Path(a.source_campaign).resolve();root=Path(a.output).resolve();post=Path(a.post_output).resolve();root.mkdir(parents=True,exist_ok=False)
 started=time.time();state=dict(state='running',phase='running',pid=os.getpid(),training_steps=0)
 def update(**kw):
  state.update(kw);state['checked_unix']=time.time();atomic_json(root/'prepare_status.json',state)
 def launch(script,log,*args):
  with log.open('x') as stream:return subprocess.Popen([sys.executable,'-u',script,*map(str,args)],stdout=stream,stderr=subprocess.STDOUT)
 register(root/'ACTIVE_PREPARE_RUN.json','STTW short recovery bank preparation',root/'prepare_status.json')
 update(stage='waiting_for_gpu_smoke')
 try:
  dependency=Path(a.smoke_status)
  while True:
   s=json.loads(dependency.read_text())
   if s['phase']=='completed':break
   if s['phase'] in ('error','failed'):raise RuntimeError('suffix smoke failed; no production launch')
   os.kill(s['pid'],0);update(stage='waiting_for_gpu_smoke',dependency_pid=s['pid']);time.sleep(15)
  atomic_json(root/'declaration.json',dict(experiment='short_recovery_v1',source_campaign=str(source),
    original_horizon_s=20.,snippet_s=4.,pulse_probability=.8,pulse_duration_s=.1,
    critic_warmup_updates=2,ppo_updates=25,policy_transitions_per_job=27*512*128,
    total_policy_transitions=4*27*512*128,maximum_bank_lower_ticks=2*512*1600,
    parent_anchor_kl=.05,actor_lr=3e-5,critic_lr=.0003,seed=910601,pulse_seed=910602,
    nominal_anchor_trajectory_s=8.,max_workers=2,training_seeds=1,automatic_extension=False,
    evaluation='same864-case20s development panel; no final test',
    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
  banks={};workers={}
  for alpha in (0,1):
   out=root/'banks'/f'alpha{alpha}';out.parent.mkdir(exist_ok=True)
   register(root/f'ACTIVE_BANK_ALPHA{alpha}.json',f'STTW real TRAIN prefix bank alpha{alpha}',out/'status.json')
   workers[alpha]=launch('learning/cli/collect_recovery_bank.py',root/f'bank_alpha{alpha}.log','--campaign',source,'--alpha',alpha,'--output',out)
  update(stage='collecting_train_banks',child_pids={k:p.pid for k,p in workers.items()})
  while workers:
   for alpha,process in list(workers.items()):
    if process.poll() is not None:
     if process.returncode:raise RuntimeError(f'TRAIN bank alpha{alpha} exited{process.returncode}')
     bank=root/'banks'/f'alpha{alpha}';s=json.loads((bank/'status.json').read_text())
     if s['phase']!='completed':raise RuntimeError('bank missing completion')
     banks[alpha]=bank;del workers[alpha]
   update(stage='collecting_train_banks',completed_banks=list(banks));time.sleep(10)
  logdir=root/'tensorboard';port=None
  for candidate in range(6011,6031):
   with socket.socket() as sock:
    try:sock.bind(('127.0.0.1',candidate));port=candidate;break
    except OSError:continue
  if port is None:raise RuntimeError('no free TensorBoard port in6011..6030')
  tb=launch('-m',root/'tensorboard.log','tensorboard.main','--logdir',logdir,'--port',port,'--host','127.0.0.1')
  url=f'http://localhost:{port}'
  for attempt in range(30):
   if tb.poll() is not None:raise RuntimeError('TensorBoard exited')
   try:
    with urllib.request.urlopen(url,timeout=2) as response:
     if response.status==200:break
   except Exception:time.sleep(1)
  else:raise RuntimeError('TensorBoard HTTP unavailable')
  atomic_json(root/'tensorboard_server.json',dict(pid=tb.pid,port=port,url=url,logdir=str(logdir),reward_scalar_verification='per-job actual HTTP after first produced update'))
  jobs=[]
  for alpha in (0,1):
   source_plan,record,parent,_=load_parent(source,alpha);cfg=short_training_config(parent['config']);cfgfile=root/f'alpha{alpha}_config.json';atomic_json(cfgfile,cfg)
   for arm in ('uniform','guided'):
    name=f'{arm}_alpha{alpha}';jobs.append(name);job=root/name;job.mkdir()
    atomic_json(job/'plan.json',dict(experiment='short_recovery_v1',arm=arm,alpha=alpha,updates=27,seed=910601,pulse_seed=910602,
      parent_manifest=source_plan['parent_manifest'],parent_record=record,config=str(cfgfile),bank_dir=str(banks[alpha]),
      tensorboard_info=str(root/'tensorboard_server.json'),run_name=name))
    atomic_json(job/'status.json',dict(state='planned',phase='planned',completed_updates=0,declared_updates=27,policy_transitions_total=0))
    register(job/'ACTIVE_RUN.json',f'STTW short recovery {name}',job/'status.json')
  atomic_json(root/'campaign.json',dict(experiment='short_recovery_v1',jobs=jobs,total_policy_transitions=4*27*512*128))
  protocol=json.loads((source/'comparison_protocol.json').read_text());protocol.update(policies=['parent','uniform_update27','guided_update27'],comparisons=[['parent','uniform_update27'],['parent','guided_update27'],['uniform_update27','guided_update27']],figure_caption='Parent baselineu50; candidates u27=2critic-only+25PPO,4s TRAIN snippets; fixed20sDEV. Seed910601+alpha. Thick:pulse; x:original-label failure.')
  atomic_json(root/'comparison_protocol.json',protocol)
  register(root/'ACTIVE_RUN.json','STTW short recovery finite campaign',root/'status.json')
  atomic_json(root/'status.json',dict(state='running',phase='running',pid=os.getpid(),stage='launching_scheduler',policy_transitions_total=0))
  scheduler=launch('learning/cli/parallel_tolerance_campaign.py',root/'scheduler.log',root,'--workers',2)
  post.parent.mkdir(exist_ok=True)
  # Registration sits outside post dir because waiter creates its immutable output.
  register(root/'ACTIVE_POST_RUN.json','STTW short recovery final evaluation',post/'status.json')
  evaluator=launch('learning/cli/finish_tolerance_campaign.py',root/'post_launch.log','--campaign',root,'--output',post)
  update(stage='training',scheduler_pid=scheduler.pid,post_pid=evaluator.pid,tensorboard_url=url)
  while scheduler.poll() is None:time.sleep(30);update(stage='training')
  if scheduler.returncode:raise RuntimeError('training scheduler failed; preserve active workers')
  final=json.loads((root/'status.json').read_text())
  if final['phase']!='completed' or final['completed_jobs']!=4:raise RuntimeError('incomplete final training receipts')
  update(state='completed',phase='completed',stage='training_complete_eval_separate',elapsed_s=time.time()-started)
 except BaseException as exc:
  update(state='error',phase='error',error=f'{type(exc).__name__}: {exc}',active_workers_preserved=True)
  if not (root/'status.json').exists():atomic_json(root/'status.json',state)
  raise

if __name__=='__main__':main()
