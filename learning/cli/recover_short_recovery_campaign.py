"""Explicitly recover the authenticated u25 boundary after missing final u27.

Reuses banks; retains failed attempt, charges unsaved updates, no implicit restart.
"""
import argparse,json,os,sys,subprocess,socket,time,urllib.request,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_short_recovery_campaign import register
from sttw_control.preference_best import atomic_json,digest


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--post-output',required=True);a=p.parse_args()
 old=Path(a.source).resolve();new=Path(a.output).resolve();post=Path(a.post_output).resolve()
 if json.loads((old/'status.json').read_text())['phase']!='error':raise ValueError('source campaign is not failed')
 # Authenticate all boundaries before mutating destination or launching services.
 plans={};records=[]
 for name in json.loads((old/'campaign.json').read_text())['jobs']:
  plan=json.loads((old/name/'plan.json').read_text());status=json.loads((old/name/'status.json').read_text())
  if plan['experiment']!='short_recovery_v1' or plan['updates']!=27:raise ValueError('unexpected recovery experiment')
  if plan['alpha']==0:
   receipt=json.loads((old/name/'last_completed.json').read_text());metrics=old/name/'metrics.jsonl';rows=[json.loads(x) for x in metrics.read_text().splitlines()];retained=[x for x in rows if x['update']<=receipt['update']]
   if not (status['phase']=='completed' and status['completed_updates']==27 and receipt['update']==25 and len(rows)==27 and len(retained)==25):raise ValueError('unexpected lost-checkpoint boundary')
   if digest(receipt['checkpoint'])!=receipt['checkpoint_sha256']:raise ValueError('source checkpoint changed')
   if not all(r['optimizer']['actor_bitwise_unchanged'] for r in retained[:2]):raise ValueError('critic-only warmup changed actor')
   import numpy as np
   with np.load(Path(plan['bank_dir'])/'anchors.npz') as anchors:count=len(anchors['observations'])
   cursor=sum(x['optimizer'].get('accepted_minibatches',0) for x in retained if x['update']>2)*512%count
   resume=dict(receipt,parent_anchor_cursor=cursor,cursor_source_metrics=str(metrics),cursor_source_metrics_sha256=digest(metrics),inherited_training_control_ticks=retained[-1]['training_control_ticks_total'],discarded_unsaved_policy_transitions=2*512*128)
   plan['resume_record']=resume;records.append(dict(job=name,**resume))
  elif status['phase']!='planned':raise ValueError('refuse restart of nonplanned alpha1')
  plans[name]=plan
 new.mkdir(exist_ok=False)
 for name in ['campaign.json','comparison_protocol.json','alpha0_config.json','alpha1_config.json']:shutil.copy2(old/name,new/name)
 atomic_json(new/'recovery_declaration.json',dict(cause='original27 final weights not saved; checkpoints every5 only',source_attempt=str(old),completed_prefix_banks_reused=True,resumed_learner_boundaries=records,retained_lineage_policy_transitions=3276800,old_attempt_executed_policy_transitions=3538944,recovery_new_declared_policy_transitions=3801088,discarded_unsaved_policy_transitions=262144,expected_joint_executed_policy_transitions=7340032,final_lineage_declared_policy_transitions=7077888,environment='bank reset at saved route counters+1, not frame-continuous',code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
 for port in range(6012,6031):
  with socket.socket() as sock:
   try:sock.bind(('127.0.0.1',port));break
   except OSError:continue
 else:raise RuntimeError('no free TensorBoard port')
 with (new/'tensorboard.log').open('x') as log:
  tb=subprocess.Popen([sys.executable,'-m','tensorboard.main','--logdir',str(new/'tensorboard'),'--port',str(port),'--host','127.0.0.1'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 url=f'http://localhost:{port}'
 for _ in range(30):
  if tb.poll() is not None:raise RuntimeError('TensorBoard exited')
  try:
   with urllib.request.urlopen(url,timeout=2) as r:
    if r.status==200:break
  except Exception:time.sleep(1)
 else:raise RuntimeError('TensorBoard HTTP unavailable')
 atomic_json(new/'tensorboard_server.json',dict(pid=tb.pid,port=port,url=url,logdir=str(new/'tensorboard'),reward_scalar_verification='per-job actual HTTP after first produced update'))
 for name,plan in plans.items():
  job=new/name;job.mkdir();plan.update(config=str(new/f"alpha{plan['alpha']}_config.json"),tensorboard_info=str(new/'tensorboard_server.json'))
  inherited=25*512*128 if plan['alpha']==0 else 0
  atomic_json(job/'plan.json',plan);atomic_json(job/'status.json',dict(state='planned',phase='planned',completed_updates=25 if inherited else 0,declared_updates=27,policy_transitions_total=inherited))
  register(job/'ACTIVE_RUN.json','STTW short recovery recovery1 '+name,job/'status.json')
 register(new/'ACTIVE_RUN.json','STTW short recovery corrected campaign',new/'status.json')
 register(new/'ACTIVE_POST_RUN.json','STTW short recovery corrected final evaluation',post/'status.json')
 env=os.environ.copy();env.update(PYTHONPATH='learning/src',JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
 with (new/'scheduler.log').open('x') as log:
  scheduler=subprocess.Popen([sys.executable,'-u','learning/cli/parallel_tolerance_campaign.py',str(new),'--workers','2'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 for _ in range(30):
  if scheduler.poll() is not None:raise RuntimeError('scheduler exited at launch')
  if (new/'status.json').exists() and json.loads((new/'status.json').read_text()).get('pid')==scheduler.pid:break
  time.sleep(.5)
 else:raise RuntimeError('scheduler missing live status')
 with (new/'post_launch.log').open('x') as log:
  evaluator=subprocess.Popen([sys.executable,'-u','learning/cli/finish_tolerance_campaign.py','--campaign',str(new),'--output',str(post)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 atomic_json(new/'launch.json',dict(scheduler_pid=scheduler.pid,post_pid=evaluator.pid,tensorboard_url=url));print('Recovery launched',scheduler.pid,evaluator.pid,url)

if __name__=='__main__':main()
