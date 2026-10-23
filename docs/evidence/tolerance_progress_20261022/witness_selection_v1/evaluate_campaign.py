"""Fixed paired witness-selection panels after the already queued GPU work."""
from pathlib import Path
import os,sys,json,time,subprocess
R=Path(__file__).resolve().parent;REPO=R.parents[1]
sys.path[:0]=[str(REPO/'learning/src'),str(REPO/'learning/cli')]
from sttw_control.preference_best import atomic_json,digest
from sttw_control.preference_length_evaluation import paired_counts
from sttw_control.preference_tolerance_reporting import report
from prepare_short_recovery_campaign import register
os.chdir(REPO);plan=json.loads((R/'evaluation_plan.json').read_text());out=R/(sys.argv[1] if len(sys.argv)>1 else 'evaluation');out.mkdir(exist_ok=False)
s=dict(phase='running',pid=os.getpid(),stage='waiting',completed_cases=0,declared_cases=1620,active_lower_ticks=0,training_steps=0)
def update(**kw):s.update(kw,checked_unix=time.time());atomic_json(out/'status.json',s)
update()
try:
 if digest(R/'plan.json')!=plan['training_plan_sha256']:raise ValueError('training plan changed')
 for dependency in [R/'status.json',Path(plan['wait_for_gpu'])]:
  while True:
   state=json.loads(dependency.read_text())
   if state['phase']=='completed':break
   if state['phase'] in ['error','failed']:raise RuntimeError('dependency failed: '+str(dependency))
   os.kill(state['pid'],0);update(stage='waiting',dependency=str(dependency));time.sleep(60)
 comparisons={}
 for seed in plan['seeds']:
  policies={}
  for arm in ['witness','random_success']:
   job=R/f'{arm}_{seed}';state=json.loads((job/'training/status.json').read_text());decl=json.loads((job/'declaration.json').read_text());manifest=json.loads((job/'training/manifest.json').read_text())
   if state['phase']!='completed' or state['completed_updates']!=400 or manifest['declaration']!=decl:raise ValueError('training identity mismatch')
   rec=json.loads((job/'training/candidate_record.json').read_text())
   for name in ['actor','checkpoint']:
    if digest(rec[name])!=rec[name+'_sha256']:raise ValueError('teaching candidate changed')
   policies[arm]=rec
  for panel,spec in plan['panels'].items():
   source=Path(spec['source'])
   if digest(source)!=spec['source_sha256']:raise ValueError('evaluation source changed')
   d=json.loads(source.read_text());folder=out/f'seed{seed}_{panel}';folder.mkdir()
   d.update(policies=policies,maximum_lower_ticks=spec['cases_per_policy']*2*2000,experiment_id=folder.name,question=plan['primary'],training_campaign=str(R),data_role=plan['role'],stop_rule='fixed final400 actors, no selection or parameter change')
   d.pop('acceptance_rule',None);atomic_json(folder/'declaration.json',d)
   register(folder/'ACTIVE_RUN.json',f'STTW_CONTROL 见证筛选消融评测 {seed} {panel}',folder/'raw/status.json')
   with (folder/'run.log').open('x') as f:p=subprocess.Popen([sys.executable,'-u','learning/cli/screen_preference_tolerance.py','--declaration',str(folder/'declaration.json'),'--output',str(folder/'raw')],stdout=f,stderr=subprocess.STDOUT)
   update(stage='evaluating',seed=seed,panel=panel,child_pid=p.pid)
   while p.poll() is None:time.sleep(60)
   if p.returncode:raise RuntimeError('evaluation failed: '+folder.name)
   state=json.loads((folder/'raw/status.json').read_text());rows=json.loads((folder/'raw/results.json').read_text())['cases']
   if state['phase']!='completed' or len(rows)!=spec['cases_per_policy']*2:raise ValueError('panel incomplete')
   comparisons[folder.name]=paired_counts([r for r in rows if r['policy']=='random_success'],[r for r in rows if r['policy']=='witness'])
   atomic_json(out/'comparisons.json',comparisons);update(completed_cases=s['completed_cases']+len(rows),active_lower_ticks=s['active_lower_ticks']+state['active_lower_ticks'],stage='reporting')
   report(folder/'raw',folder/'analysis')
 if s['completed_cases']!=1620:raise ValueError('total cases mismatch')
 update(phase='completed',stage='paired_results_ready')
except BaseException as e:update(phase='error',error=repr(e));raise
