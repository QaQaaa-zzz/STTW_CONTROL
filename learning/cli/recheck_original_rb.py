"""Reinvoke the authenticated 10k evaluation runtime on one original batch per RB."""
import argparse,os,sys,json,time,hashlib,fcntl,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
code=Path('/home/qy/DVGC/JIT/runs/monitoring/ablation_paper_audit_20261020/five_windows10k/code')
sys.path.insert(0,'/home/qy/DVGC/runs/worktrees/multidim-envelope/JIT/src');sys.path.insert(0,str(code))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ['JAX_COMPILATION_CACHE_DIR']=str(a.output/'jax_cache')
request=Path('/home/qy/DVGC/JIT/runs/experiments/recoverybridge_reviewer_controls_20261022/evaluation_10k_20261023/request.json')
cells=[c for c in json.loads(request.read_text())['cells'] if c['model']=='Full' and c['protocol']=='nominal'];specs=[];names=[];identities=[]
for cell in cells:
 name=cell['run'].replace(' ','_');spec_path=Path(cell['reuse_evaluation']).parent/'specs/batch_000.json';spec=json.loads(spec_path.read_text());specs.append(spec);names.append(name)
 assert spec['evaluation_payload']['actor_sha256']==cell['policy']['actor_sha256']
 payload=Path(cell['policy']['checkpoint'])/'payload.pkl';assert hashlib.sha256(payload.read_bytes()).hexdigest()==cell['policy']['payload_sha256']
 (a.output/(name+'_spec.json')).write_text(json.dumps(spec,indent=2))
 identities.append(dict(run=name,actor_sha256=cell['policy']['actor_sha256'],checkpoint=cell['policy']['checkpoint'],original_spec=str(spec_path),original_result=str(Path(cell['reuse_evaluation'])/'batch_000/summary.json')))
(a.output/'declaration.json').write_text(json.dumps(dict(role='original nominal-task regression, 250 existing initial states per RB; not a new10k campaign',episodes=sum(s['num_envs'] for s in specs),horizon=400,maximum_control_transitions=300000,optimizer_updates=0,original_runtime=str(code),runtime_sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in code.glob('*.py')},identities=identities),indent=2))
def status(stage,rows):
 (a.output/'status.json').write_text(json.dumps(dict(stage=stage,completed=len(rows),total=3,results=rows),indent=2))
repo=Path(__file__).resolve().parents[2]
os.chdir('/home/qy/DVGC/runs/worktrees/multidim-envelope')
status('waiting_gpu_lock',[])
with open('/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);status('running',[])
 from rollout import EvaluationSession
 from trajectory import summarize
 session=EvaluationSession();rows=[]
 for out,tape in session.collect_many(specs,[a.output/n for n in names],max_envs=250,compressed=True):
  result=summarize(tape,out/'episodes.csv');(out/'summary.json').write_text(json.dumps(result,indent=2));rows.append(dict(run=out.name,summary=result));status('running',rows);print(json.dumps(rows[-1]),flush=True)
 status('complete',rows)
subprocess.run(['/usr/bin/python3',str(repo/'learning/scripts/notify_chapter3_job.py'),'--title','原万回合RB模型复测完成','--message','三份RB各250个原始初态、原评测入口复测结束。查看chapter3_original_rb_recheck。','--receipt',str(a.output/'bark.json')],check=True)
