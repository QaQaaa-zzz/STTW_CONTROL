from pathlib import Path
import os,json,subprocess,time,fcntl
r=Path(__file__).parent;manifest=json.loads((r/'manifest.json').read_text());env=dict(os.environ,PYTHONPATH='learning/src:/home/qy/DVGC/JIT/src',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');records=[]
with open('/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU busy; no simulation started')
 for e in manifest['entries']:
  cmd=['/home/qy/mujoco_playground/.venv/bin/python','learning/cli/evaluate_playground_handoff.py','--config',e['config'],'--output',e['output'],'--skills','rb_run3','--modes','h75','--routes']+list(map(str,range(9)))+['--seconds','10','--physics-substeps','5']
  t=time.time()
  with (Path(e['output']).parent/'run.log').open('w') as log:p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=330)
  records.append(dict(**e,command=cmd,returncode=p.returncode,wall_s=time.time()-t));(r/'status.json').write_text(json.dumps(dict(status='failed' if p.returncode else 'complete' if len(records)==len(manifest['entries']) else 'running',completed=records),indent=2)+'\n');print(e['map'],e['alpha'],p.returncode,flush=True)
  if p.returncode:
   subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--title','第三章P5采集异常','--message','已停止本批后续采集，保留日志；未扩大预算。','--receipt',str(r/'error_bark.json')]);raise SystemExit(p.returncode)
subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--title','第三章P5分阶段alpha采集完成','--message','144新增物理回合采集结束，正在比较分阶段与同alpha条件及失败；不代表条件模型或P5已完成。','--receipt',str(r/'collection_bark.json')],check=True)
