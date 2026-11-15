from pathlib import Path
import json,subprocess,os,time,fcntl
r=Path(__file__).parent;records=[];env=dict(os.environ,PYTHONPATH='learning/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
with open('/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU busy')
 for job in json.loads((r/'training_commands.json').read_text()):
  t=time.time()
  with (r/f"train_seed{job['seed']}.log").open('w') as log:p=subprocess.run(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=250)
  records.append(dict(**job,returncode=p.returncode,wall_s=time.time()-t));(r/'status.json').write_text(json.dumps(dict(status='failed' if p.returncode else 'complete' if len(records)==3 else 'running',completed=records),indent=2)+'\n');print(job['seed'],p.returncode,flush=True)
  if p.returncode:raise SystemExit(p.returncode)
subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--title','第三章阶段顺序续训完成','--message','三个种子各1000监督更新完成；接下来检查best/last行为及规划，不代表任务通过。','--receipt',str(r/'training_bark.json')],check=True)
