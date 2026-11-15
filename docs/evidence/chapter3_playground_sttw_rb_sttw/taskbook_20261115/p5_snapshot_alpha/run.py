import os,json,subprocess,time,fcntl
from pathlib import Path
r=Path(__file__).parent
with open('/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU busy')
 cmd=['/home/qy/mujoco_playground/.venv/bin/python','learning/cli/evaluate_playground_handoff.py','--config',str(r/'config.json'),'--output',str(r/'evaluation'),'--skills','rb_run3','--modes','h75','--routes']+list(map(str,range(9)))+['--seconds','10','--physics-substeps','5']
 (r/'command.json').write_text(json.dumps(cmd));t=time.time()
 with (r/'run.log').open('w') as log:p=subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='learning/src:/home/qy/DVGC/JIT/src',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,timeout=700)
 (r/'status.json').write_text(json.dumps(dict(status='complete' if p.returncode==0 else 'failed',returncode=p.returncode,wall_s=time.time()-t)))
subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--title','第三章P5同快照分叉结束','--message','采集已结束，待核对相同alpha恢复基线、交回前差异与实际目标结果。','--receipt',str(r/'collection_bark.json')])
