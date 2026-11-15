from pathlib import Path
import subprocess,fcntl,os,json
r=Path(__file__).parent;source=r.parent/'taskbook_20261115_p4_cosine/evaluation'
cmd=['/home/qy/mujoco_playground/.venv/bin/python','learning/cli/audit_skill_handoff_trace.py','--state-event','--output',str(r/'audit')]
for name in ['cosine_s0_u6075','cosine_s1_u6225','cosine_s2_u5200']:
 for m in ['map06','map07','map48']:cmd+=['--prediction',str(source/name/m)]
(r/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with open('/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU busy')
 subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='learning/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),check=True,timeout=300)
