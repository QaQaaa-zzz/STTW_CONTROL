from pathlib import Path
import subprocess,json,os,fcntl
r=Path(__file__).parent;cmd=['/home/qy/mujoco_playground/.venv/bin/python','learning/cli/screen_skill_joint_checkpoints.py','--state-event','--training',str(r/'training'),'--output',str(r/'screen')]
for m in ['map06','map07','map48']:cmd+=['--prediction',str(r.parent/'taskbook_20261115_p4_cosine/evaluation/cosine_s0_u6075'/m)]
(r/'screen_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with open('/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU busy')
 subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='learning/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),check=True,timeout=300)
