from pathlib import Path
import json,subprocess,os,fcntl,time
r=Path(__file__).parent;old=r.parent/'taskbook_20261115_p4_cosine';py='/home/qy/mujoco_playground/.venv/bin/python';env=dict(os.environ,PYTHONPATH='learning/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');records=[]
with open('/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU busy')
 for name in ['cosine_s0_u6075','cosine_s1_u6225','cosine_s2_u5200']:
  for m in ['map06','map07','map48']:
   cfg=json.loads((old/'evaluation'/name/m/'contract.json').read_text());cmd=[py,'learning/cli/evaluate_skill_sequence.py','--ordered-phases','--selection',cfg['selection'],'--history',cfg['history'],'--output',str(r/'ordered'/name/m),'--oracle' if cfg['oracle'] else '--evaluation',cfg['oracle'] or cfg['evaluation']]
   t=time.time();p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120);(r/(name+'_'+m+'.log')).write_text(p.stdout+p.stderr);records.append(dict(candidate=name,map=m,command=cmd,returncode=p.returncode,seconds=time.time()-t));(r/'ordered_status.json').write_text(json.dumps(records,indent=2)+'\n');print(name,m,p.returncode,flush=True)
   if p.returncode:raise SystemExit(p.returncode)
