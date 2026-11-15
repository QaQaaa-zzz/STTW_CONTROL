from pathlib import Path
import json,subprocess,os,fcntl,time
r=Path(__file__).parent;root=r.parent;py='/home/qy/mujoco_playground/.venv/bin/python';env=dict(os.environ,PYTHONPATH='learning/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');records=[]
with open('/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU busy')
 for mapid in ['map06','map07','map48']:
  cfg=json.loads((root/'taskbook_20261115_p4_cosine/evaluation/cosine_s1_u6225'/mapid/'contract.json').read_text())
  for method in ['C','I']:
   cmd=[py,'learning/cli/evaluate_skill_sequence.py','--selection',str(r/'selected_models.json'),'--history',str(root/'taskbook_20261115_p4_direct/history_bridge/seed1/best.pt'),'--output',str(r/mapid/method),'--oracle' if cfg['oracle'] else '--evaluation',cfg['oracle'] or cfg['evaluation']]
   if method=='I':cmd+=['--typical',str(root/'taskbook_20261115_p4_typical/states.npz')]
   t=time.time();p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120);(r/(mapid+'_'+method+'.log')).write_text(p.stdout+p.stderr);records.append(dict(map=mapid,method=method,command=cmd,returncode=p.returncode,seconds=time.time()-t));(r/'status.json').write_text(json.dumps(records,indent=2)+'\n');print(mapid,method,p.returncode,flush=True)
   if p.returncode:raise SystemExit(p.returncode)
