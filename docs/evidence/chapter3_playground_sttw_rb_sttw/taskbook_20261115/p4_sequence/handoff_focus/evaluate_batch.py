from pathlib import Path
import os,json,time,hashlib,subprocess,fcntl
root=Path(__file__).parent;prior=root.parent/'taskbook_20261115_p4_low_lr'
assert json.loads((root/'status.json').read_text())['status']=='complete'
rows=[]
for arm in ['control','focus']:
 for seed in range(3):
  folder=root/'training'/arm/f'seed{seed}';logs=json.loads((folder/'log.json').read_text());best=json.loads((folder/'best.json').read_text())['update'];last=logs[-1]['update']
  for update in sorted({best,last}):
   checkpoint=folder/f'model_{update:04d}.pt';name=f'{arm}_s{seed}_u{update}'
   selection=json.loads((prior/'seed0_update3475_models.json').read_text())
   selection['models']['jump']={'path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
   (root/(name+'_models.json')).write_text(json.dumps(selection,indent=2)+'\n')
   rows.append(dict(name=name,arm=arm,seed=seed,update=update,best=update==best,last=update==last,checkpoint=str(checkpoint),sha256=selection['models']['jump']['sha256']))
(root/'evaluation_candidates.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'evaluation_contract.json').write_text(json.dumps(dict(candidates=len(rows),budget='Best/last deduplicated x3maps; max36 GPU batched queries;0 training/physics;stop20min',selection='Choose one candidate per arm by originalDEV72 raw true admissions, then smaller mean event-selected position error. Map48 outcomes excluded. Known DEV, not TEST.',invariants='Same frozen policies, ground models, guard, schemas, geometry and goal; no new thresholds'),indent=2)+'\n')
start=time.time();results=[]
with open('/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU busy; no evaluation launched')
 for row in rows:
  for mapid in ['map06','map07','map48']:
   cfg=json.loads((prior/'calibration_eval/seed0_update3475/C/contract.json').read_text()) if mapid=='map48' else json.loads((prior/'calibration_input/seed0_update3475'/mapid/'C/contract.json').read_text())
   out=root/'evaluation'/row['name']/mapid
   cmd=['/home/qy/mujoco_playground/.venv/bin/python','learning/cli/evaluate_skill_sequence.py','--selection',str(root/(row['name']+'_models.json')),'--history',cfg['history'],'--output',str(out),'--oracle' if cfg['oracle'] else '--evaluation',cfg['oracle'] or cfg['evaluation']]
   tick=time.time();p=subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='learning/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),capture_output=True,text=True,timeout=120)
   (root/(row['name']+'_'+mapid+'.log')).write_text(p.stdout+p.stderr)
   results.append(dict(candidate=row['name'],map=mapid,returncode=p.returncode,wall_s=time.time()-tick,command=cmd))
   (root/'evaluation_status.json').write_text(json.dumps(dict(completed=results,elapsed_s=time.time()-start),indent=2)+'\n')
   print(row['name'],mapid,p.returncode,flush=True)
   if p.returncode:raise SystemExit(p.returncode)
print('Evaluation complete',time.time()-start,flush=True)
