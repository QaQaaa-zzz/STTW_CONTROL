from pathlib import Path
import json,subprocess,os,fcntl,time
root=Path(__file__).parent;prior=root.parent/'taskbook_20261115_p4_low_lr';selected=json.loads((root/'selected_comparison.json').read_text());env=dict(os.environ,PYTHONPATH='learning/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');python='/home/qy/mujoco_playground/.venv/bin/python';records=[]
(root/'comparison_contract.json').write_text(json.dumps(dict(selection=str(root/'selected_comparison.json'),budget='Two arm leaders x3 independent-interface maps;6 GPU queries then CPU raw G/I/C/Oracle, own/shared/cross supported calibration.0 training/physics;stop15min',roles='Known DEV; same candidates/predictor/constraints strong backtracking retained'),indent=2)+'\n')
def run(cmd,label):
 t=time.time();p=subprocess.run([python]+cmd,env=env,capture_output=True,text=True,timeout=120);(root/(label+'.log')).write_text(p.stdout+p.stderr);records.append(dict(label=label,command=cmd,returncode=p.returncode,wall_s=time.time()-t));(root/'comparison_status.json').write_text(json.dumps(records,indent=2)+'\n');print(label,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
with open('/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise SystemExit('GPU busy; no independent query started')
 for row in selected:
  for mapid in ['map06','map07','map48']:
   cfg=json.loads((root/'evaluation'/row['name']/mapid/'contract.json').read_text());out=root/'independent'/row['name']/mapid
   run(['learning/cli/evaluate_skill_sequence.py','--selection',str(root/(row['name']+'_models.json')),'--history',cfg['history'],'--output',str(out),'--oracle' if cfg['oracle'] else '--evaluation',cfg['oracle'] or cfg['evaluation'],'--typical',str(root.parent/'taskbook_20261115_p4_typical/states.npz')],row['name']+'_I_'+mapid)
for row in selected:
 name=row['name'];dev=root/'calibration_input'/name;ev=root/'calibration_eval'/name
 for mapid in ['map06','map07','map48']:
  parent=ev if mapid=='map48' else dev/mapid;parent.mkdir(parents=True)
  for method,source in [('C',root/'evaluation'/name/mapid),('I',root/'independent'/name/mapid)]: (parent/method).symlink_to(source,target_is_directory=True)
 run(['learning/cli/calibrate_skill_sequence.py','--supported-only','--dev',str(dev),'--evaluation',str(ev),'--output',str(root/'calibration'/name)],name+'_calibration')
 run(['learning/cli/evaluate_continuation_planner.py','--continuous',str(ev/'C'),'--independent',str(ev/'I'),'--oracle-matrix','docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p2_oracle_wall/oracle_matrix.csv','--routes',str(root.parent/'rb_adaptation/approach_departure_turns/routes.npz'),'--output',str(root/'comparison'/name)],name+'_comparison')
