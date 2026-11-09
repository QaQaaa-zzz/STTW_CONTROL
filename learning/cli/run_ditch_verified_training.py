"""Run one fixed terrain pilot; only start adaptation after observed far-bank landing."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from sttw_control.ditch_terrain import ditch_support_mask
import torch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run

def status(stage,**kw):
 (r/'status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))

def execute(command):
 code=subprocess.run(command).returncode
 if code:
  status('failed',returncode=code,command=command);sys.exit(code)

def require_free_gpu():
 processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if processes:
  status('blocked',reason='Other GPU compute process still present despite shared lock',pids=processes);sys.exit(0)

require_free_gpu()
status('evaluating')
execute([sys.executable,'learning/cli/evaluate_ditch_skill.py','--config',str(r/'evaluation_config.json'),'--output',str(r/'evaluation')])
config=json.loads((r/'training_config.json').read_text());terrain=config['ditch_terrain']
summary=json.loads((r/'evaluation/summary.json').read_text());checks=[]
for case in summary['cases']:
 folder=r/'evaluation'/case['candidate']['name'];d=np.load(folder/'physics_contact.npz');tr=np.load(folder/'trajectory.npz')
 ids=[d['body_names'].tolist().index(w) for w in ('frontwheel','rearwheel')]
 wheels=d['body_state'][:,0,ids,:3];forces=d['contact'][:,0,ids];root=d['root_state_wxyz'][:,0];w,x,y,z=root[:,3:7].T
 roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
 support=ditch_support_mask(torch.from_numpy(wheels),torch.from_numpy(forces),torch.from_numpy(roll),torch.from_numpy(pitch),torch.from_numpy(root[:,7]),terrain).numpy()
 contact=(forces[:,:,2]>1).all(1);air=(np.linalg.norm(forces,axis=-1)<=1).all(1)
 ground_seen=False;flight_seen=False;start=None;longest=0.;first_landing=None
 # Exclude the whole terminal policy interval from the crossing witness.
 end=float(tr['time'][-2]) if len(tr['time'])>1 else 0.
 for i,t in enumerate(d['time']):
  if t>end:break
  if contact[i]:ground_seen=True
  if ground_seen and air[i]:flight_seen=True
  if flight_seen and support[i]:
   if start is None:start=float(t)
   longest=max(longest,float(t)-start)
   if first_landing is None:first_landing=float(t)
  else:start=None
 checks.append(dict(candidate=case['candidate'],observed_ground_then_flight=flight_seen,first_supported_far_bank_s=first_landing,longest_supported_continuation_s=longest,crossing_verified=longest>=.1,original_one_second_hold=case['ditch_cross_hold']))
(r/'crossing_gate.json').write_text(json.dumps(dict(required_support_s=.1,cases=checks,scope='Physical crossing/landing witness, not full-map task or statistical guarantee'),indent=2))
execute(['/home/qy/miniconda3/envs/mjp/bin/python','learning/cli/plot_ditch_skill.py','--run',str(r/'evaluation')])
passed=[c for c in checks if c['crossing_verified']]
if not passed:
 status('gate_failed',reason='No measured far-bank landing witness; training not started');sys.exit(0)
selected=max(passed,key=lambda c:c['longest_supported_continuation_s'])
config['initial_checkpoint']=selected['candidate']['path'];config['optimizer_initialization']='Physically verified frozen candidate weights/normalizers; fresh Adam'
(r/'training_config.json').write_text(json.dumps(config,indent=2)+'\n')
require_free_gpu()
status('training',selected=selected)
execute([sys.executable,'learning/cli/train_isaac_phase_u.py','--config',str(r/'training_config.json'),'--output',str(r/'adaptation')])
status('training_complete',selected=selected)
