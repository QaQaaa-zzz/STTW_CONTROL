"""Audit saved predicted and measured phase2 joint states; no new rollouts."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--physical',type=Path,required=True)
p.add_argument('--prediction',type=Path,action='append',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
model=mujoco.MjModel.from_binary_path(str(a.physical/'terrain.mjb'))
schema=json.loads((a.physical/'model_state_schema.json').read_text())
hand=json.loads((a.physical/'resolved_config.json').read_text())['ground_handoff']
qi=[schema['qpos_names'].index(n) for n in ['hip_joint','knee_joint']]
vi=[schema['qvel_names'].index(n) for n in ['hip_joint','knee_joint']]
home=model.key('initial_state').qpos[qi]
pt=hand['lock_position_tolerance_rad'];vt=hand['lock_velocity_tolerance_rad_s']
rows=[];summaries=[]
for path in a.prediction:
 z=np.load(path/'predictions.npz');cases=json.loads((path/'cases.json').read_text())
 pe=abs(z['handoff_qpos'][:,qi]-home).max(-1);ve=abs(z['handoff_qvel'][:,vi]).max(-1)
 group=[]
 for i,c in enumerate(cases):
  actual=np.load(c['prefix']);ids=np.flatnonzero(actual['trace'][:,-1]==2)
  measured=None
  if len(ids):
   k=ids[0];ap=float(abs(actual['qpos'][k,qi]-home).max());av=float(abs(actual['qvel'][k,vi]).max())
   measured=dict(position_error_rad=ap,speed_rad_s=av,ready=ap<pt and av<vt)
  ready=bool(pe[i]<pt and ve[i]<vt)
  row=dict(prediction=str(path),case_index=i,root=c['root'],route=c['route'],actual_success=c['actual_success'],original_admitted=bool(c['admitted']),joint_ready=ready,admitted_with_joint_gate=bool(c['admitted']) and ready,position_error_rad=float(pe[i]),speed_rad_s=float(ve[i]),measured=measured)
  rows.append(row);group.append(row)
 summaries.append(dict(prediction=str(path),cases=len(group),joint_ready=sum(x['joint_ready'] for x in group),original_admitted=sum(x['original_admitted'] for x in group),admitted_with_joint_gate=sum(x['admitted_with_joint_gate'] for x in group),measured_events=sum(x['measured'] is not None for x in group),measured_ready=sum(x['measured'] is not None and x['measured']['ready'] for x in group)))
(a.output/'cases.json').write_text(json.dumps(rows,indent=2)+'\n')
(a.output/'summary.json').write_text(json.dumps(dict(home=home.tolist(),position_tolerance_rad=pt,velocity_tolerance_rad_s=vt,groups=summaries,scope='Known DEV saved-state audit. Necessary joint check only, not full contact/body readiness. Shared prefixes are not independent episodes.'),indent=2)+'\n')
print(json.dumps(summaries,indent=2))
