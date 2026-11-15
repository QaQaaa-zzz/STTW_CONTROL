"""Verify serialized ground predictor continuation with an existing frozen model."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sttw_control.playground_arx import states
from sttw_control.playground_history_model import load_history_model
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);torch.set_num_threads(1)
model=load_history_model(a.model);cfg=json.loads((a.evaluation/'resolved_config.json').read_text());schema=json.loads((a.evaluation/'model_state_schema.json').read_text());row=json.loads((a.evaluation/'status.json').read_text())['results'][0];data=np.load(a.evaluation/row['case_dir']/'trajectory.npz');pose,velocity=states(data['qpos'][::4],data['qvel'][::4],schema);trace=data['trace'][::4];anchor=int(np.flatnonzero(trace[:,-1]==2)[0]);bank=np.load(cfg['ground_handoff']['reference_path']);tensor=lambda v:torch.as_tensor(v,device='cuda',dtype=torch.float32);route=row['route'];args=[tensor(bank['s']),tensor(bank['xy'][route:route+1]),tensor(bank['heading'][route:route+1]),tensor(bank['curvature'][route:route+1])];speed=tensor([trace[anchor,18]])
with torch.no_grad():
 state=model.initialize(tensor(pose[anchor:anchor+1]),tensor(velocity[anchor:anchor+1]),tensor(velocity[anchor-1:anchor]),tensor(data['controller_state'][::4][anchor:anchor+1]));whole=model.rollout_state(state,*args,100,speed);first=model.rollout_state(state,*args,40,speed)
 torch.save(first['next_state'],a.output/'state40.pt');restored=torch.load(a.output/'state40.pt',map_location='cuda',weights_only=False);last=model.rollout_state(restored,*args,60,speed)
 errors={}
 for key in ['pose','velocity','domain']:
  combined=torch.cat((first[key],last[key][:,1:]),1);errors[key]=float((whole[key]-combined).abs().max());torch.testing.assert_close(whole[key],combined,rtol=0,atol=0)
 for key in ['pose','velocity','previous_velocity','hidden']:torch.testing.assert_close(getattr(whole['next_state'],key),getattr(last['next_state'],key),rtol=0,atol=0)
 legacy=model.rollout(state.pose,state.velocity,state.previous_velocity,tensor(data['controller_state'][::4][anchor:anchor+1]),*args,100,speed,return_domain=True)
 for key,value in zip(['pose','velocity','domain'],legacy):torch.testing.assert_close(whole[key],value,rtol=0,atol=0)
result=dict(passed=True,model=str(a.model),case=row['case_dir'],errors=errors,steps=last['next_state'].elapsed_steps,legacy_api_exact=True,device='cuda',scope='Ground model only; no RB full-joint model and no predicted real-controller-history claim')
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
