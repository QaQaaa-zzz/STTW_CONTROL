"""Single-lane ground diagnostic on the common pendulum chassis in Isaac Lab."""
import argparse,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app
import torch
from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
from sttw_control.torch_controller import TorchController
cfg=json.loads(Path('runs/isaac_signals_parity/dynamic.json').read_text())
options=json.loads(Path('learning/configs/isaaclab_h75_offroad_selected.json').read_text())['plant_options']
plant=IsaacLabVectorPlant(a.asset,['plane'],[0],a.output,pendulum_rest=[-1.2,2.499999],**options)
controller=TorchController(cfg,1,'cuda:0')
plant.reset(speed=2.)
(a.output/'protocol.json').write_text(json.dumps(dict(asset=str(a.asset),controller=cfg,plant=plant.metadata,role='ground compatibility DEV; ECBC only, no learned policy',pendulum_rest_rad=[-1.2,2.499999],initialization_note='knee 1e-6 rad inside original upper bound to accommodate USD degree-to-radian rounding; joint limits unchanged',max_physics_steps=15000,optimizer_updates=0),indent=2))
rows=[];start=time.monotonic();failed=False
for k in range(601):
 m,pose,v=plant.observe()
 row=torch.cat((torch.tensor([[k*.005]],device='cuda:0'),pose,v[:,None],m[:,:4],plant.dofs[:,plant.pendulum_servos.indices,0]),dim=1)
 rows.append(row[0].cpu().numpy())
 failed=bool((m[:,0].abs()>torch.pi/4).any() or not torch.isfinite(row).all())
 if k==600 or failed:break
 request=torch.stack((m[:,5]*.1,m[:,2],m[:,3],m[:,0],m[:,1],torch.zeros_like(v)),dim=-1)
 out=controller.step(request,torch.ones(1,dtype=torch.bool,device='cuda:0'))
 command=torch.stack((out[:,0].clamp(-3,3),torch.full_like(v,20.)),dim=-1)
 for _ in range(25):plant.step(command)
trace=np.asarray(rows);np.savez(a.output/'trajectory.npz',trace=trace,columns=['time','x','y','yaw','speed','roll','roll_rate','steer','steer_rate','hip','knee'])
result=dict(status='complete',failed=failed,physics_steps=k*25,duration_s=k*.005,peak_roll_deg=float(np.rad2deg(abs(trace[:,5])).max()),final_speed=float(trace[-1,4]),elapsed_s=time.monotonic()-start)
(a.output/'results.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
plant.close();app.close()
