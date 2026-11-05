"""Run original Local300 with the saved MuJoCo standard command streams."""
import argparse,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--initial-state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device=s['device']).app
import torch
from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
from sttw_control.torch_controller import TorchController
from sttw_control.torch_local_lower import TorchLocalLower
from sttw_control.torch_path_reference import wrap
n=len(s['cases']);device=s['device'];torch.set_num_threads(2)
plant=IsaacLabVectorPlant('runs/isaac_urdf_drive/vehicle.urdf',['plane'],[0]*n,a.output)
x=np.load(a.initial_state);tensor=lambda v:torch.as_tensor(v,dtype=torch.float32,device=device)
root=tensor(x['root_pose']).repeat(n,1);root[:,:3]+=plant.origin;plant.robot.write_root_pose_to_sim(root);plant.robot.write_root_velocity_to_sim(tensor(x['root_velocity']).repeat(n,1))
order=[list(x['joint_names']).index(k) for k in plant.joints];plant.robot.write_joint_state_to_sim(tensor(x['joint_position'][order]).repeat(n,1),tensor(x['joint_velocity'][order]).repeat(n,1));plant.robot.reset();plant.sensor.reset();plant.refresh()
controller=TorchController(json.loads(Path(s['controller_config']).read_text()),n,device)
for key in ('eso','disturbance','gains'):getattr(controller,key)[:]=tensor(x[key])
lower=TorchLocalLower(s['actor'],s['actor_sha256'],n,device);previous=tensor(x['previous_reference']).repeat(n,1);command=tensor(x['previous_command']).repeat(n,1);scale=tensor([1.5,10.]);streams=tensor(np.stack([np.load(Path(s['reference_streams'])/(case+'.npy')) for case in s['cases']],axis=1));alive=torch.ones(n,device=device,dtype=torch.bool);rows=[];start=time.time()
(a.output/'contract.json').write_text(json.dumps(dict(settings=s,initial_state=str(a.initial_state.resolve()),plant=plant.metadata,controller=controller.config,role='original Local300 standard-plane transfer; no H75 or optimizer'),indent=2))
for tick in range(round(s['seconds']/.005)):
 m,pose,v=plant.observe();raw=streams[tick,:,1:3];valid=alive.clone();action,obs,finite=lower.prepare(m,v,raw,previous,command,controller);applied=action*scale
 row=torch.stack((m[:,5]*.1,m[:,2],m[:,3],m[:,0],m[:,1],raw[:,1]),dim=-1);output=controller.step(row,torch.full_like(alive,(int(x['physical_tick'])+tick)*.005>3.))
 steer=(output[:,0]+applied[:,0]).clamp(-3,3);steer=torch.maximum(torch.maximum(torch.full_like(steer,-3),(-.8-m[:,2])/.005),torch.minimum(torch.minimum(torch.full_like(steer,3),(.8-m[:,2])/.005),steer));command=torch.stack((steer,(raw[:,0]/.1+applied[:,1]).clamp(-60,60)),dim=-1)
 alive&=finite&(m[:,0].abs()<=.7);endpoint=plant.roots.clone();end_dofs=plant.dofs.clone();stop=torch.full((n,),25,device=device,dtype=torch.long)
 for substep in range(25):
  contact=plant.step(torch.where(alive[:,None],command,0.));endpoint=torch.where(alive[:,None],plant.roots,endpoint);end_dofs=torch.where(alive[:,None,None],plant.dofs,end_dofs);hit=alive&(contact|~torch.isfinite(plant.roots).all(-1));stop=torch.where(hit,substep+1,stop);alive&=~hit
 post,position,speed=plant.signals.observe(endpoint,end_dofs[:,:,0],end_dofs[:,:,1]);position-=plant.origin;alive&=post[:,0].abs()<=.7;lower.finish(applied,wrap(position[:,2]-pose[:,2])/.005)
 rows.append(dict(time=torch.full((n,),tick*.005,device=device),pose=position,measurement=post,speed=speed,reference=raw,action=action,command=command,valid=valid,failed=valid&~alive,stop_tick=stop));previous=raw
 if (tick+1)%200==0:print(json.dumps(dict(step=tick+1,alive=int(alive.sum()),seconds=time.time()-start)),flush=True)
np.savez_compressed(a.output/'trajectory.npz',**{k:torch.stack([r[k] for r in rows]).cpu().numpy() for k in rows[0]})
(a.output/'status.json').write_text(json.dumps(dict(stage='complete',cases=s['cases'],survived=alive.cpu().tolist(),physical_lane_steps=len(rows)*n*25,optimizer_updates=0,elapsed_s=time.time()-start),indent=2));plant.close();app.close(wait_for_replicator=False)
