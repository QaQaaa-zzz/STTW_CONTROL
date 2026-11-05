"""Bounded GPU tensor/vehicle isolation check; no PPO updates."""
import argparse,json,time
from pathlib import Path
from isaacgym import gymapi
import torch
from sttw_control.isaac_vector_env import IsaacVectorPlant
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=15);p.add_argument('--steps',type=int,default=100);a=p.parse_args()
if a.output.exists() or not 1<=a.count<=4096 or not 1<=a.steps<=1000:raise ValueError('bounded probe contract violated')
start=time.time();plant=IsaacVectorPlant('runs/isaac_urdf_drive/vehicle.urdf',['runs/isaac_flat_mesh_010','runs/isaac_training_terrain/slope','runs/isaac_training_terrain/rough_slope'],[i%3 for i in range(a.count)])
initial=plant.observe()[1].clone();command=torch.tensor([[0.,23.]],device='cuda').repeat(a.count,1);contacts=torch.zeros(a.count,dtype=torch.bool,device='cuda');torch.cuda.synchronize();initialized=time.time()
for _ in range(a.steps):contacts|=plant.step(command)
torch.cuda.synchronize();finished=time.time();m,pose,speed=plant.observe();result=dict(count=a.count,steps_per_env=a.steps,total_physics_steps=a.count*a.steps,physics_device=str(plant.roots.device),tensor_pipeline='gpu',finite=bool(torch.isfinite(plant.roots).all()),initial_pose=initial.cpu().tolist(),final_pose=pose.cpu().tolist(),speed=speed.cpu().tolist(),nonwheel_contact=contacts.cpu().tolist(),initialization_seconds=initialized-start,simulation_seconds=finished-initialized,role='GPU tensor/isolation probe; no closed-loop training or convergence claim',contact_monitor='GPU net contact force, differs from CPU individual-contact query; requires validation')
a.output.write_text(json.dumps(result,indent=2));plant.close();print(json.dumps(result))
