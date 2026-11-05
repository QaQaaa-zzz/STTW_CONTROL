"""Replay a recorded first control command through the GPU plant."""
import argparse,json
from pathlib import Path
from isaacgym import gymapi
import numpy as np
import torch
from sttw_control.isaac_vector_env import IsaacVectorPlant
p=argparse.ArgumentParser();p.add_argument('--asset',type=Path);p.add_argument('--height-offset',type=float,default=0.);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--terrain',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.receipt.read_text());raw=np.load(a.receipt.with_suffix('.npz'));plant=IsaacVectorPlant(a.asset or r['asset'],[a.terrain],[0]);plant.initial_roots[:,2]+=a.height_offset;plant.reset(r['drive_speed']);command=torch.tensor(raw['control_trace'][0,12:14][None],dtype=torch.float32,device='cuda');rows=[]
for i in range(25):
 plant.step(command);rows.append(plant.roots.cpu().numpy().copy())
measurement,pose,speed=plant.observe();result=dict(height_offset=a.height_offset,command=command.cpu().tolist(),measurement=measurement.cpu().tolist(),old_measurement=raw['control_trace'][1,1:8].tolist(),pose=pose.cpu().tolist(),speed=speed.cpu().tolist(),physics_steps=25)
a.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.output/'trace.npz',root=np.stack(rows),old_root=raw['body_states'][:25,0]);(a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));plant.close()
