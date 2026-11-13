"""Run the Playground ground-suite protocol through the original Isaac plant."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--isaac-source',type=Path,required=True);p.add_argument('--terrain-source',type=Path,required=True)
a=p.parse_args();cfg=json.loads(a.config.read_text());source=json.loads(a.isaac_source.read_text());a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app
import torch
from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
from sttw_control.playground_ground_evaluation import evaluate_ground_suite
torch.set_num_threads(2)
terrain=a.output/'terrain';terrain.mkdir();raw=a.terrain_source.read_bytes();(terrain/'terrain.npz').write_bytes(raw);(terrain/'identity.json').write_text(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),source=str(a.terrain_source))))
suite=cfg['ground_suite'];n=len(suite['cases'])*(len(suite['alphas'])+1)
options=dict(source['plant_options']);options.update(pendulum_locked=True,track_wheel_contacts=False)
plant=IsaacLabVectorPlant(source['asset'],[terrain],[0]*n,a.output,**options)
class Simulation:
    def __init__(self,plant):self.plant=plant
    def reset(self,cases,case_ids):
        roots=plant.robot.data.default_root_state.clone();positions=plant.robot.data.default_joint_pos.clone();velocities=torch.zeros_like(positions)
        for i,k in enumerate(case_ids):
            c=cases[k];yaw=c['start_yaw'];roots[i,:2]=torch.tensor(c['start_xy'],device=plant.device);roots[i,3:7]=torch.tensor([np.cos(yaw/2)/np.sqrt(2),np.cos(yaw/2)/np.sqrt(2),np.sin(yaw/2)/np.sqrt(2),np.sin(yaw/2)/np.sqrt(2)],device=plant.device);roots[i,7:9]=torch.tensor([c['speed_m_s']*np.cos(yaw),c['speed_m_s']*np.sin(yaw)],device=plant.device)
        plant.reset_idx(torch.arange(n,device=plant.device),state=dict(root_pose=roots[:,:7],root_velocity=roots[:,7:],joint_position=positions,joint_velocity=velocities));plant.set_pendulum_locked(True)
    def observe(self):
        m,pose,speed=plant.observe();q=plant.roots[:,3:7];x,y,z,w=q.unbind(-1);pitch=torch.asin((2*(w*y-z*x)).clamp(-1,1))
        # qpos here retains Isaac's native root frame and named joint order.
        roots=plant.roots[:,:7].clone();roots[:,:3]-=plant.origin;roots[:,3:7]=roots[:,[6,3,4,5]]
        return torch.cat((roots,plant.dofs[:,:,0]),-1).cpu().numpy(),m,pose,speed,pitch
    def advance(self,applied,steer):
        command=torch.stack((steer,applied[:,1]),-1)
        for _ in range(round(.005/plant.physics_dt)):plant.step(command)
hand=cfg['ground_handoff'];bundle=Path(hand['bundle']);identity=json.loads((bundle/'identity.json').read_text());hcfg=json.loads((bundle/'config.json').read_text());hcfg['follower']['nominal_steer_slew_rad_s']=hand.get('nominal_steer_slew_rad_s',hcfg['follower']['nominal_steer_slew_rad_s']);hcfg['follower'].update(hand.get('follower_overrides',{}));controller=json.loads(Path(json.loads(Path(hand['settings']).read_text())['controller_config']).read_text())
(a.output/'isaac_model.json').write_text(json.dumps(dict(metadata=plant.metadata,options=options,joint_order=plant.joints,initialization='Original upright root frame at z0.15, native rest joints, zero wheel speeds, prescribed body vx; same physical initial condition as Playground',differences='Original Isaac ground velocity steering and 2.5ms physics retained; Playground uses integrated-position steering and 1ms implicitfast'),indent=2))
evaluate_ground_suite(None,None,None,cfg,bundle,hcfg,identity,controller,a.output,round(.005/plant.physics_dt),None,simulation=Simulation(plant))
os.sync();os._exit(0)
