"""Same-start frozen preference cohorts; no snapshot or PhysX-state restoration."""
import json,math
from pathlib import Path
import numpy as np
import torch


class GravelRecoveryPreference:
    def __init__(self,env,config,output,recorder):
        self.env=env;self.config=config;self.output=Path(output);self.recorder=recorder;self.groups=config['groups'];self.completed=set();self.logs=[];self.policy_step=0
        self.stable=torch.zeros(env.n,device=env.device,dtype=torch.long);self.required=math.ceil(config['confirmation_s']/env.physics_dt);self.rear=env.plant.bodies.index('rearwheel')
        terrain=Path(config['terrain']);mesh=np.load(terrain/'terrain.npz');vertices=mesh['vertices'];identity=json.loads((terrain/'identity.json').read_text());_,heading,_=env.route.at(torch.tensor([identity['terrain_support_end_s_m']],device=env.device));self.direction=torch.stack((torch.cos(heading[0]),torch.sin(heading[0])));raised=vertices[vertices[:,2]>0];self.bound=float((raised[:,:2]@self.direction.cpu().numpy()).max())+config['wheel_bound_radius_m']

    def record(self,env,lower_tick,physics_tick):
        if self.recorder is not None:self.recorder.record(env,lower_tick,physics_tick)
        if len(self.completed)==len(self.groups):return
        rear_xy=env.plant.robot.data.body_link_pos_w[:,self.rear,:2]-env.plant.origin[:,:2]
        beyond=(rear_xy*self.direction).sum(1)>self.bound
        loaded=env.plant.contact[:,self.rear,2]>self.config['rear_support_force_N']
        self.stable=torch.where(beyond&loaded&env.alive,self.stable+1,0)

    def decision(self,step):
        self.policy_step=step
        ready=((self.stable>=self.required)&self.env.alive).cpu().tolist()
        for gi,group in enumerate(self.groups):
            lanes=group['lanes']
            if gi in self.completed or not all(ready[i] for i in lanes):continue
            self.completed.add(gi);root=self.env.plant.roots[lanes].clone();root[:,:3]-=self.env.plant.origin[lanes]
            maximum=lambda v:float((v-v[:1]).abs().max())
            prefix=dict(root_position_max_delta_m=maximum(root[:,:3]),root_velocity_max_delta_m_s=maximum(root[:,7:10]),joint_position_max_delta_rad=maximum(self.env.plant.dofs[lanes,:,0]),joint_velocity_max_delta_rad_s=maximum(self.env.plant.dofs[lanes,:,1]),observation_max_delta=maximum(self.env.middle.obs[lanes]),filtered_max_delta=maximum(self.env.policy.filtered[lanes]),offsets_max_delta=maximum(self.env.policy.offsets[lanes]),eso_max_delta=maximum(self.env.controller.eso[lanes]))
            self.logs.append(dict(group=gi,lanes=lanes,entry_alpha=group['entry_alpha'],target_alphas=group['target_alphas'],policy_step=step,time_s=step*.02,prefix_at_decision=prefix,physics_state_restored=False))
        for event in self.logs:
            fraction=min(1.,(step-event['policy_step'])*.02/self.config['ramp_s']);target=torch.tensor(event['target_alphas'],device=self.env.device)
            self.env.alpha[event['lanes']]=event['entry_alpha']+fraction*(target-event['entry_alpha'])
        if any(x['policy_step']==step for x in self.logs):(self.output/'gravel_recovery_events.json').write_text(json.dumps(self.logs,indent=2)+'\n')

    def close(self):
        if self.recorder is not None:self.recorder.close()
        (self.output/'gravel_recovery_events.json').write_text(json.dumps(self.logs,indent=2)+'\n')
        (self.output/'gravel_recovery_status.json').write_text(json.dumps(dict(groups=len(self.groups),forked_groups=len(self.completed),missing_groups=sorted(set(range(len(self.groups)))-self.completed),ramp_s=self.config['ramp_s'],required_stable_physics_steps=self.required,event='all cohort rear centers beyond entire raised support plus radius, and continuous rear positive load; next50Hz update',clearance_plane=dict(direction=self.direction.cpu().tolist(),threshold_m=self.bound),physics_state_restored=False,prefix_contract='Each candidate starts from identical saved state and entry alpha; all histories naturally evolve without reset; recorded prefix differences must be inspected'),indent=2)+'\n')
