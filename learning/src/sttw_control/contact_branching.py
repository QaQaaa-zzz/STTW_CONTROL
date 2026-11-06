"""Event-triggered frozen-alpha forks using the existing rollout recorder hook."""
import json
from pathlib import Path
import numpy as np
import torch
from .contact_events import WheelClearanceEvents
from .isaac_path_snapshot import capture,restore


class ContactBranching:
    def __init__(self,env,config,output,terrain):
        self.env=env;self.config=config;self.output=Path(output);self.groups=config['groups']
        self.events=WheelClearanceEvents(env.n,env.device,env.physics_dt,config.get('confirmation_s',.02))
        self.log=[];self.active=[];self.completed=set()
        mesh=np.load(Path(terrain)/'terrain.npz');vertices=mesh['vertices'];raised=vertices[vertices[:,2]>0]
        identity=json.loads((Path(terrain)/'identity.json').read_text())
        anchor,heading,_=env.route.at(torch.tensor([identity['terrain_support_end_s_m']],device=env.device))
        self.direction=torch.stack((torch.cos(heading[0]),torch.sin(heading[0])))
        # A supporting halfspace of ALL raised vertices, including lateral obliquity.
        # Wheel-centre clearance adds the declared bounding radius; no future trajectory is used.
        self.bound=float((raised[:,:2]@self.direction.cpu().numpy()).max())+config['wheel_bound_radius_m']
        self.rear=env.plant.bodies.index('rearwheel')
        if not env.plant.wheel_sensors or env.plant.servo_mode!='explicit':
            raise ValueError('Contact branches require wheel sensors and explicit servos')

    def record(self,env,lower_tick,physics_tick):
        ridge=[];rear_ground=None
        for wi,sensor in enumerate(env.plant.wheel_sensors):
            force,point,normal,separation,count,start=sensor.contact_physx_view.get_contact_data(dt=env.physics_dt)
            # The normal API buffers are consumed before reading the next sensor.
            count=count[:,0];start=start[:,0];slots=torch.arange(int(count.max()),device=env.device)
            valid=slots[None,:]<count[:,None];indices=(start[:,None]+slots[None,:]).clamp_max(len(force)-1)
            load=force[indices,0];height=point[indices,2]-env.plant.origin[:,None,2]
            ridge.append((valid&(load>.01)&(height>.003)).any(1))
            if wi==1:rear_ground=(torch.where(valid&(height<=.003),load,0).sum(1)>.1)
        rear_xy=env.plant.robot.data.body_link_pos_w[:,self.rear,:2]-env.plant.origin[:,:2]
        beyond=(rear_xy*self.direction).sum(1)>self.bound
        self.events.update(torch.stack(ridge,1),beyond,rear_ground,env.alive)

    def decision(self,policy_step):
        ready=self.events.take_ready(self.env.alive)
        if self.config.get('event','rear_clearance')=='front_contact':
            ready=(self.events.first_contact[:,0]>=0)&self.env.alive
        for gi,group in enumerate(self.groups):
            source=group['source_lane']
            if gi in self.completed or not bool(ready[source]):continue
            lanes=group['branch_lanes'];state=capture(self.env,source);restore(self.env,state,lanes)
            np.savez_compressed(self.output/f'contact_fork_{gi:03d}.npz',**{k:v.cpu().numpy() for k,v in state.items()})
            self.completed.add(gi)
            item=dict(group=gi,source_lane=source,branch_lanes=lanes,target_alphas=group['target_alphas'],entry_alpha=float(state['rollout.alpha']),policy_step=policy_step,time_s=policy_step*.02,first_contact_steps=self.events.first_contact[source].cpu().tolist(),last_contact_steps=self.events.last_contact[source].cpu().tolist(),confirmed_step=int(self.events.confirmed[source]))
            self.log.append(item);self.active.append(item)
        ramp=self.config.get('ramp_s',0.)
        for item in self.active:
            fraction=1. if ramp==0 else min(1.,(policy_step-item['policy_step'])*.02/ramp)
            ids=item['branch_lanes'];target=torch.tensor(item['target_alphas'],device=self.env.device)
            self.env.alpha[ids]=item['entry_alpha']+fraction*(target-item['entry_alpha'])
        (self.output/'contact_forks.json').write_text(json.dumps(self.log,indent=2)+'\n') if len(self.log)>0 and any(x['policy_step']==policy_step for x in self.log) else None

    def close(self):
        (self.output/'contact_branch_status.json').write_text(json.dumps(dict(groups=len(self.groups),forked_groups=len(self.completed),missing_groups=sorted(set(range(len(self.groups)))-self.completed),event=self.config.get('event','rear_clearance'),ramp_s=self.config.get('ramp_s',0.),clearance_plane=dict(direction=self.direction.cpu().tolist(),wheel_centre_threshold_m=self.bound),physics_caches_copied=False,comparison='source is unreset continuation; matching recovery-alpha branch quantifies restore effect'),indent=2)+'\n')
