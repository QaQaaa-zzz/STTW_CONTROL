"""Event-triggered frozen-alpha forks using the existing rollout recorder hook."""
import json
from pathlib import Path
import numpy as np
import torch
from .contact_events import WheelClearanceEvents
from .isaac_path_snapshot import capture,restore
from .contact_torque_geometry import front_contact_torques


class ContactBranching:
    def __init__(self,env,config,output,terrain):
        self.env=env;self.config=config;self.output=Path(output);self.groups=config['groups']
        self.events=WheelClearanceEvents(env.n,env.device,env.physics_dt,config.get('confirmation_s',.02))
        self.log=[];self.active=[];self.completed=set()
        self.impulses={key:torch.zeros(env.n,device=env.device) for key in ['normal_signed','normal_absolute','lateral_signed','lateral_absolute','negative_absolute','migration_signed','migration_absolute']}
        self.front_unloaded_s=torch.zeros(env.n,device=env.device)
        self.front_load=torch.zeros(env.n,device=env.device)
        mesh=np.load(Path(terrain)/'terrain.npz');vertices=mesh['vertices'];raised=vertices[vertices[:,2]>0]
        identity=json.loads((Path(terrain)/'identity.json').read_text())
        anchor,heading,_=env.route.at(torch.tensor([identity['terrain_support_end_s_m']],device=env.device))
        self.direction=torch.stack((torch.cos(heading[0]),torch.sin(heading[0])))
        # A supporting halfspace of ALL raised vertices, including lateral obliquity.
        # Wheel-centre clearance adds the declared bounding radius; no future trajectory is used.
        self.bound=float((raised[:,:2]@self.direction.cpu().numpy()).max())+config['wheel_bound_radius_m']
        self.rear=env.plant.bodies.index('rearwheel')
        if not env.plant.track_contact_wrench or env.plant.servo_mode!='explicit':
            raise ValueError('Contact branches require wheel sensors and explicit servos')

    def record(self,env,lower_tick,physics_tick):
        if len(self.completed)==len(self.groups):return
        samples=env.plant.read_wheel_contact_samples()
        ridge=[];rear_ground=None
        for wi,wheel in enumerate(['frontwheel','rearwheel']):
            force,point,normal,count,start=[samples[wheel+'_'+key] for key in ['normal_force','normal_point','normal_direction','normal_count','normal_start']]
            count=count[:,0];start=start[:,0];slots=torch.arange(int(count.max()),device=env.device)
            valid=slots[None,:]<count[:,None];indices=(start[:,None]+slots[None,:]).clamp_max(len(force)-1)
            load=force[indices,0];height=point[indices,2]-env.plant.origin[:,None,2]
            ridge.append((valid&(load>.01)&(height>.003)).any(1))
            if wi==0:self.front_load=(torch.where(valid,load,0)[:,:,None]*normal[indices]).sum(1)[:,2]
            if wi==1:rear_ground=(torch.where(valid&(height<=.003),load,0).sum(1)>.1)
        rear_xy=env.plant.robot.data.body_link_pos_w[:,self.rear,:2]-env.plant.origin[:,:2]
        beyond=(rear_xy*self.direction).sum(1)>self.bound
        self.events.update(torch.stack(ridge,1),beyond,rear_ground,env.alive)
        if not bool((self.events.first_contact[:,0]>=0).any()):return
        torque=front_contact_torques(samples,env.plant.bodies)
        active=(self.events.first_contact[:,0]>=0)&env.alive
        for prefix,field in [('normal','normal_moment'),('lateral','lateral_moment'),('migration','reference_delta')]:
            value=torch.where(active,torque[field],0.)
            self.impulses[prefix+'_signed']+=value*env.physics_dt
            self.impulses[prefix+'_absolute']+=value.abs()*env.physics_dt
        self.impulses['negative_absolute']+=torch.where(active,torque['negative_lateral'].abs(),0.)*env.physics_dt
        self.front_unloaded_s+=(active&(self.front_load.abs()<.1))*env.physics_dt

    def decision(self,policy_step):
        if len(self.completed)==len(self.groups) and not self.active:return
        ready=self.events.take_ready(self.env.alive)
        if self.config.get('event','rear_clearance')=='front_contact':
            ready=(self.events.first_contact[:,0]>=0)&self.env.alive
        ready=ready.cpu().tolist()
        for gi,group in enumerate(self.groups):
            source=group['source_lane']
            if gi in self.completed or not bool(ready[source]):continue
            lanes=group['branch_lanes'];state=capture(self.env,source);restore(self.env,state,lanes)
            np.savez_compressed(self.output/f'contact_fork_{gi:03d}.npz',**{k:v.cpu().numpy() for k,v in state.items()})
            self.completed.add(gi)
            item=dict(group=gi,source_lane=source,branch_lanes=lanes,target_alphas=group['target_alphas'],entry_alpha=float(state['rollout.alpha']),policy_step=policy_step,time_s=policy_step*.02,first_contact_steps=self.events.first_contact[source].cpu().tolist(),last_contact_steps=self.events.last_contact[source].cpu().tolist(),confirmed_step=int(self.events.confirmed[source]))
            item['contact_features_oracle']=dict(impulse_Nms={key:float(value[source]) for key,value in self.impulses.items()},front_vertical_load_N=float(self.front_load[source]),front_near_zero_vertical_load_s=float(self.front_unloaded_s[source]),scope='first actual front ridge contact through confirmed clearance decision; simulator contact truth, not assumed deployable sensing')
            self.log.append(item);self.active.append(item)
        ramp=self.config.get('ramp_s',0.)
        pending=[]
        for item in self.active:
            fraction=1. if ramp==0 else min(1.,(policy_step-item['policy_step'])*.02/ramp)
            ids=item['branch_lanes'];target=torch.tensor(item['target_alphas'],device=self.env.device)
            self.env.alpha[ids]=item['entry_alpha']+fraction*(target-item['entry_alpha'])
            if fraction<1.:pending.append(item)
        self.active=pending
        (self.output/'contact_forks.json').write_text(json.dumps(self.log,indent=2)+'\n') if len(self.log)>0 and any(x['policy_step']==policy_step for x in self.log) else None

    def close(self):
        (self.output/'contact_branch_status.json').write_text(json.dumps(dict(groups=len(self.groups),forked_groups=len(self.completed),missing_groups=sorted(set(range(len(self.groups)))-self.completed),event=self.config.get('event','rear_clearance'),ramp_s=self.config.get('ramp_s',0.),clearance_plane=dict(direction=self.direction.cpu().tolist(),wheel_centre_threshold_m=self.bound),physics_caches_copied=False,comparison='source is unreset continuation; matching recovery-alpha branch quantifies restore effect'),indent=2)+'\n')
