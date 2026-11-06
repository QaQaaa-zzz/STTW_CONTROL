"""Phase U on the common Isaac bicycle, using the native Go2 VecEnv contract."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from isaaclab.envs.mdp.events import push_by_setting_velocity
from .isaaclab_vector_env import IsaacLabVectorPlant
from .isaaclab_randomization import randomize_vehicle_properties,PendulumControlRandomization
from .pendulum_skill import TorchPendulumObservation,TorchPendulumGeometry,TorchPendulumSkill
from .phase_u_events import initial_events,advance_events,classify_terminal
from .phase_u_reward import phase_u_reward,phase_u_episode_reward


class PhaseUEnv:
    def __init__(self,cfg,output):
        self.cfg=cfg;self.device=cfg['device'];self.num_envs=cfg['num_envs'];self.dt=.02
        self.num_obs=76;self.num_privileged_obs=106;self.num_actions=4
        spec=json.loads(Path(cfg['task_spec']).read_text());self.task=spec['task'];self.rand=spec['randomization']
        self.max_episode_length=self.task['ppo']['episode_horizon'];self.all_ids=torch.arange(self.num_envs,device=self.device)
        # One source-size platform and surrounding ground, replicated with environments.
        terrain=Path(output)/'terrain';terrain.mkdir(parents=True)
        x,y,z=5.6,0.,0.;sx,sy,sz=2.,1.,.16
        vertices=np.array([[x+dx*sx,y+dy*sy,z+dz*sz] for dx,dy,dz in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]],dtype=np.float32)
        triangles=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7],[8,9,10],[8,10,11]],dtype=np.int32)
        vertices=np.concatenate((vertices,np.array([[-5,-4,0],[15,-4,0],[15,4,0],[-5,4,0]],dtype=np.float32)))
        np.savez(terrain/'terrain.npz',vertices=vertices,triangles=triangles)
        (terrain/'identity.json').write_text(json.dumps(dict(sha256=hashlib.sha256((terrain/'terrain.npz').read_bytes()).hexdigest(),platform_front_x=3.6,platform_top_z=.16)))
        self.plant=IsaacLabVectorPlant(Path(cfg['asset']),[str(terrain)],[0]*self.num_envs,Path(output),**cfg['plant_options'])
        self.physics_substeps=round(self.dt/self.plant.physics_dt)
        self.sensor_substeps=round(self.rand['sensor_dt_s']/self.plant.physics_dt)
        if self.sensor_substeps<1 or abs(self.physics_substeps*self.plant.physics_dt-self.dt)>1e-9 or abs(self.sensor_substeps*self.plant.physics_dt-self.rand['sensor_dt_s'])>1e-9:
            raise ValueError('physics_dt must divide policy and sensor periods')
        if 'jump_steering_gains' in cfg:
            self.plant.jump_steering_servo.kp.fill_(cfg['jump_steering_gains'][0]);self.plant.jump_steering_servo.kd.fill_(cfg['jump_steering_gains'][1])
        self.properties=randomize_vehicle_properties(self.plant,self.rand)
        self.randomization=PendulumControlRandomization(self.plant,self.rand)
        mount=json.loads(Path(cfg['asset_spec']).read_text())['mount_rotation_wxyz']
        self.history=TorchPendulumObservation(self.plant.joints,mount,self.num_envs,self.device)
        self.geometry=TorchPendulumGeometry(json.loads(Path(cfg['geometry']).read_text()),self.plant.bodies,self.device)
        policy=TorchPendulumSkill(cfg['bundle'],self.device)
        self.mapping=policy.mapping;self.lo=policy.lo;self.hi=policy.hi
        self.episode_length_buf=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.episode_return=torch.zeros(self.num_envs,device=self.device)
        self.last_action=torch.zeros(self.num_envs,4,device=self.device)
        self.obs_buf=torch.zeros(self.num_envs,76,device=self.device)
        self.privileged_obs_buf=torch.zeros(self.num_envs,106,device=self.device)
        self.airborne=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.events=initial_events(torch.full((self.num_envs,),1.5,device=self.device),self.task)
        self.total_physics_steps=0

    def get_observations(self):return self.obs_buf
    def get_privileged_observations(self):return self.privileged_obs_buf

    def physical(self):
        root=self.plant.roots.clone();root[:,:3]-=self.plant.origin
        contact=(self.plant.contact[:,self.plant.nonwheel].square().sum(-1)>0).any(-1)
        critic=self.history.privileged(self.obs_buf,root,self.plant.dofs[:,:,0],self.plant.dofs[:,:,1],contact)
        return root,contact,critic

    def reset_idx(self,ids):
        n=len(ids);r=self.task['reset'];plant=self.plant
        pose=plant.robot.data.default_root_state[ids,:7].clone();pose[:,0]=1.5;pose[:,1]=0.
        velocity=torch.zeros(n,6,device=self.device);velocity[:,0]=r['initial_forward_velocity']
        self.airborne[ids]=torch.rand(n,device=self.device)<r['airborne_rsi_probability']
        for tensor,column,label in ((pose,0,'x'),(pose,2,'z'),(velocity,0,'vx'),(velocity,2,'vz')):
            lo,hi=r[f'airborne_rsi_{label}_min'],r[f'airborne_rsi_{label}_max']
            tensor[:,column]=torch.where(self.airborne[ids],lo+(hi-lo)*torch.rand(n,device=self.device),tensor[:,column])
        position=plant.robot.data.default_joint_pos[ids].clone();joint_velocity=torch.zeros_like(position)
        plant.reset_idx(ids,state=dict(root_pose=pose,root_velocity=velocity,joint_position=position,joint_velocity=joint_velocity))
        self.randomization.reset(ids);self.history.frames[ids]=0;self.history.valid_count[ids]=0
        self.last_action[ids]=0;self.episode_length_buf[ids]=0;self.episode_return[ids]=0
        fresh=initial_events(pose[:,0],self.task)
        for k in self.events:self.events[k][ids]=fresh[k]
        self.obs_buf[ids]=0;self.obs_buf[ids,-1]=fresh['jump_signal'].float()
        self.privileged_obs_buf[ids]=self.physical()[2][ids]

    def reset(self):
        self.reset_idx(self.all_ids)
        return self.obs_buf,self.privileged_obs_buf

    def targets(self,action):
        m=self.mapping
        def angle(a,initial,i):return torch.where(a>=0,initial+a*(self.hi[i]-initial),initial+a*(initial-self.lo[i]))
        target=torch.stack((action[:,0]*self.hi[0],m['base_rear_speed']+action[:,1]*m['rear_speed_delta'],angle(action[:,2],m['hip_initial'],2),angle(action[:,3],m['knee_initial'],3)),-1)
        return torch.maximum(self.lo,torch.minimum(self.hi,target))

    def step(self,latent_actions):
        plant=self.plant;action=torch.tanh(latent_actions);target=self.targets(action)
        previous_forward_velocity=plant.roots[:,7].clone()
        push_ticks=round(self.rand['push_interval_s']/self.dt)
        ids=((self.episode_length_buf>0)&(self.episode_length_buf%push_ticks==0)).nonzero().flatten()
        if len(ids):push_by_setting_velocity(SimpleNamespace(scene={'robot':plant.robot}),ids,self.rand['push_velocity_range'])
        previous_root_velocity=plant.robot.data.root_link_lin_vel_w.clone()
        for tick in range(self.physics_substeps):
            delivered=self.randomization.command(target)
            command=torch.stack((torch.zeros_like(delivered[:,0]),delivered[:,1]),-1)
            plant.step(command,pendulum_target=delivered[:,2:],steering_position_target=delivered[:,:1],refresh_diagnostics=False)
            root_velocity=plant.robot.data.root_link_lin_vel_w
            root_acceleration=(root_velocity-previous_root_velocity)/plant.physics_dt;previous_root_velocity=root_velocity.clone()
            if (tick+1)%self.sensor_substeps==0:
                plant.refresh();root=plant.roots.clone();root[:,:3]-=plant.origin
                distance=3.6-(self.geometry.frontmost_x(plant.rigid)-plant.origin[:,0])
                acceleration=root_acceleration if self.cfg.get('root_acceleration_difference',False) else plant.robot.data.body_com_acc_w[:,plant.bodies.index('chasis'),:3]
                frame=self.history.make_frame(root,plant.dofs[:,:,0],plant.dofs[:,:,1],acceleration,distance,action)
                measured=self.randomization.sensor_frame(frame)
        plant.sensor.update(self.dt,force_recompute=True);plant.refresh()
        root,contact,critic=self.physical();angles=critic[:,99:102]
        nonfinite=~torch.isfinite(root).all(-1)|~torch.isfinite(plant.dofs).all(dim=(1,2))|~torch.isfinite(action).all(-1)
        inputs=dict(episode_step=self.events['episode_step'],nonfinite=nonfinite,roll=angles[:,0],pitch=angles[:,1],illegal_contact=contact,backward_exit=root[:,0]<1.5-self.task['physical_limits']['max_backward_distance'],stuck=self.events['stuck'],yaw=angles[:,2],jump_zone_seen=self.events['jump_zone_seen'])
        preliminary=classify_terminal(inputs,self.task)
        previous=self.events
        self.events=advance_events(previous,dict(x=root[:,0],z=root[:,2],forward_velocity=root[:,7],vertical_velocity=root[:,9],physical_failure=preliminary['terminated']),self.task)
        inputs.update(stuck=self.events['stuck'],jump_zone_seen=self.events['jump_zone_seen'])
        terminal=classify_terminal(inputs,self.task)
        reward_events=dict(jump_zone_seen=self.events['jump_zone_seen'],jump_signal=self.events['jump_signal'],first_apex_success=self.events['apex_seen']&~previous['apex_seen'],illegal_contact=contact,physical_failure_transition=terminal['physical_failure'],roll_pitch_failure_transition=terminal['roll_limit']|terminal['pitch_limit'],jump_zone_missed_transition=terminal['jump_zone_missed'],stuck_transition=terminal['stuck'],yaw_limit_transition=terminal['yaw_limit'],timeout_transition=terminal['timeout'])
        hip,knee=plant.pendulum_servos.indices
        reward_state=dict(previous_forward_velocity=previous_forward_velocity,policy_dt=self.dt,x=root[:,0],roll=angles[:,0],pitch=angles[:,1],yaw=angles[:,2],z=root[:,2],forward_velocity=root[:,7],pitch_rate=root[:,11],hip_velocity=plant.dofs[:,hip,1],knee_velocity=plant.dofs[:,knee,1],hip_force=plant.last_pendulum_torque[:,hip],knee_force=plant.last_pendulum_torque[:,knee])
        clipped,unclipped,parts=phase_u_reward(reward_state,action,self.last_action,reward_events,self.task['reward'])
        reward,self.episode_return,override=phase_u_episode_reward(clipped,self.episode_return,terminal['stuck'],terminal['yaw_limit'],self.task['reward'])
        done=terminal['terminated']|terminal['truncated'];self.episode_length_buf+=1;self.last_action.copy_(action)
        self.obs_buf=self.history.advance_frame(measured,self.events['jump_signal'].float())
        self.privileged_obs_buf=self.history.privileged(self.obs_buf,root,plant.dofs[:,:,0],plant.dofs[:,:,1],contact)
        info=dict(time_outs=terminal['truncated'].float(),reward_parts={**parts,'episode_return_override':override},end_code=terminal['end_code'],apex=self.events['apex_seen'].clone(),unscaled_reward=reward)
        ids=done.nonzero().flatten()
        if len(ids):
            info['episode']=dict(reward=self.episode_return[ids].mean(),length=self.episode_length_buf[ids].float().mean(),apex_fraction=self.events['apex_seen'][ids].float().mean(),physical_failure=terminal['physical_failure'][ids].float().mean())
            self.reset_idx(ids)
        self.total_physics_steps+=self.physics_substeps*self.num_envs
        return self.obs_buf,self.privileged_obs_buf,reward*self.task['ppo']['reward_scaling'],done,info
