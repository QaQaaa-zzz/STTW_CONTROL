"""Native Go2 VecEnv interface for original H75 task states and switching.

Task physics/control/rewards remain shared with the fixed-case rollout. Go2
owns sampling, storage, GAE and PPO; each finished lane resets independently.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .isaaclab_vector_env import IsaacLabVectorPlant
from .isaac_path_rollout import IsaacPathRollout
from .torch_path_policy import TorchPathPolicy
from .torch_path_reference import TorchPathReference
from .torch_middle_control import TorchMiddleControl
from .torch_observation import rotate_inverse


class Go2PathEnv:
    def __init__(self,cfg,output):
        self.cfg=cfg;self.device=cfg['device'];self.num_envs=cfg['num_envs'];self.num_obs=351;self.num_privileged_obs=352;self.num_actions=2;self.max_episode_length=500
        bundle=Path(cfg['bundle']);identity=json.loads((bundle/'identity.json').read_text());config=json.loads((bundle/'config.json').read_text())
        terrains=list(cfg['terrains'].values());self.plant=IsaacLabVectorPlant(cfg['asset'],terrains,[i%len(terrains) for i in range(self.num_envs)],output,**cfg['plant_options'])
        policy=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],config,self.num_envs,self.device)
        settings=dict(cfg,episode_s=10,diagnostics=False,zero_middle=[False]*self.num_envs,auto_reset_horizon=False)
        self.rollout=IsaacPathRollout(bundle,settings,policy,plant=self.plant)
        saved=np.load(Path(cfg['training_bank'])/'states.npz');self.bank={key:torch.as_tensor(saved[key],device=self.device) for key in saved.files if key!='joint_names'}
        order=[list(saved['joint_names']).index(name) for name in self.plant.joints]
        for key in ('joint_position','joint_velocity'):self.bank[key]=self.bank[key][:,order]
        routes=np.load(Path(cfg['training_bank'])/'routes.npz');self.speeds=torch.tensor(routes['speed'],device=self.device);self.families=torch.tensor(routes['family'],device=self.device);self.turn_end=torch.tensor(routes['turn_end'],device=self.device)
        self.route=TorchPathReference(routes,config,self.device,route_ids=torch.zeros(self.num_envs,device=self.device,dtype=torch.long));r=self.rollout;r.route=self.route;r.middle=TorchMiddleControl(r.policy,self.route,identity['frame_scales'],r.alpha,r.speed)
        for group in ('projection','follower'):
            prefix='closedloop.'+group+'.';setattr(r.middle,group,{key[len(prefix):]:torch.zeros((self.num_envs,)+value.shape[1:],device=self.device,dtype=value.dtype) for key,value in self.bank.items() if key.startswith(prefix)})
        r.middle.nominal=torch.zeros(self.num_envs,2,device=self.device);r.middle.rates=torch.zeros_like(r.middle.nominal);r.middle.pose=torch.zeros(self.num_envs,3,device=self.device);r.middle.yaw_rate=torch.zeros(self.num_envs,device=self.device);r.middle.fault=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool);r.middle.domain_exit=r.middle.fault.clone();r.middle.tick=0
        self.episode_length_buf=torch.zeros(self.num_envs,device=self.device,dtype=torch.long);self.episode_reward=torch.zeros(self.num_envs,device=self.device);self.error_squares=torch.zeros(self.num_envs,3,device=self.device)
        self.switch_enabled=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool);self.switch_progress=torch.zeros(self.num_envs,device=self.device);self.target_alpha=torch.zeros(self.num_envs,device=self.device);self.switched=torch.zeros_like(self.switch_enabled)
        self.force_levels=torch.tensor(cfg['pulse']['forces_N'],device=self.device);self.force_world=torch.zeros(self.num_envs,3,device=self.device);self.pulse_start=torch.zeros(self.num_envs,device=self.device,dtype=torch.long);self.impulse=torch.zeros_like(self.force_world)
        self.chassis=torch.tensor([self.plant.bodies.index(cfg['randomization']['base_body'])],device=self.device);self.site_position=torch.tensor(cfg['pulse']['site_body_position_m'],device=self.device).expand(self.num_envs,1,3).contiguous();self.zero_torque=torch.zeros(self.num_envs,1,3,device=self.device);r.training_pulse=self.apply_pulse
        self.terrain_center=torch.zeros(self.num_envs,3,device=self.device);self.terrain_kind_id=torch.zeros(self.num_envs,device=self.device,dtype=torch.long);self.terrain_entered=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool);self.terrain_cleared=self.terrain_entered.clone();self.terrain_contacted=self.terrain_entered.clone()
        self.sampling_steps=0;self.sampling_started=time.time()
        self.reset()

    def reset_idx(self,ids):
        r=self.rollout;count=len(ids);picked=torch.randint(len(self.speeds),(count,),device=self.device);self.route.route_ids[ids]=picked
        state={key:self.bank[key][picked].float() for key in ('root_pose','root_velocity','joint_position','joint_velocity')}
        if self.cfg.get('random_terrain_reset',False):
            layout=self.plant.metadata['terrain_layout']
            terrain=torch.randint(len(self.cfg['terrains']),(count,),device=self.device)
            clone=torch.as_tensor(layout['clone_origins'],device=self.device)[ids]
            offsets=torch.as_tensor(layout['template_offsets'],device=self.device)[terrain]
            self.terrain_center[ids]=clone+offsets;self.terrain_kind_id[ids]=terrain;self.terrain_entered[ids]=False;self.terrain_cleared[ids]=False;self.terrain_contacted[ids]=False
            origin=clone+offsets
            origin[:,:2]-=state['root_pose'][:,:2]
            origin[:,:2]+=(torch.rand(count,2,device=self.device)-.5)
            self.plant.origin[ids]=origin
        self.plant.reset_idx(ids,state=state)
        for key in ('eso','disturbance','gains'):getattr(r.controller,key)[ids]=self.bank['closedloop.physical.controller.'+key][picked]
        for key in ('frames','mask','previous_applied','yaw_rate'):getattr(r.lower,key)[ids]=self.bank['closedloop.lower.'+key][picked]
        r.policy.filtered[ids]=self.bank['closedloop.filtered'][picked];r.policy.offsets[ids]=self.bank['closedloop.offsets'][picked];r.policy.execution_mode[ids]=0
        r.previous_reference[ids]=self.bank['previous_reference'][picked];r.previous_command[ids]=self.bank['previous_command'][picked];r.alpha[ids]=self.bank['closedloop.alpha'][picked];r.speed[ids]=self.speeds[picked]
        r.task_tick[ids]=self.bank['closedloop.tick'][picked].long();r.physical_tick[ids]=self.bank['physical_tick'][picked].long();r.since_reset_tick[ids]=0;r.alive[ids]=True
        m,pose,_=self.plant.observe();r.measurement[ids]=m[ids];r.randomization.reset(ids)
        r.middle.frames[ids]=self.bank['closedloop.history.frames'][picked];r.middle.mask[ids]=self.bank['closedloop.history.mask'][picked];r.middle.nominal[ids]=self.bank['closedloop.physical.raw'][picked];r.middle.rates[ids]=self.bank['closedloop.nominal_rates'][picked];r.middle.pose[ids]=pose[ids];r.middle.yaw_rate[ids]=self.bank['closedloop.yaw_rate'][picked];r.middle.fault[ids]=False;r.middle.domain_exit[ids]=False
        for group in ('projection','follower'):
            values=getattr(r.middle,group)
            for key in values:values[key][ids]=self.bank['closedloop.'+group+'.'+key][picked]
        self.episode_length_buf[ids]=0;self.episode_reward[ids]=0;self.error_squares[ids]=0;self.switched[ids]=False
        progress=r.middle.projection['progress'][ids];self.switch_progress[ids]=progress+.5+torch.rand(count,device=self.device)
        self.switch_enabled[ids]=(torch.rand(count,device=self.device)<self.cfg['switch_probability'])&(self.families[picked]>0)&(self.switch_progress[ids]<self.turn_end[picked]-.2)
        levels=torch.tensor([0,.25,.5,.75,1],device=self.device);self.target_alpha[ids]=levels[torch.randint(5,(count,),device=self.device)]
        force=self.force_levels[torch.randint(len(self.force_levels),(count,),device=self.device)];force*=torch.rand(count,device=self.device)<self.cfg['pulse']['probability'];yaw=pose[ids,2];self.force_world[ids]=force[:,None]*torch.stack((-yaw.sin(),yaw.cos(),torch.zeros_like(yaw)),dim=-1);self.pulse_start[ids]=r.physical_tick[ids];self.impulse[ids]=0

    def apply_pulse(self,physics_tick):
        r=self.rollout;active=(r.physical_tick-self.pulse_start<round(self.cfg['pulse']['duration_s']/.005))&r.alive;force=self.force_world*active[:,None];self.impulse+=force*.0002
        local=rotate_inverse(self.plant.roots[:,3:7],force)
        self.plant.robot.permanent_wrench_composer.set_forces_and_torques(forces=local[:,None].contiguous(),torques=self.zero_torque,positions=self.site_position,body_ids=self.chassis,is_global=False)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs,device=self.device));return self.get_observations(),self.get_privileged_observations()

    def get_observations(self):
        r=self.rollout;event=self.switch_enabled&~self.switched&(r.middle.projection['progress']>=self.switch_progress);r.alpha.copy_(torch.where(event,self.target_alpha,r.alpha));self.switched|=event
        self.obs_buf,self.privileged_obs_buf=r.observation();return self.obs_buf

    def get_privileged_observations(self):return self.privileged_obs_buf

    def step(self,actions):
        if self.cfg.get('profile_first_step',False) and self.sampling_steps==0:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                out=self.rollout.step(latent=actions)
            Path(self.cfg['profile_output']).write_text(profile.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
        else:out=self.rollout.step(latent=actions)
        distance=torch.linalg.vector_norm(self.plant.roots[:,:2]-self.terrain_center[:,:2],dim=-1)
        self.terrain_entered|=(distance>=2.6)&(self.terrain_kind_id>0)
        self.terrain_cleared|=(distance>=3.4)&(self.terrain_kind_id>0)
        for sensor in self.plant.wheel_sensors:
            points=sensor.data.contact_pos_w.reshape(self.num_envs,-1,3);forces=sensor.data.force_matrix_w.reshape(self.num_envs,-1,3)
            self.terrain_contacted|=((points[:,:,2]>self.terrain_center[:,2,None]+.003)&(torch.linalg.vector_norm(forces,dim=-1)>.01)).any(-1)&(self.terrain_kind_id>0)
        self.last_exposure=dict(contacted=self.terrain_contacted.clone(),entered=self.terrain_entered.clone(),cleared=self.terrain_cleared.clone(),terrain=self.terrain_kind_id.clone())
        self.last_out=out
        self.last_terminal=dict(domain=(out['failure']&self.rollout.middle.domain_exit).clone(),fault=(out['failure']&self.rollout.middle.fault).clone())
        self.episode_length_buf+=1;self.episode_reward+=out['reward'];self.error_squares+=out['tracking_error'].square();ids=out['done'].nonzero(as_tuple=False).flatten();infos={}
        if len(ids):
            infos['episode']=dict(reward=self.episode_reward[ids].mean(),length=self.episode_length_buf[ids].float().mean()*.02,failure=out['failure'][ids].float().mean(),speed_rmse=torch.sqrt(self.error_squares[ids,0]/self.episode_length_buf[ids]).mean(),path_rmse=torch.sqrt(self.error_squares[ids,1]/self.episode_length_buf[ids]).mean(),switched=self.switched[ids].float().mean())
            if not self.cfg.get('evaluation_mode',False):self.reset_idx(ids)
        obs=self.get_observations();self.sampling_steps+=1
        if self.sampling_steps==1 or self.sampling_steps%6==0:print(json.dumps(dict(sampling_policy_steps=self.sampling_steps,elapsed_sampling_s=time.time()-self.sampling_started)),flush=True)
        return obs,self.privileged_obs_buf,out['reward'],out['done'],infos
