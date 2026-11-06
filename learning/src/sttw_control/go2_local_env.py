"""STTW lower task implementing the native Go2 RSL-RL VecEnv interface."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from types import SimpleNamespace
from isaaclab.envs.mdp.events import push_by_setting_velocity
from rsl_rl.env import VecEnv
from .isaaclab_vector_env import IsaacLabVectorPlant
from .isaaclab_randomization import randomize_vehicle_properties, VehicleControlRandomization
from .torch_controller import TorchController
from .torch_local_lower import TorchLocalLower


class Go2LocalEnv(VecEnv):
    def __init__(self, cfg, output):
        self.cfg = cfg
        self.device = cfg['device']
        self.num_envs = cfg['num_envs']
        self.num_obs = 210
        self.task_spec = json.loads(Path(cfg['task_spec']).read_text()) if cfg.get('task_spec') else None
        self.num_privileged_obs = 211 if self.task_spec else 210
        self.num_actions = 2
        self.dt = cfg['control_dt']
        self.max_episode_length = round(cfg['episode_s']/self.dt)
        self.plant = IsaacLabVectorPlant(cfg['asset'], ['plane'], [0]*self.num_envs, output, **cfg.get('plant_options', {}))
        self.decimation = round(self.dt/self.plant.sim.cfg.dt)
        self.wheel_radius_m = self.plant.wheel_radius_m
        controller_cfg=json.loads(Path(cfg['controller_config']).read_text());controller_cfg['wheel_radius_m']=self.wheel_radius_m
        self.controller = TorchController(controller_cfg, self.num_envs, self.device)
        self.history = TorchLocalLower(cfg['history_actor'], cfg['history_actor_sha256'], self.num_envs, self.device)
        data = np.load(cfg['initial_state'])
        order = [list(data['joint_names']).index(name) for name in self.plant.joints]
        self.initial = {key: torch.as_tensor(data[key], device=self.device, dtype=torch.float32)
                        for key in ('root_pose', 'root_velocity', 'eso', 'disturbance', 'gains', 'previous_reference', 'previous_command')}
        for key in ('joint_position', 'joint_velocity'):
            self.initial[key] = torch.as_tensor(data[key][...,order], device=self.device, dtype=torch.float32)
        self.initial_count = len(self.initial['root_pose']) if self.initial['root_pose'].ndim == 2 else 1
        self.properties = randomize_vehicle_properties(self.plant, cfg['randomization'])
        self.randomization = VehicleControlRandomization(self.plant, cfg['randomization'])
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reference = torch.zeros(self.num_envs, 2, device=self.device)
        self.previous_reference = torch.zeros_like(self.reference)
        self.target = torch.zeros_like(self.reference)
        self.command = torch.zeros_like(self.reference)
        self.previous_action = torch.zeros_like(self.reference)
        self.command_countdown = torch.zeros_like(self.episode_length_buf)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.all_ids = torch.arange(self.num_envs, device=self.device)
        self.total_steps = 0
        self.command_bank = None
        if cfg.get('command_bank'):
            bank=np.load(Path(cfg['command_bank'])/'commands.npz')
            self.command_bank={key:torch.as_tensor(bank[key],device=self.device,dtype=torch.float32) for key in ('rows','slew','labels')}
            self.schedule=torch.zeros(self.num_envs,16,3,device=self.device)
            self.slew=torch.zeros(self.num_envs,2,device=self.device)
            self.task_labels=torch.zeros(self.num_envs,7,device=self.device)
        self.reset()

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, ids):
        state_ids = torch.randint(self.initial_count, (len(ids),), device=self.device)
        initial = ({key: value[state_ids] for key, value in self.initial.items()}
                   if self.initial['root_pose'].ndim == 2 else self.initial)
        self.plant.reset_idx(ids, state=initial)
        self.controller.reset(ids)
        for key in ('eso', 'disturbance', 'gains'):
            getattr(self.controller, key)[ids] = initial[key]
        self.history.reset(ids)
        self.randomization.reset(ids)
        self.reference[ids] = initial['previous_reference']
        self.previous_reference[ids] = self.reference[ids]
        self.target[ids] = self.reference[ids]
        self.command[ids] = initial['previous_command']
        self.previous_action[ids] = 0
        self.command_countdown[ids] = round(self.cfg['commands']['initial_hold_s']/self.dt)
        self.episode_length_buf[ids] = 0
        self.episode_return[ids] = 0
        if self.command_bank is not None:
            grouped = self.command_bank['rows'].ndim == 4
            count = self.command_bank['rows'].shape[1 if grouped else 0]
            draw = torch.randint(count, (len(ids),), device=self.device)
            index = (state_ids, draw) if grouped else draw
            self.schedule[ids] = self.command_bank['rows'][index]
            self.slew[ids] = self.command_bank['slew'][index]
            self.task_labels[ids] = self.command_bank['labels'][index]

    def reset(self):
        self.reset_idx(self.all_ids)
        measurement, _, speed = self.plant.observe()
        self.measurement = measurement
        self._observation(speed)
        return self.obs_buf, self.privileged_obs_buf

    def _observation(self, speed):
        self.obs_buf = self.history.observation(self.measurement, speed, self.reference,
            self.previous_reference, self.command, self.controller)
        self.privileged_obs_buf = (torch.cat((self.obs_buf,(1-self.episode_length_buf/self.max_episode_length)[:,None]),-1)
                                   if self.task_spec else self.obs_buf)

    def _commands(self):
        if self.command_bank is not None:
            active=(self.schedule[:,:,0]<=self.episode_length_buf[:,None]*self.dt)&(self.schedule[:,:,0]<99.)
            indices=torch.where(active,torch.arange(16,device=self.device),0).max(-1).values
            self.target=self.schedule[self.all_ids,indices,1:]
            raw=self.reference
            proposed=raw+(self.target-raw).clamp(-self.slew*self.dt,self.slew*self.dt)
            design=self.task_spec['command_design'];roll_bound=design['rare_roll_reference']
            ratio=self.controller._system(torch.maximum(raw[:,0],self.target[:,0]))[3].abs()
            limit=torch.minimum(ratio*roll_bound,torch.full_like(ratio,design['abs_steer_max_edge']))
            desired=self.target[:,1].clamp(-limit,limit)
            delta=raw[:,1]+(desired-raw[:,1]).clamp(-self.slew[:,1]*self.dt,self.slew[:,1]*self.dt)
            safe=(delta/self.controller._system(proposed[:,0])[3]).abs()<=roll_bound+1e-7
            self.previous_reference.copy_(raw)
            self.reference=torch.stack((torch.where(safe,proposed[:,0],raw[:,0]),delta),-1)
            return
        ids = (self.command_countdown <= 0).nonzero(as_tuple=False).flatten()
        n = len(ids)
        c = self.cfg['commands']
        low, high = c['speed_m_s']
        speed = low+(high-low)*torch.rand(n, device=self.device)
        candidate_speed = self.target[:, 0].clone()
        candidate_speed[ids] = speed
        ratio = self.controller._system(candidate_speed)[3][ids].abs()
        limit = torch.minimum(ratio*c['max_reference_roll_rad'], torch.full_like(speed, c['max_steer_rad']))
        steer = (2*torch.rand(n, device=self.device)-1)*limit
        steer = torch.where(torch.rand(n, device=self.device)<c['straight_probability'], 0., steer)
        self.target[ids] = torch.stack((speed, steer), -1)
        low, high = c['hold_s']
        self.command_countdown[ids] = ((low+(high-low)*torch.rand(n, device=self.device))/self.dt).long()
        self.previous_reference.copy_(self.reference)
        slew = torch.tensor(c['slew'], device=self.device)*self.dt
        proposed = self.reference+(self.target-self.reference).clamp(-slew, slew)
        ratio = self.controller._system(torch.maximum(self.reference[:, 0], proposed[:, 0]))[3].abs()
        bound = torch.minimum(ratio*c['max_reference_roll_rad'], torch.full_like(ratio, c['max_steer_rad']))
        proposed[:, 1] = proposed[:, 1].clamp(-bound, bound)
        self.reference.copy_(proposed)
        self.command_countdown -= 1

    def step(self, actions):
        push_ticks = round(self.cfg['randomization']['push_interval_s']/self.dt)
        push_ids = ((self.episode_length_buf > 0) & (self.episode_length_buf % push_ticks == 0)).nonzero(as_tuple=False).flatten()
        if len(push_ids):
            push_by_setting_velocity(SimpleNamespace(scene={'robot': self.plant.robot}), push_ids,
                self.cfg['randomization']['push_velocity_range'])
            self.plant.refresh()
        action = torch.tanh(actions)
        m = self.plant.observe()[0] if self.cfg.get('controller_clean_feedback', False) else self.measurement
        row = torch.stack((m[:,5]*self.wheel_radius_m,m[:,2],m[:,3],m[:,0],m[:,1],self.reference[:,1]), -1)
        output = self.controller.step(row, torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        applied = action*torch.tensor([1.5,10.], device=self.device)
        steer = (output[:,0]+applied[:,0]).clamp(-3,3)
        steer = torch.maximum(((-.8-m[:,2])/self.dt).clamp(min=-3),
                              torch.minimum(((.8-m[:,2])/self.dt).clamp(max=3), steer))
        self.command = torch.stack((steer,(self.reference[:,0]/self.wheel_radius_m+applied[:,1]).clamp(-60,60)), -1)
        _, before, _ = self.plant.observe()
        failure = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        peak = torch.zeros(self.num_envs, device=self.device)
        for _ in range(self.decimation):
            contact = self.plant.step(self.randomization.command(self.command))
            clean, pose, speed = self.plant.observe()
            failure |= contact | (clean[:,0].abs()>.7) | ~torch.isfinite(self.plant.roots).all(-1) | ~torch.isfinite(clean).all(-1)
            peak = torch.maximum(peak, clean[:,0].abs())
            self.measurement = self.randomization.observe(clean)
        yaw = torch.atan2(torch.sin(pose[:,2]-before[:,2]),torch.cos(pose[:,2]-before[:,2]))/self.dt
        self.episode_length_buf += 1
        r = self.cfg['reward']
        parts = dict(speed=torch.exp(-((speed-self.reference[:,0])/r['speed_scale'])**2),
                     steer=torch.exp(-((clean[:,2]-self.reference[:,1])/r['steer_scale'])**2),
                     roll=-torch.relu(peak-r['roll_start'])**2,
                     roll_rate=-clean[:,1]**2,
                     action=-action.square().sum(-1),
                     action_change=-(action-self.previous_action).square().sum(-1))
        reward = self.dt*sum(r['weights'][key]*value for key,value in parts.items())
        reward = torch.where(failure, torch.full_like(reward,-r['failure_penalty']), reward)
        if self.task_spec:
            r=self.task_spec['reward'];caps=r['caps']
            def huber(x):return torch.where(x.abs()<=1,x*x,2*x.abs()-1)
            costs=dict(speed=r['speed_weight']*huber((speed-self.reference[:,0])/r['speed_huber_scale']),
                steer=r['steer_weight']*huber((clean[:,2]-self.reference[:,1])/r['steer_huber_scale']),
                roll=r['roll_weight']*huber(torch.relu(peak-r['roll_start'])/r['roll_scale']),
                working_roll=r['working_roll_weight']*huber(torch.relu(peak-r['working_roll_start'])/r['working_roll_scale']),
                roll_rate=r['roll_rate_weight']*huber(torch.relu(clean[:,1].abs()-r['roll_rate_start'])/r['roll_rate_scale']),
                action=r['action_weight']*action.square().sum(-1),
                action_change=r['action_change_weight']*(action-self.previous_action).square().sum(-1))
            if 'yaw_rate_weight' in r:
                yaw_target=self.reference[:,0]*torch.tan(self.reference[:,1])*math.cos(self.controller.config['caster'])/self.controller.config['wheelbase']
                costs['yaw_rate']=r['yaw_rate_weight']*huber((yaw-yaw_target)/r['yaw_rate_scale'])
            terms={key:torch.where(failure,0.,-r['scale']*self.dt*value.clamp(max=caps[key])) for key,value in costs.items()}
            remaining=(self.max_episode_length-self.episode_length_buf+1).clamp_min(0)
            gamma=self.task_spec['gamma']
            tail=remaining if gamma==1 else -torch.expm1(remaining*math.log(gamma))/(1-gamma)
            terms['failure']=torch.where(failure,-r['failure_extra']-r['scale']*self.dt*r['failure_cost_rate_bound']*tail,0.)
            reward=sum(terms.values())
            self.reward_parts=terms
        reward = torch.nan_to_num(reward, nan=-self.cfg['reward']['failure_penalty'])
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~failure
        done = failure | timeout
        self.episode_return += reward
        self.previous_action.copy_(action)
        self.history.finish(applied,yaw)
        ids = done.nonzero(as_tuple=False).flatten()
        extras = {'time_outs': torch.zeros_like(timeout) if self.task_spec else timeout}
        if len(ids):
            extras['episode'] = dict(reward=self.episode_return[ids].mean(),
                                     failure=failure[ids].float().mean(),
                                     duration_s=self.episode_length_buf[ids].float().mean()*self.dt)
            self.reset_idx(ids)
            clean, _, speed = self.plant.observe()
            self.measurement[ids] = clean[ids]
        self._commands()
        self._observation(speed)
        self.total_steps += 1
        return self.obs_buf, self.privileged_obs_buf, reward, done, extras
