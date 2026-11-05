"""STTW lower task implementing the native Go2 RSL-RL VecEnv interface."""
import json
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
        self.num_obs = self.num_privileged_obs = 210
        self.num_actions = 2
        self.dt = cfg['control_dt']
        self.max_episode_length = round(cfg['episode_s']/self.dt)
        self.plant = IsaacLabVectorPlant(cfg['asset'], ['plane'], [0]*self.num_envs, output)
        self.decimation = round(self.dt/self.plant.sim.cfg.dt)
        self.controller = TorchController(json.loads(Path(cfg['controller_config']).read_text()), self.num_envs, self.device)
        self.history = TorchLocalLower(cfg['history_actor'], cfg['history_actor_sha256'], self.num_envs, self.device)
        data = np.load(cfg['initial_state'])
        order = [list(data['joint_names']).index(name) for name in self.plant.joints]
        self.initial = {key: torch.as_tensor(data[key], device=self.device, dtype=torch.float32)
                        for key in ('root_pose', 'root_velocity', 'eso', 'disturbance', 'gains', 'previous_reference', 'previous_command')}
        for key in ('joint_position', 'joint_velocity'):
            self.initial[key] = torch.as_tensor(data[key][order], device=self.device, dtype=torch.float32)
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
        self.reset()

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, ids):
        self.plant.reset_idx(ids, state=self.initial)
        self.controller.reset(ids)
        for key in ('eso', 'disturbance', 'gains'):
            getattr(self.controller, key)[ids] = self.initial[key]
        self.history.reset(ids)
        self.randomization.reset(ids)
        self.reference[ids] = self.initial['previous_reference']
        self.previous_reference[ids] = self.reference[ids]
        self.target[ids] = self.reference[ids]
        self.command[ids] = self.initial['previous_command']
        self.previous_action[ids] = 0
        self.command_countdown[ids] = round(self.cfg['commands']['initial_hold_s']/self.dt)
        self.episode_length_buf[ids] = 0
        self.episode_return[ids] = 0

    def reset(self):
        self.reset_idx(self.all_ids)
        measurement, _, speed = self.plant.observe()
        self.measurement = measurement
        self._observation(speed)
        return self.obs_buf, self.privileged_obs_buf

    def _observation(self, speed):
        self.obs_buf = self.history.observation(self.measurement, speed, self.reference,
            self.previous_reference, self.command, self.controller)
        self.privileged_obs_buf = self.obs_buf

    def _commands(self):
        ids = (self.command_countdown <= 0).nonzero(as_tuple=False).flatten()
        n = len(ids)
        c = self.cfg['commands']
        low, high = c['speed_m_s']
        speed = low+(high-low)*torch.rand(n, device=self.device)
        ratio = self.controller._system(speed)[3].abs()
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
        m = self.measurement
        row = torch.stack((m[:,5]*.1,m[:,2],m[:,3],m[:,0],m[:,1],self.reference[:,1]), -1)
        output = self.controller.step(row, torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        applied = action*torch.tensor([1.5,10.], device=self.device)
        steer = (output[:,0]+applied[:,0]).clamp(-3,3)
        steer = torch.maximum(((-.8-m[:,2])/self.dt).clamp(min=-3),
                              torch.minimum(((.8-m[:,2])/self.dt).clamp(max=3), steer))
        self.command = torch.stack((steer,(self.reference[:,0]/.1+applied[:,1]).clamp(-60,60)), -1)
        _, before, _ = self.plant.observe()
        failure = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        peak = torch.zeros(self.num_envs, device=self.device)
        for _ in range(self.decimation):
            contact = self.plant.step(self.randomization.command(self.command))
            clean, pose, speed = self.plant.observe()
            failure |= contact | (clean[:,0].abs()>.7) | ~torch.isfinite(self.plant.roots).all(-1) | ~torch.isfinite(clean).all(-1)
            peak = torch.maximum(peak, clean[:,0].abs())
            self.measurement = self.randomization.observe(clean)
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
        reward = torch.nan_to_num(reward, nan=-r['failure_penalty'])
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~failure
        done = failure | timeout
        self.episode_return += reward
        self.previous_action.copy_(action)
        yaw = torch.atan2(torch.sin(pose[:,2]-before[:,2]),torch.cos(pose[:,2]-before[:,2]))/self.dt
        self.history.finish(applied,yaw)
        ids = done.nonzero(as_tuple=False).flatten()
        extras = {'time_outs': timeout}
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
