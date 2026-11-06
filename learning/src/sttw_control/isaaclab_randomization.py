"""Go2-style startup property randomization using Isaac Lab's event functions.

Only vehicle property selection lives here. Sampling and PhysX property writes
are upstream. Call once at startup, not at every reset: these PhysX setters use
CPU tensors even when simulation and policy run on GPU.
"""
from types import SimpleNamespace
import torch
from isaaclab.envs.mdp import events
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.buffers import DelayBuffer


def randomize_vehicle_properties(plant, config, env_ids=None):
    env = SimpleNamespace(scene={'robot': plant.robot}, num_envs=plant.count,
                          device=plant.device)
    ids = torch.arange(plant.count, device=plant.device) if env_ids is None else env_ids
    bodies = SceneEntityCfg('robot')
    base = SceneEntityCfg('robot', body_ids=[plant.bodies.index(config['base_body'])])
    material = dict(asset_cfg=bodies,
                    static_friction_range=tuple(config['friction_range']),
                    dynamic_friction_range=tuple(config['friction_range']),
                    restitution_range=(0., 0.), num_buckets=64,
                    make_consistent=True)
    nominal_materials = plant.robot.root_physx_view.get_material_properties().clone()
    term = events.randomize_rigid_body_material(
        EventTermCfg(func=events.randomize_rigid_body_material, mode='startup', params=material), env)
    term(env, ids, **material)
    # PhysX encodes compliant stiffness as negative restitution. Preserve it
    # while letting the upstream event sample both friction coefficients.
    materials = plant.robot.root_physx_view.get_material_properties()
    materials[..., 2] = torch.where(nominal_materials[..., 2] < 0,
                                    nominal_materials[..., 2], materials[..., 2])
    plant.robot.root_physx_view.set_material_properties(materials, ids.cpu())
    # One proportional density variation per body; nominal source assets remain
    # immutable. Lab scales inertia together with mass from default properties.
    links = SceneEntityCfg('robot', body_ids=[i for i in range(len(plant.bodies)) if i not in base.body_ids]) if 'base_mass_add_kg' in config else bodies
    mass = dict(asset_cfg=links, mass_distribution_params=tuple(config['mass_scale']),
                operation='scale', recompute_inertia=True)
    term = events.randomize_rigid_body_mass(
        EventTermCfg(func=events.randomize_rigid_body_mass, mode='startup', params=mass), env)
    term(env, ids, **mass)
    if 'base_mass_add_kg' in config:
        mass = dict(asset_cfg=base, mass_distribution_params=tuple(config['base_mass_add_kg']),
                    operation='add', recompute_inertia=True)
        term = events.randomize_rigid_body_mass(
            EventTermCfg(func=events.randomize_rigid_body_mass, mode='startup', params=mass), env)
        term(env, ids, **mass)
    events.randomize_rigid_body_com(env, ids,
        com_range={axis: tuple(config['base_com_m']) for axis in ('x', 'y', 'z')},
        asset_cfg=base)
    return dict(materials=plant.robot.root_physx_view.get_material_properties().cpu().tolist(),
                masses=plant.robot.root_physx_view.get_masses().cpu().tolist(),
                coms=plant.robot.root_physx_view.get_coms().cpu().tolist())


def apply_fixed_evaluation_groups(env, groups):
    """Apply frozen per-seed parameters to one scene before its first step."""
    randomization=env.randomization
    for group in groups:
        ids=torch.tensor(group['env_ids'],device=env.device,dtype=torch.long)
        cfg=group['config']
        env.randomization_properties=randomize_vehicle_properties(env.plant,cfg,ids)
        randomization.plant.servos.gain[ids]=randomization.nominal_gain*cfg['servo_gain_scale'][0]
        randomization.steer_offset[ids]=cfg['steer_zero_offset_rad'][0]
        lag=round(cfg['latency_s'][0]/env.plant.physics_dt)
        for buffer in randomization.buffers.values():
            buffer.set_time_lag(torch.full((len(ids),),lag,device=env.device,dtype=torch.int),ids)


class VehicleControlRandomization:
    """Delay at the physical clock; noise/offset only on observed signals.

    The task calls command() before each physics step and observe() afterwards.
    Reward and termination must use the plant's clean state, never this output.
    """
    def __init__(self, plant, config):
        self.plant, self.config = plant, config
        self.device, self.count = plant.device, plant.count
        self.lag_bounds = [round(seconds/plant.sim.cfg.dt)
                           for seconds in config['latency_s']]
        self.buffers = {name: DelayBuffer(self.lag_bounds[1], self.count, self.device)
                        for name in ('command', 'motor', 'imu')}
        self.nominal_gain = plant.servos.gain.clone()
        plant.servos.gain = self.nominal_gain.repeat(self.count, 1)
        self.steer_offset = torch.zeros(self.count, device=self.device)
        self.noise = torch.tensor(config['measurement_noise'], device=self.device)
        self.reset(torch.arange(self.count, device=self.device))

    def reset(self, env_ids):
        n = len(env_ids)
        for buffer in self.buffers.values():
            buffer.reset(env_ids)
            lags = torch.randint(self.lag_bounds[0], self.lag_bounds[1]+1,
                                 (n,), device=self.device, dtype=torch.int)
            buffer.set_time_lag(lags, env_ids)
        low, high = self.config['servo_gain_scale']
        self.plant.servos.gain[env_ids] = self.nominal_gain * (
            low+(high-low)*torch.rand(n, 2, device=self.device))
        low, high = self.config['steer_zero_offset_rad']
        self.steer_offset[env_ids] = low+(high-low)*torch.rand(n, device=self.device)

    def command(self, command):
        return self.buffers['command'].compute(command)

    def observe(self, measurement):
        observed = measurement.clone()
        observed[:, 2] += self.steer_offset
        observed += (2*torch.rand_like(observed)-1)*self.noise
        motor = self.buffers['motor'].compute(observed[:, [2, 3, 5, 6]])
        imu = self.buffers['imu'].compute(observed[:, [0, 1, 4]])
        observed[:, [2, 3, 5, 6]] = motor
        observed[:, [0, 1, 4]] = imu
        return observed


class PendulumControlRandomization:
    """Four physical targets at physics dt; measured frame at source 5 ms dt.

    Reuses Isaac Lab DelayBuffer. No randomness touches reward/terminal signals.
    Call sensor_frame every sensor_dt, then append its output to the 50 Hz FIFO.
    """
    def __init__(self,plant,config):
        self.plant,self.config=plant,config
        self.device,self.count=plant.device,plant.count
        self.action_bounds=[round(t/plant.sim.cfg.dt) for t in config['latency_s']]
        self.sensor_bounds=[round(t/config['sensor_dt_s']) for t in config['latency_s']]
        self.action_buffer=DelayBuffer(self.action_bounds[1],self.count,self.device)
        self.sensor_buffers={name:DelayBuffer(self.sensor_bounds[1],self.count,self.device) for name in ('imu','motor','speed')}
        self.noise=torch.tensor(config['frame_noise'],device=self.device)
        self.offset=torch.zeros(self.count,3,device=self.device)
        self.nominal={}
        # Scale all active jump servos, retaining nominal force/target bounds.
        for label,servo,fields in (('drive',plant.servos,('gain',)),('steering',plant.jump_steering_servo,('kp','kd')),('pendulum',plant.pendulum_servos,('kp','kd'))):
            for field in fields:
                value=getattr(servo,field).clone();self.nominal[label,field]=value
                setattr(servo,field,value.repeat(self.count,1))
        self.reset(torch.arange(self.count,device=self.device))

    def reset(self,ids):
        n=len(ids)
        for buffer,bounds in [(self.action_buffer,self.action_bounds)]+[(b,self.sensor_bounds) for b in self.sensor_buffers.values()]:
            buffer.reset(ids)
            buffer.set_time_lag(torch.randint(bounds[0],bounds[1]+1,(n,),device=self.device,dtype=torch.int),ids)
        low,high=self.config['servo_gain_scale']
        for label,servo,fields in (('drive',self.plant.servos,('gain',)),('steering',self.plant.jump_steering_servo,('kp','kd')),('pendulum',self.plant.pendulum_servos,('kp','kd'))):
            for field in fields:
                nominal=self.nominal[label,field]
                getattr(servo,field)[ids]=nominal*(low+(high-low)*torch.rand(n,len(nominal),device=self.device))
        low,high=self.config['joint_zero_offset_rad']
        self.offset[ids]=low+(high-low)*torch.rand(n,3,device=self.device)
        if self.plant.servo_mode=='hybrid':self.plant.sync_native_drives(ids)

    def command(self,target):
        return self.action_buffer.compute(target)

    def sensor_frame(self,frame):
        observed=frame.clone()
        observed[:,9:12]+=self.offset
        observed+=(2*torch.rand_like(observed)-1)*self.noise
        observed[:,:9]=self.sensor_buffers['imu'].compute(observed[:,:9])
        observed[:,9:17]=self.sensor_buffers['motor'].compute(observed[:,9:17])
        observed[:,21:22]=self.sensor_buffers['speed'].compute(observed[:,21:22])
        return observed
