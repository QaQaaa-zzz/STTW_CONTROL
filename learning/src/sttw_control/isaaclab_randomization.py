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


def randomize_vehicle_properties(plant, config):
    env = SimpleNamespace(scene={'robot': plant.robot}, num_envs=plant.count,
                          device=plant.device)
    ids = torch.arange(plant.count, device=plant.device)
    bodies = SceneEntityCfg('robot')
    base = SceneEntityCfg('robot', body_ids=[plant.bodies.index(config['base_body'])])
    material = dict(asset_cfg=bodies,
                    static_friction_range=tuple(config['friction_range']),
                    dynamic_friction_range=tuple(config['friction_range']),
                    restitution_range=(0., 0.), num_buckets=64,
                    make_consistent=True)
    term = events.randomize_rigid_body_material(
        EventTermCfg(func=events.randomize_rigid_body_material, mode='startup', params=material), env)
    term(env, ids, **material)
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
