"""Finite world-frame force pulse using the existing Isaac Lab wrench interface."""
import numpy as np
import torch
from .torch_observation import rotate_inverse


class EvaluationPulse:
    def __init__(self, env, config):
        self.env=env;self.config=config;self.tick=0;self.records=[]
        self.start=round(config['start_s']/env.physics_dt)
        self.stop=self.start+round(config['duration_s']/env.physics_dt)
        if self.start<0 or self.stop<=self.start:
            raise ValueError('Pulse requires nonnegative start and positive duration')
        self.body=torch.tensor([env.plant.bodies.index(config['body'])],device=env.device)
        self.force=torch.tensor(config['world_force_N'],device=env.device,dtype=torch.float32).expand(env.n,3)
        self.position=torch.tensor(config['site_body_position_m'],device=env.device).expand(env.n,1,3).contiguous()
        self.zero=torch.zeros(env.n,1,3,device=env.device)

    def __call__(self, physics_tick):
        active=self.start<=self.tick<self.stop
        if active or self.tick in (0,self.stop):
            force=self.force*self.env.alive[:,None] if active else self.zero[:,0]
            local=rotate_inverse(self.env.plant.roots[:,3:7],force)
            self.env.plant.robot.permanent_wrench_composer.set_forces_and_torques(forces=local[:,None].contiguous(),torques=self.zero,positions=self.position,body_ids=self.body,is_global=False)
            if active:self.records.append(force.clone())
        self.tick+=1

    def save(self, output):
        forces=torch.stack(self.records).cpu().numpy()
        np.savez_compressed(output/'applied_pulse.npz',time_s=np.arange(self.start,self.stop)*self.env.physics_dt,force_world_N=forces,impulse_Ns=forces.sum(axis=0)*self.env.physics_dt,site_body_position_m=self.config['site_body_position_m'])
