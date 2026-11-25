"""Clocked observations and a separate original-route scoring projection for frozen DEV."""
from collections import deque

import torch


class SampledVehicleFeedback:
    def __init__(self, config, route, request, wheel_radius):
        self.config = config
        self.period = round(200 / config['location_hz'])
        self.lag = round(config['availability_s'] / .005)
        if self.period < 1 or self.lag < 0 or abs(self.period*config['location_hz']-200) > 1e-7 or abs(self.lag*.005-config['availability_s'])>1e-9:
            raise ValueError('Sample period and availability delay must be integer multiples of 5 ms')
        self.route = route
        self.request = request
        self.radius = wheel_radius
        self.std = torch.as_tensor(config['noise_std'], device=request.device, dtype=request.dtype)
        if self.std.shape != (10,) or torch.any(self.std<0):
            raise ValueError('Ten nonnegative standard deviations are required for the seven sensor and three pose channels')
        self.generator = torch.Generator(device=request.device)
        self.frames = deque(maxlen=self.lag+1)

    def reset(self, measurement, pose, speed):
        self.generator.manual_seed(self.config['seed'])
        self.frames.clear()
        self.tick = 0
        self.truth_pose = pose.clone()
        self.projection = self.route.project(pose[:,:2],torch.zeros_like(speed),torch.zeros_like(speed))
        f = self.route.pursuit(self.projection['progress'],pose,speed,self.request)
        self.domain_exit = f['invalid'].clone()
        self._sample(measurement,pose)

    def advance(self, measurement, pose, speed):
        self.tick += 1
        delta = torch.linalg.vector_norm(pose[:,:2]-self.truth_pose[:,:2],dim=-1)
        self.projection = self.route.project(pose[:,:2],self.projection['progress'],delta)
        self.truth_pose = pose.clone()
        if self.tick % 4 == 0:
            f = self.route.pursuit(self.projection['progress'],pose,speed,self.request)
            self.domain_exit |= f['invalid'] & ~f['numerical_fault']
        self._sample(measurement,pose)

    def _sample(self, measurement, pose):
        # Consume the same noise draws every tick, including ticks with no XY fix.
        signals = torch.cat((measurement,pose),dim=-1)
        noisy = signals + self.std*torch.randn(signals.shape,device=signals.device,dtype=signals.dtype,generator=self.generator)
        if self.tick % self.period == 0:
            self.fix = noisy[:,7:9].clone()
        noisy[:,7:9] = self.fix
        self.frames.append((self.tick,noisy))
        stamp, available = self.frames[0]
        self.imu_tick, self.location_tick = stamp, stamp-stamp%self.period
        self.measurement = available[:,:7]
        self.pose = available[:,7:]
        # The native Isaac adapter already converts negative forward wheel rate to positive.
        self.speed = self.measurement[:,5]*self.radius
