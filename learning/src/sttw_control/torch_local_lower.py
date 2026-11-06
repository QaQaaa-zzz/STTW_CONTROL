"""Frozen LOCAL300 inference at 200 Hz; preserves its causal 210D interface."""
import hashlib
import io
import numpy as np
from pathlib import Path
import torch


def local_frame(measurement, true_speed, governed, previous_reference,
                previous_command, previous_applied, yaw_rate, rates, controller):
    m = measurement
    radius=controller.config.get("wheel_radius_m",.1)
    a4 = controller._system(m[:, 5]*radius)[2]
    ratio = controller._system(governed[:, 0])[3]
    frame = torch.stack((m[:,0],m[:,1],m[:,2],m[:,3],true_speed,yaw_rate,
    m[:,5]*radius,m[:,6]*radius,governed[:,0],governed[:,1],rates[:,0],rates[:,1],
    previous_reference[:,0],previous_reference[:,1],previous_command[:,0],previous_command[:,1],
    previous_applied[:,0],previous_applied[:,1],-controller.disturbance/a4,
    governed[:,1]/ratio), dim=-1)
    return frame


class TorchLocalLower:
    def __init__(self, actor_path, actor_sha256, count, device='cpu', dtype=torch.float32):
        data = Path(actor_path).read_bytes()
        if hashlib.sha256(data).hexdigest() != actor_sha256:
            raise ValueError('frozen lower actor identity mismatch')
        archive = np.load(io.BytesIO(data), allow_pickle=False)
        params = {f'Dense_{i}': {name: archive[f'Dense_{i}_{name}'] for name in ('kernel','bias')} for i in range(3)}
        shapes = [(210, 256), (256, 128), (128, 2)]
        if set(params) != {f'Dense_{i}' for i in range(3)} or any(
                params[f'Dense_{i}']['kernel'].shape != shape or
                params[f'Dense_{i}']['bias'].shape != (shape[1],)
                for i, shape in enumerate(shapes)):
            raise ValueError('LOCAL300 requires 210-256-128-2 weights')
        self.weights = [(torch.tensor(params[f'Dense_{i}']['kernel'], device=device, dtype=dtype),
                         torch.tensor(params[f'Dense_{i}']['bias'], device=device, dtype=dtype))
                        for i in range(3)]
        self.scales = torch.tensor([.3,1.5,.35,3.,3.,2.,3.,3.,3.,.35,.8,.45,3.,.35,3.,60.,1.5,10.,.3,.3], device=device, dtype=dtype)
        self.frames = torch.zeros(count, 10, 20, device=device, dtype=dtype)
        self.mask = torch.zeros(count, 10, device=device, dtype=dtype)
        self.previous_applied = torch.zeros(count, 2, device=device, dtype=dtype)
        self.yaw_rate = torch.zeros(count, device=device, dtype=dtype)

    def reset(self, indices):
        self.frames[indices] = 0
        self.mask[indices] = 0
        self.previous_applied[indices] = 0
        self.yaw_rate[indices] = 0

    @torch.no_grad()
    def observation(self, measurement, true_speed, governed, previous_reference,
                previous_command, controller):
        # Evaluate before controller.step: ESO features use the previous state.
        frame = local_frame(measurement, true_speed, governed, previous_reference,
            previous_command, self.previous_applied, self.yaw_rate,
            (governed-previous_reference)/controller.config['dt'], controller)
        self.frames = torch.cat((self.frames[:,1:],frame[:,None]),dim=1)
        self.mask = torch.cat((self.mask[:,1:],torch.ones_like(self.mask[:,:1])),dim=1)
        return torch.cat(((self.frames/self.scales).clamp(-5,5).flatten(1),self.mask),dim=-1)

    @torch.no_grad()
    def prepare(self, measurement, true_speed, governed, previous_reference,
                previous_command, controller):
        obs = self.observation(measurement, true_speed, governed,
                               previous_reference, previous_command, controller)
        action = self.infer(obs)
        return action,obs,torch.isfinite(obs).all(-1)&torch.isfinite(action).all(-1)

    @torch.no_grad()
    def infer(self, observation):
        x = observation
        for weight, bias in self.weights[:-1]:
            x = x@weight+bias
            x = torch.where(x>0,x,torch.exp(x)-1)
        weight,bias = self.weights[-1]
        return torch.tanh(x@weight+bias)

    def finish(self, applied_residual, yaw_rate):
        # Applied residual is after residual permission clipping, before final
        # actuator command clipping (closed_loop_kernel contract). Yaw rate is unwrapped
        # pose difference / lower dt, not the instantaneous IMU z measurement.
        self.previous_applied = applied_residual.clone()
        self.yaw_rate = yaw_rate.clone()
