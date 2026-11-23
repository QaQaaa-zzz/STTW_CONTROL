"""E1 time-indexed physical correction at the frozen H75 reference interface."""
import numpy as np
import torch


class OracleTargetBias:
    def __init__(self, times_s, offsets, lanes, horizon, interface, device):
        times = np.asarray(times_s, dtype=float)
        offsets = np.asarray(offsets, dtype=float)
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or times[0] != 0 or times[-1] != horizon * .02 or np.any(np.diff(times) <= 0)
                or offsets.shape != (lanes,len(times),2) or not np.isfinite(offsets).all()
                or np.any(np.abs(offsets) > np.array([1.25,.40])+1e-7)):
            raise ValueError('Invalid E1 time grid or physical bias')
        # 50 Hz publications, with the same target held for the next four 200 Hz ticks.
        t = np.arange(horizon) * .02
        left = np.searchsorted(times, t, side='right') - 1
        w = ((t-times[left])/(times[left+1]-times[left]))[None,:,None]
        interpolated = offsets[:,left,:] + w*(offsets[:,left+1,:]-offsets[:,left,:])
        make = lambda a: torch.as_tensor(a,dtype=torch.float32,device=device)
        self.schedule = make(interpolated.transpose(1,0,2))
        self.low = make([-interface['speed_negative_scale_m_s'], -interface['steer_scale_rad']])
        self.high = make([interface['speed_positive_scale_m_s'], interface['steer_scale_rad']])
        self.set_step(0)

    def set_step(self, step):
        self.bias = self.schedule[step]

    def __call__(self, target):
        # Leave signed zeros and all other unmodified channel values bit-exact.
        return torch.where(self.bias==0, target, torch.clamp(target+self.bias,min=self.low,max=self.high))
