"""Causal, evaluation-only 50 Hz nominal roll-envelope speed intervention."""
import torch


class RollEnvelopeBrake:
    """One state sample per upper tick. Shaping, actuation and reward stay downstream."""

    def __init__(self, controller, radius, interface, count, device, settings):
        self.controller = controller
        self.radius = radius
        self.speed_floor = -interface['speed_negative_scale_m_s']
        self.period = .02
        self.enter = settings['enter_margin_rad']
        self.release = settings['release_margin_rad']
        self.horizon = settings['horizon_s']
        step = settings['grid_s']
        if not (0 <= self.enter < self.release < .7 and 0 < step <= self.horizon):
            raise ValueError('Invalid roll margin thresholds or prediction grid')
        self.t = torch.linspace(0, self.horizon, round(self.horizon / step) + 1, device=device)
        self.previous = torch.full((count,), torch.nan, device=device)
        self.active = torch.zeros(count, dtype=torch.bool, device=device)

    def margins(self, phi, omega, delta, speed):
        c,b,a,*_ = self.controller._system(speed)
        lam = a.sqrt()[:, None]
        # For each signed bound, first take the extreme steering rate that
        # minimizes signed roll, then cancel the rate at the travel stop.
        def step_response(t):
            return ((c[:,None]*(torch.cosh(lam*t)-1)
                    + b[:,None]*(torch.sinh(lam*t)/lam-t))/a[:,None])
        cosh = torch.cosh(lam*self.t)
        sinh = torch.sinh(lam*self.t)
        free = phi[:,None]*cosh + omega[:,None]*sinh/lam + (b*delta/a)[:,None]*(cosh-1)
        ramp = step_response(self.t)
        lower = []
        for sign in (1., -1.):
            stop = (.8-sign*delta).clamp_min(0)/3
            tail = (self.t-stop[:,None]).clamp_min(0)
            oriented_roll = sign*free + 3*(ramp-step_response(tail))
            lower.append(.7-oriented_roll.amax(1))
        return torch.stack(lower, dim=1), (b < 0) & (c < 0) & (a > 0)

    def observe(self, measurement, alive):
        phi, delta = measurement[:,0], measurement[:,2]
        speed = self.radius * measurement[:,5]
        difference = phi-self.previous
        omega = torch.atan2(torch.sin(difference),torch.cos(difference))/self.period
        self.previous = phi.clone()
        margins, domain = self.margins(phi,omega,delta,speed)
        valid = alive & domain & torch.isfinite(margins).all(1) & (speed >= .5) & (delta.abs() <= .8) & (phi.abs() < .7)
        smallest = margins.amin(1)
        self.active = valid & torch.where(self.active,smallest < self.release,smallest < self.enter)
        return dict(guard_margin_rad=torch.where(valid[:,None],margins,torch.nan), guard_valid=valid,
                    guard_active=self.active.clone(),guard_inputs=torch.stack((phi,omega,delta,speed),dim=-1))

    def __call__(self, target):
        speed = torch.where(self.active,self.speed_floor,target[:,0])
        return torch.stack((speed,target[:,1]),dim=-1)
