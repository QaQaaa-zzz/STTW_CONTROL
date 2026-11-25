"""Finite-grid reference governor for the frozen bicycle, evaluated at its 50 Hz clock."""
import math

import torch

from .path_reference_governor import native_cost_candidates
from .torch_path_reference import wrap
from .torch_path_reward import costs


class PredictiveReferenceGovernor:
    """Local bicycle and constrained-steering preview, without future contact observations."""

    def __init__(self, env, settings):
        self.env = env
        self.dt = env.config['timing']['upper_dt_s']
        self.steps = settings['preview_steps']
        self.speed_tau = settings['speed_response_s']
        if self.steps < 1 or self.speed_tau <= 0:
            raise ValueError("Use a positive nominal preview on dedicated governor lanes")
        candidates = native_cost_candidates([0., 0.], env.config)
        # Zero first so identical forecasts retain the unmodified references.
        grid = [13, *[i for i in range(1, 26) if i != 13]]
        self.latents = torch.as_tensor(candidates['latents'][grid], device=env.device)
        self.targets = torch.as_tensor(candidates['physical_offsets'][grid], device=env.device,
                                       dtype=self.latents.dtype)
        self.speed_levels, self.speed_index = self.targets[:, 0].unique(return_inverse=True)
        self.speed_fraction = -math.expm1(-self.dt / self.speed_tau)

    @torch.no_grad()
    def action(self, measurement, speed):
        env, dt = self.env, self.dt
        middle, policy, controller = env.middle, env.policy, env.controller
        # Freeze coefficients/gains at each speed proposal's nominal mid-horizon speed.
        # The 25 steer/speed pairs share five speed levels. The plant keeps its real lower loop.
        interface = env.config['residual_interface']
        half_time = .5 * dt * self.steps
        low = -half_time * interface['correction_speed_down_m_s2']
        high = half_time * interface['correction_speed_up_m_s2']
        offset_mid = policy.offsets[:, 0:1] + (self.speed_levels - policy.offsets[:, 0:1]).clamp(low, high)
        ref_mid = (middle.nominal[:, 0:1] + offset_mid).clamp(*interface['speed_reference_bounds_m_s'])
        speed_mid = speed[:, None] + (-math.expm1(-half_time / self.speed_tau)) * (ref_mid - speed[:, None])
        systems = [controller._system(column) for column in speed_mid.unbind(1)]
        a1, a2, a4, ratio, gains = (torch.stack(field, dim=1)[:, self.speed_index]
                                     for field in zip(*systems))
        gains = gains * controller.feedback_gain_scale
        phi, omega, delta = (measurement[:, i:i+1] for i in (0, 1, 2))
        speed = speed[:, None]
        ey = middle.projection['cross_track'][:, None]
        heading_error = wrap(middle.pose[:, 2] - middle.projection['heading'])[:, None]
        nominal = middle.nominal[:, None]
        offset = policy.offsets[:, None]
        filtered = policy.filtered[:, None]
        decay = policy.decay ** (dt / env.config['timing']['lower_dt_s'])
        lo = policy.increment_low * (dt / env.config['timing']['lower_dt_s'])
        hi = policy.increment_high * (dt / env.config['timing']['lower_dt_s'])
        original_speed = env.speed[:, None]
        steer_to_curve = math.cos(env.cc['caster']) / env.cc['wheelbase']
        # A common route preview based on current speed uses only the known centerline.
        s = middle.projection['progress']
        times = torch.arange(self.steps, device=env.device) * dt
        curvature = torch.stack([env.route.at(s + t * speed[:, 0])[2] for t in times])
        ratio_req = controller._system(env.speed)[3]
        chi = ((torch.atan(curvature / steer_to_curve) / ratio_req).abs() - .18).div(.12).clamp(0, 1)
        target = self.targets.expand(env.n, -1, -1)
        peak = phi.abs().expand_as(target[..., 0])
        forecast = []
        for k in range(self.steps):
            f = decay * filtered + (1 - decay) * target
            proposed = nominal + offset + (f - offset).clamp(lo, hi)
            ref = proposed.clamp(policy.low, policy.high)
            applied = ref - nominal
            rate = (applied - offset) / dt
            filtered = torch.where(proposed != ref, applied, f)
            offset = applied
            speed = speed + self.speed_fraction * (ref[..., 0] - speed)
            steer = ref[..., 1]
            steer_rate = ((steer - delta) * gains[..., 0] +
                          (steer / ratio - phi) * gains[..., 1] - omega * gains[..., 2]).clamp(-3., 3.)
            new_delta = (delta + dt * steer_rate).clamp(-.8, .8)
            accel = a4 * phi + a2 * (delta + new_delta) / 2 + a1 * ((new_delta - delta) / dt)
            phi = phi + dt * omega + .5 * dt**2 * accel
            omega = omega + dt * accel
            peak = torch.maximum(peak, phi.abs())
            heading_rate = speed * (steer_to_curve * torch.tan((delta + new_delta) / 2) -
                                    curvature[k, :, None])
            ey = ey + dt * speed * torch.sin(heading_error + .5 * dt * heading_rate)
            heading_error = heading_error + dt * heading_rate
            delta = new_delta
            forecast.append((speed, ey, heading_error, phi, omega, offset, rate))
        v, y, heading, roll, roll_rate, offsets, rates = map(torch.stack, zip(*forecast))
        score = -costs(alpha=env.alpha[:, None], chi=chi[:, :, None], ev=v-original_speed,
                       ey=y, eh=heading, peak_roll=roll.abs(), roll_rate=roll_rate, speed=v,
                       offsets=offsets, offset_rate=rates, config=env.config)['reward'].sum(0)
        margin = .7 - peak
        feasible = margin >= 0
        winner = torch.where(feasible.any(1), score.masked_fill(~feasible, torch.inf).argmin(1),
                             margin.argmax(1))
        row = torch.arange(env.n, device=env.device)
        return self.latents[winner], dict(governor_candidate=winner,
                                          governor_margin_rad=margin[row, winner],
                                          governor_feasible_count=feasible.sum(1))
