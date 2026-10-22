"""Persistent-alpha reward, NumPy/JAX-compatible pure numerical reference.

NOT an installed repository patch. No simulator and no training is launched.
Caps are applied to endpoint cost channels BEFORE their convex combination.
The true physical/domain termination and failed-interval replacement remain
owned by the environment; callers MUST NOT add normal reward to a failure tail.
"""
from __future__ import annotations
from typing import Any
import numpy as np

VERSION = 'path_persistent_preference_v3'

def costs(*, alpha: Any, chi: Any, ev: Any, ey: Any, eh: Any,
          peak_roll: Any, roll_rate: Any, speed: Any, offsets: Any,
          offset_rate: Any, config: dict, xp: Any = np) -> dict:
    """Scalar or batched input; offsets/offset_rate have final dimension two.

    Host caller validates finite inputs, alpha/chi in [0,1], and config identity.
    xp can be numpy or jax.numpy. No Python branching on traced arrays.
    `raw` records the unbounded convex cost, `effective` the capped-channel cost.
    In v3 effective != min(raw, cap) in general; endpoint cap diagnostics are
    returned separately so log consumers cannot silently apply the old meaning.
    """
    c = config.get('path_reward', config)
    if c.get('reward_version') != VERSION:
        raise ValueError('Use an explicit v3 reward configuration.')
    eps = float(c['path_secondary_ratio'])
    if not 0 < eps <= 1:
        raise ValueError('path_secondary_ratio must be in (0,1].')
    a = xp.asarray(alpha); q = xp.asarray(chi)
    H = lambda x: xp.where(xp.abs(x) <= 1, x*x, 2*xp.abs(x)-1)
    pos = lambda x: xp.maximum(x, 0)
    caps = c['component_caps']
    v0 = (10-9*q)*H(xp.asarray(ev)/c['speed_scale_m_s'])
    v1 = 10*H(xp.asarray(ev)/c['speed_scale_m_s'])
    p = 10*(H(xp.asarray(ey)/c['cross_track_scale_m']) +
            c['path_heading_factor']*H(xp.asarray(eh)/c['path_heading_scale_rad']))
    qp = c['primary_weight']*q*H(pos(xp.abs(ey)-c['primary_path_band_m'])/c['primary_path_scale_m'])
    qv = c['primary_weight']*q*H(pos(xp.abs(ev)-c['primary_speed_band_m_s'])/c['primary_speed_scale_m_s'])
    common = dict(
        roll=c['roll_weight']*H(pos(xp.asarray(peak_roll)-c['roll_start_rad'])/c['roll_scale_rad']),
        working_roll=c['working_roll_weight']*H(pos(xp.asarray(peak_roll)-c['working_roll_start_rad'])/c['working_roll_scale_rad']),
        roll_rate=c['roll_rate_weight']*H(pos(xp.abs(roll_rate)-c['roll_rate_start_rad_s'])/c['roll_rate_scale_rad_s']),
        overspeed=c['overspeed_weight']*H(pos(xp.asarray(ev)-c['overspeed_band_m_s'])/c['overspeed_scale_m_s']),
        low_speed=c['low_speed_weight']*H(pos(c['low_speed_start_m_s']-xp.asarray(speed))/c['low_speed_scale_m_s']),
        offset_magnitude=c['offset_magnitude_weight']*xp.sum((xp.asarray(offsets)/xp.asarray(c['offset_magnitude_scales']))**2,axis=-1),
        offset_rate=c['offset_rate_weight']*xp.sum((xp.asarray(offset_rate)/xp.asarray(c['offset_rate_scales']))**2,axis=-1),
    )
    eff={k:xp.minimum(v,caps[k]) for k,v in common.items()}
    path_factor=1-(1-eps)*a
    eff.update(speed=(1-a)*xp.minimum(v0,caps['speed'])+a*xp.minimum(v1,caps['speed']),
               path=path_factor*xp.minimum(p,caps['path']),
               primary=(1-a)*xp.minimum(qp,caps['primary'])+a*xp.minimum(qv,caps['primary']))
    raw=dict(common, speed=(1-a)*v0+a*v1,path=path_factor*p,primary=(1-a)*qp+a*qv)
    endpoint_capped={'speed0':v0>caps['speed'],'speed1':v1>caps['speed'],
                     'path':p>caps['path'],'primary0':qp>caps['primary'],'primary1':qv>caps['primary']}
    return dict(raw=raw,effective=eff,endpoint_capped=endpoint_capped,
                cost_rate=sum(eff.values()),conservative_cost_rate_bound=c['failure_cost_rate_bound'])

def failure_reward(remaining_intervals: Any, config: dict, *, gamma: float=.997,
                   upper_dt: float=.02, xp: Any=np) -> Any:
    """Replace the entire failed policy interval, exactly as in the baseline.
    The unchanged bound is conservative for every alpha in v3.
    """
    c=config.get('path_reward',config)
    if not 0 < gamma < 1 or upper_dt <= 0:
        raise ValueError('Invalid discount or policy step.')
    n=xp.asarray(remaining_intervals)
    return -c['failure_extra']-c['scale']*upper_dt*c['failure_cost_rate_bound']*(1-gamma**n)/(1-gamma)
