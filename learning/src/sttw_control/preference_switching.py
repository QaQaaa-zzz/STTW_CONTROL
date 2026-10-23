"""Explicit fixed/step/ramp preference schedules; preserve all closed-loop state."""
import math

def validate_schedule(spec, horizon, dt):
    if set(spec)!={'switch_s','ramp_s','modes'}:raise ValueError('explicit switching fields required')
    t,r=spec['switch_s'],spec['ramp_s']
    if not all(math.isfinite(v) for v in (t,r)) or not 0<t<t+r<horizon:raise ValueError('switch and ramp must lie inside horizon')
    if not math.isclose(t/dt,round(t/dt),abs_tol=1e-8):raise ValueError('switch must align with policy tick')
    if not spec['modes'] or any(m not in ('fixed','step','ramp') for m in spec['modes'].values()):raise ValueError('unsupported preference schedule')

def scheduled_alpha(initial, tick, lower_dt, switch_s, ramp_s, mode):
    import jax.numpy as j
    if mode=='fixed':return initial
    t=tick*lower_dt
    w=j.where(t>=switch_s,1.,0.) if mode=='step' else j.clip((t-switch_s)/ramp_s,0.,1.)
    return initial+(1.-2.*initial)*w
