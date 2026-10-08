"""Pure-NumPy reference for proposed R196 V4 reward. Not a vehicle simulator.

Causal API: call state_cost for each physical 5 ms interval; call upper_motion_cost
once at the end of the 20 ms policy interval. Sum physical rewards then subtract
scale*policy_dt*upper_motion_cost. A failure replaces that whole interval exactly
once. All inputs use the original issued commands, never the governed commands
as tracking targets.
"""
from __future__ import annotations
import math
import numpy as np

CAPS = {'speed':100., 'steer':100., 'heading':300., 'roll':100.,
        'roll_rate':40., 'overspeed':50., 'low_speed':50., 'magnitude':10.,
        'upper_rate':10., 'upper_acceleration':20., 'yaw_damping':20.}


def huber(z):
    a = np.abs(np.asarray(z, dtype=float))
    return np.minimum(a, 1.)**2 + 2.*np.maximum(a - 1., 0.)


def pos(x):
    return np.maximum(x, 0.)


def state_cost(*, alpha: int, chi: float, recovery_weight: float,
               speed: float, steer: float, raw_speed: float, raw_steer: float,
               heading_debt: float, yaw_rate: float, roll: float, roll_rate: float,
               offsets: np.ndarray, wheelbase: float=.408,
               caster_rad: float=25*math.pi/180) -> dict:
    vals = np.asarray([chi,recovery_weight,speed,steer,raw_speed,raw_steer,
                       heading_debt,yaw_rate,roll,roll_rate,*offsets], dtype=float)
    if not np.all(np.isfinite(vals)) or alpha not in (0,1):
        raise ValueError('Invalid reward input; do not silently sanitize.')
    if not 0 <= chi <= 1 or not 0 <= recovery_weight <= 1:
        raise ValueError('chi and g must be in [0,1]')
    g = recovery_weight
    ev, ed = speed-raw_speed, steer-raw_steer
    under, over = pos(-ev), pos(ev)
    vn, dn = 8*huber(ev/.1), 8*huber(ed/.05)
    if alpha == 0:
        vc = .4*huber(under/.3)+.4*huber(over/.1)+2*huber(pos(under-.6)/.2)
        dc = 8*huber(ed/.05)
    else:
        vc = 8*huber(ev/.1)
        dc = .8*huber(ed/.05)+2*huber(pos(abs(ed)-.15)/.05)
    speed_cost = (1-g)*((1-chi)*vn+chi*vc)+g*8*huber(ev/.1)
    steer_cost = (1-g)*((1-chi)*dn+chi*dc)+g*huber(ed/.1)
    r_c = raw_speed*math.cos(caster_rad)*math.tan(raw_steer)/wheelbase
    w_mag = .05+.45*(1-chi)*math.exp(-(heading_debt/.1)**2
            -(float(pos(abs(roll)-.2))/.1)**2-(roll_rate/.6)**2)
    raw = {
      'speed':float(speed_cost), 'steer':float(steer_cost),
      'heading':float(6*g*huber(pos(abs(heading_debt)-.01)/.1)),
      'roll':float(40*huber(pos(abs(roll)-.26)/.04)),
      'roll_rate':float(huber(pos(abs(roll_rate)-.8)/.8)),
      'overspeed':float(2*huber(pos(ev-.05)/.1)),
      'low_speed':float(4*huber(pos(1.5-speed)/.2)),
      'magnitude':float(w_mag*np.sum((np.asarray(offsets)/[.5,.1])**2)),
      'yaw_damping':float(.5*g*math.exp(-(heading_debt/.15)**2)*huber((yaw_rate-r_c)/.2))}
    eff = {name:min(value,CAPS[name]) for name,value in raw.items()}
    return {'raw':raw,'effective':eff,'cost':sum(eff.values()),
            'capped':{k:raw[k]>CAPS[k] for k in raw},'magnitude_weight':w_mag}


def upper_motion_cost(offsets_now, offsets_previous, previous_rate,
                      *, dt=.02, acceleration_valid=True) -> dict:
    """Offsets at completed policy interval endpoints; never differentiate raw commands.

    Rate is valid on the first interval using initial offset zero. Acceleration
    is masked on the first interval, and all cross-reset pairs are invalid.
    """
    now, prev, prv_rate = map(lambda v:np.asarray(v,dtype=float),
                              (offsets_now,offsets_previous,previous_rate))
    if any(v.shape != (2,) for v in (now,prev,prv_rate)):
        raise ValueError('Expected [speed_correction, steer_correction]')
    rate = (now-prev)/dt
    accel = (rate-prv_rate)/dt
    raw = {'upper_rate':float(.1*np.sum((rate/[1.,.4])**2)),
           'upper_acceleration':float(.05*np.sum((accel/[10.,10.])**2))
                                if acceleration_valid else 0.}
    return {'rate':rate,'acceleration':accel,'raw':raw,
            'effective':{k:min(v,CAPS[k]) for k,v in raw.items()},
            'cost':sum(min(v,CAPS[k]) for k,v in raw.items())}


def failure_reward(remaining_intervals: int, gamma=.997, scale=.1, dt=.02):
    if not isinstance(remaining_intervals,int) or remaining_intervals<1:
        raise ValueError('remaining_intervals includes current interval')
    return -5-scale*dt*sum(CAPS.values())*(-math.expm1(remaining_intervals*math.log(gamma)))/(1-gamma)


def self_check():
    assert sum(CAPS.values()) == 800.
    base=dict(chi=1.,recovery_weight=0.,raw_speed=2.6,raw_steer=.25,
              heading_debt=0.,yaw_rate=0.,roll=.2,roll_rate=0.,offsets=np.zeros(2))
    vals=[]
    for a in (0,1):
        A=state_cost(alpha=a,speed=2.1,steer=.245,**base)
        B=state_cost(alpha=a,speed=2.57,steer=.095,**base)
        # Compare tracking costs, independent of unvalidated physical trajectories.
        av=A['raw']['speed']+A['raw']['steer'];bv=B['raw']['speed']+B['raw']['steer']
        vals.append((a,av,bv))
    assert vals[0][1] < vals[0][2] and vals[1][1] > vals[1][2]
    quiet={**base,'chi':0.,'raw_speed':2.3,'raw_steer':0.,'speed':2.3,'steer':0.,
           'recovery_weight':1.,'roll':0.}
    assert state_cost(alpha=0,**quiet)['cost']==0
    assert state_cost(alpha=1,**quiet)['cost']==0
    a=state_cost(alpha=0,**{**quiet,'heading_debt':.3})
    b=state_cost(alpha=0,**{**quiet,'heading_debt':.07})
    assert a['raw']['heading']>b['raw']['heading']>0
    m=upper_motion_cost([0,.01],[0,0],[0,0],acceleration_valid=False)
    assert abs(m['cost']-.15625)<1e-12
    s=upper_motion_cost([0,.01],[0,0],[0,.5])
    assert abs(s['raw']['upper_acceleration'])<1e-12
    assert failure_reward(800)<-5 and failure_reward(1)<-5
    print('Mathematical reference checks passed; not a dynamics validation.')
    print('Preference tracking-cost examples:',vals)
    print('Per-policy-step maximum normal cost reward:',-.1*.02*sum(CAPS.values()))

if __name__=='__main__':
    self_check()
