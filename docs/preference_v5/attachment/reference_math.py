"""Numerical checks for the proposed V5 specification; NOT vehicle simulation.

This is an executable scalar/vector reference for cost definitions.  The vehicle
implementation must use its current controller configuration and state timing.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np

@dataclass(frozen=True)
class BicycleParameters:
    wheelbase: float = .408
    caster: float = math.radians(25.)
    gravity: float = 9.8
    cg_forward: float = .164
    trail: float = .024
    cg_height: float = .2

def huber(x):
    a = np.abs(np.asarray(x, dtype=float))
    return np.where(a <= 1., a*a, 2.*a-1.)

def positive(x):
    return np.maximum(x, 0.)

def q_ratio(v, p=BicycleParameters()):
    v = np.asarray(v, dtype=float)
    if np.any(~np.isfinite(v)) or np.any((v < 1.5) | (v > 3.0)):
        raise ValueError('Inverse/closed-form check is limited to v in [1.5,3.0].')
    return p.gravity*p.wheelbase/np.cos(p.caster)/(v*v-p.cg_forward*p.trail*p.gravity/p.cg_height)

def preference_and_compatibility(alpha, raw, governed):
    if alpha not in (0, 1):
        raise ValueError('Only endpoints 0 and 1 are specified.')
    raw = np.asarray(raw, float); governed = np.asarray(governed, float)
    dv, dd = governed[...,0]-raw[...,0], governed[...,1]-raw[...,1]
    primary = 4.*((1-alpha)*huber(positive(np.abs(dd)-.01)/.05)
                  + alpha*huber(positive(np.abs(dv)-.05)/.20))
    phi_req = governed[...,1]/q_ratio(governed[...,0])
    compatibility = 2.*huber(positive(np.abs(phi_req)-.26)/.05)
    return primary, compatibility

def costs(alpha, raw, governed, actual_speed, actual_steer, roll,
          roll_rate, heading_error, yaw_rate, yaw_reference, chi, g,
          offset_rate=(0.,0.), offset_acceleration=(0.,0.)):
    """Unscaled cost rates. Smooth derivatives are computed at 20 ms endpoints.

    The implementation integrates physical cost rates at 5 ms and adds smooth
    rates once with a 20 ms multiplier. Do not charge them four times unscaled.
    """
    if alpha not in (0, 1): raise ValueError('Endpoint alpha required')
    raw=np.asarray(raw,float); gov=np.asarray(governed,float)
    ev=np.asarray(actual_speed)-raw[...,0]; ed=np.asarray(actual_steer)-raw[...,1]
    under=positive(-ev); over=positive(ev)
    vn=8*huber(positive(np.abs(ev)-.03)/.1)
    dn=8*huber(positive(np.abs(ed)-.003)/.05)
    if alpha == 0:
        vc=.4*huber(under/.3)+.4*huber(over/.1)+2*huber(positive(under-.6)/.2)
        dc=8*huber(ed/.05)
    else:
        vc=8*huber(ev/.1)
        dc=.8*huber(ed/.05)+2*huber(positive(np.abs(ed)-.15)/.05)
    vr=8*huber(positive(np.abs(ev)-.03)/.1)
    dr=huber(ed/.1)
    speed=(1-g)*((1-chi)*vn+chi*vc)+g*vr
    steer=(1-g)*((1-chi)*dn+chi*dc)+g*dr
    ep=np.asarray(heading_error)
    rho=.2+.8*(1-chi)*np.exp(-(ep/.1)**2)
    rate=np.asarray(offset_rate); acc=np.asarray(offset_acceleration)
    off=gov-raw
    magw=.05+.45*(1-chi)*np.exp(-(ep/.1)**2
        -(positive(np.abs(roll)-.2)/.1)**2-(np.asarray(roll_rate)/.6)**2)
    refpri,compat=preference_and_compatibility(alpha,raw,gov)
    return {
        'speed':speed,'steer':steer,
        'heading':6*g*huber(positive(np.abs(ep)-.01)/.1),
        'roll':40*huber(positive(np.abs(roll)-.26)/.04),
        'roll_rate':huber(positive(np.abs(roll_rate)-.8)/.8),
        'overspeed':2*huber(positive(ev-.05)/.1),
        'low_speed':4*huber(positive(1.5-np.asarray(actual_speed))/.2),
        'magnitude':magw*np.sum((off/np.asarray([.5,.1]))**2,axis=-1),
        'upper_rate':.03*rho*np.sum((rate/np.asarray([1.,.4]))**2,axis=-1),
        'upper_acceleration':.01*rho*np.sum((acc/10.)**2,axis=-1),
        'yaw_damping':.5*g*np.exp(-(ep/.15)**2)*huber((np.asarray(yaw_rate)-yaw_reference)/.2),
        'reference_priority':chi*(1-g)*refpri,
        'command_compatibility':compat,
    }

def failure_reward(n, gamma=.997, bound=1680.):
    if int(n) != n or n < 1: raise ValueError('positive integer remaining steps required')
    return -5.-.1*.02*bound*(-math.expm1(n*math.log(gamma)))/(1-gamma)

def checks():
    here=Path(__file__).resolve().parent
    c=json.loads((here/'STTW_R196_PreferenceV5_alpha0.json').read_text())
    assert sum(c['reward']['caps'].values()) == 1680.
    raw=np.array([2.6,.25]); a=np.array([2.19,.25]); b=np.array([2.6,.17])
    report={}
    for alpha in (0,1):
        ra,ca=preference_and_compatibility(alpha,raw,a)
        rb,cb=preference_and_compatibility(alpha,raw,b)
        va=costs(alpha,raw,a,a[0],a[1],.26,0.,0.,0.,0.,1.,0.)
        vb=costs(alpha,raw,b,b[0],b[1],.26,0.,0.,0.,0.,1.,0.)
        report[f'alpha{alpha}']={'A_reference_penalty':float(ra),'B_reference_penalty':float(rb),
            'A_compatibility':float(ca),'B_compatibility':float(cb),
            'A_synthetic_total':float(sum(va.values())),'B_synthetic_total':float(sum(vb.values()))}
    assert report['alpha0']['A_synthetic_total'] < report['alpha0']['B_synthetic_total']
    assert report['alpha1']['B_synthetic_total'] < report['alpha1']['A_synthetic_total']
    assert preference_and_compatibility(0,raw,raw)[0] == 0
    assert preference_and_compatibility(1,raw,raw)[0] == 0
    for phi in [.30,.33,.345,.40,.69]:
        v=40*float(huber((phi-.26)/.04))
        assert v < 900
        report[f'roll_{phi}']={'V4_after_cap':min(v,100),'V5_after_cap':min(v,900)}
    report['initial_zero_mean_probability_speed_request_below_minus_0_4']={
        str(s):.5*math.erfc(math.atanh(.4)/(s*math.sqrt(2))) for s in [.1,.3]}
    for n in [1,250,800]:
        worst=-.1*.02*1680*(-math.expm1(n*math.log(.997)))/(1-.997)
        assert failure_reward(n) < worst
    assert .03 < .10 and .003 < .04
    report['scope']='Algebra only. Synthetic A/B are NOT simulated reachable trajectories.'
    report['physical_simulations_run']=0
    return report

if __name__=='__main__':
    result=checks()
    out=Path(__file__).with_name('reference_checks.json')
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))
