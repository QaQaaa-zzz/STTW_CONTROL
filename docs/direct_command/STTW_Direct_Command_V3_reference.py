"""Numerical specification checks only; this is not a trained policy or simulator.
Run: python STTW_Direct_Command_V3_reference.py [path/to/config.json]
Requires only Python standard library and NumPy.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from typing import Any
import numpy as np


def finite_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f'{name} must be finite')
    return arr


def validate_alpha(alpha: float) -> None:
    if not math.isfinite(alpha) or alpha not in (0.0, 1.0):
        raise ValueError('alpha must equal 0 or 1')


def huber(z: Any) -> Any:
    a = np.abs(finite_array(z, 'huber input'))
    return np.minimum(a, 1.)**2 + 2.*np.maximum(a-1., 0.)


def q_ratio(v: float, c: dict) -> float:
    r=c['reference']
    if not math.isfinite(v) or not 1.5 <= v <= 3.0:
        raise ValueError('reference check only supports v in [1.5,3.0]')
    A=r['gravity_expected']*r['wheelbase_m_expected']/math.cos(math.radians(r['caster_deg_expected']))
    B=r['cg_forward_m_expected']*r['trail_m_expected']*r['gravity_expected']/r['cg_height_m_expected']
    return A/(v*v-B)


def raw_context(vc: float, dc: float, dv: float, dd: float, clock: float, c: dict) -> tuple[float,float,float]:
    """Return chi, g for CURRENT interval and clock for NEXT interval.
    Caller must not advance this clock twice between observation and reward.
    """
    finite_array([vc,dc,dv,dd,clock], 'raw context')
    r=c['reward']; dt=c['plant']['control_dt_s']; phi=dc/q_ratio(vc,c)
    chi=float(np.clip((abs(phi)-r['conflict_phi_start_rad'])/(r['conflict_phi_full_rad']-r['conflict_phi_start_rad']),0,1))
    g=float(np.clip((clock-r['recovery_settle_s'])/r['recovery_ramp_s'],0,1))
    eligible=(abs(dc)<=r['recovery_raw_steer_abs_max_rad'] and abs(phi)<=r['recovery_raw_phi_abs_max_rad']
        and abs(dv)<=r['recovery_raw_speed_rate_abs_max_m_s2'] and abs(dd)<=r['recovery_raw_steer_rate_abs_max_rad_s'])
    # When a newly issued command becomes ineligible, g must be zero now,
    # regardless of the stale accumulated clock from the previous command.
    if not eligible:
        return chi,0.,0.
    return chi,g,min(r['recovery_settle_s']+r['recovery_ramp_s'], clock+dt)


def map_latent(z: Any, c: dict) -> np.ndarray:
    z=finite_array(z,'latent')
    if z.shape != (2,):
        raise ValueError('z must have shape (2,)')
    a=np.tanh(z); d=c['action']
    return np.array([a[0]*(d['speed_positive_scale_m_s'] if a[0]>=0 else d['speed_negative_scale_m_s']),
                     d['steer_scale_rad']*a[1]])


def execute_reference(raw: Any, offsets: Any, target_offsets: Any, c: dict) -> tuple[np.ndarray,np.ndarray]:
    raw=finite_array(raw,'raw'); old=finite_array(offsets,'offset'); target=finite_array(target_offsets,'target')
    if any(x.shape!=(2,) for x in (raw,old,target)):
        raise ValueError('raw,offsets,target must have shape (2,)')
    a=c['action']; dt=c['plant']['control_dt_s']
    change=np.clip(target-old,[-a['correction_speed_decrease_rate_m_s2']*dt,-a['correction_steer_rate_rad_s']*dt],
                   [a['correction_speed_increase_rate_m_s2']*dt,a['correction_steer_rate_rad_s']*dt])
    goal=np.clip(raw+old+change,[a['speed_reference_min_m_s'],-a['steer_reference_abs_max_rad']],
                 [a['speed_reference_max_m_s'],a['steer_reference_abs_max_rad']])
    return goal,goal-raw


def weights(alpha: float, chi: float, gate: float, c: dict) -> tuple[float,float]:
    validate_alpha(alpha)
    if not 0<=chi<=1 or not 0<=gate<=1:
        raise ValueError('chi and gate must be within [0,1]')
    r=c['reward']; high=r['priority_high']; diff=high-r['priority_low']
    wv=high-diff*chi*(1-alpha)
    wd=(1-gate)*(high-diff*chi*alpha)+gate*r['recovery_steer_weight']
    return wv,wd


def cost_parts(*, alpha: float, chi: float, gate: float, speed_error: float,
               steer_error: float, heading_error: float, roll: float, roll_rate: float,
               actual_speed: float, executed_offsets: Any, final_command: Any,
               previous_final_command: Any, config: dict) -> tuple[dict,dict,float]:
    finite_array([speed_error,steer_error,heading_error,roll,roll_rate,actual_speed],'physical state')
    d=finite_array(executed_offsets,'executed offset'); u=finite_array(final_command,'final command')
    up=finite_array(previous_final_command,'previous command')
    if any(x.shape!=(2,) for x in (d,u,up)):raise ValueError('offset/command vectors must have two entries')
    r=config['reward'];wv,wd=weights(alpha,chi,gate,config)
    H=lambda x:float(huber(x))
    p={
      'speed':wv*H(speed_error/r['speed_error_scale_m_s']),
      'steer':wd*H(steer_error/r['steer_error_scale_rad']),
      'heading':r['heading_weight']*gate*H(max(abs(heading_error)-r['heading_deadband_rad'],0)/r['heading_error_scale_rad']),
      'roll':r['roll_weight']*H(max(abs(roll)-r['roll_soft_start_rad'],0)/r['roll_error_scale_rad']),
      'roll_rate':r['roll_rate_weight']*H(max(abs(roll_rate)-r['roll_rate_soft_start_rad_s'],0)/r['roll_rate_error_scale_rad_s']),
      'overspeed':r['overspeed_weight']*H(max(speed_error-r['overspeed_free_band_m_s'],0)/r['overspeed_scale_m_s']),
      'low_speed':r['low_speed_weight']*H(max(r['low_speed_start_m_s']-actual_speed,0)/r['low_speed_scale_m_s']),
      'correction':r['reference_correction_weight']*float(np.sum((d/np.array(r['correction_normalizers']))**2)),
      'command_change':r['final_command_change_weight']*float(np.sum(((u-up)/np.array(r['final_command_normalizers']))**2))}
    raw=sum(p.values()); factor=min(1.,r['cost_rate_cap']/max(raw,1e-12))
    eff={k:v*factor for k,v in p.items()}
    reward=-r['scale']*config['plant']['control_dt_s']*min(raw,r['cost_rate_cap'])
    return p,eff,reward


def failure_reward(remaining_including_current: int, c: dict) -> float:
    M=round(c['commands']['episode_seconds']/c['plant']['policy_dt_s'])
    if not isinstance(remaining_including_current,int) or not 1<=remaining_including_current<=M:
        raise ValueError('remaining steps out of range')
    r=c['reward'];gamma=c['ppo']['gamma'];N=remaining_including_current
    geometric=-math.expm1(N*math.log(gamma))/(1-gamma)
    return -r['failure_extra_penalty']-r['scale']*c['plant']['policy_dt_s']*r['cost_rate_cap']*geometric


def run_checks(c: dict) -> dict:
    n=c['network']; count=n['history_frames']*len(n['frame_fields'])+n['history_frames']+len(n['context_fields'])
    assert count==345==n['actor_input_dim'] and n['critic_input_dim']==count+1
    dims=[count,*n['hidden_sizes'],2]; params=sum((i+1)*o for i,o in zip(dims[:-1],dims[1:]))
    assert params==69186
    assert abs(c['plant']['policy_dt_s']/c['plant']['control_dt_s']-4)<1e-12
    assert c['ppo']['num_envs']*c['ppo']['rollout_policy_steps']//c['ppo']['minibatches']==c['ppo']['minibatch_size']
    assert c['budget']['default_policy_transitions']==c['ppo']['num_envs']*c['ppo']['rollout_policy_steps']*c['ppo']['default_updates']
    assert c['budget']['default_control_transitions_upper']==4*c['budget']['default_policy_transitions']
    assert sum(c['commands'][k] for k in ('nominal_fraction','conflict_fraction','random_fraction'))==1
    assert c['ppo']['num_envs']%2==0
    assert np.array_equal(map_latent([0,0],c),[0,0])
    rng=np.random.default_rng(42)
    for _ in range(10000):
        raw=np.array([rng.uniform(2,2.6),rng.uniform(-.3,.3)])
        goal,off=execute_reference(raw,[0,0],[0,0],c)
        assert np.array_equal(goal,raw) and np.array_equal(off,[0,0])
    high=map_latent([1000,1000],c); low=map_latent([-1000,-1000],c)
    assert np.allclose(high,[.25,.2]) and np.allclose(low,[-1,-.2])
    # Raw gates are independent of the policy and drop immediately on new conflict.
    chi,g,nclock=raw_context(2.3,0,0,0,.8,c); assert g==1 and nclock==.8 and chi==0
    chi,g,nclock=raw_context(2.6,.25,0,0,.8,c); assert chi==1 and g==0 and nclock==0
    examples={}
    for alpha in [0.,1.]:
        wv,wd=weights(alpha,1.,0.,c)
        examples[str(alpha)]=[wv*float(huber(ev/.1))+wd*float(huber(ed/.05)) for ev,ed in [(-.4,.01),(-.04,-.08)]]
    assert np.allclose(examples['0.0'],[5.92,17.728]) and np.allclose(examples['1.0'],[56.032,3.04])
    assert examples['0.0'][0]<examples['0.0'][1] and examples['1.0'][1]<examples['1.0'][0]
    # Compare hypothetical one-interval recovery errors (not a physical rollout).
    wv,wd=weights(0,0,1,c)
    no_recovery=4*float(huber((.3-.03)/.15))
    correction=wd*float(huber(.04/.05))+4*float(huber((.05-.03)/.15))
    assert correction<no_recovery
    M=800;worst=-c['reward']['scale']*c['plant']['policy_dt_s']*c['reward']['cost_rate_cap']
    for N in (1,2,5,100,400,800):
        brute=worst*sum(c['ppo']['gamma']**j for j in range(N))-c['reward']['failure_extra_penalty']
        assert math.isclose(failure_reward(N,c),brute,abs_tol=1e-10)
        assert failure_reward(N,c)<brute+c['reward']['failure_extra_penalty']
    for _ in range(1000):
        p,e,r=cost_parts(alpha=float(rng.integers(2)),chi=rng.random(),gate=rng.random(),
            speed_error=rng.normal(),steer_error=rng.normal(),heading_error=rng.normal()*4,
            roll=rng.normal()*.4,roll_rate=rng.normal()*2,actual_speed=rng.uniform(.5,3),
            executed_offsets=rng.normal(size=2),final_command=rng.normal(size=2),previous_final_command=rng.normal(size=2),config=c)
        assert min(p.values())>=0 and sum(e.values())<=100+1e-8
        assert math.isclose(r,-c['reward']['scale']*c['plant']['control_dt_s']*sum(e.values()),abs_tol=1e-10)
        assert -.05-1e-10<=r<=0
    for bad in (-1,.5,float('nan'),2):
        try: validate_alpha(bad)
        except ValueError:pass
        else:raise AssertionError('invalid alpha accepted')
    return {'status':'passed','scope':'pure numerical/specification checks only; no repository import, simulator rollout, or training',
            'actor_input_dim':count,'critic_input_dim':count+1,'actor_mean_parameters':params,
            'policy_transitions':c['budget']['default_policy_transitions'],
            'control_transitions_upper':c['budget']['default_control_transitions_upper'],
            'gamma_time_constant_s':-c['plant']['policy_dt_s']/math.log(c['ppo']['gamma']),
            'synthetic_command_costs_AB':examples,'failure_reward_at_start':failure_reward(800,c),
            'failure_reward_at_last_interval':failure_reward(1,c),
            'normal_zero_action_identity_samples':10000,'reward_reconstruction_random_samples':1000}


if __name__=='__main__':
    path=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('STTW_Direct_Command_V3.json')
    config=json.loads(path.read_text(encoding='utf-8'))
    result=run_checks(config)
    print(json.dumps(result,ensure_ascii=False,indent=2))
