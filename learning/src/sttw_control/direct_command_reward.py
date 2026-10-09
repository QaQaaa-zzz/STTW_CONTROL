"""V3 per-control-tick cost and finite-task physical failure accounting."""
from __future__ import annotations

import jax.numpy as jp


COMPONENTS = ('speed', 'steer', 'heading', 'roll', 'roll_rate', 'overspeed',
              'low_speed', 'correction', 'command_change')


def huber(z):
    a = jp.abs(z)
    return jp.where(a <= 1., z*z, 2.*a-1.)


def interval_cost(*, alpha, chi, g, raw, actual_speed, actual_steer,
                  heading_error, roll, roll_rate, executed_offsets,
                  final_command, previous_final_command, spec, yaw_rate=0., cc=None, peak_roll=None):
    """Score post-physics state against this interval's original raw command.

    Returns raw/effective component dictionaries and a scalar 5 ms reward.
    Invalid values remain invalid; caller records physical/policy failure.
    """
    if spec.get('smooth_v4'):
        return smooth_state_cost(alpha,chi,g,raw,actual_speed,actual_steer,heading_error,roll,roll_rate,executed_offsets,yaw_rate,spec,cc=cc,peak_roll=peak_roll)
    r = spec['reward']; dt = spec['plant']['control_dt_s']
    raw = jp.asarray(raw)
    ev = actual_speed - raw[..., 0]
    ed = actual_steer - raw[..., 1]
    priority_gap = r['priority_high']-r['priority_low']
    wv = r['priority_high'] - priority_gap*chi*(1.-alpha)
    wd_base = r['priority_high'] - priority_gap*chi*alpha
    wd = (1.-g)*wd_base + g*r['recovery_steer_weight']
    correction_normalizers = jp.asarray(r['correction_normalizers'])
    final_normalizers = jp.asarray(r['final_command_normalizers'])
    raw_components = dict(
        speed=wv*huber(ev/r['speed_error_scale_m_s']),
        steer=wd*huber(ed/r['steer_error_scale_rad']),
        heading=r['heading_weight']*g*huber(jp.maximum(jp.abs(heading_error)-r['heading_deadband_rad'], 0.)/r['heading_error_scale_rad']),
        roll=r['roll_weight']*huber(jp.maximum(jp.abs(roll)-r['roll_soft_start_rad'], 0.)/r['roll_error_scale_rad']),
        roll_rate=r['roll_rate_weight']*huber(jp.maximum(jp.abs(roll_rate)-r['roll_rate_soft_start_rad_s'], 0.)/r['roll_rate_error_scale_rad_s']),
        overspeed=r['overspeed_weight']*huber(jp.maximum(ev-r['overspeed_free_band_m_s'], 0.)/r['overspeed_scale_m_s']),
        low_speed=r['low_speed_weight']*huber(jp.maximum(r['low_speed_start_m_s']-actual_speed, 0.)/r['low_speed_scale_m_s']),
        correction=r['reference_correction_weight']*jp.sum((jp.asarray(executed_offsets)/correction_normalizers)**2, axis=-1),
        command_change=r['final_command_change_weight']*jp.sum(((jp.asarray(final_command)-jp.asarray(previous_final_command))/final_normalizers)**2, axis=-1))
    raw_cost = sum(raw_components.values())
    cap = r['cost_rate_cap']
    effective_cost = jp.minimum(raw_cost, cap)
    factor = jp.minimum(1., cap/jp.maximum(raw_cost, 1e-12))
    effective_components = {name: value*factor for name, value in raw_components.items()}
    cap_fraction = jp.where(raw_cost > cap, 1., 0.)
    return dict(raw_components=raw_components,
                effective_components=effective_components,
                raw_cost=raw_cost, effective_cost=effective_cost,
                cap_fraction=cap_fraction,
                reward=-r['scale']*dt*effective_cost,
                speed_weight=wv, steer_weight=wd)


def failure_reward(remaining_including_current, spec):
    """Replace the entire current 20 ms policy reward on first physical failure."""
    r = spec['reward']; p = spec['ppo']; gamma = p['gamma']
    n = jp.asarray(remaining_including_current)
    geometric = (1.-jp.power(gamma, n))/(1.-gamma)
    return -r['failure_extra_penalty'] - (
        r['scale']*spec['plant']['policy_dt_s']*r.get('failure_absorbing_cost_rate',r.get('cost_rate_cap'))*geometric)


def smooth_state_cost(alpha, chi, g, raw, speed, steer, debt, roll, roll_rate, offsets, yaw_rate, spec, cc=None, peak_roll=None):
    """SmoothV4: independent protections, no total-cost clipping."""
    r=spec['reward']; n=r['normal']; a0=r['conflict_alpha0']; a1=r['conflict_alpha1']; rec=r['recovery']; safe=r['safety']; mag=r['offset_magnitude']
    if spec.get('preference_v52') and peak_roll is None:raise ValueError('V5.2 requires interval substep peak_roll')
    risk=jp.maximum(jp.abs(roll),peak_roll) if spec.get('preference_v52') else jp.abs(roll)
    pos=lambda x:jp.maximum(x,0.)
    ev=speed-raw[0]; ed=steer-raw[1]; under=pos(-ev)
    vn=n['speed_weight']*huber(pos(jp.abs(ev)-n.get('speed_deadband_m_s',0.))/n['speed_scale_m_s']); dn=n['steer_weight']*huber(pos(jp.abs(ed)-n.get('steer_deadband_rad',0.))/n['steer_scale_rad'])
    v0=a0['underspeed_weight']*huber(under/a0['underspeed_scale_m_s'])+a0['overspeed_tracking_weight']*huber(pos(ev)/a0['overspeed_tracking_scale_m_s'])+a0['excess_underspeed_weight']*huber(pos(under-a0['underspeed_soft_band_m_s'])/a0['excess_scale_m_s'])
    v1=a1['speed_weight']*huber(ev/a1['speed_scale_m_s'])
    d0=a0['steer_weight']*huber(ed/a0['steer_scale_rad'])
    d1=a1['steer_weight']*huber(ed/a1['steer_scale_rad'])+a1['excess_steer_weight']*huber(pos(jp.abs(ed)-a1['steer_soft_band_rad'])/a1['excess_scale_rad'])
    ref=spec['reference']; rc=raw[0]*jp.cos(ref['caster_deg_expected']*jp.pi/180)*jp.tan(raw[1])/ref['wheelbase_m_expected']
    wm=mag['weight_floor']+mag['neutral_extra_weight']*(1-chi)*jp.exp(-(debt/mag['heading_scale_rad'])**2-(pos(jp.abs(roll)-mag['roll_start_rad'])/mag['roll_scale_rad'])**2-(roll_rate/mag['roll_rate_scale_rad_s'])**2)
    rawc=dict(speed=(1-g)*((1-chi)*vn+chi*jp.where(alpha==0,v0,v1))+g*rec['speed_weight']*huber(pos(jp.abs(ev)-rec.get('speed_deadband_m_s',0.))/rec['speed_scale_m_s']),
      steer=(1-g)*((1-chi)*dn+chi*jp.where(alpha==0,d0,d1))+g*rec['steer_weight']*huber(ed/rec['steer_scale_rad']),
      heading=rec['heading_weight']*g*huber(pos(jp.abs(debt)-rec['heading_deadband_rad'])/rec['heading_scale_rad']),
      roll=safe['roll_weight']*huber(pos(risk-safe['roll_start_rad'])/safe['roll_scale_rad']),
      roll_rate=safe['roll_rate_weight']*huber(pos(jp.abs(roll_rate)-safe['roll_rate_start_rad_s'])/safe['roll_rate_scale_rad_s']),
      overspeed=safe['overspeed_weight']*huber(pos(ev-safe['overspeed_band_m_s'])/safe['overspeed_scale_m_s']),
      low_speed=safe['low_speed_weight']*huber(pos(safe['low_speed_start_m_s']-speed)/safe['low_speed_scale_m_s']),
      magnitude=wm*jp.sum((offsets/jp.asarray(mag['normalizers']))**2))
    if spec.get('preference_v52'):
        work=r['working_roll_excess']
        rawc['working_roll_excess']=work['weight']*huber(pos(risk-work['start_rad'])/work['scale_rad'])
    if spec.get('priority_recovery_v51'):
        primary=r['primary_excess']
        steer_excess=huber(pos(jp.abs(ed)-primary['alpha0_steer_tolerance_rad'])/primary['alpha0_steer_scale_rad'])
        speed_excess=huber(pos(jp.abs(ev)-primary['alpha1_speed_tolerance_m_s'])/primary['alpha1_speed_scale_m_s'])
        rawc['primary_excess']=primary['weight']*chi*(1-g)*((1-alpha)*steer_excess+alpha*speed_excess)
        yaw_debt=jp.clip(rec['yaw_debt_gain_per_s']*debt,-rec['yaw_debt_rate_limit_rad_s'],rec['yaw_debt_rate_limit_rad_s'])
        rawc['yaw_recovery']=rec['yaw_recovery_weight']*g*huber(((yaw_rate-rc)-yaw_debt)/rec['yaw_scale_rad_s'])
    else:
        rawc['yaw_damping']=rec['yaw_damping_weight']*g*jp.exp(-(debt/rec['yaw_damping_heading_scale_rad'])**2)*huber((yaw_rate-rc)/rec['yaw_scale_rad_s'])
    if spec.get('preference_v5'):
        from .direct_command_policy import q_ratio
        if cc is None:
            raise ValueError('V5 compatibility cost requires the live controller configuration')
        priority = r['reference_priority']; compat = r['command_compatibility']
        offsets = jp.asarray(offsets)
        governed = jp.asarray(raw) + offsets
        rawc['reference_priority'] = priority['weight']*chi*(1-g)*(
            (1-alpha)*huber(pos(jp.abs(offsets[1])-priority['alpha0_steer_correction_deadband_rad'])/priority['alpha0_steer_correction_scale_rad'])
            + alpha*huber(pos((-offsets[0] if priority.get('alpha1_downward_only',False) else jp.abs(offsets[0]))-priority['alpha1_speed_correction_deadband_m_s'])/priority['alpha1_speed_correction_scale_m_s']))
        phi_requested = governed[1]/q_ratio(governed[0],cc)
        rawc['command_compatibility'] = compat['weight']*huber(pos(jp.abs(phi_requested)-compat['roll_reference_limit_rad'])/compat['scale_rad'])
    caps=r['independent_component_caps']; eff={k:jp.minimum(v,caps[k]) for k,v in rawc.items()}
    return dict(raw_components=rawc,effective_components=eff,raw_cost=sum(rawc.values()),effective_cost=sum(eff.values()),cap_fraction=jp.any(jp.stack([v>caps[k] for k,v in rawc.items()])).astype(jp.float32),reward=-r['scale']*spec['plant']['control_dt_s']*sum(eff.values()))


def upper_motion_cost(now, previous, previous_rate, valid, spec, chi=0., heading_error=0.):
    r=spec['reward']; dt=r['rate_sample_dt_s']; rate=(now-previous)/dt; acc=(rate-previous_rate)/dt
    rho = .2 + .8*(1-chi)*jp.exp(-(heading_error/.1)**2) if spec.get('preference_v5') else 1.
    raw={'upper_rate':rho*r['upper_rate']['weight']*jp.sum((rate/jp.asarray(r['upper_rate']['normalizers']))**2),
         'upper_acceleration':jp.where(valid,rho*r['upper_acceleration']['weight']*jp.sum((acc/jp.asarray(r['upper_acceleration']['normalizers']))**2),0.)}
    effective={k:jp.minimum(v,r['independent_component_caps'][k]) for k,v in raw.items()}
    return rate,acc,raw,effective
