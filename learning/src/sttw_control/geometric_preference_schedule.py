"""Causal spatial-preview rules, independent of route names and simulation.

The heuristic acceleration threshold is a DEV scheduling rule, not a physical
safety bound. This file is loaded explicitly by the sealed native runner.
"""
import math


RULE_KEYS = frozenset(('preview_m', 'engage_accel_m_s2', 'release_accel_m_s2',
                       'active_alpha', 'alpha_rate_per_s', 'upper_dt_s'))


def validate_rule(rule):
    if set(rule) != RULE_KEYS:
        raise ValueError('schedule rule fields differ from declared schema')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in rule.values()):
        raise ValueError('schedule rule values must be finite numbers')
    if (rule['preview_m'] <= 0 or rule['release_accel_m_s2'] < 0 or
            rule['engage_accel_m_s2'] <= rule['release_accel_m_s2'] or
            not 0 <= rule['active_alpha'] <= 1 or rule['alpha_rate_per_s'] <= 0 or
            rule['upper_dt_s'] != .02):
        raise ValueError('invalid preview, hysteresis, alpha or native50Hz timing')
    return dict(rule)


def preview_max_curvature(path_s, curvature, progress, preview_m):
    """Maximum of the piecewise linear path over a closed spatial interval.

    Include every covered knot and interpolated endpoints, so a narrow curve
    cannot disappear between a small fixed number of preview samples. Only the
    current progress and this spatial preview enter the rule, never future state,
    elapsed time, route labels, disturbance labels or actual speed.
    """
    import jax.numpy as j
    lo=j.clip(progress,path_s[0],path_s[-1])
    hi=j.minimum(lo+preview_m,path_s[-1])
    covered=(path_s>=lo)&(path_s<=hi)
    knots=j.max(j.where(covered,j.abs(curvature),0.))
    return j.maximum(knots,j.maximum(j.abs(j.interp(lo,path_s,curvature)),
                                    j.abs(j.interp(hi,path_s,curvature))))


def advance(alpha, latched, original_speed, path_s, curvature, progress, rule):
    """One50Hz decision; caller preserves the complete native physical carry."""
    import jax.numpy as j
    max_k=preview_max_curvature(path_s,curvature,progress,rule['preview_m'])
    acceleration=original_speed**2*max_k
    active=j.where(latched,acceleration>=rule['release_accel_m_s2'],
                   acceleration>rule['engage_accel_m_s2'])
    target=j.where(active,rule['active_alpha'],1.)
    step=rule['alpha_rate_per_s']*rule['upper_dt_s']
    executed=alpha+j.clip(target-alpha,-step,step)
    cap=j.minimum(original_speed,j.sqrt(rule['engage_accel_m_s2']/j.maximum(max_k,1e-12)))
    cap=j.where(active,cap,original_speed)
    return executed,active,dict(rule_active=active,rule_preview_max_curvature=max_k,
        rule_accel_estimate=acceleration,rule_alpha_target=target,rule_speed_target=cap)


def cap_speed_latent(z, nominal, speed_cap, active, config):
    """Cap H75 proposed speed within its ORIGINAL two-channel authority.

    Preserve a stronger H75 deceleration, the steering latent, and inactive
    exact identity. The caller still uses the unchanged native filters, slew and
    amplitude bounds. A cap below representable authority is logged as limited;
    it does not grant an enlarged speed residual or alter original v_user.
    """
    import jax.numpy as j
    c=config['residual_interface'];eps=1e-6
    scale=j.where(z[0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s'])
    proposed=nominal[0]+j.tanh(z[0])*scale
    requested=j.minimum(proposed,speed_cap)
    correction=requested-nominal[0]
    inverse_scale=j.where(correction>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s'])
    fraction=correction/inverse_scale
    inverse=j.arctanh(j.clip(fraction,-1+eps,1-eps))
    # Also avoid a roundtrip when the geometric cap does not bind H75.
    binds=active&(proposed>speed_cap)
    output=z.at[0].set(j.where(binds,inverse,z[0]))
    realized=nominal[0]+j.tanh(output[0])*j.where(output[0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s'])
    return output,dict(rule_h75_proposed_speed=proposed,
        rule_requested_speed=j.where(active,requested,proposed),
        rule_realizable_speed=realized,rule_speed_cap_binding=binds,
        rule_speed_authority_limited=binds&(j.abs(fraction)>1-eps))
