"""NumPy checks of actual permission and the unchanged correction interface.

No JAX action helper is called, and no saved action is rewritten for an audit.
The original disabled-all-zero verifier remains the default for older runs.
"""
import numpy as np


def physical_target(latent, config):
    c=config['residual_interface'];t=np.tanh(np.asarray(latent))
    scales=np.stack([np.where(t[...,0]>=0,c['speed_positive_scale_m_s'],
                             c['speed_negative_scale_m_s']),
                     np.full(t.shape[:-1],c['steer_scale_rad'])],axis=-1)
    return t*scales


def verify_disabled_support(raw, config, *, enabled):
    active=np.asarray(raw['active_tick'],bool)
    legal=np.asarray(raw['actor_action_valid'],bool)
    profile=np.asarray(raw['relay_profile'])
    expected=active&~legal&(profile==7)&bool(enabled)
    np.testing.assert_array_equal(raw['support_permission_override'],expected)
    # This experiment is original causal-support mode1, never constant support.
    np.testing.assert_array_equal(np.asarray(raw['relay_active'])[active],profile[active]==7)
    issued=np.asarray(raw['issued_latent'])
    np.testing.assert_array_equal(issued[active&~legal&~expected],0.)
    np.testing.assert_array_equal(issued[expected,1],0.)
    if not np.isfinite(issued[active]).all():raise ValueError('nonfinite issued action')
    # A decision is held for four lower ticks. Use its pre-advance nominal,
    # rather than later nominal samples modified by the follower's slew.
    indices=np.flatnonzero(active);starts=indices[(np.asarray(raw['physical_tick'])[active]%4)==0]
    if len(indices) and (not len(starts) or starts[0]!=indices[0]):
        raise ValueError('full control-boundary initial state required')
    for start in starts:
        block=np.arange(start,min(start+4,len(active)));block=block[active[block]]
        np.testing.assert_array_equal(expected[block],np.repeat(expected[start],len(block)))
        np.testing.assert_array_equal(issued[block],np.tile(issued[start],(len(block),1)))
        if expected[start]:
            nominal=float(raw['nominal_reference'][start,0])
            positive=config['residual_interface']['speed_positive_scale_m_s']
            request=max(0.,1.5-nominal)
            target=np.clip(request/positive,-1+1e-6,1-1e-6)*positive
            np.testing.assert_allclose(physical_target(issued[block],config),
                np.tile([target,0.],(len(block),1)),atol=3e-6,rtol=0)
    return dict(override_lower_ticks=int(expected.sum()),actor_disabled_lower_ticks=int((active&~legal).sum()))


def verify_correction_chain(raw, config, *, initial_filtered, initial_offsets):
    """Check targets, filter, slew, clipping and antiwindup against real logs."""
    c=config['residual_interface'];dt=config['timing']['lower_dt_s']
    active=np.asarray(raw['active_tick'],bool);indices=np.flatnonzero(active)
    if not np.array_equal(indices,np.arange(len(indices))):raise ValueError('active physical prefix required')
    filtered=np.asarray(initial_filtered,dtype=float);offset=np.asarray(initial_offsets,dtype=float)
    decay=np.exp(-dt/np.asarray(c['filter_tau_s']))
    down=np.array([-c['correction_speed_down_m_s2'],-c['correction_steer_rate_rad_s']])*dt
    up=np.array([c['correction_speed_up_m_s2'],c['correction_steer_rate_rad_s']])*dt
    low=np.array([c['speed_reference_bounds_m_s'][0],-c['steer_reference_bound_rad']])
    high=np.array([c['speed_reference_bounds_m_s'][1],c['steer_reference_bound_rad']])
    for i in indices:
        target=physical_target(raw['issued_latent'][i],config)
        f=decay*filtered+(1-decay)*target
        increment=np.clip(f-offset,down,up)
        before=np.asarray(raw['nominal_reference'][i])+offset+increment
        governed=np.clip(before,low,high);out=governed-raw['nominal_reference'][i]
        filtered=np.where(governed!=before,out,f);offset=out
        np.testing.assert_allclose(raw['target_offset'][i],target,atol=3e-6,rtol=0)
        np.testing.assert_allclose(raw['reference'][i],governed,atol=4e-6,rtol=0)
        np.testing.assert_allclose(raw['filtered_offset'][i],filtered,atol=4e-6,rtol=0)
        np.testing.assert_allclose(raw['correction_offset'][i],offset,atol=4e-6,rtol=0)
        # Audit each real transition from its recorded predecessor. Recursively
        # integrating a separate float64 filter would instead accumulate a
        # different arithmetic trajectory over2000 ticks. Both successor
        # values were checked above before they become the next inputs.
        filtered=np.asarray(raw['filtered_offset'][i],dtype=float)
        offset=np.asarray(raw['correction_offset'][i],dtype=float)
    return dict(correction_ticks=len(indices),checked='original filter/slew/bounds/antiwindup')
