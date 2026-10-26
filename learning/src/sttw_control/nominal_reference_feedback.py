"""Frozen nominal-response deviation feedback and an existing-gate control.

The nominal table is acquired without perturbations, indexed by current path
progress, and never updated from disturbed outcomes. This is a DEV mechanism
probe, not a stability certificate or a claim of novel feedback control.
"""
import numpy as np
import jax.numpy as j

from .reference_task_feedback import context_geometry, steering_feedback

NOMINAL_ROLES = ('feedback_nominal', 'feedback_gate')


def nominal_table(progress, ey, eh, *, initial_progress, initial_ey, initial_eh):
    progress, ey, eh = [np.asarray(x, float) for x in (progress, ey, eh)]
    if (progress.shape != (2000,) or ey.shape != progress.shape or eh.shape != progress.shape
            or not np.isfinite(np.r_[progress, ey, eh, initial_progress, initial_ey, initial_eh]).all()
            or not np.all(np.diff(np.r_[initial_progress, progress]) > 0)):
        raise ValueError('complete finite strictly increasing nominal progress required')
    return dict(progress=np.r_[initial_progress, progress].astype(np.float32),
        # Mirror the existing observation scale .2 and clip 5, not privileged ey.
        path_error=np.clip(np.r_[initial_ey, ey], -1., 1.).astype(np.float32),
        heading_error=np.unwrap(np.r_[initial_eh, eh]).astype(np.float32))


def lookup_nominal(progress, table):
    covered=(progress >= table['progress'][0]) & (progress <= table['progress'][-1])
    ey=j.interp(progress, table['progress'], table['path_error'])
    eh=j.interp(progress, table['progress'], table['heading_error'])
    return ey, eh, covered


def adjusted_errors(ey, eh, nominal_ey, nominal_eh):
    difference=eh-nominal_eh
    return ey-nominal_ey, j.arctan2(j.sin(difference), j.cos(difference))


class DeviationActor:
    def __init__(self, actor, config, spec):
        self.actor, self.config, self.spec=actor, config, spec

    def apply(self, parameters, observation):
        original=self.actor.apply(parameters['actor'], observation)
        ey, eh=context_geometry(observation, self.config)
        ey, eh=adjusted_errors(ey, eh, *parameters['reference_error'])
        return steering_feedback(original, ey, eh, parameters['effective_gain'], self.config, self.spec)[0]


def make_nominal_evaluation_step(env, actor, *, horizon_s, spec):
    from .finite_value_evaluation import make_evaluation_step
    from .reference_reentry import actor_permission
    from .reference_reserve_gate import gate_step
    advance=make_evaluation_step(env, DeviationActor(actor,env.config,spec), horizon_s=horizon_s)

    def step(carry, parameters, path, speed, pulse, mode):
        s=carry.state
        stopped=s.physical.failed | s.fault | (s.tick >= round(horizon_s/env.cc.dt))
        enabled=actor_permission(s.domain_exit,s.follower.invalid,carry.clean_blocks,j.bool_(False),stopped)
        obs, _, _=env.observation(s,path,speed,horizon_s)
        safe=j.where(enabled,obs,j.zeros_like(obs))
        original=actor.apply(parameters['actor'],safe)
        ey, eh=context_geometry(safe,env.config)
        nominal_ey, nominal_eh, covered=lookup_nominal(s.projection.progress,parameters['nominal_table'])
        measured, _, _=env.physics.observe(s.physical.data)
        _, _, gate=gate_step(carry.relay,j.zeros(2),rho=j.float32(0),
                            roll_rate=measured[1],mode=2,dt=env.config['timing']['upper_dt_s'])
        is_nominal=parameters['feedback_kind']==1
        ref_ey=j.where(is_nominal,nominal_ey,0.)
        ref_eh=j.where(is_nominal,nominal_eh,0.)
        on=j.where(is_nominal,covered,(mode==1)&gate['active'])
        gain=j.where(on,parameters['feedback_gain_rad'],0.)
        errors=adjusted_errors(ey,eh,ref_ey,ref_eh)
        proposed, requested, realized=steering_feedback(original,*errors,gain,env.config,spec)
        supplied=dict(parameters,reference_error=j.stack([ref_ey,ref_eh]),effective_gain=gain)
        end, logs=advance(carry,supplied,path,speed,pulse,mode)
        active=logs['active_tick'];legal=logs['actor_action_valid']
        logs.update(original_actor_latent=j.where(legal[:,None],j.tile(original,(4,1)),j.nan),
            feedback_actor_latent=j.where(legal[:,None],j.tile(proposed,(4,1)),j.nan),
            feedback_path_error_m=j.where(active,ey,j.nan),
            feedback_heading_error_rad=j.where(active,eh,j.nan),
            feedback_requested_steer_rad=j.where(legal,requested,0.),
            feedback_realized_steer_rad=j.where(legal,realized,0.),
            feedback_effective_gain_rad=j.where(active,gain,j.nan),
            nominal_progress_m=j.where(active,s.projection.progress,j.nan),
            nominal_profile_covered=active&covered,
            nominal_path_error_m=j.where(active,ref_ey,j.nan),
            nominal_heading_error_rad=j.where(active,ref_eh,j.nan))
        return end,logs
    return step


def verify_nominal_feedback(raw, case, config, spec, table):
    """NumPy replay of interpolation, gate, mapping and original permissions."""
    active=np.asarray(raw['active_tick'],bool)
    legal=np.asarray(raw['actor_action_valid'],bool)
    disabled=active&~legal
    np.testing.assert_array_equal(raw['issued_latent'][disabled],np.zeros((disabled.sum(),2)))
    progress=np.asarray(raw['nominal_progress_m'])[active]
    covered=(progress>=table['progress'][0])&(progress<=table['progress'][-1])
    np.testing.assert_array_equal(raw['nominal_profile_covered'][active],covered)
    if case['feedback_kind']==1:
        ey0=np.interp(progress,table['progress'],table['path_error'])
        eh0=np.interp(progress,table['progress'],table['heading_error'])
        gain=np.where(covered,case['feedback_gain_rad'],0.)
    else:
        ey0=eh0=np.zeros(len(progress))
        gain=np.where(raw['relay_active'][active],case['feedback_gain_rad'],0.)
    np.testing.assert_allclose(raw['nominal_path_error_m'][active],ey0,atol=2e-6)
    np.testing.assert_allclose(raw['nominal_heading_error_rad'][active],eh0,atol=2e-6)
    np.testing.assert_allclose(raw['feedback_effective_gain_rad'][active],gain,atol=1e-7)
    ey=np.asarray(raw['feedback_path_error_m'])[active]-ey0
    eh=np.asarray(raw['feedback_heading_error_rad'])[active]-eh0
    eh=np.arctan2(np.sin(eh),np.cos(eh))
    requested=gain*np.tanh(eh/spec['heading_scale_rad']-ey/spec['path_scale_m'])
    np.testing.assert_allclose(raw['feedback_requested_steer_rad'][active],np.where(legal[active],requested,0.),atol=2e-6)
    original=np.asarray(raw['original_actor_latent'])[legal]
    request=requested[legal[active]]
    scale=config['residual_interface']['steer_scale_rad']
    target=np.clip(np.tanh(original[:,1])+request/scale,-1+1e-6,1-1e-6)
    expected=np.where(request==0,np.tanh(original[:,1]),target)
    np.testing.assert_allclose(np.tanh(raw['issued_latent'][legal,1]),expected,atol=3e-6)
    np.testing.assert_array_equal(raw['feedback_actor_latent'][legal,0],original[:,0])
