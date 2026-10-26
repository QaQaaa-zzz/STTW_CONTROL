"""Small causal steering-reference feedback, a simple control comparison."""
import copy
import numpy as np
import jax.numpy as j

from .path_command_policy import map_latent
from .reference_prefix import inverse_offset_target

PROBE_ROLES = ('feedback_005', 'feedback_010')


def validate_spec(spec):
    if spec['gains_rad'] != dict(feedback_005=.05, feedback_010=.10):
        raise ValueError('the two predeclared gains must both be retained')
    if spec['path_scale_m'] != .5 or spec['heading_scale_rad'] != .3:
        raise ValueError('predeclared feedback scales changed')


def feedback_cases(source_cases, spec):
    validate_spec(spec)
    events = [c for c in source_cases if c['role']=='parent_plain']
    if len(events)!=30 or len({c['case_id'] for c in events})!=30:
        raise ValueError('thirty unique inherited events required')
    cases=[]
    for event in events:
        for role,gain in spec['gains_rad'].items():
            c=copy.deepcopy(event)
            c.update(role=role,actor_role='free_GAE',mode_index=1,profile=0,
                     causal_support=True,feedback_gain_rad=gain,
                     case_id=event['case_id'].rsplit('/',1)[0]+'/'+role)
            cases.append(c)
    return cases


def context_geometry(observation, config):
    """Decode named fields of the EXISTING clipped/scaled Actor observation."""
    fields=config['network']['context_fields']
    names=[f['name'] for f in fields]
    if len(set(names))!=len(names):
        raise ValueError('unique context fields required')
    def value(name):
        idx=names.index(name)
        return observation[observation.shape[-1]-len(fields)+idx]*fields[idx]['scale']
    ey=value('signed_cross_track_error_m')
    eh=j.arctan2(value('sin_path_heading_error'),value('cos_path_heading_error'))
    return ey,eh


def steering_feedback(original, ey, eh, gain, config, spec):
    base=map_latent(original,config)
    requested=gain*j.tanh(eh/spec['heading_scale_rad']-ey/spec['path_scale_m'])
    target=base.at[1].add(requested)
    proposed,realized,_=inverse_offset_target(target,config)
    # Never round-trip the speed channel. Zero intervention is exact even near
    # tanh saturation; only an actual steering increment uses the inverse.
    action=original.at[1].set(j.where(requested==0,original[1],proposed[1]))
    return action,requested,j.where(requested==0,0.,realized[1]-base[1])


class FeedbackActor:
    def __init__(self, actor, config, spec):
        validate_spec(spec)
        self.actor,self.config,self.spec=actor,config,spec

    def apply(self, parameters, observation):
        original=self.actor.apply(parameters['actor'],observation)
        ey,eh=context_geometry(observation,self.config)
        return steering_feedback(original,ey,eh,parameters['feedback_gain_rad'],self.config,self.spec)[0]


def make_feedback_evaluation_step(env, actor, *, horizon_s, spec):
    """Wrap the unchanged execution kernel, with separately audited source output."""
    from .finite_value_evaluation import make_evaluation_step
    from .reference_reentry import actor_permission
    advance=make_evaluation_step(env,FeedbackActor(actor,env.config,spec),horizon_s=horizon_s)
    def step(carry,parameters,path,speed,pulse,mode):
        s=carry.state
        stopped=s.physical.failed|s.fault|(s.tick>=round(horizon_s/env.cc.dt))
        enabled=actor_permission(s.domain_exit,s.follower.invalid,carry.clean_blocks,j.bool_(False),stopped)
        obs,_,_=env.observation(s,path,speed,horizon_s)
        safe=j.where(enabled,obs,j.zeros_like(obs))
        original=actor.apply(parameters['actor'],safe)
        ey,eh=context_geometry(safe,env.config)
        proposed,requested,realized=steering_feedback(original,ey,eh,parameters['feedback_gain_rad'],env.config,spec)
        end,logs=advance(carry,parameters,path,speed,pulse,mode)
        active=logs['active_tick'];legal=logs['actor_action_valid']
        # Preserve the meaning of original_actor_latent: the frozen NN itself,
        # before this comparison's feedback and before the existing speed rule.
        logs.update(original_actor_latent=j.where(legal[:,None],j.tile(original,(4,1)),j.nan),
                    feedback_actor_latent=j.where(legal[:,None],j.tile(proposed,(4,1)),j.nan),
                    feedback_path_error_m=j.where(active,ey,j.nan),
                    feedback_heading_error_rad=j.where(active,eh,j.nan),
                    feedback_requested_steer_rad=j.where(legal,requested,0.),
                    feedback_realized_steer_rad=j.where(legal,realized,0.))
        return end,logs
    return step


def verify_feedback_log(raw, case, config, spec):
    legal=np.asarray(raw['actor_action_valid'],bool)
    disabled=np.asarray(raw['active_tick'],bool)&~legal
    np.testing.assert_array_equal(raw['issued_latent'][disabled],np.zeros((disabled.sum(),2)))
    if not np.any(legal):
        return
    original=np.asarray(raw['original_actor_latent'])[legal]
    # Batched NumPy reconstruction is separate from the JAX runtime formula.
    ey=np.asarray(raw['feedback_path_error_m'])[legal]
    eh=np.asarray(raw['feedback_heading_error_rad'])[legal]
    request=case['feedback_gain_rad']*np.tanh(eh/spec['heading_scale_rad']-ey/spec['path_scale_m'])
    scale=config['residual_interface']['steer_scale_rad']
    normalized=np.clip((scale*np.tanh(original[:,1])+request)/scale,-1+1e-6,1-1e-6)
    expected=np.where(request==0,original[:,1],np.arctanh(normalized))
    np.testing.assert_allclose(raw['feedback_requested_steer_rad'][legal],request,atol=2e-6)
    # Compare physical targets near inverse-tanh saturation, where tiny float
    # differences amplify in latent units but are immaterial to the interface.
    np.testing.assert_allclose(np.tanh(raw['issued_latent'][legal,1]),np.tanh(expected),atol=3e-6)
    np.testing.assert_array_equal(raw['feedback_actor_latent'][legal,0],original[:,0])
