from types import SimpleNamespace
import copy
import jax
import jax.numpy as j
import numpy as np
import pytest

from sttw_control.preference_speed_authority import (
    PreferenceSpeedActor, scale_physical_speed, make_authority_step, verify_authority_log)
from sttw_control.path_command_policy import map_latent


CFG = dict(network=dict(context_fields=[dict(name='alpha',scale=1.)]),
           residual_interface=dict(speed_positive_scale_m_s=.7,speed_negative_scale_m_s=.3,steer_scale_rad=.2))


def test_actual_physical_scaling_with_asymmetric_scales_and_saturation():
    native=j.array([[x,.63] for x in [-25.,-2.,-.1,0.,.1,2.,25.]])
    for alpha in [0.,1e-9,.25,.5,.75,1.]:
        scaled=jax.jit(jax.vmap(lambda z:scale_physical_speed(z,j.float32(alpha))))(native)
        original=jax.vmap(lambda z:map_latent(z,CFG))(native)
        mapped=jax.vmap(lambda z:map_latent(z,CFG))(scaled)
        np.testing.assert_allclose(mapped[:,0],(1-alpha)*original[:,0],atol=2e-7)
        np.testing.assert_array_equal(scaled[:,1],native[:,1])
        assert np.isfinite(scaled).all()
        if alpha==0:np.testing.assert_array_equal(scaled,native)
        if alpha==1:np.testing.assert_array_equal(scaled[:,0],np.zeros(len(native)))
    assert not np.isclose(np.tanh(2*.5),.5*np.tanh(2))


def test_original_alpha_reaches_native_and_invalid_inputs_are_not_hidden():
    seen=[]
    class Native:
        def apply(self,params,obs):
            seen.append(obs);return params
    obs=j.array([.4,.5]);p=j.array([.7,-.8])
    wrapped=PreferenceSpeedActor(Native(),CFG)
    np.testing.assert_array_equal(wrapped.apply(p,obs),scale_physical_speed(p,.5))
    assert seen[0] is obs
    for alpha in [-.1,1.1,float('nan')]:
        assert np.isnan(scale_physical_speed(p,j.float32(alpha))).all()
    assert np.isnan(scale_physical_speed(j.array([j.inf,0.]),j.float32(1.))).all()
    for scale in [0.,-1.,float('nan')]:
        bad=copy.deepcopy(CFG);bad['network']['context_fields'][0]['scale']=scale
        with pytest.raises(ValueError):PreferenceSpeedActor(Native(),bad)


def test_single_physical_advance_and_native_source_logs(monkeypatch):
    import sttw_control.reference_task_feedback as feedback
    import sttw_control.reference_reentry as reentry
    calls=[]
    class Native:
        def apply(self,params,obs):return params
    def factory(env,actor,**kwargs):
        def advance(carry,parameters,*args):
            calls.append(carry)
            out=actor.apply(parameters['actor'],j.array([.5]))
            return carry,dict(actor_action_valid=j.ones(4,bool),original_actor_latent=j.tile(out,(4,1)))
        return advance
    monkeypatch.setattr(feedback,'make_feedback_evaluation_step',factory)
    monkeypatch.setattr(reentry,'actor_permission',lambda *args:j.bool_(True))
    state=SimpleNamespace(physical=SimpleNamespace(failed=False),fault=False,tick=0,
                          domain_exit=False,follower=SimpleNamespace(invalid=False))
    carry=SimpleNamespace(state=state,clean_blocks=16)
    env=SimpleNamespace(config=CFG,cc=SimpleNamespace(dt=.005),
                        observation=lambda *args:(j.array([.5]),None,None))
    native=j.array([.8,-.9]);end,logs=make_authority_step(env,Native(),horizon_s=10,spec={})(
        carry,dict(actor=native),None,1,None,1)
    assert end is carry and calls==[carry]
    np.testing.assert_array_equal(logs['original_actor_latent'],j.tile(native,(4,1)))
    np.testing.assert_array_equal(logs['source_actor_latent'],j.tile(scale_physical_speed(native,.5),(4,1)))
    np.testing.assert_array_equal(logs['speed_authority_user_alpha'],np.full(4,.5))


def test_independent_audit_rejects_latent_instead_of_physical_scaling():
    from sttw_control.reference_task_feedback import steering_feedback
    native=j.array([2.,-.7]);source=scale_physical_speed(native,.5)
    spec=dict(path_scale_m=.5,heading_scale_rad=.3)
    action,request,_=steering_feedback(source,.3,.1,.1,CFG,spec)
    raw=dict(active_tick=np.ones(2,bool),actor_action_valid=np.ones(2,bool),
        original_actor_latent=np.tile(native,(2,1)),source_actor_latent=np.tile(source,(2,1)),
        speed_authority_user_alpha=np.full(2,.5),feedback_actor_latent=np.tile(action,(2,1)),
        issued_latent=np.tile(action,(2,1)),feedback_path_error_m=np.full(2,.3),
        feedback_heading_error_rad=np.full(2,.1),feedback_requested_steer_rad=np.full(2,request))
    case=dict(alpha=.5,feedback_gain_rad=.1)
    verify_authority_log(raw,case,CFG,spec)
    bad=copy.deepcopy(raw);bad['source_actor_latent'][:,0]=1.
    with pytest.raises(AssertionError):verify_authority_log(bad,case,CFG,spec)


def test_new_role_retains_both_strong_controls_in_goal_and_cost_reports():
    from sttw_control.finite_value_evaluation import paired_report
    from sttw_control.finite_value_evaluation_report import goal_report,role_labels
    roles=['feedback_010','steer_only_feedback','preference_speed_authority'];rows=[]
    for force in [0.,16.]:
        for role in roles:
            ok=role=='preference_speed_authority'
            rows.append(dict(case_id=f'{force}/{role}',role=role,alpha=.5,context='left',
                event_id=str(force),force_N=force,category='safe_task' if ok else 'safe_task_failed',
                physical_outcome='survived',goal_arrived=ok,scored_cost=1.,
                task_path_rmse_m=1.,task_speed_rmse_m_s=1.,task_heading_rmse_rad=1.,feedback_gain_rad=.1))
    pair=paired_report(rows,roles=roles);goal=goal_report(rows,roles)
    for base in roles[:2]:
        key='preference_speed_authority_vs_'+base
        assert pair['comparisons'][key]['task_gains']==2
        assert goal['disturbed']['paired'][key]['arrival_gains']==1
    assert 'x(1-alpha)' in role_labels(rows)['preference_speed_authority']


def test_midpoint_builder_preserves_original_task_event_and_permissions():
    from sttw_control.preference_speed_authority import midpoint_cases,AUTHORITY_ROLE
    originals=[dict(case_id=f'context/a.5/F{i}/feedback_010',role='feedback_010',alpha=.5,
        feedback_gain_rad=.1,mode_index=1,causal_support=True,force_N=i,bank_index=2,
        pulse_onset_task_tick=600,world_direction=[0,1,0]) for i in range(20)]
    before=copy.deepcopy(originals);cases=midpoint_cases(originals)
    assert originals==before
    for a,b in zip(originals,cases):
        assert b['role']==AUTHORITY_ROLE and b['preference_speed_authority'] is True
        for k in a:
            if k not in ['role','case_id']:assert a[k]==b[k]
    with pytest.raises(ValueError):midpoint_cases(originals[:-1])


def test_disabled_actions_are_still_zero_and_not_audited_as_real_alpha():
    raw=dict(active_tick=np.ones(2,bool),actor_action_valid=np.zeros(2,bool),
             issued_latent=np.zeros((2,2)),source_actor_latent=np.full((2,2),np.nan))
    verify_authority_log(raw,dict(alpha=.5),CFG,{})
    raw['issued_latent'][0,0]=.1
    with pytest.raises(AssertionError):verify_authority_log(raw,dict(alpha=.5),CFG,{})
