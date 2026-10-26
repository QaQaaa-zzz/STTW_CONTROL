import copy
import numpy as np
import pytest
import jax
import jax.numpy as j
from sttw_control.zero_middle_feedback import source_latent, SourceActor, zero_cases, verify_zero_middle
from sttw_control.path_command_policy import map_latent


def test_source_selection_jit_vmap_and_zero_physical_reference_offsets():
    native=j.array([[2.,-3.],[-4.,5.]])
    selected=jax.jit(jax.vmap(source_latent))(native,j.array([True,False]))
    np.testing.assert_array_equal(selected,j.array([[0.,0.],[-4.,5.]]))
    cfg={'residual_interface':dict(speed_positive_scale_m_s=.5,speed_negative_scale_m_s=.3,steer_scale_rad=.2)}
    np.testing.assert_array_equal(map_latent(selected[0],cfg),j.zeros(2))
    class Native:
        def apply(self, params, observation):return params+observation[:2]
    source=SourceActor(Native());obs=j.array([2.,3.]);params=j.array([.1,.2])
    np.testing.assert_array_equal(source.apply(dict(native=params,zero=False),obs),params+obs)
    np.testing.assert_array_equal(source.apply(dict(native=params,zero=True),obs),j.zeros(2))


def test_zero_cases_preserve_each_full_event_and_permissions_without_mutating_source():
    originals=[dict(case_id=f'context/a0/F{i}/free_GAE',role='free_GAE',actor_role='free_GAE',
                    mode_index=1,causal_support=True,feedback_gain_rad=0.,alpha=0.,force_N=i,
                    pulse_onset_task_tick=600,world_direction=[0,1,0]) for i in range(60)]
    before=copy.deepcopy(originals);result=zero_cases(originals)
    assert originals==before and len(result)==120
    changed={'case_id','role','actor_role','feedback_gain_rad','actor_source_zero'}
    for original,a,b in zip(originals,result[::2],result[1::2]):
        assert a['actor_source_zero'] and b['actor_source_zero']
        assert a['feedback_gain_rad']==0 and b['feedback_gain_rad']==.1
        assert {k:v for k,v in a.items() if k not in changed}=={k:v for k,v in original.items() if k not in changed}
    with pytest.raises(ValueError):zero_cases(originals[:-1])
    originals[0]['mode_index']=0
    with pytest.raises(ValueError):zero_cases(originals)


def test_zero_source_audit_separates_native_output_and_rejects_issued_speed_or_source_leak():
    from sttw_control.reference_task_feedback import steering_feedback
    from sttw_control.finite_value_evaluation_report import role_labels
    cfg={'residual_interface':dict(speed_positive_scale_m_s=.5,speed_negative_scale_m_s=.3,steer_scale_rad=.2)}
    spec=dict(path_scale_m=.5,heading_scale_rad=.3)
    case=dict(role='zero_middle_feedback',actor_source_zero=True,feedback_gain_rad=.1,causal_support=True)
    action,request,realized=steering_feedback(j.zeros(2),.3,.1,.1,cfg,spec)
    raw=dict(active_tick=np.ones(2,bool),actor_action_valid=np.ones(2,bool),
             actor_source_zero=np.ones(2,bool),original_actor_latent=np.ones((2,2)),source_actor_latent=np.zeros((2,2)),
             feedback_actor_latent=np.tile(action,(2,1)),issued_latent=np.tile(action,(2,1)),
             feedback_path_error_m=np.full(2,.3),feedback_heading_error_rad=np.full(2,.1),
             feedback_requested_steer_rad=np.full(2,request))
    verify_zero_middle(raw,case,cfg,spec)
    assert 'zero neural middle' in role_labels([case])[case['role']]
    wrong=copy.deepcopy(raw);wrong['source_actor_latent'][0,0]=.1
    with pytest.raises(AssertionError):verify_zero_middle(wrong,case,cfg,spec)
    wrong=copy.deepcopy(raw);wrong['feedback_actor_latent'][0,0]=.1
    with pytest.raises(AssertionError):verify_zero_middle(wrong,case,cfg,spec)


def test_factorial_reports_include_learning_increment_under_same_feedback():
    from sttw_control.finite_value_evaluation import paired_report
    from sttw_control.finite_value_evaluation_report import goal_report
    roles=['zero_middle','free_GAE','zero_middle_feedback','feedback_010'];rows=[]
    for force in [0.,16.]:
        for role in roles:
            success=role=='feedback_010'
            rows.append(dict(case_id=f'{force}/{role}',role=role,category='safe_task' if success else 'safe_task_failed',
                goal_arrived=success,physical_outcome='survived',context='left',alpha=0.,event_id=str(force),force_N=force,
                scored_cost=1.,task_path_rmse_m=1.,task_speed_rmse_m_s=1.,task_heading_rmse_rad=1.))
    paired=paired_report(rows,roles=roles)
    assert paired['comparisons']['feedback_010_vs_zero_middle_feedback']['task_gains']==2
    goal=goal_report(rows,roles)
    assert goal['disturbed']['paired']['feedback_010_vs_zero_middle_feedback']['arrival_gains']==1
    assert goal['disturbed']['paired']['feedback_010_vs_free_GAE']['arrival_gains']==1


def test_parent_and_trained_aliases_keep_their_actual_native_metadata_contracts():
    from sttw_control.zero_middle_feedback import verify_alias_identity
    parent=dict(actor='p',actor_sha256='a',checkpoint='c',checkpoint_sha256='b')
    verify_alias_identity(dict(parent),parent)
    trained=dict(parent,fingerprint='f',value_contract={'mode':'free'})
    verify_alias_identity(copy.deepcopy(trained),trained)
    with pytest.raises(ValueError):verify_alias_identity(dict(parent,fingerprint='made_up'),parent)
    with pytest.raises(ValueError):verify_alias_identity(dict(trained,fingerprint='changed'),trained)
