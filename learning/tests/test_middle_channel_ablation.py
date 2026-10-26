import copy
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as j
import pytest

from sttw_control.middle_channel_ablation import (select_channels,ChannelActor,channel_cases,
    make_channel_step,verify_channel_log)


def test_jit_mask_endpoints_match_native_and_zero_and_preserve_selected_channel():
    native=j.array([.61,-.73])
    masks=j.array([[True,True],[False,False],[True,False],[False,True]])
    selected=jax.jit(jax.vmap(lambda mask:select_channels(native,mask)))(masks)
    np.testing.assert_array_equal(selected,j.array([[.61,-.73],[0,0],[.61,0],[0,-.73]]))
    class Native:
        def apply(self,params,obs):return params+obs[:2]
    source=ChannelActor(Native())
    np.testing.assert_array_equal(source.apply(dict(native=native,keep_channels=masks[2]),j.zeros(2)),selected[2])


def test_case_builder_only_changes_neural_channels_not_events_alpha_or_permission():
    originals=[dict(case_id=f'context/a0/F{i}/feedback_010',role='feedback_010',actor_role='free_GAE',
        alpha=0.,mode_index=1,profile=0,causal_support=True,feedback_gain_rad=.1,force_N=i,
        pulse_onset_task_tick=600,world_direction=[0,1,0]) for i in range(60)]
    before=copy.deepcopy(originals);rows=channel_cases(originals)
    assert originals==before and len(rows)==120
    changed={'case_id','role','source_keep_channels'}
    for original,a,b in zip(originals,rows[::2],rows[1::2]):
        assert a['source_keep_channels']==[True,False] and b['source_keep_channels']==[False,True]
        for row in [a,b]:
            assert {k:v for k,v in row.items() if k not in changed}=={k:v for k,v in original.items() if k not in changed}
    with pytest.raises(ValueError):channel_cases(originals[:-1])
    originals[0]['feedback_gain_rad']=.05
    with pytest.raises(ValueError):channel_cases(originals)


def test_wrapper_advances_once_uses_masked_source_and_keeps_native_log_separate(monkeypatch):
    import sttw_control.reference_task_feedback as feedback
    import sttw_control.reference_reentry as reentry
    calls=[]
    class Native:
        def apply(self,params,obs):return params+obs[:2]
    def factory(env,actor,**kwargs):
        def advance(carry,parameters,path,speed,pulse,mode):
            calls.append(carry)
            source=actor.apply(parameters['actor'],j.ones(2))
            return carry,dict(actor_action_valid=j.ones(4,bool),original_actor_latent=j.tile(source,(4,1)))
        return advance
    monkeypatch.setattr(feedback,'make_feedback_evaluation_step',factory)
    monkeypatch.setattr(reentry,'actor_permission',lambda *args:j.bool_(True))
    env=SimpleNamespace(cc=SimpleNamespace(dt=.005),observation=lambda *args:(j.ones(2),None,None))
    state=SimpleNamespace(physical=SimpleNamespace(failed=False),fault=False,tick=0,domain_exit=False,
                          follower=SimpleNamespace(invalid=False))
    carry=SimpleNamespace(state=state,clean_blocks=16)
    params=dict(actor=dict(native=j.array([.2,.3]),keep_channels=j.array([True,False])))
    end,logs=make_channel_step(env,Native(),horizon_s=10,spec={})(carry,params,None,1,None,1)
    assert end is carry and calls==[carry]
    np.testing.assert_array_equal(logs['original_actor_latent'],j.tile(j.array([1.2,1.3]),(4,1)))
    np.testing.assert_array_equal(logs['source_actor_latent'],j.tile(j.array([1.2,0]),(4,1)))


def test_channel_log_audit_and_reports_retain_both_vs_each_partial_contrasts():
    from sttw_control.reference_task_feedback import steering_feedback
    from sttw_control.finite_value_evaluation import paired_report
    from sttw_control.finite_value_evaluation_report import goal_report,role_labels
    cfg={'residual_interface':dict(speed_positive_scale_m_s=.5,speed_negative_scale_m_s=.3,steer_scale_rad=.2)}
    spec=dict(path_scale_m=.5,heading_scale_rad=.3);native=j.array([.4,-.7]);source=j.array([.4,0.])
    action,request,_=steering_feedback(source,.3,.1,.1,cfg,spec)
    case=dict(role='speed_only_feedback',source_keep_channels=[True,False],feedback_gain_rad=.1,causal_support=True)
    raw=dict(active_tick=np.ones(2,bool),actor_action_valid=np.ones(2,bool),source_keep_channels=np.tile([True,False],(2,1)),
        original_actor_latent=np.tile(native,(2,1)),source_actor_latent=np.tile(source,(2,1)),
        feedback_actor_latent=np.tile(action,(2,1)),issued_latent=np.tile(action,(2,1)),
        feedback_path_error_m=np.full(2,.3),feedback_heading_error_rad=np.full(2,.1),
        feedback_requested_steer_rad=np.full(2,request))
    verify_channel_log(raw,case,cfg,spec)
    bad=copy.deepcopy(raw);bad['source_actor_latent'][0,1]=-.7
    with pytest.raises(AssertionError):verify_channel_log(bad,case,cfg,spec)
    assert 'speed only' in role_labels([case])[case['role']]
    roles=['zero_middle_feedback','speed_only_feedback','steer_only_feedback','feedback_010'];rows=[]
    for force in [0.,16.]:
        for role in roles:
            success=role=='feedback_010'
            rows.append(dict(case_id=f'{force}/{role}',role=role,category='safe_task' if success else 'safe_task_failed',
                goal_arrived=success,physical_outcome='survived',context='left',alpha=0.,event_id=str(force),force_N=force,
                scored_cost=1.,task_path_rmse_m=1.,task_speed_rmse_m_s=1.,task_heading_rmse_rad=1.))
    pair=paired_report(rows,roles=roles);goal=goal_report(rows,roles)
    for base in ['speed_only_feedback','steer_only_feedback','zero_middle_feedback']:
        assert pair['comparisons']['feedback_010_vs_'+base]['task_gains']==2
        assert goal['disturbed']['paired']['feedback_010_vs_'+base]['arrival_gains']==1
