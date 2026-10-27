import jax.numpy as j
import numpy as np
from sttw_control.reference_reentry import actor_permission,next_clean_blocks,reentry_cases


def test_history_gate_waits_full16_blocks_and_never_enables_invalid_or_terminal():
    assert actor_permission(False,False,0,False,False)
    assert not actor_permission(True,False,15,False,False)
    assert actor_permission(True,False,16,False,False)
    assert not actor_permission(True,True,16,False,False)
    assert not actor_permission(True,False,16,True,False)
    assert not actor_permission(True,False,16,False,True)
    n=j.int32(0)
    for _ in range(16):n=next_clean_blocks(n,j.ones(4,bool),j.zeros(4,bool),j.zeros(4,bool))
    assert int(n)==16
    assert int(next_clean_blocks(n,j.ones(4,bool),j.array([0,1,0,0],bool),j.zeros(4,bool)))==0
    assert int(next_clean_blocks(n,j.array([1,1,1,0],bool),j.zeros(4,bool),j.zeros(4,bool)))==0
    assert int(next_clean_blocks(n,j.ones(4,bool),j.zeros(4,bool),j.array([0,0,0,1],bool)))==0


def test_reentry_cases_preserve_all_events_users_and_same_actor():
    original=[]
    for role in ['task_only','physical_credit','floor']:
        for alpha in [.5,.75]:
            for event in ['zero','t380_F-32','t380_F+32','t580_F-32','t580_F+32']:
                original.append(dict(case_id=f'{role}/{alpha}/{event}',role=role,actor_role=role,alpha=alpha,state_alpha=alpha,event_id=event,stage='timing'))
    new=reentry_cases(original)
    assert len(new)==20 and all(r['stage']=='reentry' for r in new)
    assert all(r['alpha']==r['state_alpha'] for r in new)
    assert len({r['case_id'] for r in new})==20
    assert {r['actor_role'] for r in new}=={'task_only','physical_credit'}
    assert original[0]['stage']=='timing' and new[0]['source_case_id']==original[0]['case_id']


def test_real_step_handback_uses_current_history_and_keeps_cumulative_exit(monkeypatch):
    from flax import struct
    from sttw_control.anticipatory_reference import make_reference_step
    from sttw_control.reference_reentry import ReentryEvaluationState
    from sttw_control.causal_reference_baseline import initial_gate
    from sttw_control.path_command_reward import BASE_COST_KEYS
    from types import SimpleNamespace
    import sttw_control.live_path_continuation as live
    @struct.dataclass
    class Physical:
        failed: object
        raw: object
    @struct.dataclass
    class State:
        physical: object
        fault: object
        tick: object
        domain_exit: object
        follower: object
        offsets: object
        projection: object
        history_value: object
    @struct.dataclass
    class Follower:
        invalid: object
    @struct.dataclass
    class Projection:
        progress: object
    config=dict(path_reward={},path=dict(straight_prefix_m=2.5),residual_interface=dict(speed_negative_scale_m_s=1.,speed_positive_scale_m_s=.25,steer_scale_rad=.2,speed_reference_bounds_m_s=[1.,3.5]))
    class Env:
        cc=SimpleNamespace(dt=.005)
        def __init__(self):self.config=config
        def observation(self,s,path,speed,horizon):return j.zeros(351).at[0].set(s.history_value),None,j.bool_(False)
    def factory(env,*,horizon_s):
        def advance(s,action,path,speed,pulse):
            names=['active_tick','actual_xy','actual_forward_speed','wheel_speed_proxy','phi','phi_dot','actual_delta','peak_roll','contact_failure','physical_failure','nonfinite','lower_fault','policy_fault','time','path_cross_track','path_heading_error','path_progress','goal_section_signed_distance','applied_force_impulse','tick_reward','scored_tick_reward','physical_tick','chi','yaw_unwrapped','governed','nominal','filtered_offset','offsets','target_offset']
            log={k:j.zeros(4) for k in names};log.update(active_tick=j.ones(4,bool),domain_exit=j.repeat(s.follower.invalid,4),raw_components={k:j.zeros(4) for k in BASE_COST_KEYS},effective_components={k:j.zeros(4) for k in BASE_COST_KEYS})
            for k in ['nonfinite','lower_fault','policy_fault','contact_failure']:log[k]=j.zeros(4,bool)
            return s.replace(history_value=s.history_value+.1,tick=s.tick+4),log
        return advance
    monkeypatch.setattr(live,'make_live_reference_step',factory)
    actor=SimpleNamespace(apply=lambda p,obs:j.stack([obs[0],j.float32(0)]))
    env=Env();path=SimpleNamespace(turn_end=j.float32(5.))
    state=State(Physical(j.bool_(False),j.array([1.5,0.])),j.bool_(False),j.int32(800),j.bool_(True),Follower(j.bool_(False)),j.zeros(2),Projection(j.float32(3.)),j.float32(.1))
    carry=ReentryEvaluationState(state,initial_gate(),j.int32(16));step=make_reference_step(env,actor,horizon_s=10.,allow_reentry=True)
    end,logs=step(carry,None,path,j.float32(1.5),j.zeros(5),j.int32(0),j.float32(1.),j.bool_(False))
    np.testing.assert_allclose(logs['issued_latent'][:,0],.1);assert logs['reentry_actor_active'].all() and end.state.domain_exit
    end2,logs2=step(end,None,path,j.float32(1.5),j.zeros(5),j.int32(0),j.float32(1.),j.bool_(False))
    np.testing.assert_allclose(logs2['issued_latent'][:,0],.2);assert end2.state.domain_exit
    bad=carry.replace(state=state.replace(follower=Follower(j.bool_(True))))
    end,logs=step(bad,None,path,j.float32(1.5),j.zeros(5),j.int32(0),j.float32(1.),j.bool_(False))
    assert not logs['actor_action_valid'].any() and int(end.clean_blocks)==0 and end.state.domain_exit

    # Keep the network disabled and preserve its history/reentry contract,
    # while allowing only the original relay's speed-support candidate.
    low=bad.replace(state=bad.state.replace(physical=bad.state.physical.replace(raw=j.array([1.1,0.]))))
    end,supported=step(low,{'support_independent_permission':j.bool_(True)},path,
        j.float32(1.5),j.zeros(5),j.int32(7),j.float32(1.),j.bool_(False))
    assert supported['support_permission_override'].all()
    assert not supported['actor_action_valid'].any()
    assert np.isnan(supported['original_actor_latent']).all()
    np.testing.assert_allclose(np.tanh(supported['issued_latent'][:,0])*.25,.25,atol=1e-6)
    np.testing.assert_array_equal(supported['issued_latent'][:,1],0.)
    assert end.state.domain_exit and int(end.clean_blocks)==0
    np.testing.assert_allclose(end.state.history_value,low.state.history_value+.1)
    import jax
    for before,after in zip(jax.tree.leaves(low.gate),jax.tree.leaves(end.gate)):
        np.testing.assert_array_equal(after,before)
    _,legacy=step(low,None,path,j.float32(1.5),j.zeros(5),j.int32(7),j.float32(1.),j.bool_(False))
    assert not legacy['support_permission_override'].any()
    np.testing.assert_array_equal(legacy['issued_latent'],np.zeros((4,2)))


def test_explicit_roles_give_every_controller_same_permission_without_changing_rules():
    import pytest
    roles={'parent':'parent_reentry','task_only':'task_reentry','physical_credit':'physical_reentry','floor':'floor_reentry','causal_eso_allalpha':'eso_reentry'}
    original=[dict(case_id=f'{role}/{alpha}/{event}',role=role,actor_role=role if role in ['task_only','physical_credit'] else 'parent',alpha=alpha,state_alpha=alpha,event_id=event,stage='timing',profile=5 if role=='causal_eso_allalpha' else 0,reference_speed_floor_m_s=1.5 if role=='floor' else 1.)
              for role in roles for alpha in [.5,.75] for event in ['zero','t380_F-32','t380_F+32','t580_F-32','t580_F+32']]
    allcases=reentry_cases(original,roles=roles);assert len(allcases)==50
    right=reentry_cases(original,roles={k:roles[k] for k in ['parent','floor','causal_eso_allalpha']});assert len(right)==30
    for c in allcases:
        old=next(x for x in original if x['case_id']==c['source_case_id'])
        for key in ['alpha','state_alpha','event_id','actor_role','profile','reference_speed_floor_m_s']:assert c[key]==old[key]
    with pytest.raises(ValueError):reentry_cases(original,roles={'zero_middle':'z'})
    with pytest.raises(ValueError):reentry_cases(original[:-1],roles=roles)
    with pytest.raises(ValueError):reentry_cases(original+original[:1],roles=roles)


def test_learned_floor_cases_change_only_declared_floor_and_identity():
    import pytest
    original=[dict(case_id=f'{role}/{alpha}/{event}',role=role,actor_role=role,
        alpha=alpha,state_alpha=alpha,event_id=event,stage='timing',profile=0,
        reference_speed_floor_m_s=1.,initial_state_index=7,pulse_duration_s=.1)
        for role in ['task_only','physical_credit'] for alpha in [.5,.75]
        for event in ['zero','t380_F-32','t380_F+32','t580_F-32','t580_F+32']]
    roles={'task_only':'task_floor_reentry','physical_credit':'physical_floor_reentry'}
    cases=reentry_cases(original,roles=roles,reference_speed_floor_m_s=1.5)
    assert len(cases)==20
    for old,new in zip(original,cases):
        assert old['reference_speed_floor_m_s']==1.
        assert new['reference_speed_floor_m_s']==1.5
        for k in old:
            if k not in ['case_id','role','stage','reference_speed_floor_m_s']:assert new[k]==old[k]
        assert new['source_case_id']==old['case_id']
    assert all(c['reference_speed_floor_m_s']==1. for c in reentry_cases(original,roles=roles))
    for invalid in [float('nan'),float('inf'),.9]:
        with pytest.raises(ValueError):reentry_cases(original,roles=roles,reference_speed_floor_m_s=invalid)


def test_declared_support_profiles_clone_actor_event_matrix_without_floor_change():
    import pytest
    original=[dict(case_id=f'{role}/{a}/{event}',role=role,actor_role=role,alpha=a,state_alpha=a,event_id=event,stage='timing',profile=0,reference_speed_floor_m_s=1.)
        for role in ['parent','task_only'] for a in [.5,.75]
        for event in ['zero','t380_F-32','t380_F+32','t580_F-32','t580_F+32']]
    roles={'parent':'parent_support','task_only':'task_support'}
    cases=reentry_cases(original,roles=roles,reference_profiles=(6,7))
    assert len(cases)==40 and len({c['case_id'] for c in cases})==40
    for c in cases:
        old=next(x for x in original if x['case_id']==c['source_case_id'])
        assert c['reference_speed_floor_m_s']==1. and c['profile'] in [6,7]
        for k in ['actor_role','alpha','state_alpha','event_id']:assert c[k]==old[k]
    for profiles in [(6,6),(0,),()]:
        with pytest.raises(ValueError):reentry_cases(original,roles=roles,reference_profiles=profiles)
