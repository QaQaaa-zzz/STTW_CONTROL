"""Fixed endpoint inventory and original-alpha reward semantics, CPU only."""
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.goal_evaluation import evaluation_cases,rescore_original_components
from sttw_control.path_command_reward import costs,configured_cost_keys


def test_evaluation_has_fixed_unique_physics_and_explicit_alpha_aliases():
    cases=evaluation_cases()
    assert len(cases)==len({c['case_id'] for c in cases})==128
    assert sum(len(c['user_alphas']) for c in cases)==192
    assert sum(c['suite']=='transfer' for c in cases)==80
    for role in range(4):
        for suite in ['transfer','training_geometry']:
            for context in ['left','right']:
                zero=[c for c in cases if (c['role'],c['suite'],c['context'],c['force_N'])==(role,suite,context,0.)]
                assert len(zero)==(3 if role<2 else 1)
    assert all(c['controller_alpha']==0 and c['user_alphas']==[0.,.5,1.] for c in cases if c['role']>=2)


def test_original_alpha_rescore_matches_actual_v3_formula_and_retains_other_channels():
    # Real frozen configuration, synthetic physical metrics; no plant or rollout.
    cfg=json.loads(Path('runs/live_preference_collector_v1_attempt2/effective_config.json').read_text())
    import jax.numpy as j
    ev=np.array([.05,.8,-.3],np.float32);ey=np.array([.01,.4,2.],np.float32)
    eh=np.array([.02,-.2,.5],np.float32);chi=np.array([0.,.7,1.],np.float32)
    inputs=dict(chi=j.array(chi),ev=j.array(ev),ey=j.array(ey),eh=j.array(eh),peak_roll=j.array([.1,.3,.45]),
        roll_rate=j.array([.1,.5,2.]),speed=j.array(ev+1.5),offsets=j.zeros((2,)),offset_rate=j.zeros((2,)),config=cfg)
    # costs() is a single-tick function; vmap physical rows to preserve its exact contract.
    def formula(a):
        rows=[]
        for i in range(3):
            one={k:(v[i] if k in ['chi','ev','ey','eh','peak_roll','roll_rate','speed'] else v) for k,v in inputs.items()}
            r=costs(alpha=a,**one)
            rows.append([float(r['effective'][k]) for k in configured_cost_keys(cfg)])
        return np.array(rows)
    p=dict(active_tick=np.ones(3,bool),task_reward_valid=np.ones(3,bool),effective_costs=formula(.5),
        actual_forward_speed=ev+1.5,requested_speed=np.full(3,1.5),path_cross_track=ey,path_heading_error=eh,chi=chi)
    for a in (0.,.5,1.):
        out=rescore_original_components(p,user_alpha=a,config=cfg,recorded_alpha=.5)
        np.testing.assert_allclose(out,formula(a),rtol=3e-5,atol=2e-5)
        np.testing.assert_array_equal(out[:,3:],p['effective_costs'][:,3:])
    bad=dict(p,effective_costs=p['effective_costs'].copy());bad['effective_costs'][0,0]+=1
    with pytest.raises(ValueError,match='recorded alpha'):
        rescore_original_components(bad,user_alpha=1.,config=cfg,recorded_alpha=.5)


@pytest.mark.parametrize('role,expected_raw,profile,zero',[(0,[-.2,.3],7,False),(1,[-.2,.3],7,False),(2,[0.,0.],0,True),(3,[0.,0.],7,False)])
def test_deterministic_wrapper_forwards_distinct_zero_and_support_permissions(monkeypatch,role,expected_raw,profile,zero):
    from types import SimpleNamespace
    from flax import struct
    import jax.numpy as j
    from sttw_control.goal_evaluation import GoalEvaluationState,make_goal_evaluation_step
    @struct.dataclass
    class Physical:
        data: object
        failed: object
    @struct.dataclass
    class Follower:
        invalid: object
    @struct.dataclass
    class State:
        physical: object
        follower: object
        fault: object
        domain_exit: object
        tick: object
    observed={}
    def factory(env,actor,**contract):
        assert contract['allow_reentry'] and contract['horizon_s']==10.
        def advance(carry,params,path,speed,pulse,prof,lower,z):
            observed.update(raw=np.asarray(actor.apply(params,j.zeros(351))),profile=int(prof),zero=bool(z),override=params['support_independent_permission'])
            logs=dict(active_tick=j.ones(4,bool),actual_xy=j.tile(j.array([3.,2.]),(4,1)),
                contact_failure=j.zeros(4,bool),nonfinite=j.zeros(4,bool),lower_fault=j.zeros(4,bool),
                policy_fault=j.zeros(4,bool),horizon_reached=j.zeros(4,bool),actor_action_valid=j.full(4,not bool(z)))
            return carry,logs
        return advance
    monkeypatch.setattr('sttw_control.anticipatory_reference.make_reference_step',factory)
    env=SimpleNamespace(cc=SimpleNamespace(dt=.005),config=dict(residual_interface=dict(speed_reference_bounds_m_s=[.7,3.2])),
        physics=SimpleNamespace(observe=lambda data:(None,data,None)),
        observation=lambda *args:(j.ones(351),j.ones(352),j.bool_(False)))
    s=State(Physical(j.array([1.,2.,0.]),j.bool_(False)),Follower(j.bool_(False)),j.bool_(False),j.bool_(False),j.int32(0))
    carry=GoalEvaluationState(s,j.zeros(2),j.int32(0),j.bool_(False))
    callback=lambda params,*views:(j.array([-.2,.3]),j.array(.5),j.array(0.),j.array(.5))
    step=make_goal_evaluation_step(env,callback)
    end,logs=step(carry,{},None,j.array(1.5),j.zeros(5),j.int32(role),j.array([3.,2.]))
    np.testing.assert_allclose(observed['raw'],expected_raw)
    assert observed['profile']==profile and observed['zero']==zero and not observed['override']
    assert bool(end.arrived) and np.all(np.asarray(logs['neural_active'])==(role<2))


def synthetic_endpoint():
    cfg=json.loads(Path('runs/live_preference_collector_v1_attempt2/effective_config.json').read_text())
    n=12;channels=np.zeros((n,len(configured_cost_keys(cfg))));channels[:,-1]=.2
    p=dict(active_tick=np.ones(n,bool),actor_action_valid=np.ones(n,bool),physical_tick=np.arange(n),
        actual_xy=np.column_stack([np.arange(n)*.1,np.zeros(n)]),actual_forward_speed=np.ones(n)*1.5,
        requested_speed=np.ones(n)*1.5,path_cross_track=np.zeros(n),path_heading_error=np.zeros(n),chi=np.zeros(n),
        effective_costs=channels,task_reward_valid=np.ones(n,bool),tick_reward=np.full(n,-.0001),scored_tick_reward=np.full(n,-.0001),
        contact_failure=np.zeros(n,bool),nonfinite=np.zeros(n,bool),lower_fault=np.zeros(n,bool),policy_fault=np.zeros(n,bool),
        horizon_reached=np.arange(n)==n-1,applied_force_impulse=np.zeros((n,3)))
    case=dict(case_id='synthetic',suite='unit_fixture',context='left',role_name='zero',controller_alpha=0.,user_alphas=[0.,.5,1.],force_N=0.,pulse_onset_task_tick=0)
    return cfg,p,case


@pytest.mark.parametrize('mode',['exact','shifted','early_contact','zero_force'])
def test_pulse_timing_distinguishes_shift_from_same_integral_and_truncated_observation(mode):
    from sttw_control.goal_evaluation import pulse_timing_diagnostic
    dt=.005;n=40;onset=10;duration=.1
    p=dict(active_tick=np.ones(n,bool),physical_tick=np.arange(n),applied_force_impulse=np.zeros((n,3)))
    c=dict(force_N=32.,world_direction=[0.,1.,0.],pulse_onset_task_tick=onset,pulse_duration_s=duration)
    p['applied_force_impulse'][10:30,1]=32*dt
    if mode=='shifted':
        p['applied_force_impulse'][:]=0;p['applied_force_impulse'][11:31,1]=32*dt
        assert p['applied_force_impulse'].sum()==pytest.approx(32*duration)
    elif mode=='early_contact':
        p['active_tick'][15:]=False;p['applied_force_impulse'][15:]=0
    elif mode=='zero_force':
        c['force_N']=0.;p['applied_force_impulse'][:]=0.
    d=pulse_timing_diagnostic(p,c,dt=dt)
    assert d['pulse_timing_status']==('mismatch' if mode=='shifted' else 'matched_observed_ticks')
    assert d['pulse_observation']==('partial_window' if mode=='early_contact' else 'zero_force' if mode=='zero_force' else 'full_window')
    if mode=='early_contact':
        assert d['first_applied_task_tick']==10 and d['last_applied_task_tick']==14


def test_pulse_timing_preserves_unknown_and_unobserved_pre_pulse_tail():
    from sttw_control.goal_evaluation import pulse_timing_diagnostic
    p=dict(active_tick=np.ones(4,bool),physical_tick=np.arange(4),applied_force_impulse=np.zeros((4,3)))
    c=dict(force_N=-32.,world_direction=[1.,0.,0.],pulse_onset_task_tick=10,pulse_duration_s=.1)
    d=pulse_timing_diagnostic(p,c,dt=.005)
    assert d['pulse_observation']=='before_pulse' and d['pulse_timing_status']=='matched_observed_ticks'
    assert d['first_applied_task_tick'] is None
    p['applied_force_impulse'][0]=np.nan
    assert pulse_timing_diagnostic(p,c,dt=.005)['pulse_timing_status']=='unknown'


def test_pulse_boundary_float_spill_uses_full_tick_scale_not_zero_expected_relative_tolerance():
    from sttw_control.goal_evaluation import pulse_timing_diagnostic
    p=dict(active_tick=np.ones(40,bool),physical_tick=np.arange(40),applied_force_impulse=np.zeros((40,3)))
    c=dict(force_N=32.,world_direction=[0.,1.,0.],pulse_onset_task_tick=10,pulse_duration_s=.1)
    p['applied_force_impulse'][10:30,1]=.16
    p['applied_force_impulse'][30,1]=2e-5
    d=pulse_timing_diagnostic(p,c,dt=.005)
    assert d['pulse_timing_status']=='matched_observed_ticks' and d['last_applied_task_tick']==29
    p['applied_force_impulse'][30,1]=.16
    assert pulse_timing_diagnostic(p,c,dt=.005)['pulse_timing_status']=='mismatch'


def test_first_goal_is_retained_after_later_contact_and_legacy_terminal_block_is_not_running_cost():
    from sttw_control.goal_evaluation import score_endpoint_case
    cfg,p,case=synthetic_endpoint();p['contact_failure'][-1]=True;p['scored_tick_reward'][-4:]=[0,0,0,-1234.]
    views=score_endpoint_case(p,case,[0.,0.],config=cfg,horizon_s=.06,goal_radius_m=.01)
    for v in views:
        assert (v['goal'],v['survival'],v['goal_time_s'])==('arrived','contact',.005)
        assert v['pre_goal_cost']==pytest.approx(.0001)
        assert v['legacy_scored_return']==pytest.approx(-1234.0008)
        assert v['physical_ticks']==12


def test_same_tick_goal_contact_is_not_arrival_even_with_zero_measured_distance():
    from sttw_control.goal_evaluation import score_endpoint_case
    cfg,p,case=synthetic_endpoint()
    p['active_tick'][1:]=False;p['actor_action_valid'][1:]=False;p['horizon_reached'][:]=False
    p['contact_failure'][0]=True;p['scored_tick_reward'][:]=0.;p['scored_tick_reward'][0]=-1234.
    v=score_endpoint_case(p,case,[0.,0.],config=cfg,horizon_s=.06,goal_radius_m=.01)[0]
    assert (v['goal'],v['survival'])==('not_arrived','contact')
    assert v['goal_time_s'] is None and v['minimum_goal_distance_m']==0.
    assert v['pre_goal_cost']==0. and v['legacy_scored_return']==-1234.
    assert v['minimum_prefatal_goal_distance_m'] is None and v['speed_rmse_m_s'] is None


def test_prepare_rejects_training_config_drift_against_original_seal(tmp_path):
    from sttw_control.goal_endpoint_run import prepare
    from sttw_control.preference_best import atomic_json,digest
    training=tmp_path/'training';training.mkdir();out=tmp_path/'evaluation'
    atomic_json(training/'status.json',dict(phase='completed',collections_completed=64))
    atomic_json(training/'plan.json',{})
    atomic_json(training/'effective_config.json',dict(version='original'))
    atomic_json(training/'seal.json',dict(plan_sha256=digest(training/'plan.json'),config_sha256=digest(training/'effective_config.json')))
    atomic_json(training/'effective_config.json',dict(version='drifted'))
    with pytest.raises(ValueError,match='original training environment config changed'):prepare(training,out)
    assert not out.exists()


def test_analysis_binds_original_generated_goal_before_scoring(tmp_path):
    from sttw_control.goal_endpoint_analysis import analyze
    from sttw_control.preference_best import atomic_json,digest
    root=tmp_path/'evaluation';root.mkdir()
    atomic_json(root/'status.json',dict(phase='completed',completed_cases=128))
    atomic_json(root/'plan.json',{});atomic_json(root/'case_index.json',[])
    np.savez(root/'path.npz',s=[0.,1.],xy=[[0.,0.],[1.,0.]],goal=1.)
    atomic_json(root/'path_manifest.json',{'transfer/left':dict(file='path.npz',sha256=digest(root/'path.npz'),original_goal_xy=[9.,0.])})
    atomic_json(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'physics_receipt.json',dict(plan_sha256=digest(root/'plan.json'),case_index_sha256=digest(root/'case_index.json'),path_manifest_sha256=digest(root/'path_manifest.json')))
    with pytest.raises(ValueError,match='original path goal mismatch'):analyze(root)
    assert not (root/'analysis').exists()
