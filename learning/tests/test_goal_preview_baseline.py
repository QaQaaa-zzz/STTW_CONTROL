import copy
import json
from pathlib import Path
from types import SimpleNamespace
from flax import struct
import jax
import jax.numpy as j
import numpy as np
import pytest

from sttw_control.geometric_path import Path as GeometricPath
from sttw_control.path_reference import pursuit,publish
from sttw_control.goal_preview_baseline import GoalPreviewEnv,goal_follower,goal_cases,GOAL_ROLES

CFG=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Feedback_V1.json').read_text())


def path():
    s=j.linspace(0,20,101)
    return GeometricPath(s,j.stack([s,j.zeros_like(s)],axis=1),j.zeros_like(s),j.zeros_like(s),j.array(4.),j.array(10.))


def test_goal_target_uses_same_geometric_curvature_and_keeps_speed():
    p=path();pose=j.array([9.,1.,0.]);base=pursuit(p,9.2,pose,2.,1.3,CFG)
    base=base.replace(lookahead=j.array(1.5))
    result,used=jax.jit(lambda b:goal_follower(b,p,9.2,pose,CFG))(base)
    assert bool(used);np.testing.assert_allclose(result.preview_world,[10,0])
    np.testing.assert_allclose(result.preview_body,[1,-1]);assert float(result.kappa)==pytest.approx(-1.)
    assert float(result.command[0])==float(base.command[0]);assert float(result.command[1])<0
    assert float(result.lookahead)==float(base.lookahead)


@pytest.mark.parametrize('progress,pose,look',[(3.,[3.,1.,0.],1.5),(10.1,[9.,1.,0.],1.5),
    (9.8,[9.8,.1,0.],1.5),(9.2,[10.5,1.,0.],1.5),(8.,[9.,1.,0.],1.)])
def test_inactive_inside_passed_or_behind_retains_original_exactly(progress,pose,look):
    p=path();pose=j.array(pose);base=pursuit(p,progress,pose,2.,1.3,CFG).replace(lookahead=j.array(look))
    result,used=goal_follower(base,p,progress,pose,CFG)
    assert not bool(used)
    for a,b in zip(jax.tree.leaves(result),jax.tree.leaves(base)):np.testing.assert_array_equal(a,b)


@struct.dataclass
class Physical:
    raw:object
    data:object
    sentinel:object


@struct.dataclass
class State:
    physical:object
    follower:object
    projection:object
    tick:object
    nominal_rates:object
    fault:object
    domain_exit:object
    lower:object


def test_proxy_one_physical_tick_one_slew_and_preserves_real_fault_history():
    p=path();pose=j.array([9.,1.,0.]);f=pursuit(p,9.2,pose,2.,1.3,CFG)
    f=f.replace(lookahead=j.array(1.5),command=j.array([1.3,.2]))
    state=State(Physical(j.array([1.3,0.]),pose,j.array(0)),f,SimpleNamespace(progress=j.array(9.2)),
                j.int32(3),j.zeros(2),j.bool_(True),j.bool_(True),j.array([37.]))
    # Projection needs to be a JAX pytree for the conditional update.
    from sttw_control.geometric_path import Projection
    state=state.replace(projection=Projection(j.array(9.2),j.array(9.2),j.array([9.2,0.]),j.array(1.),j.array(0.)))
    calls=[]
    class Base:
        config=CFG
        physics=SimpleNamespace(observe=lambda data:(None,data,j.array(2.)))
        def tick(self,s,*args,**kwargs):
            calls.append(s)
            raw,rates=publish(s.physical.raw,s.follower.command,self.config)
            end=s.replace(tick=s.tick+1,physical=s.physical.replace(raw=raw,sentinel=s.physical.sentinel+1),nominal_rates=rates)
            return end,dict(domain_exit=j.bool_(False),tracking_domain_failure=j.bool_(False),nominal=s.physical.raw,
                            preview_world=s.follower.preview_world)
    env=GoalPreviewEnv(Base());end,logs=env.tick(state,j.zeros(2),p,1.3)
    assert len(calls)==1 and int(end.physical.sentinel)==1
    follower,used=goal_follower(f,p,9.2,pose,CFG);assert bool(used)
    expected,rates=publish(state.physical.raw,follower.command,CFG)
    np.testing.assert_array_equal(end.physical.raw,expected);np.testing.assert_array_equal(end.nominal_rates,rates)
    assert float(end.physical.raw[1])<0 # Double publishing would yield a different value.
    assert bool(end.fault) and bool(end.domain_exit)
    np.testing.assert_array_equal(end.lower,state.lower)
    np.testing.assert_array_equal(logs['nominal'],state.physical.raw)
    np.testing.assert_array_equal(logs['preview_world'],state.follower.preview_world)
    nonboundary=state.replace(tick=j.int32(0));out,_=env.tick(nonboundary,j.zeros(2),p,1.3)
    np.testing.assert_array_equal(out.follower.preview_world,f.preview_world)


def test_copied_proxy_delegates_methods_to_same_local_config():
    class Base:
        def __init__(self):self.config=dict(marker=1)
        def observation(self):return self.config['marker']
    source=GoalPreviewEnv(Base());local=copy.copy(source);local.config=dict(marker=2)
    assert local is not source and local.base is not source.base
    assert local.observation()==2 and source.observation()==1
    assert local.tick.__self__ is local


def test_three_sources_share_exact_task_events_and_no_new_alpha_rule():
    originals=[dict(role='feedback_010',case_id=f'c/a.5/F{i}/feedback_010',alpha=.5,
        feedback_gain_rad=.1,mode_index=1,causal_support=True,force_N=i) for i in range(60)]
    before=copy.deepcopy(originals);cases=goal_cases(originals)
    assert originals==before and len(cases)==180
    for original,group in zip(originals,[cases[i:i+3] for i in range(0,180,3)]):
        for c in group:
            assert c['source_keep_channels']==GOAL_ROLES[c['role']][1]
            for k in ['alpha','force_N','feedback_gain_rad','mode_index','causal_support']:assert c[k]==original[k]
    with pytest.raises(ValueError):goal_cases(originals[:-1])


def test_goal_ahead_does_not_override_original_heading_domain():
    p=path();pose=j.array([9.,-2.,j.pi/2+.2]);base=pursuit(p,9.2,pose,2.,1.3,CFG)
    base=base.replace(lookahead=j.array(1.5))
    result,used=goal_follower(base,p,9.2,pose,CFG)
    assert bool(used) and bool(result.invalid)
    with pytest.raises(ValueError):goal_follower(base,p,9.2,pose,CFG,radius_m=.6)


def test_goal_reports_retain_all_six_roles_and_main_factorial_contrasts():
    from sttw_control.finite_value_evaluation import paired_report
    from sttw_control.finite_value_evaluation_report import goal_report,role_labels
    roles=['zero_middle_feedback','steer_only_feedback','feedback_010',*GOAL_ROLES];rows=[]
    for force in [0.,16.]:
        for role in roles:
            ok=role in GOAL_ROLES
            rows.append(dict(case_id=f'{force}/{role}',role=role,alpha=.5,context='left',event_id=str(force),
                force_N=force,category='safe_task' if ok else 'safe_task_failed',physical_outcome='survived',
                goal_arrived=ok,scored_cost=1.,task_path_rmse_m=1.,task_speed_rmse_m_s=1.,
                task_heading_rmse_rad=1.,feedback_gain_rad=.1,goal_preview=ok))
    pairs=paired_report(rows,roles=roles);goals=goal_report(rows,roles)
    for a,(b,_) in GOAL_ROLES.items():
        key=a+'_vs_'+b
        assert pairs['comparisons'][key]['task_gains']==2
        assert goals['disturbed']['paired'][key]['arrival_gains']==1
        assert 'Goal-preview PP' in role_labels(rows)[a]


def test_numpy_audit_uses_preblock_target_and_rejects_double_publish_or_target_tampering():
    from sttw_control.goal_preview_baseline import verify_goal_log
    p=path();n=12;user=1.3
    xy=np.tile([0.,1.],(n,1));xy[4:]=[9.,1.]
    progress=np.zeros(n);progress[4:]=9.2
    world=np.zeros((n,2));body=np.zeros((n,2));commands=np.zeros((n,2))
    for start in [0,4,8]:
        pre=max(start-1,0);pose=j.array([*xy[pre],0.]);original=pursuit(p,progress[pre],pose,2.,user,CFG)
        follower,_=goal_follower(original,p,progress[pre],pose,CFG)
        world[start:start+4]=follower.preview_world;body[start:start+4]=follower.preview_body
        commands[start:start+4]=follower.command
    nominal=np.zeros((n,2),np.float32);nominal[0]=[user,0.]
    for i in range(1,n):nominal[i]=publish(j.array(nominal[i-1]),j.array(commands[i]),CFG)[0]
    raw=dict(active_tick=np.ones(n,bool),actor_action_valid=np.zeros(n,bool),issued_latent=np.zeros((n,2)),
        source_keep_channels=np.zeros((n,2),bool),source_actor_latent=np.full((n,2),np.nan),
        original_actor_latent=np.full((n,2),np.nan),physical_tick=np.arange(n),path_progress=progress,
        actual_forward_speed=np.full(n,2.),actual_xy=xy,actual_yaw=np.zeros(n),
        goal_preview_world=world,goal_preview_body=body,goal_preview_command=commands,nominal_reference=nominal)
    case=dict(role='goal_zero_feedback',requested_speed_m_s=user)
    arrays={k:np.asarray(getattr(p,k)) for k in p.__dataclass_fields__}
    receipt=verify_goal_log(raw,case,CFG,{},arrays);assert receipt['changed_target_ticks']==4
    bad=copy.deepcopy(raw);bad['nominal_reference'][9,1]+=.001
    with pytest.raises(AssertionError):verify_goal_log(bad,case,CFG,{},arrays)
    bad=copy.deepcopy(raw);bad['goal_preview_world'][8,0]+=.1
    with pytest.raises(AssertionError):verify_goal_log(bad,case,CFG,{},arrays)
    rounded=copy.deepcopy(raw);rounded['goal_preview_body'][8:,0]+=4e-5
    verify_goal_log(rounded,case,CFG,{},arrays)
    bad=copy.deepcopy(raw);bad['goal_preview_body'][8,0]+=.001
    with pytest.raises(AssertionError):verify_goal_log(bad,case,CFG,{},arrays)
