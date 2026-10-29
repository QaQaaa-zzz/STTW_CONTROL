"""Bounded online transport/decision checks without a plant or policy rollout."""
import inspect
import json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import pytest
from sttw_control.native_goal_run import five_alpha_online_inventory,planning_chunk_steps
from sttw_control.tolerance_coordination import coordinate,choose_alpha,candidate_features,OnlineCarry


def config():return json.loads((Path(__file__).parents[1]/'configs/h75_five_alpha_online.json').read_text())


def test_exact_fourteen_and_unchanged_source_routes():
    p=config();rows=five_alpha_online_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==14
    assert {r['alpha'] for r in rows}=={.5} and {r['force_N'] for r in rows}=={-64.,64.}
    assert sum(planning_chunk_steps(10.,10.))*4*len(rows)==28000
    source=json.loads((Path(__file__).parents[1]/'configs/middle_selection.json').read_text())
    assert p['route_contexts']==source['route_contexts']
    assert not any('event' in k or 'force' in k or 'time' in k for k in inspect.signature(coordinate).parameters)


@pytest.mark.parametrize('key,value',[('arm',1),('hypothetical_duration_s',.1),('candidate_alphas',[0,.5,1]),('maximum_lower_ticks',28001),('user_alpha',1.),('planning_frequency_hz',1.),('onset_s',3.),('task_aware_fallback',True)])
def test_reject_contract_changes(key,value):
    p=config();p[key]=value
    with pytest.raises(ValueError):five_alpha_online_inventory(p)


def test_five_candidate_duration_features_and_permission_restore_jit():
    interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2)
    x=candidate_features(j.zeros(351),j.zeros((5,2)),.2,interface,candidate_alphas=(0.,.25,.5,.75,1.),hypothetical_duration_s=.15)
    assert x.shape==(5,4,41)
    np.testing.assert_allclose(x[...,39],1.5)
    targets=j.tile(j.array([[2.6,0.]]),(5,1));q=j.array([.1,.9,.9,.1,.1])
    choose=jax.jit(lambda permission,prev,latch:choose_alpha(targets,0.,j.array([2.6,0.]),2.6,q,permission,latch,prev,user_alpha=.5,candidate_alphas=(0.,.25,.5,.75,1.)))
    denied=choose(False,.25,True)
    assert float(denied['alpha'])==.25 and bool(denied['latched'])
    restored=choose(True,.25,True)
    assert float(restored['alpha'])==.5 and not bool(restored['latched'])
    native=object();sidecar=OnlineCarry(native,j.bool_(True),j.array(.25))
    rebound=sidecar._replace(native='rebound native geometry')
    assert rebound.latched==sidecar.latched and rebound.prev_alpha==sidecar.prev_alpha and sidecar.native is native


def test_arm_zero_inference_uses_immutable_common_observation():
    # Match the production query construction: candidate alpha belongs only to
    # Actor queries; q sees the shared current observation and explicit candidates.
    obs=j.zeros(351).at[336].set(.5);grid=(0.,.25,.5,.75,1.)
    queries=jax.vmap(lambda a:obs.at[336].set(a))(j.array(grid))
    np.testing.assert_array_equal(queries[:,336],grid)
    assert float(obs[336])==.5
    params={k:j.zeros(shape) for k,shape in dict(w0=(1,3,41,16),b0=(1,3,16),w1=(1,3,16,16),b1=(1,3,16),w2=(1,3,16,1),b2=(1,3,1)).items()}
    interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2,speed_reference_bounds_m_s=[1.,3.5],steer_reference_bound_rad=.65)
    decide=jax.jit(lambda o:coordinate(params,o,j.zeros((5,2)),0.,j.array([2.6,0.]),2.6,0.,j.array([0.,10.]),j.zeros(2),True,False,.5,interface,arm=0,user_alpha=.5,candidate_alphas=grid,hypothetical_duration_s=.15))
    result=decide(obs)
    assert result['probabilities'].shape==(5,4) and result['qmin'].shape==(5,)
    assert result['alpha']==.5 and result['degraded']
    assert obs[336]==.5


def test_nominal_panel_all_users_zero_force_exact_budget_and_frozen_candidate():
    base=config();p=json.loads((Path(__file__).parents[1]/'configs/h75_five_alpha_online_nominal_users.json').read_text());rows=five_alpha_online_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==35
    assert sum(planning_chunk_steps(10.,10.))*4*len(rows)==70000
    assert p['predictor_training']==base['predictor_training'] and p['export_manifest_sha256']==base['export_manifest_sha256']
    assert p['candidate_alphas']==base['candidate_alphas'] and p['alpha_dwell_s']==.2
    assert p['route_contexts']==base['route_contexts']
    for route in {r['route'] for r in rows}:
        rr=[r for r in rows if r['route']==route]
        assert [r['user_alpha'] for r in rr]==[0.,.25,.5,.75,1.]
        assert all(r['alpha']==r['query_alpha']==r['user_alpha']==r['effective_user_alpha'] and r['force_N']==0 and r['event']=='zero' for r in rr)


@pytest.mark.parametrize('user',[0.,.25,.5,.75,1.])
def test_each_user_restores_its_own_index_and_fallback_remains_half(user):
    grid=(0.,.25,.5,.75,1.);targets=j.tile(j.array([[2.6,0.]]),(5,1));q=j.zeros(5).at[grid.index(user)].set(.9)
    choose=lambda scores:choose_alpha(targets,0.,j.array([2.6,0.]),2.6,scores,True,True,1.-user,user_alpha=user,candidate_alphas=grid)
    restored=choose(q)
    assert restored['alpha']==user and not restored['latched']
    fallback=choose(j.zeros(5))
    assert fallback['alpha']==.5 and fallback['latched'] and fallback['degraded']


@pytest.mark.parametrize('field,value',[('expected_cases',34),('maximum_lower_ticks',70001),('user_alphas',[0,.5,1]),('forces_N',[-64,64]),('alpha_dwell_s',0.)])
def test_nominal_panel_rejects_changed_scope(field,value):
    p=json.loads((Path(__file__).parents[1]/'configs/h75_five_alpha_online_nominal_users.json').read_text());p[field]=value
    with pytest.raises(ValueError):five_alpha_online_inventory(p)


def test_nominal_initial_audit_neutralizes_only_user_and_keeps_hidden_state():
    from flax import struct
    from sttw_control.native_goal_run import audit_five_alpha_online_initial
    @struct.dataclass
    class Physical:
        physical_tick: object
    @struct.dataclass
    class State:
        alpha: object
        tick: object
        physical: object
        history: object
    @struct.dataclass
    class Carry:
        state: object
        latch_history: object
    p=json.loads((Path(__file__).parents[1]/'configs/h75_five_alpha_online_nominal_users.json').read_text());cases=five_alpha_online_inventory(p)
    users=j.array([c['alpha'] for c in cases]);history=j.repeat(j.arange(7),5)
    initial=Carry(State(users,j.zeros(35,j.int32),Physical(j.full(35,700,j.int32)),history),history+10)
    audit=audit_five_alpha_online_initial(initial,cases,700,panel_kind='nominal_five_user')
    assert audit['alpha_neutralized_only'] and audit['initial_hold_until_tick']==0
    np.testing.assert_array_equal(initial.state.alpha,users)
    assert audit['initial_user_alphas']==[c['alpha'] for c in cases]
    bad=initial.replace(state=initial.state.replace(history=history.at[1].set(999)))
    with pytest.raises(ValueError,match='neutralization'):audit_five_alpha_online_initial(bad,cases,700,panel_kind='nominal_five_user')
    bad=initial.replace(state=initial.state.replace(alpha=users.at[0].set(.5)))
    with pytest.raises(ValueError,match='clocks/user'):audit_five_alpha_online_initial(bad,cases,700,panel_kind='nominal_five_user')
