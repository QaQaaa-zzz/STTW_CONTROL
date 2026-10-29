import numpy as np
import pytest

from sttw_control.preference_outcome_prediction import outcome_features,STATE_INDICES,plan_alpha_intervals,FrozenAlphaRangePlanner
from sttw_control.tolerance_coordination import select_from_alpha_range
from sttw_control.preference_outcome_prediction import merge_outcome_datasets,audit_query_feature_alignment,SCHEMA


def example():
    obs=np.arange(351,dtype=np.float32)/100.;obs[336]=.5
    case=dict(alpha=.5,source_task_start_s=1.6,preparation_s=.6,onset_s=2.2,force_N=24.,duration_s=.4,force_direction_world=[0.,1.,0.])
    interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2)
    return obs,np.array([.4,-.3]),case,interface


def test_features_use_source_observation_and_explicit_counterfactual_not_future_outcomes():
    obs,z,case,interface=example();x=outcome_features(obs,z,np.pi/2,case,interface)
    assert x.shape==(42,);np.testing.assert_array_equal(x[:34],obs[list(STATE_INDICES)])
    np.testing.assert_allclose(x[-8:],[.5,.25*np.tanh(.4),np.tanh(-.3),.75,0.,1.,1.,.84],atol=1e-6)
    changed=dict(case,goal_arrived=True,contact_failure=True,future_roll=100.,future_speed=-100.)
    np.testing.assert_array_equal(x,outcome_features(obs,z,np.pi/2,changed,interface))


def test_feature_query_rejects_alpha_and_preparation_clock_mismatch():
    obs,z,case,interface=example()
    with pytest.raises(ValueError,match='candidate alpha'):outcome_features(obs,z,0.,dict(case,alpha=1.),interface)
    with pytest.raises(ValueError,match='preparation'):outcome_features(obs,z,0.,dict(case,onset_s=2.3),interface)


def test_range_planner_uses_worst_declared_hypothesis_and_preserves_empty_set():
    obs,z,case,interface=example();export=dict(mean=np.zeros(42),scale=np.ones(42),w0=np.zeros((1,42,16)),b0=np.zeros((1,16)),w1=np.zeros((1,16,16)),b1=np.zeros((1,16)),w2=np.zeros((1,16,4)),b2=np.array([[2.,3.,np.log1p(.2),np.log1p(2.)]]))
    export['w0'][0,37,0]=1.;export['w1'][0,0,0]=1.;export['w2'][0,0,0]=-2.
    nominal=dict(force_N=0.,duration_s=.4,preparation_s=0.)
    gust=dict(force_N=24.,duration_s=.4,preparation_s=.6,force_direction_world=[0.,1.,0.])
    kwargs=dict(no_fall_threshold=.8,task_threshold=.8,max_path_error_m=.3)
    safe=plan_alpha_intervals(export,obs,[0.,.5,1.],np.zeros((3,2)),np.pi/2,1.6,[nominal],interface,**kwargs)
    worst=plan_alpha_intervals(export,obs,[0.,.5,1.],np.zeros((3,2)),np.pi/2,1.6,[nominal,gust],interface,**kwargs)
    assert safe['estimated_intervals']==[[0.,1.]]
    assert worst['empty'] and worst['estimated_intervals']==[]


def test_original_user_preference_is_separate_from_current_conditioning_alpha():
    planner=FrozenAlphaRangePlanner.__new__(FrozenAlphaRangePlanner)
    planner.parameters=dict(mean=np.zeros(42),scale=np.ones(42),w0=np.zeros((1,42,16)),b0=np.zeros((1,16)),w1=np.zeros((1,16,16)),b1=np.zeros((1,16)),w2=np.zeros((1,16,4)),b2=np.array([[2.,3.,np.log1p(.2),np.log1p(2.)]]))
    planner.middle_query=lambda x:np.zeros((x.shape[0],2))
    planner.manifest=dict(source_model=dict(registry_id='fixture'),parameters_sha256='fixture')
    obs,_,_,planner.interface=example()
    result=planner.query(obs,0.,1.6,[dict(force_N=0.,duration_s=.4,preparation_s=0.)],user_alpha=.3,candidate_alphas=[0.,.25,.5])
    assert result['user_alpha']==.3 and result['conditioning_alpha_before_query']==.5
    assert .3 in result['feasible_candidates']


def test_range_selection_preserves_user_and_restores_after_temporary_deviation():
    plan=dict(candidate_alphas=[0.,.25,.5,.75,1.],feasible_mask=[False,False,True,False,False])
    first=select_from_alpha_range(plan,user_alpha=0.,previous_alpha=0.)
    assert first['selected_alpha']==.5 and first['changed'] and not first['degraded']
    plan['feasible_mask']=[True,True,True,True,True]
    returned=select_from_alpha_range(plan,user_alpha=0.,previous_alpha=.5,switch_weight=100.)
    assert returned['selected_alpha']==0. and returned['reason']=='original_preference_feasible'


def test_empty_range_and_permission_denial_do_not_invent_feasible_actions():
    plan=dict(candidate_alphas=[0.,.25,.5,.75,1.],feasible_mask=[False]*5)
    held=select_from_alpha_range(plan,user_alpha=0.,previous_alpha=.75)
    assert held['selected_alpha']==.75 and held['degraded'] and not held['selected_is_predicted_feasible']
    plan['feasible_mask'][2]=True
    denied=select_from_alpha_range(plan,user_alpha=0.,previous_alpha=.75,publication_permitted=False)
    assert denied['selected_alpha']==.75 and denied['reason']=='publication_denied'


def test_disconnected_range_selection_never_commands_unqueried_interior():
    plan=dict(candidate_alphas=[0.,.25,.5,.75,1.],feasible_mask=[True,True,False,True,True])
    result=select_from_alpha_range(plan,user_alpha=.5,previous_alpha=.75)
    assert result['selected_alpha']==.75 and result['selected_is_predicted_feasible']


def test_refit_merge_releases_known_dev_explicitly_without_changing_parent_roles(tmp_path):
    import json,hashlib
    def parent(name,case,role,actor='actor'):
        p=tmp_path/name;p.mkdir()
        np.savez_compressed(p/'dataset.npz',features=np.zeros((1,42)),targets=np.zeros((1,4)),regression_mask=np.ones((1,2)),roles=np.array([role]))
        meta=dict(schema=SCHEMA,feature_dimension=42,state_indices=list(STATE_INDICES),physical_contract_sha256='physics-fixture',query_features=['fixture'],targets=['fixture'],
            source_model=dict(actor_sha256=actor),records=[dict(case_id=case,role=role)],partition={role:1},inputs_sha256={},
            dataset_sha256=hashlib.sha256((p/'dataset.npz').read_bytes()).hexdigest())
        (p/'manifest.json').write_text(json.dumps(meta));return p
    a=parent('a','a','training');b=parent('b','b','development');before=(b/'manifest.json').read_bytes()
    out=tmp_path/'merged';meta=merge_outcome_datasets([a,b],out,release_note='Known DEV outcomes released for a new refit; new validation required.')
    with np.load(out/'dataset.npz') as z:assert z['roles'].tolist()==['training','training']
    assert meta['records'][1]['parent_role']=='development' and (b/'manifest.json').read_bytes()==before
    with pytest.raises(ValueError,match='duplicate physical'):merge_outcome_datasets([a,a],tmp_path/'duplicate',release_note='explicit')
    c=parent('c','c','training',actor='other')
    with pytest.raises(ValueError,match='different feature'):merge_outcome_datasets([a,c],tmp_path/'different',release_note='explicit')
    d=parent('d','d','training');changed=json.loads((d/'manifest.json').read_text());changed['physical_contract_sha256']='other-physics';(d/'manifest.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='different feature'):merge_outcome_datasets([a,d],tmp_path/'physics',release_note='explicit')
    with pytest.raises(ValueError,match='explicit release'):merge_outcome_datasets([a],tmp_path/'implicit',release_note='')


def test_bounded_actor_roundoff_does_not_relax_state_or_feasibility_checks():
    export=dict(mean=np.zeros(42),scale=np.ones(42),w0=np.zeros((1,42,16)),b0=np.zeros((1,16)),w1=np.zeros((1,16,16)),b1=np.zeros((1,16)),w2=np.zeros((1,16,4)),b2=np.array([[2.,3.,np.log1p(.2),np.log1p(2.)]]))
    frozen=np.zeros((1,42));observed=frozen.copy();observed[0,36]=1.2e-4
    audit=audit_query_feature_alignment(frozen,observed,export)
    assert audit['candidate_steer_error_rad']==pytest.approx(2.4e-5) and audit['frozen_predictions_preserved']
    altered=frozen.copy();altered[0,0]=1.2e-4
    with pytest.raises(ValueError,match='exceed bounded'):audit_query_feature_alignment(frozen,altered,export)
    altered=frozen.copy();altered[0,36]=1e-3
    with pytest.raises(ValueError,match='exceed bounded'):audit_query_feature_alignment(frozen,altered,export)
    export['w0'][0,36,0]=1.;export['w1'][0,0,0]=1.;export['w2'][0,0,0]=20.;export['b2'][0,0]=np.log(.7998/(1-.7998))
    with pytest.raises(ValueError,match='changes default feasibility'):audit_query_feature_alignment(frozen,observed,export)
