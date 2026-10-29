"""CPU-only contracts for the five-alpha label readback, no dynamics."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('five_alpha_analysis',ROOT/'learning/cli/analyze_native_goal_confirmation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def inventory():
    p=json.loads((ROOT/'learning/configs/h75_five_alpha_tolerance_labels.json').read_text())
    rows=[]
    for g in p['source_groups']:
        for a,prep,f in m.itertools.product(p['candidate_alphas'],p['preparation_s'],p['forces_N']):
            rows.append(dict(g,candidate_alpha=a,query_alpha=a,alpha=.5,preparation_s=prep,force_N=f,duration_s=.15,onset_s=g['source_task_s']+prep,primary_window_start_s=g['source_task_s'],primary_window_end_s=g['source_task_s']+prep+.15+3))
    return p,rows


def test_complete_parent_and_child_products():
    p,rows=inventory();m.validate_five_alpha_cases(rows,p)
    child=dict(p,source_groups=p['source_groups'][7:]);m.validate_five_alpha_cases(rows[140:],child)
    with pytest.raises(ValueError,match='incomplete'):m.validate_five_alpha_cases(rows[:-1],p)
    with pytest.raises(ValueError,match='duplicate'):m.validate_five_alpha_cases(rows+[rows[0]],p)


def test_source_alpha_may_not_be_rewritten_to_candidate():
    p,rows=inventory();rows[0]['alpha']=0.
    with pytest.raises(ValueError,match='identity'):m.validate_five_alpha_cases(rows,p)


def raw_label(contact_tick=None,fault_tick=None):
    n=670;raw=dict(time=6.+np.arange(n)*.005,active_tick=np.ones(n,bool))
    for k in ('policy_fault','lower_fault','nonfinite','continuation_geometry_fault','contact_failure','physical_failure'):raw[k]=np.zeros(n,bool)
    if contact_tick is not None:raw['contact_failure'][contact_tick]=True;raw['active_tick'][contact_tick+1:]=False
    if fault_tick is not None:raw['policy_fault'][fault_tick]=True;raw['active_tick'][fault_tick+1:]=False
    return raw


def test_preparation_fall_negative_and_engineering_fault_invalid():
    c=dict(primary_window_start_s=6.,primary_window_end_s=9.35,onset_s=6.2)
    assert m.samestate_primary_label(raw_label(),c)['label_status']=='no_fall'
    r=m.samestate_primary_label(raw_label(contact_tick=5),c)
    assert r['label_status']=='fall' and r['prep_contact']
    assert m.samestate_primary_label(raw_label(fault_tick=5),c)['label_status']=='invalid_fault'


def test_features_use_prephysics_half_state_and_event_invariant_queries():
    cases=[dict(case_id=str(i),source_index=7,candidate_alpha=.25,preparation_s=prep,force_N=f,duration_s=.15) for i,(prep,f) in enumerate(m.itertools.product([.2,.6],[-64.,64.]))]
    state=np.zeros(351,np.float32);state[336]=.5;query=state.copy();query[336]=.25
    raws={c['case_id']:dict(active_tick=np.ones(1,bool),reference_target_permitted=np.ones(1,bool),pre_upper_yaw=np.array([.2]),query_observation=query[None].copy(),composed_latent=np.array([[.1,-.2]],np.float32)) for c in cases}
    src=dict(native_observation=np.tile(state,(4,1)),source_world_yaw=np.full(4,.2))
    interface=dict(speed_positive_scale_m_s=1.,speed_negative_scale_m_s=1.,steer_scale_rad=.2)
    x=m.five_alpha_feature_rows(cases,src,raws,interface)
    assert x.shape==(4,41);np.testing.assert_array_equal(x[:,34],.25)
    np.testing.assert_allclose(x[:,39],1.5)
    raws['3']['composed_latent'][0,0]=.3
    with pytest.raises(AssertionError):m.five_alpha_feature_rows(cases,src,raws,interface)


def test_role_metadata_uses_sealed_training_only_bank4_protocol():
    p=json.loads((ROOT/'learning/configs/h75_five_alpha_bank4_labels.json').read_text())
    role=m.five_alpha_role_metadata(dict(protocol=p,data_role='stale outer description'))
    assert role['data_role']==p['data_role']
    assert role['data_partition']=='training_only'
    assert role['source_bank_index']==4
    assert (role['declared_training_rows'],role['declared_development_rows'])==(560,0)
    old,_=inventory();legacy=m.five_alpha_role_metadata(dict(protocol=old))
    assert legacy['data_partition']=='mixed'
    assert (legacy['declared_training_rows'],legacy['declared_development_rows'])==(160,120)


def test_limited_collection_partition_and_scope_do_not_claim_h10():
    import copy
    p,_=inventory();p=copy.deepcopy(p);p.update(collection_end_s=5.76,expected_cases=140,expected_source_states=7,require_ongoing_sources=True)
    p['source_groups']=p['source_groups'][:7]
    for g in p['source_groups']:g['source_task_s']=2.
    p['fitting_split'].update(expected_training_rows=80,expected_development_rows=60)
    result=m.label_analysis_partition(p)
    assert result['cases']==140 and result['training_rows']==80
    assert m.label_plot_outcome(dict(collection_outcome=dict(physical_outcome='survived'),horizons={}))['physical_outcome']=='survived'
    p['source_groups'][0]['expected_ever_arrived']=True
    with pytest.raises(ValueError,match='ongoing'):m.label_analysis_partition(p)


def test_primary_label_uses_exact_end_not_longer_collection_boundary():
    n=752;raw=dict(time=2.+np.arange(n)*.005,active_tick=np.ones(n,bool))
    for k in ('policy_fault','lower_fault','nonfinite','continuation_geometry_fault','contact_failure','physical_failure'):raw[k]=np.zeros(n,bool)
    contact=680  # task5.4: after prep.2 labelend5.35, before prep.6 labelend5.75.
    raw['contact_failure'][contact]=True;raw['active_tick'][contact+1:]=False
    short=dict(primary_window_start_s=2.,primary_window_end_s=5.35,onset_s=2.2)
    long=dict(primary_window_start_s=2.,primary_window_end_s=5.75,onset_s=2.6)
    assert m.samestate_primary_label(raw,short)['label_status']=='no_fall'
    assert m.samestate_primary_label(raw,long)['label_status']=='fall'
    assert m.samestate_primary_label(raw,short)['observed_window_ticks']==670
