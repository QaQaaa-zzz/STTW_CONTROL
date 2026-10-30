"""Paired equivalent lateral load, without integrating physical dynamics."""
import copy
import itertools
import json
from pathlib import Path

import jax.numpy as j
import numpy as np
import pytest
from flax import struct
from sttw_control.native_goal_run import (
    fixed_h75_capability_inventory,force_pulses,audit_five_alpha_online_initial)


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
    hidden: object

def config():
    return json.loads((Path(__file__).parents[1]/'configs/h75_low_speed_lateral_load_screen.json').read_text())


def test_nn_boundary_inventory_only_runs_selected_points_and_new_gust_timings():
    p=json.loads((Path(__file__).parents[1]/'configs/u170_nn_range_boundary.json').read_text())
    rows=fixed_h75_capability_inventory(p)
    assert len(rows)==4 and {r['alpha'] for r in rows}=={.25,.5}
    assert {r['preparation_s'] for r in rows}=={.3,.5}
    assert all(r['force_N']==24. for r in rows)
    assert all(r['alpha']==(.5 if r['original_user_alpha']==0. else .25) for r in rows)
    assert [h['preparation_s'] for h in p['range_planner_boundary']['hypotheses']]==[.2,.6]
    assert not any('onsets_s' in h for h in p['range_planner_boundary']['hypotheses'])
    assert p['maximum_lower_ticks']==8000 and p['optimizer_updates']==0


def test_preparation_coverage_reuses_four_completed_cases_without_dropping_other_alphas():
    p=json.loads((Path(__file__).parents[1]/'configs/u170_preparation_time_coverage.json').read_text())
    rows=fixed_h75_capability_inventory(p);excluded={r['case_id'] for r in p['excluded_completed_cases']}
    assert len(rows)==26 and not ({r['case_id'] for r in rows}&excluded)
    assert {r['alpha'] for r in rows}=={0.,.25,.5,.75,1.}
    assert {r['preparation_s'] for r in rows}=={.3,.4,.5}
    assert len({r['case_id'] for r in rows}|excluded)==30
    bad=copy.deepcopy(p);bad['excluded_completed_cases'][0]['case_id']='not-a-declared-case'
    with pytest.raises(ValueError,match='outside declared'):fixed_h75_capability_inventory(bad)


def test_full_load_grid_and_identical_world_force_across_alpha():
    cfg=config();rows=fixed_h75_capability_inventory(cfg)
    assert len(rows)==len({row['case_id'] for row in rows})==34
    assert {(r['force_N'],r['onset_s'],r['alpha']) for r in rows if r['force_N']}==set(itertools.product(cfg['forces_N'],cfg['onsets_s'],cfg['alphas']))
    assert {(r['force_N'],r['onset_s'],r['alpha']) for r in rows if not r['force_N']}=={(0.,0.,0.),(0.,0.,1.)}
    for i in range(0,34,2):
        a,b=rows[i:i+2]
        assert a['alpha']==0. and b['alpha']==1.
        assert {k:v for k,v in a.items() if k not in ('alpha','query_alpha','user_alpha','effective_user_alpha','case_id')}=={k:v for k,v in b.items() if k not in ('alpha','query_alpha','user_alpha','effective_user_alpha','case_id')}
        pulses=force_pulses([a,b],3.5)
        np.testing.assert_array_equal(pulses[0],pulses[1])
        np.testing.assert_allclose(pulses[0],[3.5+a['onset_s'],4.1+a['onset_s'],0,a['force_N'],0])
    assert cfg['maximum_lower_ticks']==len(rows)*2000
    assert {r['bank_index'] for r in rows}=={0}
    assert all(r['path_key'].startswith('bank0_') for r in rows)
    assert cfg['low_speed_evidence_gate']['true_forward_speed_max_m_s']==2.1


@pytest.mark.parametrize('key,value',[('alphas',[0,.5,1]),('forces_N',[-16,16]),('onsets_s',[2.]),('duration_s',-1.),('force_frame','bodyY'),('maximum_lower_ticks',68001),('candidate_model',{'registry_id':'candidate'})])
def test_changed_screen_contract_rejected(key,value):
    cfg=config();cfg[key]=value
    with pytest.raises(ValueError):fixed_h75_capability_inventory(cfg)


def test_adopted_base_and_configured_event_grid_keep_pairing_and_budget():
    path=Path(__file__).parents[1]/'configs/u170_low_speed_lateral_load_screen.json'
    cfg=json.loads(path.read_text());rows=fixed_h75_capability_inventory(cfg)
    assert {r['model_id'] for r in rows}=={cfg['candidate_model']['registry_id']}
    assert cfg['route_contexts'][0]['radius_m']==1.3
    assert len(rows)==34
    smaller=copy.deepcopy(cfg);smaller.update(forces_N=[-12.,12.],onsets_s=[2.2],expected_cases=6,maximum_lower_ticks=12000)
    rows=fixed_h75_capability_inventory(smaller)
    assert len(rows)==6
    for a,b in zip(rows[::2],rows[1::2]):
        np.testing.assert_array_equal(force_pulses([a],3.5),force_pulses([b],3.5))


def test_common_fixed_world_vector_is_not_rotated_per_alpha():
    cfg=config();cfg.update(force_frame='world_fixed_vector',force_direction_world=[-.8,.6,0.])
    rows=fixed_h75_capability_inventory(cfg);a,b=next((a,b) for a,b in zip(rows[::2],rows[1::2]) if a['force_N']==16.)
    pulses=force_pulses([a,b],3.5)
    np.testing.assert_array_equal(pulses[0],pulses[1]);np.testing.assert_allclose(pulses[0][2:],[-12.8,9.6,0.])
    cfg['force_direction_world']=[-1.,1.,0.]
    with pytest.raises(ValueError,match='unit horizontal'):fixed_h75_capability_inventory(cfg)


def test_conditional_case_grid_uses_source_clock_plus_preparation_and_full_groups():
    path=Path(__file__).parents[1]/'configs/u170_conditional_load_labels.json'
    cfg=json.loads(path.read_text());rows=fixed_h75_capability_inventory(cfg)
    assert len(rows)==100 and len({r['case_id'] for r in rows})==100
    assert {r['source_task_start_s'] for r in rows}=={1.6}
    assert {round(r['onset_s'],6) for r in rows if r['force_N']}=={1.8,2.2}
    for i in range(0,100,5):
        pulses=force_pulses(rows[i:i+5],3.5)
        for p in pulses[1:]:np.testing.assert_array_equal(p,pulses[0])
    assert sum(r['force_N']==20. and r['preparation_s']==.6 for r in rows)==20


@pytest.mark.parametrize('alphas',[[0.,1.],[0.,.25,.5,.75,1.]])
def test_two_alpha_per_event_reset_audit_preserves_complete_pairing(alphas):
    cfg=config();n=17*len(alphas);cfg.update(alphas=alphas,expected_cases=n,maximum_lower_ticks=n*2000)
    rows=fixed_h75_capability_inventory(cfg);alpha=j.array([r['alpha'] for r in rows])
    values=j.repeat(j.arange(17),len(alphas))
    initial=Carry(State(alpha,j.zeros(n,j.int32),Physical(j.full(n,700,j.int32)),values),values+10)
    receipt=audit_five_alpha_online_initial(initial,rows,700,panel_kind='disturbed_declared_user',expected_route_count=1)
    assert receipt['paired_reset_all_leaves_exact']
    continued=initial.replace(state=initial.state.replace(tick=j.full(n,320,j.int32),physical=Physical(j.full(n,1020,j.int32))))
    assert audit_five_alpha_online_initial(continued,rows,1020,panel_kind='disturbed_declared_user',expected_route_count=1,task_tick=320)['paired_reset_all_leaves_exact']
    with pytest.raises(ValueError,match='clocks/user'):
        audit_five_alpha_online_initial(continued,rows,1020,panel_kind='disturbed_declared_user',expected_route_count=1)
    bad=initial.replace(hidden=initial.hidden.at[1].set(999))
    with pytest.raises(ValueError,match='neutralization'):
        audit_five_alpha_online_initial(bad,rows,700,panel_kind='disturbed_declared_user',expected_route_count=1)


def test_mirrored_nominal_sources_preserve_direction_and_have_no_force():
    p=config();p.update(alphas=[0.],forces_N=[],onsets_s=[],include_nominal=True,expected_cases=2,maximum_lower_ticks=4000)
    left=copy.deepcopy(p['route_contexts'][0]);right=copy.deepcopy(left)
    left.update(route='left',direction=1.);right.update(route='right',direction=-1.)
    p['route_contexts']=[left,right]
    rows=fixed_h75_capability_inventory(p)
    assert len(rows)==2 and all(r['force_N']==0. for r in rows)
    assert {r['route'] for r in rows}=={'left','right'}
    p['route_contexts'][1]['direction']=0.
    with pytest.raises(ValueError,match='geometries'):fixed_h75_capability_inventory(p)


def test_missing_half_of_disturbance_grid_is_rejected():
    p=config();p['forces_N']=[]
    with pytest.raises(ValueError,match='grid'):fixed_h75_capability_inventory(p)


def test_track_segments_are_explicit_and_preserve_event_pairing():
    cfg=config()
    cfg['route_contexts'][0].update(family=2,angle_deg=60.)
    with pytest.raises(ValueError,match='geometries'):
        fixed_h75_capability_inventory(cfg)
    cfg['geometry_scope']='low_speed_track_segments'
    rows=fixed_h75_capability_inventory(cfg)
    assert len(rows)==cfg['expected_cases']
    for a,b in zip(rows[::2],rows[1::2]):
        np.testing.assert_array_equal(force_pulses([a],3.5),force_pulses([b],3.5))
    cfg['route_contexts'][0]['speed']=2.8
    with pytest.raises(ValueError,match='geometries'):
        fixed_h75_capability_inventory(cfg)


def test_mixed_speed_scope_preserves_old_low_speed_contract():
    cfg=json.loads((Path(__file__).parents[1]/'configs/u170_mixed_speed_capability.json').read_text())
    rows=fixed_h75_capability_inventory(cfg)
    assert len(rows)==40 and len({r['case_id'] for r in rows})==40
    assert {r['alpha'] for r in rows}=={0.,.25,.5,.75,1.}
    assert {r['speed'] for r in rows}=={1.6,2.,3.2}
    assert all(r['bank_index']==0 and r['force_N']==0 for r in rows)
    cfg['geometry_scope']='low_speed_track_segments'
    with pytest.raises(ValueError,match='speed/angle scope'):
        fixed_h75_capability_inventory(cfg)
    cfg['geometry_scope']='mixed_speed_track_segments'
    cfg['route_contexts'][0]['speed']=3.6
    with pytest.raises(ValueError,match='speed/angle scope'):
        fixed_h75_capability_inventory(cfg)


def test_multiple_prepared_states_keep_five_alpha_pairing_and_separate_paths():
    p=json.loads((Path(__file__).parents[1]/'configs/lower_domain_transfer_dev.json').read_text())
    p.update(bank_indices=[0,6],expected_cases=40,maximum_lower_ticks=80000)
    rows=fixed_h75_capability_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==40
    for bank in [0,6]:
        selected=[r for r in rows if r['bank_index']==bank]
        assert len(selected)==20
        assert {r['alpha'] for r in selected}=={0.,.25,.5,.75,1.}
        assert all(r['path_key'].startswith(f'bank{bank}_') for r in selected)
    alpha=j.array([r['alpha'] for r in rows]);values=j.repeat(j.arange(8),5)
    initial=Carry(State(alpha,j.zeros(40,j.int32),Physical(j.full(40,700,j.int32)),values),values+10)
    assert audit_five_alpha_online_initial(initial,rows,700,panel_kind='disturbed_declared_user',expected_route_count=8)['paired_reset_all_leaves_exact']
    p['bank_indices']=[0,0]
    with pytest.raises(ValueError,match='unique sorted'):fixed_h75_capability_inventory(p)
