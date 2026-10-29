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
