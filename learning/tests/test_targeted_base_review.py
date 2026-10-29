"""Targeted candidate review inventory/reset checks; no physical integration."""
import json
from pathlib import Path
import jax.numpy as j
import pytest
from flax import struct
from sttw_control.native_goal_run import fixed_h75_capability_inventory,audit_five_alpha_online_initial


def config():return json.loads((Path(__file__).parents[1]/'configs/h75_mixed_u170_targeted_base_review.json').read_text())


def test_targeted_review_exact_twelve_bankzero_candidate_cases():
    p=config();rows=fixed_h75_capability_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==12
    assert {r['alpha'] for r in rows}=={0.,.5,1.}
    assert {r['bank_index'] for r in rows}=={0}
    assert {r['model_id'] for r in rows}=={p['candidate_model']['registry_id']}
    assert all(r['force_N']==0 and r['onset_s']==0 and r['path_key'].startswith('bank0_') for r in rows)
    assert p['maximum_lower_ticks']==12*2000
    assert p['route_contexts'][-1]['goal_distance_m']==19.65486717224121
    assert [c['speed'] for c in p['route_contexts']]==[1.6,2.,2.6,2.8]


@pytest.mark.parametrize('key,value',[('bank_indices',[6]),('alphas',[0,1]),('expected_cases',20),('forces_N',[4]),('candidate_model',None)])
def test_targeted_review_rejects_changed_contract(key,value):
    p=config();p[key]=value
    with pytest.raises(ValueError):fixed_h75_capability_inventory(p)


def test_targeted_three_alpha_complete_reset_pairing():
    @struct.dataclass
    class Physical:physical_tick:object
    @struct.dataclass
    class State:
        alpha:object
        tick:object
        physical:object
        history:object
    @struct.dataclass
    class Carry:
        state:object
        hidden:object
    rows=fixed_h75_capability_inventory(config());values=j.repeat(j.arange(4),3)
    initial=Carry(State(j.array([r['alpha'] for r in rows]),j.zeros(12,j.int32),Physical(j.full(12,700,j.int32)),values),values)
    r=audit_five_alpha_online_initial(initial,rows,700,panel_kind='targeted_three_user',expected_route_count=4)
    assert r['paired_reset_all_leaves_exact']
    with pytest.raises(ValueError):audit_five_alpha_online_initial(initial.replace(hidden=values.at[1].set(55)),rows,700,panel_kind='targeted_three_user',expected_route_count=4)
