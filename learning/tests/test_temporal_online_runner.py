"""Two temporal rank identities reuse the original120 comparison physics."""
import copy
import json
from pathlib import Path
import pytest
from sttw_control.native_goal_run import tolerance_inventory,validate_temporal_comparison_contract


def protocol():return json.loads((Path(__file__).parents[1]/'configs/h75_tolerance_online40.json').read_text())


def test_only_two_new_rank_arms_are_allocated():
    p=protocol();rows=tolerance_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==40
    assert len(rows)*10/.005==p['maximum_lower_ticks']==80000
    assert {r['mode'] for r in rows}=={'temporal_rank_selected','temporal_rank_last'}
    groups={}
    for r in rows:groups.setdefault((r['bank_index'],r['route'],r['event']),[]).append(r)
    assert len(groups)==20 and all(len(x)==2 for x in groups.values())
    assert p['comparison_total_records']==160 and p['cached_physical_cases']==120
    assert p['arm_to_export_index']=={'temporal_rank_selected':0,'temporal_rank_last':1}


@pytest.mark.parametrize('key,value',[('expected_cases',120),('maximum_lower_ticks',240000),('arms',['q_rank','q_bce']),('arm_to_export_index',{'temporal_rank_selected':1,'temporal_rank_last':0})])
def test_reject_changed_rank_identity_or_physics_budget(key,value):
    p=copy.deepcopy(protocol());p[key]=value
    with pytest.raises(ValueError):tolerance_inventory(p)


def test_cached_comparison_requires_matched_full_controller_and_conditions():
    p=protocol();cache=dict(mode='tolerance_online120',cases=120,config={'physics':'original'},initial_source={'bank':'original'},models={'H75':'frozen'},contexts=[{'path':'original'}],protocol=copy.deepcopy(p))
    args=(cache['config'],cache['initial_source'],cache['models'],cache['contexts'])
    assert validate_temporal_comparison_contract(p,cache,*args)
    changed=copy.deepcopy(cache);changed['protocol']['onsets_s']=[2.2,3.4]
    with pytest.raises(ValueError,match='condition differs'):validate_temporal_comparison_contract(p,changed,*args)
    changed=copy.deepcopy(cache);changed['config']['physics']='different'
    with pytest.raises(ValueError,match='identity differs'):validate_temporal_comparison_contract(p,changed,*args)
