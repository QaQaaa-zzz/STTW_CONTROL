"""New-physics capability inventory, no preparation/evaluation integration."""
import json
from pathlib import Path
import pytest
from sttw_control.native_goal_run import fixed_h75_capability_inventory,planning_chunk_steps


def config():return json.loads((Path(__file__).parents[1]/'configs/h75_friction08_capability.json').read_text())


def test_frozen_only_h75_thirtyfive_nominal_and_original_routes():
    p=config();rows=fixed_h75_capability_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==35
    assert {r['mode'] for r in rows}=={'H75'}
    assert {r['alpha'] for r in rows}=={0.,.25,.5,.75,1.}
    assert all(r['alpha']==r['query_alpha']==r['user_alpha'] and r['force_N']==0 for r in rows)
    assert sum(planning_chunk_steps(10.,10.))*4*len(rows)==70000
    reference=json.loads((Path(__file__).parents[1]/'configs/middle_selection.json').read_text())
    assert p['route_contexts']==reference['route_contexts']
    assert all(len([r for r in rows if r['route']==route])==5 for route in {r['route'] for r in rows})


@pytest.mark.parametrize('key,value',[('model_id','other'),('alphas',[0,.5,1]),('expected_cases',34),('maximum_lower_ticks',70001),('forces_N',[-64,64]),('planning_frequency_hz',1),('bank_indices',[3])])
def test_changed_capability_contract_rejected(key,value):
    p=config();p[key]=value
    with pytest.raises(ValueError):fixed_h75_capability_inventory(p)
