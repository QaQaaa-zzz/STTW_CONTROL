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


def test_s_goal_extension_changes_only_goal_on_authenticated_saved_exit():
    import copy
    import numpy as np
    from sttw_control.native_goal_run import validate_goal_extension_source
    p=json.loads((Path(__file__).parents[1]/'configs/h75_friction08_s_goal_extension.json').read_text())
    rows=fixed_h75_capability_inventory(p)
    assert len(rows)==10 and p['maximum_lower_ticks']==2000*len(rows)
    old={c['route']:c for c in config()['route_contexts']}
    for c in p['route_contexts']:
        assert {k:v for k,v in c.items() if k!='goal_distance_m'}==old[c['route']]
        r=p['goal_source']['paths'][c['route']]
        assert r['new_goal_s']-r['old_goal_s']==3.
        with np.load(r['file']) as z:
            assert np.all(z['curvature'][(z['s']>=r['old_goal_s'])&(z['s']<=r['new_goal_s'])]==0)
    source=json.loads((Path(p['goal_source']['run'])/'plan.json').read_text())['initial_source']
    validate_goal_extension_source(p,source)
    bad=copy.deepcopy(p);bad['goal_source']['paths'][p['route_contexts'][0]['route']]['new_goal_s']+=1.
    with pytest.raises(ValueError):validate_goal_extension_source(bad,source)


def test_low_speed_tight_turn_has_four_routes_five_alphas_frozen_budget():
    p=json.loads((Path(__file__).parents[1]/'configs/h75_friction08_low_speed_tight_turn.json').read_text())
    rows=fixed_h75_capability_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==20
    assert {r['alpha'] for r in rows}=={0.,.25,.5,.75,1.}
    assert {(c['speed'],c['radius_m']) for c in p['route_contexts']}=={(1.6,.9),(1.6,1.3),(2.,.9),(2.,1.3)}
    assert all(c['family']==1 and c['direction']==1. and c['angle_deg']==90. and c['goal_offset_after_turn_m']==4. for c in p['route_contexts'])
    assert sum(planning_chunk_steps(10.,10.))*4*len(rows)==40000
    assert all(r['force_N']==0. and r['bank_index']==6 for r in rows)
    assert not any('goal_extension_m' in r for r in rows)
    p['route_contexts'][0]['speed']=1.2
    with pytest.raises(ValueError):fixed_h75_capability_inventory(p)
