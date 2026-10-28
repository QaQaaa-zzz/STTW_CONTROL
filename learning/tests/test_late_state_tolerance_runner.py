"""Late same-state panel contract; no physics or predictor fitting."""
from pathlib import Path
import copy
import json
import pytest
from sttw_control.native_goal_run import late_state_inventory,bank144_inventory,bank144_prefix_cases,late_source_rows


def protocol():return json.loads((Path(__file__).parents[1]/'configs/h75_late_state_tolerance48.json').read_text())


def test_late_panel_budget_source_ancestry_and_primary_windows():
    p=protocol();rows=late_state_inventory(p)
    assert rows==bank144_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==48
    assert p['maximum_prefix_lower_ticks']==0
    assert p['historical_prefix_lower_ticks_reused']==4*6/.005==4800
    assert p['maximum_lower_ticks']==48*4/.005==38400
    assert p['maximum_total_lower_ticks']==38400
    for source in range(4):
        own=[r for r in rows if r['source_index']==source]
        assert len(own)==12
        assert {(r['candidate_alpha'],r['preparation_s'],r['force_N']) for r in own}=={(a,t,f) for a in (0.,.5,1.) for t in (.2,.6) for f in (-64.,64.)}
        assert {r['alpha'] for r in own}=={1.}
        assert {r['primary_window_start_s'] for r in own}=={6.}
        assert all(r['primary_window_end_s']<=10 for r in own)
    prefixes=bank144_prefix_cases(p)
    assert len(prefixes)==4
    assert {(r['speed_alpha'],r['steer_alpha'],r['alpha']) for r in prefixes}=={(.5,.5,1.)}
    assert all(r['force_N']==0 for r in prefixes)


@pytest.mark.parametrize('key,value',[('source_task_s',5.),('new_horizon_s',5.),('preparation_s',[0.,.2,.6]),('maximum_prefix_lower_ticks',4801),('maximum_lower_ticks',40000),('maximum_total_lower_ticks',43201),('optimizer_updates',1)])
def test_reject_silent_late_protocol_changes(key,value):
    p=copy.deepcopy(protocol());p[key]=value
    with pytest.raises(ValueError):late_state_inventory(p)


def test_saved_source_matching_is_metadata_only_and_unique():
    groups=protocol()['source_groups']
    cases=[dict(bank_index=g['bank_index'],route=g['route'],mode='fixed_half',force_N=0.) for g in groups]
    assert late_source_rows(cases,groups)==[0,1,2,3]
    cases.append(dict(cases[0]))
    with pytest.raises(ValueError):late_source_rows(cases,groups)
