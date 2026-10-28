import copy
import json
from pathlib import Path
import pytest
from sttw_control.native_goal_run import bank144_inventory,bank144_prefix_cases,late_source_rows,new_confirmation_sources


def protocol():return json.loads((Path(__file__).parents[1]/'configs/h75_user_late_state192.json').read_text())


def test_all16_sources_and_twelve_each_original_user_preserved():
    p=protocol();rows=bank144_inventory(p)
    assert len(rows)==len({c['case_id'] for c in rows})==192
    assert p['maximum_lower_ticks']==192*4/.005==153600
    assert p['maximum_prefix_lower_ticks']==0
    assert new_confirmation_sources(p)
    for src in p['source_groups']:
        own=[r for r in rows if r['source_index']==src['source_index']]
        assert len(own)==12
        assert {r['candidate_alpha'] for r in own}=={0.,.5,1.}
        assert all(r['alpha']==src['source_user_alpha']==r['effective_user_alpha'] for r in own)
        assert all(r['query_alpha']==r['candidate_alpha'] for r in own)
        assert max(r['primary_window_end_s'] for r in own)==pytest.approx(9.7)
    prefixes=bank144_prefix_cases(p)
    assert len(prefixes)==16
    assert [r['speed'] for r in prefixes]==[r['speed'] for r in p['source_groups']]


def test_nominal_lane_selection_uses_origin_and_preserves_source_order():
    p=protocol();cases=[]
    for g in reversed(p['source_groups']):
        cases += [dict(bank_index=g['bank_index'],route=g['route'],mode=g['origin'],force_N=f) for f in (-64.,0.,64.)]
    selected=late_source_rows(cases,p['source_groups'])
    assert len(set(selected))==16
    assert [cases[i]['mode'] for i in selected]==[g['origin'] for g in p['source_groups']]
    with pytest.raises(ValueError):late_source_rows(cases+[cases[selected[0]]],p['source_groups'])


@pytest.mark.parametrize('field,value',[('expected_cases',48),('maximum_lower_ticks',38400),('bank_indices',[2,5]),('preparation_s',[0.,.2,.6])])
def test_cannot_silently_change_declared_suffix_budget(field,value):
    p=protocol();p[field]=value
    with pytest.raises(ValueError):bank144_inventory(p)


def test_no_score_based_source_drop_or_duplicate():
    p=protocol();p['source_groups'][1]=copy.deepcopy(p['source_groups'][0]);p['source_groups'][1]['source_index']=1
    with pytest.raises(ValueError):bank144_inventory(p)
