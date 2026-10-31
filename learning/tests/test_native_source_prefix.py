"""Seven nominal training-source prefixes, with no new plant simulation."""
import json
from pathlib import Path
import pytest
from sttw_control.native_goal_run import native_source_prefix_inventory,planning_chunk_steps,snapshot_block_for_task,five_alpha_label_inventory


def config():return json.loads((Path(__file__).parents[1]/'configs/h75_bank4_source_prefix.json').read_text())


def test_prefix_has_only_frozen_h75_same_seven_routes_and_bounded_six_seconds():
    p=config();rows=native_source_prefix_inventory(p)
    original=json.loads((Path(__file__).parents[1]/'configs/middle_selection.json').read_text())
    assert p['route_contexts']==original['route_contexts']
    assert len(rows)==7 and {r['mode'] for r in rows}=={'H75'}
    assert {r['bank_index'] for r in rows}=={4}
    assert all(r['alpha']==r['query_alpha']==.5 and r['force_N']==0 for r in rows)
    steps=planning_chunk_steps(10.,p['collection_s'])
    assert steps==[5]*60 and sum(steps)*4*len(rows)==8400
    assert [snapshot_block_for_task(dict(chunk_upper_steps=steps),t) for t in p['capture_task_s']]==[9,19,39,59]
    assert p['horizon_s']==10 and p['collection_s']==6


@pytest.mark.parametrize('key,value',[('alpha',1.),('collection_s',10.),('horizon_s',6.),('maximum_lower_ticks',8401),('bank_indices',[6]),('planning_frequency_hz',1.),('model_id','another_model')])
def test_prefix_rejects_permission_budget_or_history_changes(key,value):
    p=config();p[key]=value
    with pytest.raises(ValueError):native_source_prefix_inventory(p)


def test_seven_lane_prefix_is_a_valid_declarative_label_source():
    prefix=config();source_cases=native_source_prefix_inventory(prefix)
    p=json.loads((Path(__file__).parents[1]/'configs/h75_five_alpha_tolerance_labels.json').read_text())
    for g in p['source_groups']:
        c=source_cases[g['source_index']%7]
        g.update(source_lane=g['source_index']%7,source_case_id=c['case_id'],route=c['route'],path_key=c['path_key'],bank_index=4)
    rows=five_alpha_label_inventory(p,source_cases,source_plan=dict(chunk_upper_steps=[5]*60,models={'H75':{'registry_id':p['model_id']}}))
    assert len(rows)==280 and {r['bank_index'] for r in rows}=={4}
    assert {r['source_lane'] for r in rows}==set(range(7))
    assert all('bank4/' in r['source_group_identity'] for r in rows)
