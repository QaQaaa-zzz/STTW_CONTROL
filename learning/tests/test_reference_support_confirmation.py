"""New tasks retain real carry and share events without leaking future pulses."""
import dataclasses
from types import SimpleNamespace

import jax.numpy as j
import numpy as np

from sttw_control.finite_value_evaluation import confirmation_inventory, rebind_path_context, paired_report
from sttw_control.finite_value_evaluation_report import goal_report
from sttw_control.geometric_path import Path


def test_ninety_cases_share_every_event_across_three_controls():
    protocol=dict(seed=2631,alphas=[0.,.5,1.],forces_N=[0.,-16.,16.,-32.,32.],
                  onset_upper_tick_range=[140,241],pulse_duration_s=.1)
    contexts={name:dict(speed_m_s=speed,world_direction=[0.,1.,0.]) for name,speed in [('left',1.2),('right',1.5)]}
    cases=confirmation_inventory(protocol,contexts)
    assert len(cases)==len({c['case_id'] for c in cases})==90
    assert sum(c['force_N']==0 for c in cases)==18
    assert all(560<=c['pulse_onset_task_tick']<=960 and c['pulse_onset_task_tick']%4==0 for c in cases)
    for i in range(0,90,3):
        group=cases[i:i+3]
        assert [c['mode_index'] for c in group]==[0,1,2]
        common=lambda c:{k:v for k,v in c.items() if k not in ['case_id','role','actor_role','mode_index','profile','causal_support']}
        assert common(group[0])==common(group[1])==common(group[2])


def test_candidate_extension_adds_only_missing_policies_with_identical_events():
    from sttw_control.finite_value_evaluation import candidate_extension_cases
    protocol=dict(seed=2631,alphas=[0.,.5,1.],forces_N=[0.,-16.,16.,-32.,32.],
                  onset_upper_tick_range=[140,241],pulse_duration_s=.1)
    contexts={name:dict(speed_m_s=speed,world_direction=[0.,1.,0.]) for name,speed in [('left',1.2),('right',1.5)]}
    original=confirmation_inventory(protocol,contexts)
    candidates=candidate_extension_cases(original,['free_GAE','finite_GAE','free_MC'])
    assert len(candidates)==90
    assert not {c['case_id'] for c in candidates}&{c['case_id'] for c in original}
    byevent={(c['context'],c['alpha'],c['event_id']):c for c in original if c['role']=='parent_plain'}
    for c in candidates:
        baseline=byevent[c['context'],c['alpha'],c['event_id']]
        assert c['mode_index']==0 and c['profile']==0 and not c['causal_support']
        for key in ['force_N','world_direction','pulse_onset_task_tick','pulse_duration_s','requested_speed_m_s']:
            assert c[key]==baseline[key]


def test_exact_cache_join_is_unique_and_immutable(tmp_path):
    import json
    import pytest
    from sttw_control.finite_value_evaluation_report import evaluation_inputs
    from sttw_control.preference_best import digest
    (tmp_path/'physical').mkdir()
    for file,value in [('inventory.json',dict(cases=[dict(case_id='new')])),
                       ('physical/results.json',[dict(case_id='new')]),
                       ('physical/case_index.json',[dict(case_id='new')])]:
        (tmp_path/file).write_text(json.dumps(value))
    cache=tmp_path/'cache_receipt.json'
    payload=dict(declared=[dict(case_id='old')],records=[dict(case_id='old')],indices=[dict(case_id='old')],active_lower_ticks=123)
    cache.write_text(json.dumps(payload))
    plan=dict(declared_cases=1,cached_cases=1,sealed_files={str(cache.resolve()):digest(cache)})
    d,r,i,cost=evaluation_inputs(tmp_path,plan)
    assert len(d)==len(r)==len(i)==2 and cost==123
    payload['declared'][0]['case_id']='new';cache.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='cache receipt changed'):evaluation_inputs(tmp_path,plan)
    plan['sealed_files'][str(cache.resolve())]=digest(cache)
    with pytest.raises(ValueError,match='unique'):evaluation_inputs(tmp_path,plan)


def test_rebind_changes_only_task_projection_and_preview():
    @dataclasses.dataclass(frozen=True)
    class State:
        physical: object; lower: object; history: object; offsets: object
        filtered: object; alpha: object; tick: object; projection: object; follower: object
        def replace(self,**kw):return dataclasses.replace(self,**kw)
    state=State(SimpleNamespace(data=object()),object(),object(),object(),object(),.5,0,None,None)
    config=dict(path=dict(projection_backward_m=.05,projection_forward_floor_m=.3,
        projection_forward_motion_multiplier=2.,projection_forward_padding_m=.1),
        follower=dict(lookahead_time_s=.7,lookahead_min_speed_m_s=1.5,lookahead_min_m=.8,
            lookahead_max_m=1.8,wheelbase_m_expected=.408,caster_deg_expected=25.,
            steer_reference_max_rad=.6,preview_body_x_min_m=.05))
    env=SimpleNamespace(config=config,physics=SimpleNamespace(observe=lambda data:(j.zeros(6),j.array([0.,.05,.01]),j.float32(2.))))
    path=Path(j.array([0.,1.,2.]),j.array([[0.,0.],[1.,0.],[2.,0.]]),j.zeros(3),j.zeros(3),j.float32(1),j.float32(2))
    end=rebind_path_context(state,path,j.float32(1.2),env)
    for name in ['physical','lower','history','offsets','filtered']:
        assert getattr(end,name) is getattr(state,name)
    assert end.alpha==state.alpha and end.tick==0
    assert float(end.projection.cross_track)==np.float32(.05)
    assert not bool(end.follower.invalid)


def test_primary_arrival_and_later_contact_remain_separate():
    rows=[]
    for role,hit,physical in [('parent_plain',False,'survived'),('parent_causal',True,'contact')]:
        rows.append(dict(case_id=role,context='new',alpha=.5,event_id='pulse',force_N=32.,
            role=role,goal_arrived=hit,physical_outcome=physical,scored_cost=10.,
            category='contact' if physical=='contact' else 'safe_task_failed'))
    got=goal_report(rows,['parent_plain','parent_causal'])
    assert got['all']['paired']['parent_causal_vs_parent_plain']['arrival_gains']==1
    assert got['all']['counts']['parent_causal']['contact']==1
    assert paired_report(rows,roles=['parent_plain','parent_causal'])['comparisons']['parent_causal_vs_parent_plain']['physical_losses']==1
