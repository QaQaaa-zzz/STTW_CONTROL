"""Fixed physical-event matching and lower-time flattening, without physics."""
from collections import defaultdict
import numpy as np
from sttw_control.native_goal_run import inventory,flatten


def test_all_eighty_cases_match_world_events_across_methods_and_alpha():
    rows=inventory()
    assert len(rows)==len({r['case_id'] for r in rows})==80
    groups=defaultdict(list)
    for row in rows:groups[row['route'],row['event']].append(row)
    assert len(groups)==10
    for event in groups.values():
        assert {(r['mode'],r['alpha']) for r in event}=={
            (mode,a) for mode in ('H75','current','zero','equilibrium') for a in (0.,1.)}
        assert len({(r['speed'],r['force_N'],r['onset_s'],r['duration_s']) for r in event})==1
    for route in {r['route'] for r in rows}:
        events={(r['onset_s'],r['force_N']) for r in rows if r['route']==route}
        assert events=={(1.6,0.),(1.6,-32.),(1.6,32.),(2.6,-32.),(2.6,32.)}


def test_flatten_preserves_each_lane_and_all_four_lower_timestamps():
    source=np.arange(3*80*4*2).reshape(3,80,4,2)
    result=flatten(dict(example=source))['example']
    assert result.shape==(12,80,2)
    for upper in range(3):
        for lower in range(4):
            np.testing.assert_array_equal(result[upper*4+lower],source[upper,:,lower])


def test_response_inventory_is_fixed_54_with_six_fresh_nominals_and_world_y_pairs():
    from sttw_control.native_goal_run import response_inventory,response_protocol
    rows=response_inventory()
    assert len(rows)==len({r['case_id'] for r in rows})==54
    assert sum(r['force_N']==0 for r in rows)==6
    assert response_protocol()['maximum_lower_ticks']==54*2000
    groups=defaultdict(list)
    for r in rows:groups[r['route'],r['event']].append(r)
    assert len(groups)==18
    for group in groups.values():
        assert {(r['speed_alpha'],r['steer_alpha']) for r in group}=={(.5,.5),(1.,1.),(1.,.5)}
        assert len({(r['onset_s'],r['force_N'],r['duration_s']) for r in group})==1
        assert {r['speed'] for r in group}=={2.8}
    assert {r['force_N'] for r in rows}=={0.,-64.,-32.,32.,64.}
    assert {r['onset_s'] for r in rows}=={1.6,4.}


def test_channel_query_composition_changes_only_alpha_and_does_not_advance_state():
    import jax.numpy as j
    from sttw_control.native_goal_run import channel_queries
    obs=j.arange(351,dtype=float);queries=[]
    def actor(x):
        queries.append(np.asarray(x));return j.array([x[336]*2,x[336]*3])
    z,half,full=channel_queries(obs,actor,1.,.5)
    np.testing.assert_allclose(z,[2.,1.5])
    assert len(queries)==2
    np.testing.assert_array_equal(np.delete(queries[0],336),np.delete(queries[1],336))
    np.testing.assert_array_equal(np.delete(queries[0],336),np.delete(np.asarray(obs),336))
    np.testing.assert_array_equal(half,[1.,1.5]);np.testing.assert_array_equal(full,[2.,3.])


def test_response_pulses_are_world_y_and_use_native_physical_clock():
    from sttw_control.native_goal_run import response_inventory,force_pulses
    rows=response_inventory();pulses=np.asarray(force_pulses(rows,3.25))
    np.testing.assert_array_equal(pulses[:,[2,4]],0)
    np.testing.assert_array_equal(pulses[:,3],[r['force_N'] for r in rows])
    np.testing.assert_allclose(pulses[:,0],[3.25+r['onset_s'] for r in rows])
    np.testing.assert_allclose(pulses[:,1]-pulses[:,0],.1)


def test_response_metadata_is_extensible_but_budget_contract_is_not():
    from sttw_control.native_goal_run import response_protocol,response_inventory
    import pytest
    protocol=response_protocol();protocol['review_note']='additional descriptive provenance'
    assert len(response_inventory(protocol))==54
    protocol['maximum_lower_ticks']+=1
    with pytest.raises(ValueError,match='maximum_lower_ticks'):
        response_inventory(protocol)


def test_response_original_5001_support_matches_replanner_and_legacy_remains_4001():
    from sttw_control.native_goal_run import response_protocol,extend_straight_support
    from sttw_control.goal_replanning import dubins_goal_path
    p=response_protocol()
    original,_=dubins_goal_path([0.,0.,0.],[8.,3.],length_m=p['planner_support_m'],spacing_m=p['planner_spacing_m'])
    supported=extend_straight_support(original,p['planner_points'],p['planner_spacing_m'])
    assert supported is original
    assert len(original.s)==p['planner_points']==5001
    replanned,_=dubins_goal_path([1.,2.,.2],[8.,3.],length_m=p['planner_support_m'],spacing_m=p['planner_spacing_m'])
    assert len(replanned.s)==len(supported.s)
    legacy,_=dubins_goal_path([0.,0.,0.],[8.,3.])
    assert len(legacy.s)==4001
    extended=extend_straight_support(legacy,p['planner_points'],p['planner_spacing_m'])
    for field in ('s','xy','heading','curvature'):
        np.testing.assert_array_equal(np.asarray(getattr(extended,field))[:4001],np.asarray(getattr(legacy,field)))
    assert float(extended.goal)==float(legacy.goal)
    np.testing.assert_allclose(np.asarray(extended.s)[4001:]-float(legacy.s[-1]),np.arange(1,1001)*.02,atol=4e-6)


def test_switching_inventory_has_bounded_36_and_origin_state_alpha_one():
    from sttw_control.native_goal_run import switching_protocol,switching_inventory
    rows=switching_inventory(switching_protocol())
    assert len(rows)==len({r['case_id'] for r in rows})==36
    assert {r['alpha'] for r in rows}=={1.}
    assert {r['switch_s'] for r in rows}=={1.,1.4,1.6}
    groups=defaultdict(list)
    for r in rows:groups[r['route'],r['origin']].append(r)
    assert len(groups)==4 and all(len(v)==9 for v in groups.values())
    assert sum(r['force_N']==0 for r in rows)==12


def test_switch_mapping_only_changes_declared_query_pairs_at_each_upper_boundary():
    from sttw_control.native_goal_run import switch_mapping
    for tick in (200,280,320):
        before=switch_mapping(tick-4,1.,.5,tick,[.5,.5])
        at=switch_mapping(tick,1.,.5,tick,[.5,.5])
        np.testing.assert_array_equal(np.asarray(before),[1.,.5,False])
        np.testing.assert_array_equal(np.asarray(at),[.5,.5,True])


def test_capture_validates_all_state_leaves_across_nine_lanes_and_rejects_hidden_history_difference():
    from sttw_control.native_goal_run import switching_protocol,switching_inventory,assert_shared_switch_carry
    import pytest
    rows=switching_inventory(switching_protocol())
    groups={(r['route'],r['origin']):i for i,r in enumerate(rows)}
    histories=np.array([[groups[r['route'],r['origin']]]*7 for r in rows])
    carry=dict(tick=np.full(36,200),hidden_history=histories,clean=np.zeros(36))
    receipt=assert_shared_switch_carry(carry,rows)
    assert receipt['leaf_count']==3 and receipt['groups']==4
    histories[1,4]+=1
    with pytest.raises(ValueError,match='complete task1.0 carry'):
        assert_shared_switch_carry(carry,rows)


def test_stay_prefix_audit_uses_saved_raw_and_rejects_physical_mismatch(tmp_path):
    from sttw_control.native_goal_run import audit_stay_prefix
    import pytest
    cases=[dict(route='S',origin='fixed1',force_N=0.)]*2
    path=tmp_path/'stay.npz';np.savez(path,time=np.arange(200)*.005,actual_xy=np.zeros((200,2)))
    logs=dict(time=np.broadcast_to((np.arange(200)*.005)[:,None],(200,2)).copy(),actual_xy=np.zeros((200,2,2)))
    cached=[dict(cases[0],raw_file=str(path))]
    p=dict(prefix_comparison=dict(fields=['time','actual_xy']))
    assert audit_stay_prefix(logs,cases,cached,p)['all_fields_within_tolerance']
    logs['actual_xy'][100,1,0]=.001
    with pytest.raises(ValueError,match='prefix mismatch'):
        audit_stay_prefix(logs,cases,cached,p)


def test_own_switch_prefix_checks_280_tick_preparation_with_predeclared_float_tolerance(tmp_path):
    from sttw_control.native_goal_run import audit_stay_prefix
    import pytest
    path=tmp_path/'stay.npz';np.savez(path,time=np.arange(400)*.005,actual_xy=np.zeros((400,2)),active_tick=np.ones(400,bool))
    cases=[dict(route='S',origin='hybrid',force_N=-64.,switch_s=1.4)]
    cached=[dict(cases[0],raw_file=str(path))]
    logs=dict(time=(np.arange(400)*.005)[:,None],actual_xy=np.zeros((400,1,2)),active_tick=np.ones((400,1),bool))
    p=dict(prefix_comparison=dict(fields=list(logs),absolute_tolerance=1e-5,relative_tolerance=1e-6))
    logs['actual_xy'][279,0,0]=5e-6;logs['actual_xy'][280,0,0]=1.
    r=audit_stay_prefix(logs,cases,cached,p,own_switch=True)
    assert r['prefix_lower_ticks']==[280] and r['maximum_absolute_difference']['actual_xy']==5e-6
    logs['active_tick'][279,0]=False
    with pytest.raises(ValueError,match='prefix mismatch'):
        audit_stay_prefix(logs,cases,cached,p,own_switch=True)


def test_continuation_keeps_original36_inventory_and_counts_only_remaining_physics():
    from pathlib import Path
    import json
    from sttw_control.native_goal_run import continuation_inventory,switching_inventory,switching_protocol
    p=json.loads((Path(__file__).resolve().parents[1]/'configs/h75_switching_continuation.json').read_text())
    assert continuation_inventory(p)==switching_inventory(switching_protocol())
    assert p['maximum_lower_ticks']==36*9*200
    assert p['maximum_lower_ticks']+p['consumed_source_lower_ticks']==79200
    assert p['prefix_comparison']['field_absolute_tolerance']=={'phi_dot':1e-4}


def test_schedule_audit_rejects_switch_using_wrong_boundary():
    from sttw_control.native_goal_run import audit_scheduled_pairs
    import pytest
    cases=[dict(switch_s=1.4,speed_alpha=1.,steer_alpha=.5)]
    logs=dict(active_tick=np.ones((8,1),bool),time=(np.arange(276,284)*.005)[:,None],channel_alpha_pair=np.array([[[1.,.5]]]*4+[[[.5,.5]]]*4),switch_active=np.array([[False]]*4+[[True]]*4))
    assert audit_scheduled_pairs(logs,cases)['actual_pairs_and_switch_timing_verified']
    logs['channel_alpha_pair'][3,0]=[.5,.5]
    with pytest.raises(ValueError,match='pair/timing'):
        audit_scheduled_pairs(logs,cases)


def test_schedule_audit_checks_world_y_force_sign_and_declared_interval():
    from sttw_control.native_goal_run import audit_scheduled_pairs
    import pytest
    cases=[dict(switch_s=1.6,speed_alpha=1.,steer_alpha=.5,force_N=-64.,onset_s=1.6,duration_s=.1)]
    logs=dict(active_tick=np.ones((4,1),bool),time=(np.arange(320,324)*.005)[:,None],channel_alpha_pair=np.full((4,1,2),.5),switch_active=np.ones((4,1),bool),applied_force_impulse=np.tile([0.,-.32,0.],(4,1,1)))
    assert audit_scheduled_pairs(logs,cases)['worldY_force_vector_verified']
    logs['applied_force_impulse'][0,0,1]=.32
    with pytest.raises(ValueError,match='sign/timing'):
        audit_scheduled_pairs(logs,cases)


def test_descriptive_float_prefix_reports_exceedance_but_still_rejects_discrete(tmp_path):
    from sttw_control.native_goal_run import audit_stay_prefix
    import pytest
    file=tmp_path/'stay.npz';np.savez(file,time=np.arange(4)*.005,phi=np.zeros(4),active_tick=np.ones(4,bool))
    cases=[dict(route='S',origin='fixed1',force_N=0.)];cache=[dict(cases[0],raw_file=str(file))]
    logs=dict(time=(np.arange(4)*.005)[:,None],phi=np.full((4,1),.01),active_tick=np.ones((4,1),bool))
    protocol=dict(prefix_comparison=dict(fields=list(logs),absolute_tolerance=1e-5,relative_tolerance=1e-6,floating_descriptive_only=True))
    r=audit_stay_prefix(logs,cases,cache,protocol)
    assert not r['all_fields_within_tolerance'] and r['exceeded_lane_field_counts']=={'phi':1}
    assert r['maximum_absolute_difference']['phi']==.01
    logs['active_tick'][0,0]=False
    with pytest.raises(ValueError,match='prefix mismatch'):
        audit_stay_prefix(logs,cases,cache,protocol)


def test_online_scalar_inventory_is18_and_preserves_world_pulse_pairing():
    import json
    from pathlib import Path
    from sttw_control.native_goal_run import online_inventory,force_pulses
    p=json.loads((Path(__file__).resolve().parents[1]/'configs/h75_online_scalar.json').read_text())
    rows=online_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==18
    assert {r['alpha'] for r in rows}=={1.}
    assert sum(r['force_N']==0 for r in rows)==2
    assert p['maximum_lower_ticks']==18*10*200
    assert {r['mode'] for r in rows}=={'online_scalar'}
    pulses=np.asarray(force_pulses(rows,3.))
    np.testing.assert_array_equal(pulses[:,[2,4]],0.)


def test_confirmation100_inventory_unique_paths_and_paired_bank_event_arms():
    import json
    from pathlib import Path
    from sttw_control.native_goal_run import confirmation_inventory,case_path_key
    p=json.loads((Path(__file__).resolve().parents[1]/'configs/h75_online_confirmation100.json').read_text())
    rows=confirmation_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==100
    assert {r['bank_index'] for r in rows}=={0,1}
    assert len({case_path_key(r) for r in rows})==4
    groups=defaultdict(list)
    for r in rows:groups[r['bank_index'],r['route'],r['event']].append(r)
    assert len(groups)==20 and all(len(v)==5 for v in groups.values())
    assert all({r['alpha'] for r in v}=={1.} for v in groups.values())
    assert {r['query_alpha'] for r in rows if r['mode']=='fixed0'}=={0.}
    assert {r['query_alpha'] for r in rows if r['mode']=='fixed_half'}=={.5}
    assert {r['onset_s'] for r in rows}=={2.2,3.4}


def test_existing_speed_floor_is_exact_zero_at_native_nominal_and_bounded_if_needed():
    import jax.numpy as j
    from sttw_control.native_goal_run import existing_floor_latent
    cfg=dict(residual_interface=dict(speed_positive_scale_m_s=.25))
    for speed in (2.,2.3,2.8):
        z,log=existing_floor_latent(j.array([speed,.2]),cfg)
        np.testing.assert_array_equal(z,[0.,0.]);assert not bool(log['support_active'])
    z,log=existing_floor_latent(j.array([1.4,.2]),cfg)
    np.testing.assert_allclose(np.tanh(float(z[0]))*.25,.1,atol=1e-7)
    assert z[1]==0 and bool(log['support_active'])
    z,log=existing_floor_latent(j.array([1.,.2]),cfg)
    assert float(log['support_realized_target_offset'])<.25 and np.isfinite(np.asarray(z)).all()
