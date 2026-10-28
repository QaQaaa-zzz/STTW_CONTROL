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
