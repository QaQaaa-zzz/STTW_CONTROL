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
