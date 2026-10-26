import numpy as np
from sttw_control.entry_history_confirmation import transform_path,inventory


def test_fixed_task_transform_preserves_geometry_and_original():
    path=dict(s=np.array([0.,1.,2.]),xy=np.array([[4.,3.],[5.,3.],[6.,3.]]),
              heading=np.zeros(3),curvature=np.zeros(3),goal=np.array(2.),turn_end=np.array(0.))
    result,rotation=transform_path(path,[4.,3.,0.],[8.,2.,np.pi/2])
    np.testing.assert_allclose(result['xy'],[[8.,2.],[8.,3.],[8.,4.]])
    np.testing.assert_array_equal(path['xy'],[[4.,3.],[5.,3.],[6.,3.]])
    for key in ['s','curvature','goal','turn_end']:np.testing.assert_array_equal(result[key],path[key])
    np.testing.assert_allclose(rotation@np.array([0.,1.]),[-1.,0.],atol=1e-12)


def test_shared_events_and_complete_three_arm_inventory():
    contexts={f'p{i}__g{g}':dict(geometry_name=f'g{g}',prefix_name=f'p{i}',world_direction=[0.,1.,0.],speed_m_s=1.1) for i in range(6) for g in range(2)}
    cases=inventory(contexts,seed=20261026113)
    assert len(cases)==540 and len({c['case_id'] for c in cases})==540
    cells={}
    for c in cases:cells.setdefault((c['context'],c['force_N']),[]).append(c)
    assert len(cells)==60
    for rows in cells.values():
        assert len(rows)==9 and len({r['pulse_onset_task_tick'] for r in rows})==1
        assert all(560<=r['pulse_onset_task_tick']<=960 for r in rows)
        assert {r['alpha'] for r in rows}=={0.,.5,1.}
        assert {tuple(r['source_keep_channels']) for r in rows}=={(False,False),(False,True),(True,True)}
