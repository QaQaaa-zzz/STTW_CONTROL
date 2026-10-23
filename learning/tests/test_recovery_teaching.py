import numpy as np
import pytest
from sttw_control.recovery_teaching import teaching_enabled, extract_teaching, select_training_rows


def test_opt_in_requires_training_contract():
    assert not teaching_enabled({})
    with pytest.raises(ValueError):
        teaching_enabled({'teaching_records': {'role': 'DEV', 'observation_dim': 351}})
    assert teaching_enabled({'teaching_records': {'role': 'TRAIN', 'observation_dim': 351}, 'recovery_intervention': {'declared': True}})


def test_pre_action_alignment_partial_terminal_and_parent_separate():
    d=dict(time=np.arange(8)*.005,active_tick=np.array([1,1,1,1,1,0,0,0],bool),
        teaching_observation=np.repeat(np.array([[11.]*351,[22.]*351]),4,axis=0),
        teaching_target=np.repeat([[0.,.7],[.2,.8]],4,axis=0),
        teaching_parent=np.repeat([[.5,.7],[.2,.8]],4,axis=0))
    d['teaching_observation'][5:]=np.nan
    got=extract_teaching(d)
    np.testing.assert_allclose(got['time'],[0,.02])
    np.testing.assert_allclose(got['observations'][:,0],[11,22])
    np.testing.assert_allclose(got['targets'][:,0],[0,.2])
    np.testing.assert_allclose(got['parent_means'][:,0],[.5,.2])
    d['teaching_observation'][4,0]=np.nan
    with pytest.raises(ValueError):extract_teaching(d)


def test_selection_keeps_regression_controls_not_failed_parent_actions():
    common=dict(route='left',speed=2.3,alpha=0.,onset_s=1.7,domain_failure=False,physical_failure=False)
    rows=[]
    for force,base_ok,teacher_ok,gain in [(24,False,True,1),(-24,True,False,1),(16,False,True,-1)]:
        rows.extend([dict(common,force_N=force,policy='control',task_recovered=base_ok,logged_return=-10),
                     dict(common,force_N=force,policy='speed_off',task_recovered=teacher_ok,logged_return=-10+gain)])
    recovery,retention=select_training_rows(rows)
    assert [r['force_N'] for r in recovery]==[24]
    assert [r['force_N'] for r in retention]==[-24]
    with pytest.raises(ValueError):select_training_rows(rows+[rows[0]])
