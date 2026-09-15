"""Task changes must not change reward, authority or actor observations."""
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
import jax.numpy as jp
import pytest
from sttw_control.env import RecoveryEnv,load_config
from sttw_control.path import integrated_reference,ReferencePaths

CONFIG=Path(__file__).parents[1]/'configs'


def test_integrated_reference_preserves_speed_yaw_intent():
    t=integrated_reference([[0,2.,.5]],1.,count=121)
    # Extra reference tail is five seconds, so total arc=12, yaw=3.
    np.testing.assert_allclose(t[-1,:4],[12,4*np.sin(3),4*(1-np.cos(3)),3],atol=1e-10)
    assert np.all(t[:,4]==.25)


def test_only_reference_task_changes():
    old=asdict(load_config(CONFIG/'path_priority_recovery.json'));new=asdict(load_config(CONFIG/'path_reference_recovery.json'))
    assert {k for k in old if old[k]!=new[k]}=={'reference_paths','speed_schedule'}
    assert new['observation']['history_steps']==10


def test_reference_reset_history_and_reward_match_selected_case():
    cfg=load_config(CONFIG/'path_reference_recovery.json');env=RecoveryEnv(cfg)
    for i in range(4):
        state=env.reset(3,reference_id=i)
        assert int(state.path_id)==i and state.obs.shape==(280,)
        frozen=RecoveryEnv(replace(cfg,reference_paths=replace(cfg.reference_paths,selected=i)))
        other=frozen.reset(3)
        np.testing.assert_array_equal(state.obs,other.obs)
        state=env.step(state,jp.zeros(2));other=frozen.step(other,jp.zeros(2))
        np.testing.assert_allclose(state.obs,other.obs,atol=1e-6)
        assert float(state.reward)==pytest.approx(float(sum(state.tracking_components.values())),abs=1e-6)
        # Real command schedules differ without changing the input schema.
        expected=cfg.reference_paths.cases[i]['commands'][1][1]
        tick=round(cfg.reference_paths.cases[i]['commands'][1][0]/cfg.controller.dt)
        assert float(env.speed_command(tick,i))==pytest.approx(expected)


def test_reference_selection_is_not_revealed_as_an_actor_feature():
    cfg=load_config(CONFIG/'path_reference_recovery.json')
    from sttw_control.observation import observation_fields
    assert 'path_id' not in observation_fields(cfg.observation)
    with pytest.raises(ValueError):ReferencePaths(cases=cfg.reference_paths.cases,selected=4)


def test_crossing_projection_stays_on_current_branch_and_allows_reverse():
    from sttw_control.path import bend_features,features_at_progress
    table=jp.asarray([[0,-1,0,0,0],[2,1,0,0,0],[4,0,-1,1.57,0],[6,0,1,1.57,0]],float)
    pose=jp.array([.01,.02,0.])
    _,global_s=bend_features(pose,table)
    assert float(global_s)>4
    features,s=bend_features(pose,table,jp.asarray(1.),jp.asarray(.1))
    assert float(s)==pytest.approx(1.01,abs=1e-5)
    np.testing.assert_allclose(features,features_at_progress(pose,table,s),atol=1e-5)
    _,back=bend_features(jp.array([-.04,0.,0.]),table,s,jp.asarray(.1))
    assert float(back)<float(s)
    _,edge=bend_features(jp.array([100.,0.,0.]),table,s,jp.asarray(.1))
    assert float(edge)<=float(s)+.10001
