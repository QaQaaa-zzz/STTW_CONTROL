"""CPU interface checks: conflict requests and diagnostic budget isolation."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import jax.numpy as j
from sttw_control.path_command_training import PathBatch,check_conditional_dev_budget
ROOT=Path(__file__).parents[2]

def config():return json.loads((ROOT/'learning/configs/STTW_Path_Conflict_Exposure.json').read_text())

def test_budget_lightweight_is_not_full_candidate():
    cfg=config();s={'validation_control_ticks_total':720000}
    assert check_conditional_dev_budget(s,cfg,[0,1],case_count=4)==32000
    with pytest.raises(RuntimeError):check_conditional_dev_budget(s,cfg,[0,.25,.5,.75,1])

def test_current_speed_observation_never_exposes_future():
    cfg=config()
    class Env:
        def observation(self,state,path,request,horizon):return request
    batch=PathBatch(Env(),None,cfg,911501,None)
    route=SimpleNamespace(path=None,speed=2.3,speed_values=j.array([2.3,2.8,2.3]),speed_switch_times=j.array([1.,5.]))
    for tick,value in [(0,2.3),(199,2.3),(200,2.8),(999,2.8),(1000,2.3)]:
        ep=SimpleNamespace(route=route,state=SimpleNamespace(tick=j.int32(tick)))
        assert float(batch.observation(ep))==pytest.approx(value)

def test_request_refresh_is_explicit_new_version_only():
    from sttw_control.path_command_env import refresh_speed_request
    s=SimpleNamespace(follower=SimpleNamespace(command=j.array([2.3,.1])))
    assert refresh_speed_request(s,j.array(2.8),config()).shape==(2,)
    np.testing.assert_allclose(refresh_speed_request(s,j.array(2.8),config()),[2.8,.1])


def test_active_event_statistics_bind_current_request_and_sampled_stage():
    from test_path_conditional_statistics import conditional
    from sttw_control import path_command_statistics as stats
    logs,route,_=conditional(0.)
    route.sampling_stage=j.int32(3)
    logs['v_user']=j.array([2.,2.8,2.3,np.nan])
    cfg=config();data=stats.step_statistics(logs,route,cfg)
    np.testing.assert_allclose(data['route_speed_moments'],[7.1,2.**2+2.8**2+2.3**2],rtol=1e-6)
    summary=stats.summarize_statistics(data,cfg)
    cross=summary['exposure_cross_statistics']
    assert cross['active_tick_count'][2][2][0]==3
    assert cross['strong_demand_tick_count'][2][2][0]==1
    assert summary['route_distribution']['speed']['mean']==pytest.approx(7.1/3)


def test_conflict_contract_is_fresh_and_scientific_diagnostics_only():
    import copy
    from sttw_control.path_command_training import load_plan
    from sttw_control.path_command_conditioning import validate_conditional_plan
    plan,cfg,_=load_plan(ROOT/'learning/configs/path_conflict_exposure_run.json')
    assert plan['initialization']==dict(mode='scratch',parent_alpha=None,checkpoint=None,checkpoint_sha256=None)
    assert cfg['training_future_phase_C']['validation_updates']==[0,100,200]
    assert cfg['training_future_phase_C']['lightweight_validation_updates']==[50,150]
    bad=copy.deepcopy(plan);bad['scientific_threshold_policy']='stop_on_failed_precheck'
    with pytest.raises(ValueError):validate_conditional_plan(bad,cfg)


def test_full_and_light_role_caps_cannot_borrow_across_resume():
    from sttw_control.path_command_training import check_conditional_dev_budget
    from sttw_control.path_command_training import load_plan
    from pathlib import Path
    import pytest
    plan,cfg,_=load_plan(Path(__file__).parents[2]/'learning/configs/path_conflict_exposure_run.json')
    status=dict(validation_control_ticks_total=0,parent_consumed_validation_control_ticks=500000,
        parent_consumed_full_dev_control_ticks=500000,parent_consumed_lightweight_control_ticks=0)
    with pytest.raises(RuntimeError,match='full_dev'):
        check_conditional_dev_budget(status,cfg,plan['alphas'])
    status=dict(validation_control_ticks_total=0,parent_consumed_validation_control_ticks=33000,
        parent_consumed_full_dev_control_ticks=0,parent_consumed_lightweight_control_ticks=33000)
    with pytest.raises(RuntimeError,match='lightweight'):
        check_conditional_dev_budget(status,cfg,[0.,1.],case_count=4,role='lightweight')
