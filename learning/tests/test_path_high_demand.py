"""High-demand configuration changes only the declared task/reward profile."""
from pathlib import Path
import json
import jax,jax.numpy as j
import numpy as np
import pytest
from sttw_control.path_command_training import load_plan,prepare,check_conditional_dev_budget
from sttw_control.path_command_scenarios import sample_training_route
from sttw_control.path_command_selection import cases
from sttw_control.path_command_reward import costs,failure_reward
from sttw_control.geometric_path import build_path
ROOT=Path(__file__).parents[2];PLAN=ROOT/'learning/configs/path_high_demand_shared_run.json'


def test_profile_load_and_preparation_fresh_baseline_budget(tmp_path):
    plan,cfg,source=load_plan(PLAN)
    assert source['candidate']['source_update']==300 and plan['baseline_parent'] is None
    assert cfg['training_sampler']['version']=='demand_exposure_v3'
    assert cfg['path']['turn_radius_m']==1.6
    assert cfg['residual_interface']['speed_reference_bounds_m_s']==[1.5,3.5]
    assert cfg['residual_interface']['correction_speed_down_m_s2']==1.
    assert cfg['stage_C_evaluation']['route_ids']==['straight','left90','right90']
    assert {v for _,_,v in cases(cfg)}=={2.8,3.2}
    declaration=prepare(PLAN,tmp_path/'declaration')
    assert declaration['maximum_baseline_lower_ticks']==24000
    assert declaration['maximum_validation_lower_ticks']==624000
    assert declaration['complete_experiment_lower_tick_budget']==53053312
    check_conditional_dev_budget(dict(validation_control_ticks_total=504000),cfg,plan['alphas'])
    with pytest.raises(RuntimeError):check_conditional_dev_budget(dict(validation_control_ticks_total=504001),cfg,plan['alphas'])


def test_train_sampler_and_test_geometry_use_tighter_faster_profile():
    _,cfg,_=load_plan(PLAN)
    sample=jax.jit(jax.vmap(lambda eid:sample_training_route(7,eid,0,j.zeros(3),cfg)))(j.arange(512))
    np.testing.assert_array_equal(np.bincount(sample.sample_group),[102,204,206])
    assert np.all((np.asarray(sample.speed[:102])>=2.6)&(np.asarray(sample.speed[:102])<=3.2))
    assert np.all((np.asarray(sample.speed[102:])>=2.8)&(np.asarray(sample.speed[102:])<=3.2))
    assert np.all((np.asarray(sample.radius[102:])>=1.6)&(np.asarray(sample.radius[102:])<=1.8))
    for name in ['left90','right90']:
        p=build_path(name,np.zeros(3),cfg)
        assert float(abs(p.curvature).max())==pytest.approx(1/1.6)
    gentle=(np.asarray(sample.family[:102])==0)&(abs(np.asarray(sample.angle[:102]))>0)
    assert gentle.any() and np.all(np.asarray(sample.radius[:102])[gentle]==3.)


def test_posture_penalties_double_including_caps_and_failure_bound():
    _,cfg,_=load_plan(PLAN)
    old=json.loads((ROOT/'learning/configs/STTW_Path_Coverage_Shared.json').read_text())
    args=dict(chi=1.,ev=0.,ey=0.,eh=0.,roll_rate=0.,speed=3.,offsets=j.zeros(2),offset_rate=j.zeros(2))
    for alpha in [0.,.5,1.]:
        for roll in [.26,.32,.4,.69]:
            a=costs(alpha=alpha,peak_roll=roll,config=old,**args);b=costs(alpha=alpha,peak_roll=roll,config=cfg,**args)
            for component in ['roll','working_roll']:
                assert float(b['effective'][component])==pytest.approx(2*float(a['effective'][component]),rel=1e-6,abs=1e-6)
    assert cfg['path_reward']['failure_cost_rate_bound']==pytest.approx(8980.2)
    assert sum(cfg['path_reward']['component_caps'].values())==pytest.approx(8980.2)
    assert float(failure_reward(100,cfg))<float(failure_reward(100,old))


def test_engineering_declaration_has_only_its_own_short_budget(tmp_path):
    engineering=ROOT/'learning/configs/path_high_demand_shared_engineering.json'
    plan,_,_=load_plan(engineering)
    assert plan['budget']['training_policy_transitions']==128
    assert plan['budget']['training_lower_ticks']==512
    assert plan['budget']['total_lower_ticks']==512
    d=prepare(engineering,tmp_path/'engineering')
    assert d['actual_batch_dimensions']==[8,16]
    assert d['training_lower_ticks_total']==512
    assert d['maximum_validation_lower_ticks']==0
    assert d['complete_experiment_lower_tick_budget']==512
