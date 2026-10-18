"""Strict fork allowlist and extension counterexamples; no simulator steps."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.path_command_fork import validate_fork_contract, extension_decision

ROOT=Path(__file__).resolve().parents[2]

def fork_inputs():
    parent=ROOT/'runs/path_reward_peak_review_20261011'
    manifest=json.loads((parent/'manifest.json').read_text())
    spec=json.loads((ROOT/'docs/plans/path_coverage_v2/path_coverage_iteration_v2.json').read_text())
    cfg=copy.deepcopy(manifest['config']);cfg['training_sampler']=spec['sampler']
    cfg['training_future_phase_C']['validation_updates']=[50,100,150,200]
    plan=copy.deepcopy(manifest['plan']);plan.update(design_config='new.json',updates=100,sampler_version='demand_exposure_v2')
    plan['sampler_resolution']['training_sampler']=spec['sampler']
    plan['fork']=dict(parent_run=str(parent),identities=str(ROOT/'docs/evidence/path_feedback_v1_20261010/phase_C_results_20261011/paired/identities.json'),expected_parents=spec['parent'],lower_actor_sha256=spec['frozen']['lower_actor_sha256'])
    return plan,cfg,manifest['source']


def test_original_parent_identity_and_exact_whitelist():
    plan,cfg,source=fork_inputs();receipt=validate_fork_contract(plan,cfg,source)
    assert receipt['identities']['alpha0']['source_update']==100
    assert receipt['identities']['alpha1']['source_update']==150
    assert receipt['original_identities']['alpha0']['config_key']!=__import__('sttw_control.preference_best',fromlist=['config_hash']).config_key(cfg)
    for field in ('seed','best_resolution','terminal_contract'):
        bad=copy.deepcopy(plan);bad[field]='mutated'
        with pytest.raises(ValueError):validate_fork_contract(bad,cfg,source)
    bad=copy.deepcopy(cfg);bad['timing']['lower_dt_s']=.006
    with pytest.raises(ValueError):validate_fork_contract(plan,bad,source)
    bad=copy.deepcopy(source);bad['seed']=-1
    with pytest.raises(ValueError):validate_fork_contract(plan,cfg,bad)
    bad=copy.deepcopy(plan);bad['sampler_resolution']['ordinary_fraction']=.2
    with pytest.raises(ValueError):validate_fork_contract(bad,cfg,source)


def gate_inputs():
    names=['straight_2.0','straight_2.6','left90_R2_2.0','left90_R2_2.6','right90_R2_2.0','right90_R2_2.6']
    cfg={'timing':{'lower_dt_s':.005},'stage_B_baseline':{'working_roll_rad':.302},'stage_C_evaluation':{'route_ids':['straight','left90_R2','right90_R2'],'low_speed_m_s':2.,'conflict_speed_m_s':2.6}}
    summary={'all_declared_traces_complete':True,'task_failures':0,'working_failures':2,'goal_hold_failures':0,'cases':{n:{'goal':True,'final_hold':True} for n in names}}
    traces={n:{'chi':np.ones(20),'path_curvature':np.ones(20)*.5,'path_cross_track':np.ones(20)*.2,'actual_forward_speed':np.ones(20)*2.7,'v_user':np.ones(20)*2.6,'peak_roll':np.ones(20)*.32} for n in names}
    current=copy.deepcopy(traces)
    for d in current.values():d['path_cross_track']*=.9
    previous={'window_updates':25,'engineering_failure_rate':0.,'domain_failure_rate':0.,'windows':{'strong_demand':{'samples':100,'lateral_rmse_m':.2,'speed_rmse_m_s':.1}}}
    latest=copy.deepcopy(previous);latest['windows']['strong_demand']['lateral_rmse_m']=.18
    return [summary,copy.deepcopy(summary),traces,current,previous,latest,.06,0,cfg]


def test_extension_requires_all_gates_and_keeps_counterexamples():
    args=gate_inputs();assert extension_decision(*args)['allow']
    for mutate in [lambda a:a.__setitem__(6,.049),lambda a:a[1]['cases']['straight_2.0'].__setitem__('final_hold',False),lambda a:a[5].__setitem__('domain_failure_rate',.01),lambda a:a[5]['windows']['strong_demand'].__setitem__('lateral_rmse_m',.21),lambda a:a[3]['left90_R2_2.6'].__setitem__('peak_roll',np.ones(20)*.33),lambda a:a[1].__setitem__('task_failures',1)]:
        altered=copy.deepcopy(args);mutate(altered)
        assert not extension_decision(*altered)['allow']


def test_fewer_work_failures_cannot_hide_worse_safety_peak():
    args=gate_inputs();args[1]['working_failures']=1;args[3]['left90_R2_2.6']['peak_roll'][:]=.4
    assert not extension_decision(*args)['allow']


def test_empty_or_nonfinite_evidence_fails_closed():
    args=gate_inputs();args[5]['windows']['strong_demand']['samples']=0
    assert not extension_decision(*args)['allow']
    args=gate_inputs();args[3]['left90_R2_2.6']['path_cross_track'][:]=np.nan
    assert not extension_decision(*args)['allow']


def test_restore_original_complete_learner_rng_without_updates():
    import random
    import torch
    from tensordict import TensorDict
    from sttw_control.path_command_training import make_algorithm
    from sttw_control.path_command_fork import restore_fork
    plan,cfg,source=fork_inputs();receipt=validate_fork_contract(plan,cfg,source)
    obs=TensorDict({'policy':torch.zeros(4,351),'critic':torch.zeros(4,352)},batch_size=[4])
    algo=make_algorithm(obs,2,cfg,'cpu')
    for alpha in (0,1):
        saved=restore_fork(algo,receipt,alpha)
        assert saved['update']==(100 if alpha==0 else 150)
        for key,value in saved['policy'].items():torch.testing.assert_close(algo.policy.state_dict()[key],value,rtol=0,atol=0)
        restored=algo.optimizer.state_dict()
        assert restored['param_groups']==saved['optimizer']['param_groups']
        for key,state in saved['optimizer']['state'].items():
            for field,value in state.items():torch.testing.assert_close(restored['state'][key][field],value,rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),saved['torch_rng'],rtol=0,atol=0)
        assert random.getstate()==saved['python_rng']
        np.testing.assert_array_equal(np.random.get_state()[1],saved['numpy_rng'][1])


def test_extension_curvature_fallback_uses_same_window_definition():
    args=gate_inputs();args[2]['left90_R2_2.6']['chi'][:]=0
    result=extension_decision(*args)
    assert result['allow']
    assert result['evidence']['high_speed_main_error']['cases']['left90_R2_2.6']['window']=='abs(path_curvature)>1e-4 for both traces'
    args[2]['left90_R2_2.6']['path_curvature'][:]=0
    assert not extension_decision(*args)['allow']


def test_declared_other_sampler_fields_cannot_drift():
    plan,cfg,source=fork_inputs()
    for field in ('network','reward','follower','filter'):
        if field not in cfg:continue
        bad=copy.deepcopy(cfg);bad[field]={'changed':True}
        with pytest.raises(ValueError):validate_fork_contract(plan,bad,source)
    for field in ('checkpoint_sha256','actor_sha256','source_update'):
        bad=copy.deepcopy(plan);bad['fork']['expected_parents']['alpha0'][field]='wrong'
        with pytest.raises(ValueError):validate_fork_contract(bad,cfg,source)
