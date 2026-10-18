import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.path_command_cache import _rescore_trace
from sttw_control.path_command_training import load_plan
from sttw_control.path_command_reporting import reward_series
ROOT=Path(__file__).parents[2]

def test_plan_and_independent_rescore_preserve_raw_motion():
    plan,cfg,source=load_plan(ROOT/'learning/configs/path_safety_candidate_run.json')
    assert plan['updates']==50 and plan['initialization']['mode']=='weights_only_reward_fork'
    assert cfg['training_future_phase_C']['validation_updates']==[50]
    d=dict(np.load(ROOT/'runs/path_reward_peak_review_20261011/alpha1/final_best/left90_R2_2.6.npz'))
    copy={k:v.copy() for k,v in d.items()};result=_rescore_trace(d,1,cfg)
    for k in d:np.testing.assert_array_equal(d[k],copy[k])
    for k in ('actual_forward_speed','phi','phi_dot','peak_roll','path_cross_track','path_progress','actual_xy','offsets'):
        np.testing.assert_array_equal(result[k],d[k])
    np.testing.assert_allclose(result['scored_tick_reward'],reward_series(d,1,cfg)[0])
    from sttw_control.path_command_reporting import reconstruct_costs
    reconstructed=reconstruct_costs(d,1,cfg)
    for k,v in reconstructed['raw'].items():np.testing.assert_array_equal(result['raw_cost_'+k],v)
    for k,v in reconstructed['effective'].items():np.testing.assert_array_equal(result['effective_cost_'+k],v)
    assert np.any(result['diagnostic_subcosts_predictive_roll']>0)
    cplan,control,_=load_plan(ROOT/'learning/configs/path_safety_control_run.json')
    assert control.get('path_reward',{}).get('reward_version') is None
    assert cplan['seed']==plan['seed'] and cplan['sampler_resolution']==plan['sampler_resolution']

def test_parent_freeze_rejects_running_stage(tmp_path):
    from sttw_control.path_command_fork import freeze_reward_parents
    (tmp_path/'status.json').write_text('{"state":"running"}')
    with pytest.raises(ValueError,match='completed'):freeze_reward_parents(tmp_path)

def test_initial_reward_candidate_and_final_panel_reuse_end_to_end(tmp_path):
    import torch
    from tensordict import TensorDict
    from sttw_control.path_command_training import make_algorithm,save_checkpoint
    from sttw_control.path_command_fork import load_weights_only
    from sttw_control.path_command_cache import reuse_reward_parent,reuse_selected_validation
    from sttw_control.geometric_path import Path as FixedPath
    from sttw_control.preference_best import digest
    from sttw_control.path_command_policy import config_key
    parent=ROOT/'runs/path_reward_peak_review_20261011'
    manifest=json.loads((parent/'manifest.json').read_text());cfg=json.loads(json.dumps(manifest['config']))
    additions=load_plan(ROOT/'learning/configs/path_safety_candidate_run.json')[1]['path_reward']
    cfg['path_reward']=additions;cfg['task_selection_schema']='path_safety_task_v1'
    record=json.loads((parent/'alpha0/best_task_model.json').read_text())
    identity={k:record[k] for k in ('schema','alpha','lower_actor_sha256','config_key','sampler_key','best_key','seed')}
    identity['config_key']=config_key(cfg)
    obs=TensorDict({'policy':torch.zeros(8,351),'critic':torch.zeros(8,352)},batch_size=[8]);algo=make_algorithm(obs,16,cfg,'cpu')
    load_weights_only(algo,torch.load(record['checkpoint'],map_location='cpu',weights_only=False))
    ck=save_checkpoint(tmp_path/'alpha0/checkpoints',algo,0,identity)
    receipt={'parent_run':str(parent),'parent_manifest':manifest,'parent_manifest_sha256':digest(parent/'manifest.json'),'identities':{'alpha0':record}}
    expected={}
    for route in cfg['stage_C_evaluation']['route_ids']:
        p=dict(np.load(parent/'zero_upper'/f'{route}_2.6_path.npz'))
        expected[route]=FixedPath(*(p[k] for k in ('s','xy','heading','curvature','turn_end','goal')))
    traces=reuse_reward_parent(receipt,tmp_path/'alpha0',0,cfg,identity,ck,expected)
    assert len(traces)==6
    best=json.loads((tmp_path/'alpha0/best_safety_task_model.json').read_text())
    assert best['source_update']==0
    assert len(reuse_selected_validation(tmp_path/'alpha0',best,cfg))==6
    from sttw_control.path_command_cache import inherit_reward_validation
    resumed=tmp_path/'resumed/alpha0'
    assert inherit_reward_validation(tmp_path,resumed,0,cfg,identity)==1
    assert len(reuse_selected_validation(resumed,best,cfg))==6
    # Inherited candidate weights live at their original immutable receipt.
    # A second resume must not assume checkpoint0 exists in the first resume.
    from sttw_control.path_command_training import rescore_saved_candidates
    second=tmp_path/'second/alpha0'
    assert not (resumed/'checkpoints/update_0000.pt').exists()
    rescore_saved_candidates(tmp_path/'resumed',second,0,cfg,identity)
    assert inherit_reward_validation(tmp_path/'resumed',second,0,cfg,identity)==1
    second_best=json.loads((second/'best_safety_task_model.json').read_text())
    assert len(reuse_selected_validation(second,second_best,cfg))==6
    final=dict(np.load(tmp_path/'alpha0/final_best/left90_R2_2.6.npz'))
    np.testing.assert_array_equal(final['scored_tick_reward'],reward_series(final,0,cfg)[0])
    expected['left90_R2']=expected['left90_R2'].replace(xy=expected['left90_R2'].xy+1)
    with pytest.raises(AssertionError,match='original path mismatch'):
        reuse_reward_parent(receipt,tmp_path/'reject',0,cfg,identity,ck,expected)

@pytest.mark.parametrize('field,value', [('terminal_contract','terminal_contract_v1'),('http_failure_policy','warn'),('coverage_monitor',{'target':.06,'check_every':10,'consecutive_low_windows':2})])
def test_new_plan_rejects_semantic_or_visibility_changes(tmp_path,field,value):
    original=ROOT/'learning/configs/path_safety_candidate_run.json';plan=json.loads(original.read_text())
    plan['design_config']=str((original.parent/plan['design_config']).resolve())
    plan['integration_manifest']=str((original.parent/plan['integration_manifest']).resolve())
    plan[field]=value;file=tmp_path/'plan.json';file.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='contract|monitor'):load_plan(file)
