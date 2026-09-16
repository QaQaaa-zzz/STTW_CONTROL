from copy import deepcopy
import numpy as np
from sttw_control.selection import rank_candidate


def row():
    return {'radial_rmse':[.15,.16],'speed_rmse':[.06,.07],
            'nominal_radial_rmse':[.14,.15],'failed':[False,False],
            'nominal_failed':[False,False],'event_present':[True,True],
            'post_event_hold_complete':[True,True],'settling_seconds':[2.,3.],
            'post_event_peak':[.2,.3]}


def test_rank_rejects_speed_and_nominal_regression_even_if_path_improves():
    base=row();candidate=deepcopy(base);candidate['radial_rmse']=[.01,.01]
    candidate['speed_rmse']=[.09,.07]
    assert rank_candidate(candidate,base)[0] is None
    candidate=deepcopy(base);candidate['nominal_radial_rmse']=[.2,.15]
    assert rank_candidate(candidate,base)[0] is None


def test_recovery_rank_prefers_completed_hold_over_lower_whole_episode_rmse():
    base=row();missing=deepcopy(base)
    missing['post_event_hold_complete']=[True,False]
    missing['settling_seconds']=[2.,10.]
    missing['radial_rmse']=[.01,.01]
    assert rank_candidate(base,base)[0]<rank_candidate(missing,base)[0]
    invalid=deepcopy(base);invalid['post_event_peak']=[np.nan,0.]
    assert rank_candidate(invalid,base)[0] is None


def test_invalid_reference_cannot_disable_regression_guards():
    base=row();candidate=row()
    base['speed_rmse']=[np.nan,.07]
    candidate['speed_rmse']=[.19,.07]
    assert rank_candidate(candidate,base)[1]=='nonfinite baseline reference'
    base=row();base['nominal_radial_rmse']=[None,.15]
    assert rank_candidate(candidate,base)[0] is None
    candidate=row();candidate['settling_seconds']=[None,3.]
    assert rank_candidate(candidate,row())[1]=='nonfinite validation'


def test_command_selection_prioritizes_survival_then_nominal_then_return():
    from sttw_control.selection import rank_command_candidate, command_improved
    base=dict(failed=[False],initial_speed_rmse=[.02],initial_yaw_rmse=[.02],episode_return=[0.])
    good=dict(base,episode_return=[-10.])
    failed=dict(base,failed=[True],episode_return=[100.])
    regression=dict(base,initial_speed_rmse=[.3],episode_return=[100.])
    score=lambda x:rank_command_candidate(x,base)[0]
    assert score(good)<score(failed) and score(good)<score(regression)
    assert not command_improved((0.,0.,9.95),(0.,0.,10.),.1)
    assert command_improved((0.,0.,9.),(0.,0.,10.),.1)
    assert rank_command_candidate(dict(good,episode_return=[float('nan')]),base)[0] is None


def test_command_early_stop_requires_minimum_and_patience():
    from sttw_control.selection import command_should_stop
    assert not command_should_stop(12,4,16,4)
    assert not command_should_stop(16,3,16,4)
    assert command_should_stop(16,4,16,4)
    assert not command_should_stop(32,8,16,0)


def test_command_full_episode_rank_catches_late_regression_and_missing_return():
    from sttw_control.selection import rank_command_candidate,command_improved
    base=dict(failed=[False,True],episode_return=[4.,-100.],speed_rmse=[.03,2.],yaw_rmse=[.03,3.],terminal_tracking_hold=[True,False])
    good=dict(base,failed=[False,False],episode_return=[5.,-10.],speed_rmse=[.04,.1],yaw_rmse=[.04,.1],terminal_tracking_hold=[True,True])
    late=dict(good,episode_return=[100.,100.],yaw_rmse=[.6,.1])
    missing=dict(good,terminal_tracking_hold=[True,False])
    rank=lambda v:rank_command_candidate(v,base,scope='full_episode')[0]
    assert rank(good)<rank(late) and rank(good)<rank(missing)
    assert command_improved(rank(good),rank(late),.1)
    assert rank(good)[-1]==-45.5
    assert rank_command_candidate(dict(good,speed_rmse=[float('nan'),.1]),base,scope='full_episode')[0] is None


def test_best_reward_model_is_independent_of_last_and_acceptance(tmp_path):
    import json
    from sttw_control.selection import refresh_best_reward_model
    base=dict(failed=[False,True],episode_return=[4.,-100.],speed_rmse=[.03,2.],yaw_rmse=[.03,3.],terminal_tracking_hold=[True,False])
    (tmp_path/'baseline_validation.json').write_text(json.dumps(base))
    (tmp_path/'declaration.json').write_text(json.dumps({'training':{'command_speed_slack':.05,'command_yaw_slack':.05}}))
    for update,score in [(50,-80),(100,-40),(200,-70)]:
        folder=tmp_path/'checkpoints'/f'update_{update:04d}';folder.mkdir(parents=True)
        v=dict(base,episode_return=[score,score])
        (folder/'training.json').write_text(json.dumps({'update':update,'validation':v}))
    result=refresh_best_reward_model(tmp_path)
    assert result['update']==100
    assert result['development_gates_passed'] is False
    assert (tmp_path/'best_model').resolve().name=='update_0100'
    assert result['criterion']=='maximum mean fixed-development episode return'
    assert len(result['candidates'])==3


def test_best_reward_model_rejects_missing_or_nonfinite_scores(tmp_path):
    import json
    from sttw_control.selection import refresh_best_reward_model
    (tmp_path/'baseline_validation.json').write_text('{}')
    for update,v in [(1,None),(2,{'episode_return':[float('nan')]})]:
        p=tmp_path/'checkpoints'/f'update_{update:04d}';p.mkdir(parents=True)
        (p/'training.json').write_text(json.dumps({'update':update,'validation':v}))
    assert refresh_best_reward_model(tmp_path) is None
    assert not (tmp_path/'best_model').exists()


def test_best_reward_resume_competes_only_with_same_validation_contract(tmp_path):
    import json
    from sttw_control.selection import refresh_best_reward_model
    base=dict(failed=[False],episode_return=[0.],speed_rmse=[.03],yaw_rmse=[.03],terminal_tracking_hold=[True])
    prev=tmp_path/'previous';cp=prev/'checkpoints/update_0150';cp.mkdir(parents=True)
    config={'validation_seeds':[1],'command_validation_schedules':[[[0,2,0,.5]]]}
    (prev/'declaration.json').write_text(json.dumps({'training':config,'policy_identity':{'config':'same'}}))
    (cp/'training.json').write_text(json.dumps({'update':150,'validation':base}))
    current=tmp_path/'current';current.mkdir()
    declaration={'training':dict(config,resume_checkpoint=str(cp)),'policy_identity':{'config':'same'}}
    (current/'declaration.json').write_text(json.dumps(declaration))
    (current/'baseline_validation.json').write_text(json.dumps(base))
    assert refresh_best_reward_model(current)['update']==150
    declaration['training']['validation_seeds']=[2]
    (current/'declaration.json').write_text(json.dumps(declaration))
    assert refresh_best_reward_model(current) is None


def test_best_reward_reports_unevaluated_update(tmp_path):
    import json
    from sttw_control.selection import refresh_best_reward_model
    base=dict(failed=[False],episode_return=[1.],speed_rmse=[.03],yaw_rmse=[.03],terminal_tracking_hold=[True])
    (tmp_path/'baseline_validation.json').write_text(json.dumps(base))
    for update in [1,2]:
        folder=tmp_path/'checkpoints'/f'update_{update:04d}';folder.mkdir(parents=True)
        entry={'update':update}
        if update==1:entry['validation']=base
        (folder/'training.json').write_text(json.dumps(entry))
    result=refresh_best_reward_model(tmp_path)
    assert result['selection_coverage']['unevaluated_updates']==[2]
    assert not result['selection_coverage']['all_saved_updates_evaluated']


def test_every_update_best_selection_overrides_sparse_schedule():
    from sttw_control.training import TrainingConfig,should_validate,should_plot
    c=TrainingConfig(updates=10,validation_updates=(1,10),best_model_every_update=True,plot_interval=0)
    assert all(should_validate(c,i) for i in range(1,11))
    assert not should_plot(c,2)
    assert should_plot(c,10)
