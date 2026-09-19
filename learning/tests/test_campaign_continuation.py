"""Budget and no-evaluation guards for an explicitly extended campaign."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('campaign',Path('learning/cli/asymmetric_priority_comparison.py'))
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def test_continuation_budget_is_total_not_additional():
    original={'updates':48,'num_envs':1024,'rollout_steps':128,'learning_rate':.0003}
    result=campaign.continuation_config(original,250,48,'saved/checkpoint',True)
    assert result['updates']==202
    assert result['resume_checkpoint']=='saved/checkpoint'
    assert result['training_reward_selection'] is True
    assert result['evaluation_reward_best'] is False
    assert original['updates']==48
    assert result['updates']*result['num_envs']*result['rollout_steps']==26476544


def test_fresh_control_uses_total_budget():
    result=campaign.continuation_config({'updates':48},250,0,None,True)
    assert result['updates']==250 and result['resume_checkpoint'] is None


def test_completed_target_refuses_extra_training():
    with pytest.raises(ValueError,match='target'):
        campaign.continuation_config({},250,250,'checkpoint',True)


def test_no_evaluation_disables_first_middle_and_final():
    from sttw_control.training import TrainingConfig,should_validate
    original={'trainer':'rsl','num_envs':1024,'rollout_steps':128,'minibatch_size':32768}
    config=TrainingConfig(**campaign.continuation_config(original,250,48,'checkpoint',True))
    assert not any(should_validate(config,i) for i in (1,24,48,202))


def test_named_three_arm_pipeline_runs_in_order(tmp_path,monkeypatch):
    import json,sys
    training=tmp_path/'training.json';training.write_text(json.dumps({'updates':200}))
    tasks=[]
    for name in ['ecbc1','ecbc08','direct']:
        path=tmp_path/(name+'.json');path.write_text('{}');tasks.append((name,path))
    seen=[]
    def fake_train(task,output,config):
        seen.append((output.parent.name,config.updates,config.resume_checkpoint))
        return {'complete':True,'control_transitions':1,'last_checkpoint':'fixture'}
    monkeypatch.setattr(campaign,'train',fake_train)
    args=['campaign','--training',str(training),'--output',str(tmp_path/'out')]
    for name,path in tasks:args+=['--arm',f'{name}={path}']
    monkeypatch.setattr(sys,'argv',args);campaign.main()
    assert seen==[(name,200,None) for name,_ in tasks]
    assert json.loads((tmp_path/'out/status.json').read_text())['complete']
