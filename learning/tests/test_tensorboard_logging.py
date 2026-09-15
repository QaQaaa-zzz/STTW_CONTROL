import json
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def test_event_roundtrip_uses_global_update_and_no_fake_missing_values(tmp_path):
    from sttw_control.tensorboard_logging import TrainingEvents
    with TrainingEvents(tmp_path/'events') as writer:
        writer.write({'update':99,'mean_step_reward':-.2,'loss_metrics':[1,2,3,4],
                      'reward_components_mean_step':{'speed':-.1,'yaw':-.105,'alive':.005},
                      'optimizer_audit':{'accepted_minibatches':66,'full_update_rolled_back':True},
                      'validation':{'episode_return':[None,0.]}})
        writer.write({'update':100,'mean_step_reward':-.1})
    events=EventAccumulator(str(tmp_path/'events')).Reload()
    reward=events.Scalars('train/mean_step_reward')
    assert [x.step for x in reward]==[99,100]
    assert [x.value for x in reward]==pytest.approx([-.2,-.1])
    assert events.Scalars('optimizer/retained_minibatches')[0].value==0
    assert 'validation/sample_000/episode_return' not in events.Tags()['scalars']
    assert events.Scalars('validation/sample_001/episode_return')[0].value==0
    assert len(events.Scalars('reward_components/speed'))==1


def test_close_on_exception_preserves_events(tmp_path):
    from sttw_control.tensorboard_logging import TrainingEvents
    with pytest.raises(RuntimeError):
        with TrainingEvents(tmp_path/'events') as writer:
            writer.write({'update':1,'mean_step_reward':.25})
            raise RuntimeError('training failed')
    assert EventAccumulator(str(tmp_path/'events')).Reload().Scalars('train/mean_step_reward')[0].value==.25


def test_import_rejects_duplicates_and_preserves_sources(tmp_path):
    from sttw_control.tensorboard_logging import import_metrics
    p=tmp_path/'metrics.jsonl';p.write_text(json.dumps({'update':1,'mean_step_reward':.1})+'\n')
    dest=tmp_path/'events'
    with pytest.raises(ValueError,match='increasing'):import_metrics([p,p],dest)
    assert not dest.exists()
    assert import_metrics([p],dest)==1
    assert (dest/'sources.json').exists()
    with pytest.raises(FileExistsError):import_metrics([p],dest)


def test_command_labels_match_validator_order():
    from sttw_control.tensorboard_logging import validation_labels
    labels=validation_labels({'task':{'motion_commands':{},'priority':{'validation_alphas':[0,.5,1]}},
                              'training':{'command_validation_schedules':[[],[]],'validation_seeds':[3,4]}})
    assert len(labels)==12
    assert labels[3]=='case_00/alpha_0.5/seed_4'
    assert labels[6]=='case_01/alpha_0/seed_3'


def test_training_entrypoint_closes_logger_on_failure(tmp_path,monkeypatch):
    from sttw_control import training
    def fail(task,output,config,events):
        events.write({'update':5,'mean_step_reward':-.4})
        raise RuntimeError('simulated trainer error')
    monkeypatch.setattr(training,'_train',fail)
    with pytest.raises(RuntimeError,match='simulated'):
        training.train(tmp_path/'task.json',tmp_path/'training')
    events=EventAccumulator(str(tmp_path/'training/tensorboard')).Reload()
    assert events.Scalars('train/mean_step_reward')[0].step==5


def test_event_index_links_to_run_and_does_not_link_itself(tmp_path):
    from sttw_control.tensorboard_logging import register_run
    (tmp_path/'PROJECT.md').touch()
    log=tmp_path/'runs/example/training/tensorboard';log.mkdir(parents=True)
    register_run(log);register_run(log)
    index=tmp_path/'runs/tensorboard';links=list(index.iterdir())
    assert len(links)==1 and links[0].resolve()==log
    register_run(index)
    assert len(list(index.iterdir()))==1


def test_core_profile_keeps_rewards_and_kl_without_sample_explosion():
    from sttw_control.tensorboard_logging import scalar_values
    r={'mean_step_reward':.1,'loss_metrics':[1,2,3,4],
       'validation':{'episode_return':[2,None,4]},'device_memory_stats':{'used':200},
       'optimizer_audit':{'accepted_minibatches':3,'unwanted':9}}
    values=scalar_values(r,profile='core')
    assert values['validation/mean_episode_return']==3
    assert values['train/mean_step_reward']==.1
    assert values['kl/approx_kl']==4
    assert not any('sample_' in k or k.startswith('memory/') or k.endswith('unwanted') for k in values)
