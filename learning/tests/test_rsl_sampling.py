from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jp
import pytest
torch=pytest.importorskip("torch", reason="RSL optional dependency; full RSL CI installs it")
from sttw_control.rsl_sampling import phase_spread, EpisodeStatistics

class State(NamedTuple):
    tick: object
    history: object
    done: object


def test_phase_spread_advances_real_history_and_preserves_full_state():
    s=State(jp.zeros(8,dtype=int),jp.zeros(8),jp.zeros(8,dtype=bool))
    def step(s,a):return State(s.tick+1,s.history+s.tick+1,(s.tick+1)>=8)
    def reset(k):return s
    result,_,stats=phase_spread(s,jax.random.PRNGKey(5),8,8,step,reset,lambda s:jp.zeros((8,2)))
    np.testing.assert_array_equal(np.sort(result.tick),np.arange(8))
    np.testing.assert_array_equal(result.history,result.tick*(result.tick+1)/2)
    assert int(stats['active_transitions'])==28
    assert int(stats['reset_count'])==0


def test_phase_spread_resets_real_early_termination():
    s=State(jp.zeros(8,dtype=int),jp.zeros(8),jp.zeros(8,dtype=bool))
    def step(s,a):return State(s.tick+1,s.history+1,(s.tick+1)>=3)
    result,_,stats=phase_spread(s,jax.random.PRNGKey(5),8,8,step,lambda k:s,lambda s:jp.zeros((8,2)))
    assert int(stats['reset_count'])>0
    np.testing.assert_array_equal(result.tick,result.history)
    assert np.max(result.tick)<3


def test_episode_statistics_excludes_prefill_and_spans_updates():
    stats=EpisodeStatistics(torch.tensor([False,True]))
    stats.add(torch.tensor([9.,1.]),torch.tensor([True,False]),torch.tensor([False,False]))
    assert stats.flush()['count']==0
    stats.add(torch.tensor([2.,3.]),torch.tensor([False,True]),torch.tensor([False,False]))
    summary=stats.flush()
    assert summary['count']==1 and summary['mean_return']==4 and summary['mean_length_steps']==2
    stats.add(torch.tensor([-10.,5.]),torch.tensor([True,False]),torch.tensor([True,False]))
    summary=stats.flush()
    assert summary['count']==1 and summary['mean_return']==-8 and summary['failure_fraction']==1


def test_new_controls_preserve_old_defaults_and_reject_legacy_warmup():
    import pytest
    from sttw_control.training import TrainingConfig,should_validate
    old=TrainingConfig()
    assert not old.phase_spread_initialization and old.training_reward_best_enabled
    with pytest.raises(ValueError,match='no legacy warmup'):
        TrainingConfig(trainer='rsl',phase_spread_initialization=True,warmup_steps=2)
    cfg=TrainingConfig(trainer='rsl',training_reward_selection=True,training_reward_best_enabled=False,phase_spread_initialization=True)
    assert not should_validate(cfg,cfg.updates)


def test_core_tensorboard_keeps_full_episode_and_phase_metrics():
    from sttw_control.tensorboard_logging import scalar_values
    got=scalar_values({'complete_training_episodes':{'count':2,'mean_return':-10.,'scope':'not selection'},
                      'phase_tracking':{'phase_2_speed_rmse':.2}},profile='core')
    assert got['complete_training_episodes/mean_return']==-10.
    assert got['phase_tracking/phase_2_speed_rmse']==.2
    assert 'complete_training_episodes/scope' not in got
