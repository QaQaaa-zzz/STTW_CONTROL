"""Actual paired exposure and authenticated native-source initialization."""
import copy
import pytest
import torch

from sttw_control.finite_value_schedule import make_schedule, fresh_learners
from sttw_control.path_command_policy import make_torch_path_network
from sttw_control.survival_online_learner import OnlineSurvivalLearner


def test_three_arms_share_each_event_and_noise_stream_without_support():
    schedule = make_schedule([dict(context='left', speed=1.8), dict(context='right', speed=1.5)],
                             seed=19, updates=2)
    assert len(schedule) == 2
    for iteration, cases in enumerate(schedule):
        assert len(cases) == 180
        groups = {}
        for c in cases:
            assert c['profile'] == 0 and c['collection_index'] == iteration
            assert 320 <= c['pulse_onset_task_tick'] <= 680
            groups.setdefault(c['rng_fold_in'], []).append(c)
        assert len(groups) == 60
        for rows in groups.values():
            assert {c['arm'] for c in rows} == {'free_GAE', 'finite_GAE', 'free_MC'}
            assert len({(c['context'], c['alpha'], c['force_N'], c['replicate'],
                         c['pulse_onset_task_tick']) for c in rows}) == 1
    assert schedule == make_schedule([dict(context='left', speed=1.8), dict(context='right', speed=1.5)],
                                     seed=19, updates=2)
    assert [c['pulse_onset_task_tick'] for c in schedule[0]] != [c['pulse_onset_task_tick'] for c in schedule[1]]


def native_source():
    cfg = {'network': {'architecture': 'alpha_modulated_v1'}}
    torch.manual_seed(41)
    actor = make_torch_path_network(351, 2, cfg)
    critic = make_torch_path_network(352, 1, cfg)
    policy = {'actor.' + k: v for k, v in actor.state_dict().items()}
    policy.update({'critic.' + k: v for k, v in critic.state_dict().items()})
    policy['log_std'] = torch.tensor([-1., -2.])
    return cfg, OnlineSurvivalLearner(cfg, policy, beta=0., seed=42).checkpoint()


def test_native_actor_and_physical_head_preserved_but_all_optimizers_fresh():
    cfg, source = native_source()
    before = copy.deepcopy(source)
    learners = fresh_learners(cfg, source, seed=43)
    for obj in learners.values():
        assert obj.optimizer_steps == [0, 0, 0] and obj.next_collection == 0
        assert all(len(o.state) == 0 for o in obj.optimizers)
        assert all(torch.equal(v, source['actor'][k]) for k, v in obj.actor.state_dict().items())
        assert all(torch.equal(v, source['physical_critic'][k]) for k, v in obj.physical_critic.state_dict().items())
        assert torch.equal(obj.log_std, source['log_std'])
    assert all(torch.equal(v, before['task_critic'][k]) for k, v in source['task_critic'].items())
    source['config_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='source/config'):
        fresh_learners(cfg, source, seed=43)


def test_fixed_evaluation_has_one_zero_and_four_hidden_events_per_route_alpha():
    from sttw_control.finite_value_schedule import make_evaluation_schedule
    rows = make_evaluation_schedule([dict(context='left', speed=1.8), dict(context='right', speed=1.5)], seed=23)
    assert len(rows) == 300
    events = {}
    for c in rows:
        events.setdefault((c['context'], c['alpha'], c['force_N'], c['replicate']), []).append(c)
    assert len(events) == 50
    for key, group in events.items():
        assert len(group) == 6
        assert len({c['pulse_onset_task_tick'] for c in group}) == 1
        assert len({c['actor_role'] for c in group}) == 6
        assert sum(c['causal_support'] for c in group) == 1
        if key[2] == 0.:
            assert key[3] == 0
