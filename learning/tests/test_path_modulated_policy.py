"""Architecture identity, conditioning and cross-framework actor parity."""
import copy
import jax
import jax.numpy as j
import numpy as np
import pytest
import torch
from sttw_control import path_command_policy as pp

MOD = {'network': {'architecture': 'alpha_modulated_v1'}}
LEGACY = {'network': {}}
MOD_KEYS = ('gain_2', 'shift_2', 'gain_3', 'shift_3')


def test_factories_keep_legacy_default_and_add_exactly_384_parameters():
    assert hasattr(pp, 'make_path_actor'), 'explicit architecture factory required'
    assert pp.architecture_identity(LEGACY) == 'legacy'
    for input_dim, output_dim, factory in [(351, 2, pp.make_path_actor), (352, 1, pp.make_path_critic)]:
        legacy = factory(LEGACY).init(jax.random.key(3), j.zeros(input_dim))
        mod = factory(MOD).init(jax.random.key(3), j.zeros(input_dim))
        count = lambda tree: sum(x.size for x in jax.tree_util.tree_leaves(tree))
        assert count(mod) - count(legacy) == 384
        for key in MOD_KEYS:
            np.testing.assert_array_equal(mod['params'][key], 0.)
        for key in legacy['params']:
            for field in ('kernel', 'bias'):
                np.testing.assert_array_equal(legacy['params'][key][field], mod['params'][key][field])
        np.testing.assert_array_equal(factory(LEGACY).apply(legacy, j.ones(input_dim)), factory(MOD).apply(mod, j.ones(input_dim)))
        net = pp.make_torch_path_network(input_dim, output_dim, MOD)
        assert sum(p.numel() for p in net.parameters()) == count(mod)
        assert set(net.state_dict()) == {f'{i}.{k}' for i in (0, 2, 4, 6) for k in ('weight', 'bias')} | set(MOD_KEYS)


def test_nonzero_modulation_parity_gradients_optimizer_and_restore():
    assert hasattr(pp, 'make_torch_path_network'), 'Torch architecture factory required'
    torch.manual_seed(7)
    net = pp.make_torch_path_network(351, 2, MOD)
    with torch.no_grad():
        net[0].weight[:, 336].zero_()
        for i, key in enumerate(MOD_KEYS):
            getattr(net, key).copy_(torch.linspace(-.12, .17, getattr(net, key).numel()) + .02 * i)
    obs = torch.randn(5, 351)
    obs[:, 336] = torch.linspace(0., 1., 5)
    params = pp.export_torch_actor(net, MOD)
    pp.validate_actor_parameters(params, MOD)
    expected = net(obs)
    actual = pp.make_path_actor(MOD).apply(params, j.asarray(obs.numpy()))
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=2e-6, rtol=2e-5)
    single = obs[0].repeat(2, 1)
    single[:, 336] = torch.tensor([0., 1.])
    assert torch.max(torch.abs(net(single)[0] - net(single)[1])) > 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    before = {key: getattr(net, key).detach().clone() for key in MOD_KEYS}
    net(obs).square().sum().backward()
    flax_grad = jax.grad(lambda p: j.sum(pp.make_path_actor(MOD).apply({'params': p}, j.asarray(obs.numpy()))**2))(params['params'])
    for key in MOD_KEYS:
        assert torch.isfinite(getattr(net, key).grad).all()
        assert torch.count_nonzero(getattr(net, key).grad) > 0
        np.testing.assert_allclose(flax_grad[key], getattr(net, key).grad.numpy(), atol=2e-6, rtol=3e-5)
    optimizer.step()
    for key in MOD_KEYS:
        assert not torch.equal(before[key], getattr(net, key))
    restored = pp.make_torch_path_network(351, 2, MOD)
    restored.load_state_dict(copy.deepcopy(net.state_dict()))
    torch.testing.assert_close(restored(obs), net(obs), rtol=0., atol=0.)


def test_wrong_architecture_and_malformed_parameters_are_rejected():
    assert hasattr(pp, 'make_path_actor'), 'explicit architecture factory required'
    legacy = pp.PathCommandActor().init(jax.random.key(1), j.zeros(351))
    mod = pp.make_path_actor(MOD).init(jax.random.key(1), j.zeros(351))
    for params, config in ((mod, LEGACY), (legacy, MOD)):
        with pytest.raises(ValueError, match='architecture'):
            pp.validate_actor_parameters(params, config)
        with pytest.raises(ValueError, match='architecture'):
            pp.make_path_actor(config).apply(params, j.zeros(351))
    with pytest.raises(ValueError, match='architecture'):
        pp.PathCommandActor().apply(mod, j.zeros(351))
    wrong = copy.deepcopy(mod)
    wrong['params']['gain_2'] = j.zeros(127)
    with pytest.raises(ValueError):
        pp.validate_actor_parameters(wrong, MOD)
    with pytest.raises(ValueError):
        pp.architecture_identity({'network': {'architecture': 'unknown'}})
