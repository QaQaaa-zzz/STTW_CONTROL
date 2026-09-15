import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('rsl_rl')
from tensordict import TensorDict
from sttw_control.rsl_training import make_algorithm, export_actor, bootstrap_timeout, torch_from_jax, jax_from_torch


def test_rsl_latent_action_and_export_identity():
    from sttw_control.network import ResidualActor
    obs=TensorDict({'policy':torch.randn(4,20)},batch_size=[4])
    algo=make_algorithm(obs,steps=4,epochs=1,minibatches=1,device='cpu')
    with torch.no_grad():
        assert torch.count_nonzero(algo.policy.act_inference(obs))==0
        z=algo.act(obs)
        assert z.shape==(4,2)
        assert torch.all(torch.abs(torch.tanh(z))<=1)
        # Compare a nonzero trained/exported output, not only the zero initialization.
        layers=[m for m in algo.policy.actor.modules() if isinstance(m,torch.nn.Linear)]
        layers[-1].bias.fill_(.17)
        params=export_actor(algo.policy)
        got=ResidualActor().apply(params,np.asarray(obs['policy']))
        np.testing.assert_allclose(got,torch.tanh(algo.policy.act_inference(obs)).numpy(),atol=1e-6)


def test_timeout_uses_next_value_and_failure_never_bootstraps():
    got=bootstrap_timeout(torch.tensor([1.,1.,1.]),torch.tensor([4.,8.,9.]),torch.tensor([True,True,False]),torch.tensor([False,True,False]),.9)
    torch.testing.assert_close(got,torch.tensor([4.6,1.,1.]))


def test_real_rsl_ppo_update_is_finite():
    obs=TensorDict({'policy':torch.randn(4,20)},batch_size=[4])
    algo=make_algorithm(obs,steps=4,epochs=2,minibatches=2,device='cpu')
    before=[p.clone() for p in algo.policy.parameters()]
    with torch.no_grad():
        for _ in range(4):
            z=algo.act(obs)
            algo.process_env_step(obs,-(z[:,0]-.2)**2,torch.zeros(4,dtype=torch.bool),{})
        algo.compute_returns(obs)
    metrics=algo.update()
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a,b) for a,b in zip(before,algo.policy.parameters()))


def test_dlpack_roundtrip_cpu():
    import jax.numpy as jp
    x=jp.arange(12,dtype=jp.float32).reshape(3,4)
    t=torch_from_jax(x)
    np.testing.assert_array_equal(jax_from_torch(t),x)


def test_geometric_best_uses_geometric_gate(tmp_path,monkeypatch):
    import json
    from sttw_control import selection
    (tmp_path/'declaration.json').write_text(json.dumps({'task':{'tracking':{}},'training':{}}))
    (tmp_path/'baseline_validation.json').write_text('{}')
    checkpoint=tmp_path/'checkpoints/update_0001';checkpoint.mkdir(parents=True)
    (checkpoint/'training.json').write_text(json.dumps({'update':1,'validation':{'episode_return':[1.],'failed':[False]}}))
    monkeypatch.setattr(selection,'rank_tracking_candidate',lambda *a,**kw:((-2.,.1),'qualified'))
    result=selection.refresh_best_reward_model(tmp_path)
    assert result['development_gates_passed']
    assert result['task_success_verified'] is False


def test_elu_export_and_load(tmp_path):
    from sttw_control.network import save_policy,load_policy
    obs=TensorDict({'policy':torch.randn(4,16)},batch_size=[4])
    algo=make_algorithm(obs,steps=2,epochs=1,minibatches=1,device='cpu',activation='elu')
    with torch.no_grad():
        for layer in algo.policy.actor.modules():
            if isinstance(layer,torch.nn.Linear):layer.bias.add_(.2)
    identity={'history_steps':1}
    save_policy(tmp_path/'policy',export_actor(algo.policy),np.zeros(16),np.ones(16),identity,activation='elu')
    loaded=load_policy(tmp_path/'policy',expected=identity)
    with torch.no_grad():expected=torch.tanh(algo.policy.act_inference(obs)).numpy()
    np.testing.assert_allclose(loaded(obs['policy'].numpy()),expected,atol=1e-6)
