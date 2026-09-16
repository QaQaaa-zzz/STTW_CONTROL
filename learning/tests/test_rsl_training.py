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


def test_three_hidden_layers_export_nonzero_policy(tmp_path):
    from sttw_control.network import save_policy,load_policy,ResidualActor
    from sttw_control.training import TrainingConfig
    c=TrainingConfig(trainer='rsl',activation='elu',hidden_sizes=[128,128,128])
    obs=TensorDict({'policy':torch.randn(4,16)},batch_size=[4])
    algo=make_algorithm(obs,steps=2,epochs=1,minibatches=1,device='cpu',
                        activation=c.activation,hidden_sizes=c.hidden_sizes)
    assert [m.out_features for m in algo.policy.critic.modules() if isinstance(m,torch.nn.Linear)]==[128,128,128,1]
    with torch.no_grad():
        layers=[m for m in algo.policy.actor.modules() if isinstance(m,torch.nn.Linear)]
        torch.nn.init.normal_(layers[-1].weight,std=.05)
        layers[-1].bias.fill_(.1)
        expected=torch.tanh(algo.policy.act_inference(obs)).numpy()
    assert np.ptp(expected[:,0])>1e-5
    params=export_actor(algo.policy)
    np.testing.assert_allclose(ResidualActor(tuple(c.hidden_sizes),activation='elu').apply(params,obs['policy'].numpy()),expected,atol=1e-6)
    identity={'history_steps':1}
    save_policy(tmp_path/'policy',params,np.zeros(16),np.ones(16),identity,
                hidden_sizes=c.hidden_sizes,activation=c.activation)
    loaded=load_policy(tmp_path/'policy',expected=identity)
    np.testing.assert_allclose(loaded(obs['policy'].numpy()),expected,atol=1e-6)


@pytest.mark.parametrize('sizes',[[],[0,128],[128,False],[128,1.5]])
def test_invalid_hidden_sizes_rejected(sizes):
    from sttw_control.training import TrainingConfig
    with pytest.raises(ValueError,match='hidden'):
        TrainingConfig(trainer='rsl',hidden_sizes=sizes)


def test_reward_best_endpoint_never_silently_uses_last():
    from sttw_control.pipeline import select_endpoint
    result={'last_checkpoint':'last','best_checkpoint':'eligible','best_reward_checkpoint':'reward_best'}
    assert select_endpoint(result,conditioned=True,reward_best=True)=='reward_best'
    with pytest.raises(ValueError,match='reward.best'):
        select_endpoint({**result,'best_reward_checkpoint':None},conditioned=True,reward_best=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA restore regression')
def test_resume_rng_loaded_onto_cuda_is_restored_on_cpu():
    from sttw_control.rsl_training import restore_cuda_rng
    torch.cuda.manual_seed_all(123)
    states = [state.to('cuda') for state in torch.cuda.get_rng_state_all()]
    expected = torch.rand(8, device='cuda')
    restore_cuda_rng(states)
    torch.testing.assert_close(torch.rand(8, device='cuda'), expected)
