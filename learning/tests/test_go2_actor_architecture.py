"""Native checkpoints, alpha conditioning, exports and frozen inference for the two actors."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from sttw_control.torch_path_policy import make_torch_path_network, TorchPathPolicy

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT/'docs/evidence/h75_recent_20261109/inputs/bundle'
sys.path.insert(0, '/home/qy/projects/My_unitree_go2_gym/rsl_rl')
from sttw_control.go2_ppo_adapter import H75ActorCritic


def native(architecture, config, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = make_torch_path_network(351, 2, dict(config, network=dict(config['network'], architecture=architecture)))
        critic = make_torch_path_network(352, 1, config)
    return H75ActorCritic(actor, critic, torch.tensor([.25, .1]).log())


def test_dense_initialization_matches_reference():
    config = json.loads((REFERENCE/'config.json').read_text())
    concat, film = [native(a, config, 7101) for a in ('legacy','alpha_modulated_v1')]
    for name, value in concat.state_dict().items():
        torch.testing.assert_close(value, film.state_dict()[name], atol=0, rtol=0)
    assert set(film.state_dict()) - set(concat.state_dict()) == {
        'actor.'+s for s in ('gain_2','gain_3','shift_2','shift_3')}
    obs = torch.randn(1,351).repeat(5,1)
    obs[:,336] = torch.arange(5)*.25
    torch.testing.assert_close(concat.actor(obs), film.actor(obs), atol=0, rtol=0)
    assert not torch.equal(concat.actor(obs)[0], concat.actor(obs)[-1])


@pytest.mark.parametrize('architecture', ['legacy','alpha_modulated_v1'])
def test_resumed_native_export_matches_frozen(architecture, tmp_path):
    torch.set_num_threads(1)
    config = json.loads((REFERENCE/'config.json').read_text())
    policy = native(architecture, config, 7101)
    obs = torch.randn(65,351, generator=torch.Generator().manual_seed(7902))
    obs[:,336] = torch.arange(65)%5*.25
    critic_obs = torch.cat((obs,torch.ones(len(obs),1)), dim=1)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)

    def update(model, optimizer):
        optimizer.zero_grad()
        (model.actor(obs).square().mean() + model.critic(critic_obs).square().mean()).backward()
        optimizer.step()

    update(policy,opt)
    restored = native(architecture, config, 1000)
    restored.load_state_dict(deepcopy(policy.state_dict()))
    resumed_opt = torch.optim.Adam(restored.parameters())
    resumed_opt.load_state_dict(deepcopy(opt.state_dict()))
    update(policy,opt)
    update(restored,resumed_opt)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[name],atol=0,rtol=0)
    other = native('legacy' if architecture!='legacy' else 'alpha_modulated_v1', config, 7101)
    with pytest.raises(RuntimeError):
        other.load_state_dict(policy.state_dict())

    cfg = {'task': {'updates':2, 'policy_initialization':'fresh'}}
    if architecture=='legacy':
        cfg['task']['actor_architecture']=architecture
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    checkpoint = tmp_path/'model_2.pt'
    payload = {'iter':2, 'model_state_dict':restored.state_dict()}
    # Previously recorded FiLM checkpoints have no explicit architecture marker.
    if architecture=='legacy':
        payload['actor_architecture']=architecture
    torch.save(payload,checkpoint)
    bundle=tmp_path/'bundle'
    subprocess.run([sys.executable, ROOT/'learning/cli/export_go2_local_actor.py', '--training-run',tmp_path,
                    '--checkpoint',checkpoint,'--output',bundle,'--h75-reference-bundle',REFERENCE],check=True)
    identity = json.loads((bundle/'identity.json').read_text())
    assert identity['actor_architecture']==architecture and identity['training_critic_architecture']=='alpha_modulated_v1'
    assert identity['native_checkpoint']['sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert (bundle/'route.npz').read_bytes()==(REFERENCE/'route.npz').read_bytes()
    configuration = json.loads((bundle/'config.json').read_text())
    if architecture=='alpha_modulated_v1':
        assert (bundle/'config.json').read_bytes()==(REFERENCE/'config.json').read_bytes()
    frozen=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],configuration,len(obs))
    torch.testing.assert_close(frozen.latent(obs),policy.act_inference(obs),atol=2e-6,rtol=2e-6)
    assert len(frozen.weights)==(8 if architecture=='legacy' else 12)
    wrong_config=deepcopy(configuration)
    wrong_config['network']['architecture']='alpha_modulated_v1' if architecture=='legacy' else 'legacy'
    with pytest.raises(ValueError,match='parameters do not match'):
        TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],wrong_config,len(obs))
    # Forward parity extends through the unchanged 200 Hz filter/rate/clip state.
    expected=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],configuration,len(obs))
    nominal=torch.tensor([2.6,.18]).expand(len(obs),2)
    with torch.no_grad():
        for step in range(8):
            x=obs.roll(step,0)
            applied,_=frozen.correct(frozen.latent(x),nominal)
            baseline,_=expected.correct(policy.actor(x),nominal)
            torch.testing.assert_close(applied,baseline,atol=2e-6,rtol=2e-6)


def test_paired_initialization_names_and_bits_are_checked():
    from sttw_control.go2_ppo_adapter import verify_initial_weights
    reference = {"w": torch.tensor([.0, 1.]), "mod": torch.zeros(2)}
    assert verify_initial_weights({"w": reference["w"]}, reference, ["mod"])["identical_shared"]
    with pytest.raises(ValueError, match="names"):
        verify_initial_weights({"w": reference["w"]}, reference)
    with pytest.raises(ValueError, match="mismatch"):
        verify_initial_weights({"w": torch.tensor([-.0, 1.])}, reference, ["mod"])
