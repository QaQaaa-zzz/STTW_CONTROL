"""CPU checks of V5 Gaussian exploration and settled-pair regularization."""
import json
from pathlib import Path
import pytest
import torch
from test_direct_command_ppo import _obs, _rollout
from sttw_control.direct_command_ppo import make_algorithm

ROOT = Path(__file__).resolve().parents[2]

def spec():
    s = json.loads((ROOT/'learning/configs/r196_smooth_alpha0.json').read_text())
    s['preference_v5'] = json.loads((ROOT/'docs/preference_v5/attachment/STTW_R196_PreferenceV5_alpha0.json').read_text())
    s['ppo'].update(initial_latent_std=[.30,.10], latent_std_min=[.08,.03], latent_std_max=[.40,.20], minibatches=2, epochs=1)
    return s

def algorithm():
    s = spec()
    a = make_algorithm(_obs(), 3, s, 'cpu')
    a.temporal_spec = s
    return a

def test_channel_std_initialization_and_projection():
    a = algorithm()
    torch.testing.assert_close(a.policy.log_std.exp(), torch.tensor([.3,.1]))
    with torch.no_grad():
        a.policy.log_std.copy_(torch.tensor([.01,.9]).log())
    _rollout(a, _obs(), 3)
    a.hard_kl = 1e6
    result = a.update()
    assert result['accepted_epochs'] == 1
    torch.testing.assert_close(a.policy.log_std.exp(), torch.tensor([.08,.20]))

@pytest.mark.parametrize('updates,weight', [(0,0),(19,0),(20,0),(30,.001),(40,.002),(60,.002)])
def test_effective_update_ramp(updates, weight):
    a = algorithm()
    a.accepted_policy_updates = updates
    a.prepare_temporal_pairs()
    assert a.temporal_coefficient == pytest.approx(weight)

@pytest.mark.parametrize('index,physical', [(340,.2),(337,.05),(304,.05),(308,.05),(302,.02),(309,.02),(300,.22),(301,.4),(310,.101),(311,.021)])
def test_both_endpoints_must_be_settled(index, physical):
    a = algorithm()
    fields = a.temporal_spec['network']['frame_fields' if index < 320 else 'context_fields']
    scale = fields[index-(300 if index < 320 else 336)]['scale']
    a.storage.observations['policy'][1,0,index] = physical/scale
    a.prepare_temporal_pairs()
    assert a.temporal_denominator == 8
    assert not a.temporal_mask[0] and not a.temporal_mask[4]
    assert int(a.temporal_mask.sum()) == 6

def test_loss_uses_current_unslewed_physical_mean_and_all_valid_pairs():
    a = algorithm()
    a.accepted_policy_updates = 40
    a.storage.dones[0,1] = True
    a.storage.observations['policy'][1,0,337] = 1  # exclude both adjacent pairs
    a.storage.observations['policy'][1,2,299] = 1  # older history, no mask change
    actor = torch.nn.Linear(345,2,bias=False)
    with torch.no_grad():
        actor.weight.zero_()
        actor.weight[0,299] = .5
        actor.weight[1,299] = -.25
    a.policy.actor = actor
    a.prepare_temporal_pairs()
    c = a.temporal_spec['action']
    expected = .002 * 2 * ((torch.tanh(torch.tensor(.5))*c['speed_positive_scale_m_s']/.1)**2 + (torch.tanh(torch.tensor(-.25))*c['steer_scale_rad']/.02)**2)/7
    loss = a.temporal_loss(torch.arange(8))/a.num_mini_batches
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert actor.weight.grad.abs().sum() > 0

@pytest.mark.parametrize('hard_kl,accepted', [(-1.,0),(1e6,1)])
def test_only_accepted_policy_update_advances_ramp(hard_kl, accepted):
    a = algorithm()
    a.accepted_policy_updates = 30
    a.hard_kl = hard_kl
    _rollout(a, _obs(), 3)
    result = a.update()
    assert result['accepted_epochs'] == accepted
    assert a.accepted_policy_updates == 30 + accepted
    assert result['temporal_coefficient'] == pytest.approx(.001)


def test_empty_valid_pairs_has_zero_differentiable_loss():
    a = algorithm()
    a.accepted_policy_updates = 40
    a.storage.dones.fill_(True)
    a.prepare_temporal_pairs()
    assert a.temporal_denominator == 0
    loss = a.temporal_loss(torch.arange(8))
    assert float(loss) == 0
    loss.backward()
