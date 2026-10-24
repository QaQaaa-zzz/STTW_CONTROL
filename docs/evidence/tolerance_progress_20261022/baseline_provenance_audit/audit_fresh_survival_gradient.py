"""Consume authenticated fresh decisions in the real Torch loss; never step Adam."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sttw_control.path_command_policy import make_torch_path_network
from sttw_control.preference_best import digest, atomic_json, config_hash
from sttw_control.survival_credit import survival_ppo_batch, resolve_survival_credit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fixed-denominator', type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve() if args.output else run/'fresh_gradient_receipt.json'
    if output.exists():
        raise FileExistsError('preserve the prior audit receipt; choose a new --output')
    status = json.loads((run/'status.json').read_text())
    plan = json.loads((run/'plan.json').read_text())
    results = json.loads((run/'results.json').read_text())
    prelaunch = json.loads((run/'prelaunch_receipt.json').read_text())
    assert digest(run/'plan.json') == prelaunch['plan_sha256']
    assert plan['collection_policy_id'] == config_hash(plan['sampling'])
    assert status['phase'] == 'completed' and status['completed_cases'] == 4
    assert results['optimizer_updates'] == 0 and not results['actor_parameters_changed']
    assert results['collection_policy_id'] == plan['collection_policy_id']
    model = plan['model']
    for kind in ['actor', 'checkpoint']:
        assert digest(model[kind]) == model[kind+'_sha256']
    cfg = json.loads(Path(plan['source_declaration']).read_text())['config']
    actor = make_torch_path_network(351, 2, cfg)
    checkpoint = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
    actor.load_state_dict({k[len('actor.'):]:v for k,v in checkpoint['policy'].items() if k.startswith('actor.')}, strict=True)
    actor.eval()
    log_std = torch.nn.Parameter(checkpoint['policy']['log_std'].detach().clone())
    # An explicit untrained head represents exactly the .5 baseline collected.
    value = torch.nn.Sequential(torch.nn.Linear(351,1),torch.nn.Sigmoid())
    with torch.no_grad():
        value[0].weight.zero_()
        value[0].bias.zero_()
    initial = [p.detach().clone() for p in list(actor.parameters())+[log_std]+list(value.parameters())]
    arrays = {k: [] for k in ['observation','raw_action','log_prob','physical_value','valid','target']}
    receipts = {str(run/'plan.json'):digest(run/'plan.json'),str(run/'results.json'):digest(run/'results.json')}
    for row in results['cases']:
        raw = run/f"case_{row['case']:03d}.npz"
        assert digest(raw) == row['raw_sha256']
        target_file = run/f"case_{row['case']:03d}_credit.npz"
        receipts[str(raw)] = digest(raw)
        receipts[str(target_file)] = digest(target_file)
        with np.load(raw, allow_pickle=False) as data, np.load(target_file, allow_pickle=False) as target:
            decision = np.zeros_like(data['active_tick'])
            decision[np.flatnonzero(data['decision_valid'])*4] = True
            rebuilt = resolve_survival_credit(active_tick=data['active_tick'], actor_action_valid=decision,
                contact_failure=data['contact_failure'], engineering_fault=data['nonfinite']|data['lower_fault']|data['policy_fault'],
                horizon_reached=data['horizon_reached'], physical_tick=data['physical_tick'],
                window_start_tick=0, window_ticks=2000, collection_policy_id=plan['collection_policy_id'],
                learner_policy_id=plan['collection_policy_id'])
            np.testing.assert_array_equal(target['physical_targets'], rebuilt['targets'])
            np.testing.assert_array_equal(target['actor_mask'], rebuilt['actor_mask'])
            np.testing.assert_array_equal(target['decision_targets'], rebuilt['targets'][::4])
            for name in arrays:
                arrays[name].append(target['decision_targets'] if name == 'target' else data['decision_'+name])
    data = {k:torch.as_tensor(np.concatenate(v,axis=0)) for k,v in arrays.items()}
    valid = data['valid']
    assert int(valid.sum()) == sum(r['actor_decisions'] for r in results['cases'])
    assert torch.isnan(data['observation'][~valid]).all()
    assert torch.isfinite(data['target'][valid]).all()
    forward_counts=[]
    def evaluate(obs, actions):
        assert torch.isfinite(obs).all() and torch.isfinite(actions).all()
        forward_counts.append(obs.shape[0])
        mean=actor(obs)
        return torch.distributions.Normal(mean,log_std.exp()).log_prob(actions).sum(-1),value(obs).squeeze(-1)
    loss=survival_ppo_batch(evaluate,data['observation'],data['raw_action'],data['log_prob'],
                           data['physical_value'],data['target'],valid,
                           fixed_denominator=args.fixed_denominator)
    # Unit coefficient for this graph/gradient check only; not a declared
    # combination with the original task objective or a training recipe.
    (loss['actor_loss']+loss['value_loss']).backward()
    def grad_norm(parameters):
        grads=[p.grad for p in parameters if p.grad is not None]
        assert grads and all(torch.isfinite(x).all() for x in grads)
        return float(torch.sqrt(sum(x.detach().square().sum() for x in grads)))
    actor_norm=grad_norm(list(actor.parameters())+[log_std])
    value_norm=grad_norm(value.parameters())
    assert actor_norm > 0 and value_norm > 0
    current=list(actor.parameters())+[log_std]+list(value.parameters())
    assert all(torch.equal(old,p.detach()) for old,p in zip(initial,current))
    assert digest(model['actor'])==model['actor_sha256'] and digest(model['checkpoint'])==model['checkpoint_sha256']
    atomic_json(output,dict(
        collection_policy_id=plan['collection_policy_id'], raw_decision_slots=int(valid.numel()),
        legal_decisions=int(valid.sum()), excluded_hold_or_padding=int((~valid).sum()),
        actor_reduction_denominator=args.fixed_denominator or int(valid.sum()),
        actor_reduction='fixed_declared_slots' if args.fixed_denominator else 'legal_sample_mean',
        forward_sample_counts=forward_counts, actor_loss=float(loss['actor_loss'].detach()),
        value_loss=float(loss['value_loss'].detach()), actor_gradient_norm=actor_norm,
        physical_value_gradient_norm=value_norm, all_gradients_finite=True,
        optimizer_created=False, optimizer_steps=0, all_parameters_unchanged=True,
        physical_value_head='untrained zero-initialized sigmoid linear head reproducing constant.5',
        derived_targets_rebuilt_from_authenticated_raw=True,
        new_physics_ticks=0, input_sha256=receipts,
        source_sha256={str(Path(__file__).resolve()):digest(__file__),
            str(Path('learning/src/sttw_control/survival_credit.py').resolve()):digest('learning/src/sttw_control/survival_credit.py')},
        scope='Real fresh-data masked forward/backward engineering only, not a trained risk predictor or performance result.'))
    print(json.dumps(dict(legal_decisions=int(valid.sum()), excluded=int((~valid).sum()),
                          actor_grad=actor_norm,value_grad=value_norm,optimizer_steps=0)))


if __name__ == '__main__':
    main()
