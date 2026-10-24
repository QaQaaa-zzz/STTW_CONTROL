"""Readonly final-policy input sensitivity on inherited real anchors.

Removing feature kernel columns is a computational counterfactual, not a new
trained baseline or evidence of closed-loop robustness. Outputs are latent
Actor coordinates, before the real reference filters and limits.
"""
import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from sttw_control.feature_comparison_evaluation import final_actor_record
from sttw_control.path_command_policy import make_path_actor, make_torch_path_network
from sttw_control.preference_best import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--arms', nargs='+', required=True,
                        choices=['uniform', 'physics', 'polynomial'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.campaign), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    declaration = json.loads((root/'extension_declaration.json').read_text())
    anchor = declaration['retention_anchor']
    if digest(anchor['file']) != anchor['sha256']:
        raise ValueError('Anchor identity changed')
    with np.load(anchor['file']) as data:
        observations, sources = data['observations'], data['source']
    torch.set_num_threads(4)
    obs = torch.from_numpy(observations)
    receipts = {}
    for arm in args.arms:
        record, config = final_actor_record(root/(arm+'_stage2'), arm, declaration)
        saved = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)
        network = make_torch_path_network(351, 2, config)
        network.load_state_dict({k[6:]: v for k, v in saved['policy'].items()
                                 if k.startswith('actor.')}, strict=True)
        network.eval()
        with torch.no_grad():
            actual = network(obs).numpy()
        if not np.isfinite(actual).all():
            raise ValueError('Nonfinite anchor output')
        stripped = None
        if arm != 'uniform':
            counterfactual = copy.deepcopy(network)
            with torch.no_grad():
                counterfactual[0].weight[:, 351:] = 0
                stripped = counterfactual(obs).numpy()
                np.testing.assert_array_equal(network(obs).numpy(), actual)
        # Verify the deployed backend on a balanced sample from both sources
        # and every alpha; this is an implementation check, not evaluation.
        sample = np.concatenate([np.flatnonzero((sources == s) &
                                 np.isclose(observations[:, 336], alpha))[:16]
                                 for s in [0, 1] for alpha in np.unique(observations[:, 336])])
        import jax.numpy as j
        with Path(record['actor']).open('rb') as stream:
            parameters = pickle.load(stream)
        deployed = np.asarray(make_path_actor(config).apply(parameters, j.asarray(observations[sample])))
        np.testing.assert_allclose(deployed, actual[sample], rtol=3e-5, atol=5e-6)
        groups = []
        for source, label in [(0, 'low'), (1, 'high')]:
            for alpha in np.unique(observations[:, 336]):
                mask = (sources == source) & np.isclose(observations[:, 336], alpha)
                values = actual[mask]
                row = dict(source=label, alpha=float(alpha), rows=int(mask.sum()),
                           latent_mean=values.mean(0).tolist(),
                           latent_std=values.std(0).tolist())
                if stripped is not None:
                    effect = values-stripped[mask]
                    row.update(feature_removal_latent_rms=np.sqrt((effect**2).mean(0)).tolist(),
                               feature_removal_latent_max_abs=np.abs(effect).max(0).tolist())
                groups.append(row)
        receipts[arm] = dict(record=record, groups=groups,
                             deployed_backend_rows=len(sample),
                             max_torch_jax_error=float(np.abs(deployed-actual[sample]).max()),
                             extra_kernel_l2=float(torch.linalg.vector_norm(network[0].weight[:, 351:].detach()))
                             if stripped is not None else None)
    atomic_json(output, dict(anchor_sha256=anchor['sha256'], anchor_rows=len(obs),
                             source_sha256=digest(__file__), arms=receipts,
                             new_physics_ticks=0, new_optimizer_updates=0,
                             scope='Inherited parent-distribution anchors; latent sensitivity only. '
                                   'Not a trained ablation, new-policy rollout, preference-retention '
                                   'proof, or robustness result. Alpha rows are correlated.'))
    print(output)


if __name__ == '__main__':
    main()
