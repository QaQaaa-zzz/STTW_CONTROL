"""Declare an isolated, common-authority frozen-policy nominal diagnostic."""
import argparse
import copy
import json
import pickle
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from sttw_control.preference_best import atomic_json, digest
    from sttw_control.path_command_policy import make_path_actor, validate_actor_parameters
    import jax
    import jax.numpy as jnp
    import numpy as np

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path('runs/reserve_feature_expansion_v1/post_fixed_final_v1/low_new_geometry_confirmation/H75/declaration.json').resolve()
    original = json.loads(source.read_text())
    config = copy.deepcopy(original['config'])
    changes = [
        ('follower', 'steer_reference_max_rad', .6),
        ('residual_interface', 'steer_reference_bound_rad', .65),
        ('residual_interface', 'speed_reference_bounds_m_s', [1., 3.5]),
    ]
    differences = []
    for section, field, value in changes:
        differences.append(dict(section=section, field=field,
                                old=config[section][field], new=value))
        config[section][field] = value
    if config['training_future_phase_C']['episode_s'] != 10:
        raise ValueError('expected inherited ten-second protocol')
    assert config['residual_interface']['steer_reference_bound_rad'] < original['source']['actuator']['steer_limit']
    params = pickle.loads(Path(original['policies']['H75']['actor']).read_bytes())
    zero = jax.tree.map(jnp.zeros_like, params)
    validate_actor_parameters(zero, config)
    # The entire exported baseline, not just a selected observation, has zero
    # parameters. Check application with ordinary and large finite inputs too.
    actor = make_path_actor(config)
    outputs = np.asarray(actor.apply(zero, jnp.stack([jnp.zeros(351), jnp.ones(351), -5*jnp.ones(351)])))
    if not np.array_equal(outputs, np.zeros((3, 2))):
        raise ValueError('zero baseline is not zero latent')
    actor_file = root/'zero_actor.pkl'
    actor_file.write_bytes(pickle.dumps(jax.device_get(zero)))
    zero_record = dict(actor=str(actor_file), actor_sha256=digest(actor_file),
                       checkpoint=str(actor_file), checkpoint_sha256=digest(actor_file),
                       source_update=None, role='synthetic exact-zero latent, no trained checkpoint')
    runtime = [Path(p) for p in original['source_sha256'] if p.endswith('.py')]
    runtime += [Path(__file__).resolve(), Path('learning/src/sttw_control/path_reference.py').resolve()]
    runtime = list(dict.fromkeys(runtime))
    archive = root/'source_at_launch'
    archive.mkdir()
    for i, file in enumerate(runtime):
        (archive/f'{i:02d}_{file.name}').write_bytes(file.read_bytes())
    sources = {str(file): digest(file) for file in runtime}
    panels = []
    for radius in (.9, 1.1):
        name = 'R'+str(radius).replace('.', 'p')
        folder = root/name
        folder.mkdir()
        decl = copy.deepcopy(original)
        decl['config'] = copy.deepcopy(config)
        decl['config']['path']['turn_radius_m'] = radius
        decl.update(experiment_id='low_tight_preference_nominal_v1_'+name,
                    reference_protocol='tight_common_authority_v1',
                    record_policy_inputs=True,
                    question='Do frozen preference choices have useful nominal leverage in low-speed tight turns?',
                    policies=dict(H75=original['policies']['H75'], zero=zero_record),
                    routes=[dict(name='left90_'+name,builder='left90_R2'),
                            dict(name='right90_'+name,builder='right90_R2')],
                    speeds=[1.5,1.8], alphas=[0.,1.], forces_N=[0.],
                    maximum_lower_ticks=32000, source_sha256=sources,
                    code_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                    data_role='new deterministic development scenario; not independent TEST',
                    stop_rule='16 nominal cases once; preserve contact/domain failures; no automatic force or training',
                    reference_protocol_changes=differences,
                    model_config_sha256=None, recorded_unix=time.time())
        atomic_json(folder/'declaration.json', decl)
        panels.append(dict(name=name, radius_m=radius, declaration=str(folder/'declaration.json'),
                           output=str(folder/'evaluation'), cases=16, maximum_lower_ticks=32000))
    plan = dict(experiment='low_tight_preference_nominal_v1', panels=panels,
                source_declaration=str(source), source_declaration_sha256=digest(source),
                source_sha256=sources, changes=differences,
                total_cases=32, maximum_lower_ticks=64000, training_steps=0,
                inherited_models_unchanged=True, command_slew_unchanged=True,
                zero_alpha_duplicates='same controller, distinct reward contexts; not independent replicates',
                limitation='OOD reference authority and tight geometry; no claim of trained capability or causal tolerance gain',
                next_rule='Only consider paired force branches after examining nominal task and physical feasibility; no automatic expansion')
    atomic_json(root/'plan.json', plan)
    atomic_json(root/'zero_actor_receipt.json', dict(outputs=outputs.tolist(),all_parameters_zero=True,
                actor_sha256=digest(actor_file), physical_steps=0, optimization_steps=0))
    print(json.dumps(dict(root=str(root),cases=32,maximum_lower_ticks=64000)))


if __name__ == '__main__':
    main()
