"""Authenticate the fixed native endpoint before any terminal comparison.

This is identity verification, not checkpoint selection, training resumption,
policy adoption or a claim about task/survival performance.
"""
import json
import pickle
from pathlib import Path
from .preference_best import digest, config_hash
from .survival_endpoint_protocol import validate_endpoint_metadata


def verify_native_export(config, native, exported, record):
    """Match native weights/std to saved content fingerprints and Flax export."""
    import jax
    import numpy as np
    import torch
    from .survival_online_learner import OnlineSurvivalLearner
    from .path_command_policy import export_torch_actor
    rng = torch.get_rng_state().clone()
    try:
        obj = OnlineSurvivalLearner.restore(config, native)
        if (obj.fingerprint() != record['state_sha256']
                or obj.fingerprint(sampling_only=True) != record['sampling_sha256']):
            raise ValueError('native endpoint content fingerprint mismatch')
        expected = export_torch_actor(obj.actor, config)
        if jax.tree.structure(expected) != jax.tree.structure(exported):
            raise ValueError('exported Actor structure mismatch')
        for a,b in zip(jax.tree.leaves(expected), jax.tree.leaves(exported)):
            a,b = np.asarray(a),np.asarray(b)
            if a.dtype != b.dtype or not np.isfinite(a).all() or not np.array_equal(a,b):
                raise ValueError('exported Actor is not the native checkpoint weights')
        std = obj.log_std.detach().numpy()
        if std.shape != (2,) or not np.isfinite(std).all():
            raise ValueError('actual saved action distribution invalid')
        return dict(log_std=std.tolist(), state_sha256=obj.fingerprint(),
                    sampling_sha256=obj.fingerprint(sampling_only=True))
    finally:
        torch.set_rng_state(rng)


def authenticate_endpoint(training_run):
    """Read completed64 only; verify provenance, recovery, native and exports.

    No physics, optimizer calls or output files. Zero correction has no trained
    model file; floor is explicitly an alias of parent with a reference rule.
    """
    import torch
    root = Path(training_run).resolve()
    status = json.loads((root/'status.json').read_text())
    if status['phase'] != 'completed' or status['completed_updates'] != 64:
        raise ValueError('completed64 training required; no intermediate endpoint selection')
    plan = json.loads((root/'plan.json').read_text())
    def checked(path, expected):
        if digest(path) != expected: raise ValueError('changed artifact: '+str(path))
        return Path(path)
    checked(root/'effective_config.json', plan['effective_config_sha256'])
    config = json.loads((root/'effective_config.json').read_text())
    for path, sha in plan['source_sha256'].items(): checked(path, sha)
    for key in ('source_plan', 'source_declaration'): checked(plan[key], plan[key+'_sha256'])
    for kind in ('actor', 'checkpoint'): checked(plan['model'][kind], plan['model'][kind+'_sha256'])
    recovery = plan.get('recovery')
    if recovery:
        source = Path(recovery['source_run'])
        checked(source/'plan.json', recovery['source_plan_sha256'])
        checked(source/'status.json', recovery['source_status_sha256'])
        for name,sha in recovery['collection_files'].items(): checked(Path(recovery['source_collection'])/name, sha)
        for arm,path in recovery['initial_learners'].items(): checked(path, recovery['initial_learner_sha256'][arm])
        checked(recovery['successful_task_result'], recovery['successful_task_result_sha256'])
    pair_path = root/'checkpoint_pair_0064.json'
    pair = json.loads(pair_path.read_text())
    if len(pair['models']) != 2 or {m['arm'] for m in pair['models']} != {'task_only','physical_credit'}:
        raise ValueError('both fixed endpoint model records required')
    native = {}; exported = {}
    for model in pair['models']:
        arm = model['arm']
        checked(model['checkpoint'], model['checkpoint_sha256']); checked(model['actor'], model['actor_sha256'])
        native[arm] = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
        with Path(model['actor']).open('rb') as stream: exported[arm] = pickle.load(stream)
    metadata = validate_endpoint_metadata(plan,pair,native,
        plan_sha256=digest(root/'plan.json'),config_sha256=config_hash(config))
    for key,value in pair['cost_counters'].items():
        if status[key] != value: raise ValueError('terminal/pair cost accounting mismatch: '+key)
    models = {}
    for model in pair['models']:
        arm = model['arm']
        identity = verify_native_export(config,native[arm],exported[arm],model)
        models[arm] = dict(model, **identity)
    parent = dict(plan['model'])
    parent_state = torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)
    parent['log_std'] = parent_state['policy']['log_std'].detach().numpy().tolist()
    models.update(parent=parent, parent_speed_floor=dict(parent, alias_of='parent', speed_reference_floor_m_s=1.5),
                  zero_middle=dict(kind='exact_zero_correction', actor=None, checkpoint=None))
    return dict(models=models, metadata=metadata, training_plan_sha256=digest(root/'plan.json'),
                endpoint_pair=str(pair_path), endpoint_pair_sha256=digest(pair_path),
                recovery_authenticated=bool(recovery), config=config,
                scope='authenticated fixed endpoint, not evaluated or adopted')
