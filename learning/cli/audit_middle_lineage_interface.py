"""Audit frozen historical interfaces and exact zero-FiLM inference embedding.

No physics, model replacement or PPO resume. Historical tracked code does not
prove absence of untracked changes in the original execution.
"""
import argparse
import ast
import hashlib
import json
import pickle
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from sttw_control.path_command_policy import (
    MODULATION_SHAPES, PathCommandActor, export_torch_actor, make_torch_path_network,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from sttw_control.lineage_inference import embed_legacy_actor


def function_ast(source, name):
    matches = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(matches) != 1:
        raise ValueError('ambiguous function '+name)
    return ast.dump(matches[0], include_attributes=False)


def audit(registry_path, entry, historical, out):
    registry = json.loads(registry_path.read_text()); root = Path(registry['root'])
    if sha(root/'manifest.json') != registry['manifest_sha256']:
        raise ValueError('frozen manifest identity changed')
    manifest = json.loads((root/'manifest.json').read_text())
    current = json.loads((entry/'common_environment_config.json').read_text())
    source = json.loads((entry/'source_declaration.json').read_text())['source']
    frozen_source = manifest['common_frozen_lower']['source']
    common_lower = {k: source['candidate'][k] == frozen_source['candidate'][k]
        for k in ['actor_sha256', 'checkpoint_sha256', 'normalizer']}
    if not all(common_lower.values()):
        raise ValueError('different frozen lower identity/normalizer')
    spec_path = Path(source['upper_run'])/'alpha0/frozen_config_stage52.json'
    frozen_spec = Path(frozen_source['upper_run'])/'alpha0/frozen_config_stage52.json'
    spec = json.loads(spec_path.read_text())
    if sha(spec_path) != sha(frozen_spec):
        raise ValueError('frame specification differs')
    # Use saved real observations from both endpoint preferences, not fabricated states.
    index = json.loads((historical/'case_index.json').read_text())
    metadata = json.loads((historical/'task_analysis/task_results.json').read_text())
    byid = {r['case_id']: r for r in index}; samples = []; sample_sources = []
    for alpha in [0., 1.]:
        row = next(r for r in metadata if r['alpha'] == alpha)
        item = byid[row['case_id']]; path = historical/item['file']
        if sha(path) != item['sha256']:
            raise ValueError('saved observation identity changed')
        with np.load(path) as raw:
            valid = np.flatnonzero(raw['decision_valid'])
            take = valid[np.linspace(0, len(valid)-1, 8).astype(int)]
            samples.append(raw['decision_observation'][take])
        sample_sources.append(dict(case_id=row['case_id'], file=str(path), sha256=item['sha256'], decision_indices=take.tolist()))
    obs = np.concatenate(samples).astype(np.float32)
    if obs.shape != (16,351) or not np.isfinite(obs).all():
        raise ValueError('finite real Actor351 observations required')
    identities = ['DUAL_COVERAGE_U200_A0', 'DUAL_COVERAGE_U200_A1', 'DISTILLED_V3_U325', 'FULLTAIL_PPO75_U77']
    checks = {'path_command_policy.py':['map_latent','correction_tick','assemble_observation'],
              'path_command_env.py':['record_frame','observation'],
              'direct_command_policy.py':['make_frame']}
    reports=[]
    for suffix in identities:
        mid = 'STTW_MIDDLE_'+suffix
        node = next(n for n in manifest['nodes'] if n['id']==mid)
        record = registry['records'][mid]
        cfg_path = root/node['resolved_config']['path']
        if sha(cfg_path) != node['resolved_config']['sha256']:
            raise ValueError('historical resolved config changed')
        cfg = json.loads(cfg_path.read_text())
        for key in ['actor','checkpoint']:
            if sha(record[key]) != record[key+'_sha256']:
                raise ValueError('historical weights changed')
        with Path(record['actor']).open('rb') as f: params=pickle.load(f)
        checkpoint=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
        net=make_torch_path_network(351,2,cfg)
        state={k[len('actor.'):]:v for k,v in checkpoint['policy'].items() if k.startswith('actor.')}
        net.load_state_dict(state,strict=True)
        exported=export_torch_actor(net,cfg)
        if jax.tree.structure(exported)!=jax.tree.structure(params):
            raise ValueError('checkpoint export structure differs')
        for actual, expected in zip(jax.tree.leaves(exported),jax.tree.leaves(params)):
            np.testing.assert_array_equal(actual, expected)
        arch=cfg['network'].get('architecture','legacy')
        output=np.asarray(PathCommandActor(architecture=arch).apply(params,jnp.asarray(obs)))
        with torch.no_grad(): torch_output=net(torch.from_numpy(obs)).numpy()
        np.testing.assert_allclose(output,torch_output,atol=2e-5,rtol=2e-5)
        adapter_error=None
        if arch=='legacy':
            adapted=embed_legacy_actor(params)
            embedded=np.asarray(PathCommandActor(architecture='alpha_modulated_v1').apply(adapted,jnp.asarray(obs)))
            np.testing.assert_allclose(output,embedded,atol=1e-6,rtol=1e-6)
            adapter_error=float(np.max(np.abs(output-embedded)))
        code=manifest['code_archives'][node['code_commit']]; archive=root/code['path']
        if sha(archive)!=code['sha256']:raise ValueError('historical code archive changed')
        ast_checks={}
        with tarfile.open(archive) as tar:
            for filename, functions in checks.items():
                rel='learning/src/sttw_control/'+filename
                old=tar.extractfile(rel).read().decode(); new=Path(rel).read_text()
                for name in functions:
                    ast_checks[filename+':'+name]=function_ast(old,name)==function_ast(new,name)
        observation_keys=['history_frames','frame_fields','history_dt_s','mask_fields','context_fields','actor_dim','norm']
        obs_checks={k:cfg['network'][k]==current['network'][k] for k in observation_keys}
        reports.append(dict(id=mid,architecture=arch,code_commit=node['code_commit'],
            observation_config_equal=obs_checks,tracked_interface_ast_equal=ast_checks,
            checkpoint_actor_export_equal=True,offline_output_finite=bool(np.isfinite(output).all()),
            torch_jax_max_abs_error=float(np.max(np.abs(output-torch_output))),
            zero_modulation_embedding_max_abs_error=adapter_error,
            historical_lower_actor_sha256=checkpoint['identity'].get('lower_actor_sha256'),
            residual_interface=cfg['residual_interface'],follower=cfg['follower'],
            native_network_metadata=cfg['network']))
    out.mkdir(exist_ok=False)
    report=dict(schema='sttw_middle_lineage_interface_audit_v1',common_lower=common_lower,
        plant_frame_spec=dict(path=str(spec_path),sha256=sha(spec_path),
            frame_fields=spec['network']['frame_fields'],
            note='Only the 20 physical frame fields are inherited; legacy plant spec actor345/context9 is NOT the current geometric Actor351/context15 contract.'),
        sample_sources=sample_sources,actual_observations=16,models=reports,
        common_current_residual_interface=current['residual_interface'],
        common_current_follower=current['follower'],new_physics_steps=0,optimizer_updates=0,
        scope='Offline interface and parameter authentication; not historical runtime reproduction, control performance, architecture ablation or physical replay.')
    (out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=audit(Path('learning/configs/frozen_middle_registry.json'),Path('runs/entry_history_confirmation_v1'),Path('runs/online_survival_endpoint_evaluation_v2'),args.out)
    print(json.dumps(dict(models=len(result['models']),observations=16,new_physics_steps=0)))
