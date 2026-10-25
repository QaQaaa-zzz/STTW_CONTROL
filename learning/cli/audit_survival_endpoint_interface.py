"""Trace actual endpoint collection branches without advancing physics or training."""
import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path
from sttw_control.preference_best import atomic_json, digest


def main(training_run, output):
    output.mkdir(parents=True, exist_ok=False)
    state = dict(phase='running', stage='load_real_environment', pid=os.getpid(),
                 physics_steps=0, optimizer_steps=0, completed_modes=0)
    def update(**kw):
        state.update(kw, checked_unix=time.time())
        atomic_json(output/'status.json', state)
    update()
    from prepare_short_recovery_campaign import register
    register(output/'ACTIVE_RUN.json', 'STTW_CONTROL 终点评测真实接口零积分核验', output/'status.json')
    try:
        import jax
        import jax.numpy as j
        import numpy as np
        import torch
        from sttw_control.contact_only_environment import load_environment_contact_only
        from sttw_control.path_command_policy import make_path_actor
        from sttw_control.geometric_path import build_path
        from sttw_control.survival_collection import SurvivalCollectionState
        from sttw_control.survival_endpoint_collection import evaluation_config, make_endpoint_collection_step
        from sttw_control.survival_endpoint_protocol import choose_evaluation_action
        plan = json.loads((training_run/'plan.json').read_text())
        for path, sha in plan['source_sha256'].items():
            if digest(path) != sha: raise ValueError('active training source changed: '+path)
        for key in ('source_plan', 'source_declaration'):
            if digest(plan[key]) != plan[key+'_sha256']: raise ValueError('source changed: '+key)
        if digest(training_run/'effective_config.json') != plan['effective_config_sha256']:
            raise ValueError('training config changed')
        declaration = json.loads(Path(plan['source_declaration']).read_text())
        cfg = json.loads((training_run/'effective_config.json').read_text())
        sources = dict(plan['source_sha256'])
        for path in [Path(__file__), Path('learning/src/sttw_control/survival_endpoint_collection.py'),
                     Path('learning/src/sttw_control/survival_endpoint_protocol.py')]:
            sources[str(path.resolve())] = digest(path)
        atomic_json(output/'plan.json', dict(project='STTW_CONTROL', purpose=__doc__,
            training_plan=str(training_run/'plan.json'), training_plan_sha256=digest(training_run/'plan.json'),
            maximum_physics_steps=0, maximum_optimizer_steps=0, modes=['mean', 'gaussian', 'zero', 'floor_mean'],
            source_sha256=sources, scope='real branch tracing and NN inference; no actual physical transitions'))
        env, initial = load_environment_contact_only(cfg, declaration['source'], 'terminal_contract_v2')
        pose = np.asarray(env.physics.helpers.pose(initial.data))
        route = declaration['routes'][0]
        path = build_path(route['builder'], pose, cfg)
        case = json.loads(Path(plan['source_plan']).read_text())['cases'][0]
        speed = j.float32(case['speed'])
        initial_state = env.reset(initial, path, speed, j.float32(0.))
        env.set_log_template(initial_state, path, speed)
        carry = SurvivalCollectionState(initial_state, jax.random.PRNGKey(2026102429), j.zeros(2), j.bool_(False))
        with Path(plan['model']['actor']).open('rb') as stream:
            params = jax.tree.map(j.asarray, pickle.load(stream))
        checkpoint = torch.load(plan['model']['checkpoint'], map_location='cpu', weights_only=False)
        for kind in ('actor', 'checkpoint'):
            if digest(plan['model'][kind]) != plan['model'][kind+'_sha256']: raise ValueError('parent changed')
        log_std = j.asarray(checkpoint['policy']['log_std'].numpy())
        pulse = j.zeros(5)
        rows = []
        for label in ('mean', 'gaussian', 'zero', 'floor_mean'):
            mode = 'mean' if label == 'floor_mean' else label
            local = copy.copy(env)
            local.config = evaluation_config(cfg, speed_floor=1.5 if label == 'floor_mean' else None)
            actor = None if mode == 'zero' else make_path_actor(cfg)
            step = make_endpoint_collection_step(local, actor, horizon_s=10., mode=mode)
            abstract = jax.eval_shape(step, carry, params, log_std, path, speed, pulse)
            decision, logs = abstract[1]
            if decision['raw_action'].shape != (2,) or logs['reference'].shape != (4, 2):
                raise ValueError('actual environment output interface mismatch')
            obs, _, fault = local.observation(initial_state, path, speed, 10.)
            mean = j.zeros(2) if mode == 'zero' else actor.apply(params, obs)
            action, logp, valid, key = choose_evaluation_action(mean, log_std, carry.key, mode=mode)
            if bool(fault) or not np.isfinite(action).all(): raise ValueError('initial inference fault')
            rows.append(dict(mode=label, traced=True, action=np.asarray(action).tolist(),
                density_valid=bool(valid), log_probability=float(logp) if bool(valid) else None,
                rng_advanced=not np.array_equal(key, carry.key), lower_log_fields=sorted(logs),
                output_shapes={k:list(v.shape) for k,v in logs.items()}))
            update(stage='tracing_real_branches', completed_modes=len(rows))
        if int(carry.state.tick) != 0: raise ValueError('shape audit advanced task state')
        if any(digest(p) != sha for p, sha in sources.items()): raise ValueError('audit source changed during execution')
        atomic_json(output/'receipt.json', dict(modes=rows, physics_steps=0, optimizer_steps=0,
            unchanged_training_source_count=len(plan['source_sha256']),
            initial_task_tick=int(carry.state.tick), parent_actor_sha256=plan['model']['actor_sha256'],
            limits='Abstract branch agreement and real NN inference only; physical stepping and performance unverified.'))
        update(phase='completed', stage='actual_branches_traced_no_physics')
    except BaseException as exc:
        update(phase='error', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    main(args.training_run.resolve(), args.output.resolve())
