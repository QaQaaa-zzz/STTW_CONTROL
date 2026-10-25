"""Five 16-tick real collector branches; engineering only, not efficacy evaluation."""
import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path
from sttw_control.preference_best import atomic_json, digest


def main(training_run, interface_audit, output):
    output.mkdir(parents=True, exist_ok=False)
    state = dict(phase='running', stage='loading', pid=os.getpid(), active_lower_ticks=0,
                 maximum_lower_ticks=80, optimizer_steps=0, completed_branches=0)
    def update(**kw):
        state.update(kw, checked_unix=time.time())
        atomic_json(output/'status.json', state)
    update()
    from prepare_short_recovery_campaign import register
    register(output/'ACTIVE_RUN.json', 'STTW_CONTROL 终点评测采集器80步工程验证', output/'status.json')
    try:
        import jax
        import jax.numpy as j
        import numpy as np
        import torch
        from sttw_control.contact_only_environment import load_environment_contact_only
        from sttw_control.path_command_policy import make_path_actor
        from sttw_control.geometric_path import build_path
        from sttw_control.survival_collection import SurvivalCollectionState, make_survival_collection_step
        from sttw_control.survival_endpoint_collection import evaluation_config, make_endpoint_collection_step
        from sttw_control.runtime import configure_compilation_cache
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        if json.loads((interface_audit/'status.json').read_text())['phase'] != 'completed':
            raise ValueError('actual zero-integration interface audit must be complete')
        audited = json.loads((interface_audit/'plan.json').read_text())
        for path, sha in audited['source_sha256'].items():
            if digest(path) != sha: raise ValueError('audited source changed: '+path)
        plan = json.loads((training_run/'plan.json').read_text())
        if digest(training_run/'plan.json') != audited['training_plan_sha256']:
            raise ValueError('training plan changed')
        sources = dict(audited['source_sha256']); sources[str(Path(__file__).resolve())] = digest(__file__)
        atomic_json(output/'plan.json', dict(project='STTW_CONTROL', purpose=__doc__,
            training_run=str(training_run), interface_audit=str(interface_audit),
            source_sha256=sources, maximum_lower_ticks=80, optimizer_steps=0,
            branches=['gaussian', 'legacy_gaussian', 'mean', 'zero', 'floor_mean'],
            horizon_s=10., actual_probe_s_per_branch=.08, seed=2026102429,
            stop='Exactly4 upper intervals per branch, no extension, performance inference or automatic retry'))
        cfg = json.loads((training_run/'effective_config.json').read_text())
        declaration = json.loads(Path(plan['source_declaration']).read_text())
        env, initial = load_environment_contact_only(cfg, declaration['source'], 'terminal_contract_v2')
        pose = np.asarray(env.physics.helpers.pose(initial.data)); route = declaration['routes'][0]
        path = build_path(route['builder'], pose, cfg)
        case = json.loads(Path(plan['source_plan']).read_text())['cases'][0]
        speed = j.float32(case['speed']); s = env.reset(initial, path, speed, j.float32(0.))
        env.set_log_template(s, path, speed)
        carry = SurvivalCollectionState(s, jax.random.PRNGKey(2026102429), j.zeros(2), j.bool_(False))
        with Path(plan['model']['actor']).open('rb') as stream: params = jax.tree.map(j.asarray, pickle.load(stream))
        checkpoint = torch.load(plan['model']['checkpoint'], map_location='cpu', weights_only=False)
        std = j.asarray(checkpoint['policy']['log_std'].numpy()); pulse = j.zeros(5)
        for kind in ('actor', 'checkpoint'):
            if digest(plan['model'][kind]) != plan['model'][kind+'_sha256']: raise ValueError('parent identity changed')
        with (output/'initial_carry.pkl').open('wb') as stream: pickle.dump(jax.device_get(carry), stream)
        rows = []; finals = {}; raw = {}
        for label in ('gaussian', 'legacy_gaussian', 'mean', 'zero', 'floor_mean'):
            local = copy.copy(env)
            local.config = evaluation_config(cfg, speed_floor=1.5 if label == 'floor_mean' else None)
            mode = 'mean' if label == 'floor_mean' else label
            actor = None if mode == 'zero' else make_path_actor(cfg)
            one = (make_survival_collection_step(local, actor, horizon_s=10.) if label == 'legacy_gaussian'
                   else make_endpoint_collection_step(local, actor, horizon_s=10., mode=mode))
            def run(c):
                return jax.lax.scan(lambda st, _: one(st, params, std, path, speed, pulse), c, None, length=4)
            update(stage='compile_and_step_'+label)
            started = time.monotonic()
            end, (decision, logs) = jax.jit(run)(carry); jax.block_until_ready(end)
            end, decision, logs = jax.device_get((end, decision, logs))
            logs = {k:v.reshape((16,)+v.shape[2:]) for k,v in logs.items()}
            np.savez_compressed(output/(label+'.npz'), **{'decision_'+k:v for k,v in decision.items()}, **logs)
            with (output/(label+'_final.pkl')).open('wb') as stream: pickle.dump(end, stream)
            count = int(logs['active_tick'].sum()); state['active_lower_ticks'] += count
            rows.append(dict(mode=label, active_lower_ticks=count, elapsed_s=time.monotonic()-started,
                actual_final_tick=int(end.state.tick), contact=bool(end.state.physical.failed),
                fault=bool(end.state.fault), floor_clipped_ticks=int(logs.get('reference_clip_channels', np.zeros((16,2),bool))[:,0].sum())))
            raw[label] = dict(decision=decision, logs=logs); finals[label] = end
            update(completed_branches=len(rows))
        # Raw logs and full final states precede all outcome assertions.
        if state['active_lower_ticks'] > 80: raise ValueError('engineering tick limit exceeded')
        if any(r['fault'] for r in rows): raise ValueError('engineering fault in real collector branch')
        for mode in ('mean', 'floor_mean', 'zero'):
            d = raw[mode]['decision']; valid = d['valid']
            if np.any(d['density_valid'][valid]) or np.any(np.isfinite(d['log_prob'][valid])):
                raise ValueError('deterministic branch claimed Gaussian density')
            if not np.array_equal(finals[mode].key, np.asarray(carry.key)):
                raise ValueError('deterministic inference consumed RNG')
            expected = np.zeros_like(d['raw_action']) if mode == 'zero' else d['mean']
            if not np.array_equal(d['raw_action'][valid], expected[valid]):
                raise ValueError('deterministic action logging mismatch')
        diffs = {}
        for key in ('raw_action', 'mean', 'log_prob'):
            a=raw['gaussian']['decision'][key]; b=raw['legacy_gaussian']['decision'][key]
            diffs['decision_'+key+'_max_abs']=float(np.nanmax(np.abs(a-b)))
        for key in ('actual_xy', 'phi', 'actual_forward_speed', 'reference'):
            a=raw['gaussian']['logs'][key]; b=raw['legacy_gaussian']['logs'][key]
            diffs[key+'_max_abs']=float(np.nanmax(np.abs(a-b)))
        if any(digest(p) != sha for p,sha in sources.items()): raise ValueError('source changed during probe')
        atomic_json(output/'receipt.json', dict(branches=rows, gaussian_vs_legacy=diffs,
            active_lower_ticks=state['active_lower_ticks'], optimizer_steps=0,
            scope='Only initial80 total lower ticks; no pulse, complete-window survival, threshold or learned benefit tested.'))
        np.savez_compressed(output/'reference_path.npz', xy=np.asarray(path.xy))
        update(phase='completed', stage='five_initial_branches_checked')
    except BaseException as exc:
        update(phase='error', error=repr(exc))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run', required=True, type=Path)
    p.add_argument('--interface-audit', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a=p.parse_args(); main(a.training_run.resolve(), a.interface_audit.resolve(), a.output.resolve())
