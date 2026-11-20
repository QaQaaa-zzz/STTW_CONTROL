"""Paired Test 0 on the archived native step, with chapter-three physics timing."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ARCHIVE = Path('/home/qy/DVGC/JIT/runs/experiments/recoverybridge_reviewer_controls_20261022/nominal')
CODE = Path('/home/qy/DVGC/JIT/runs/monitoring/ablation_paper_audit_20261020/five_windows10k/code')
ORIGINAL = Path('/home/qy/DVGC/runs/worktrees/multidim-envelope')
POLICIES = {'source': 'pi0', 'rb_run1': 'full_8113001_r9'}


def run(out, count, dt_ms, integrator):
    sys.path[:0] = [str(CODE), str(ORIGINAL / 'JIT/src')]
    os.chdir(ORIGINAL)
    import jax
    import jax.numpy as jp
    import mujoco
    import rollout
    from jit_dvgc.model import put_warp_model
    from sttw_control.playground_contact import normal_loads

    if jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required')
    for name, directory in POLICIES.items():
        parent = ARCHIVE / directory
        spec = json.loads((parent / 'specs/batch_000.json').read_text())
        bank = np.load(spec['initial_state_bank'])
        q, v = bank['qpos'][:count], bank['qvel'][:count]
        spec.update(num_envs=count, pulse_start_schedule=spec['pulse_start_schedule'][:count])
        session = rollout.EvaluationSession()
        context = session._context(spec)
        env = context[0]
        m = env.mj_model
        m.opt.timestep = env._sim_dt = dt_ms / 1000
        m.opt.integrator = getattr(mujoco.mjtIntegrator, 'mjINT_' + integrator.upper())
        env._bundle = put_warp_model(replace(env._bundle, mjx_model=None))
        assert np.isclose(env.n_substeps * m.opt.timestep, .020) and np.isclose(env.dt, .020)
        step = m.geom('step').id
        floor = m.geom('floor').id
        wheels = tuple(m.geom(w + 'wheel_collision').id for w in ('front', 'rear'))
        bodies = jp.asarray([i for i in range(m.ngeom) if m.geom_bodyid[i] and m.geom_contype[i] and i not in wheels])
        solid = jp.asarray(np.isin(np.arange(m.ngeom), [floor, step]))
        top = jp.asarray(np.arange(m.ngeom) == step)
        height = float(m.geom_pos[step, 2] + m.geom_size[step, 2])

        def loads(data):
            all_load = normal_loads(data, solid, wheels, bodies)
            upper = normal_loads(data, top, wheels, bodies)
            return {prefix + str(i): values[:, i]
                    for prefix, values in [('all_load', all_load), ('step_load', upper)]
                    for i in range(3)}

        initial_trace = rollout.physical_trace
        step_trace = rollout.physical_trace_from_step
        def trace_initial(model, state):
            return {**initial_trace(model, state), **loads(state.data)}

        def trace_step(state):
            return {**step_trace(state), **loads(state.data)}

        print(f'{name}: executing {count} worlds, GPU {env.n_substeps} x {dt_ms} ms {integrator}', flush=True)
        with patch.object(rollout, 'physical_trace', trace_initial), patch.object(rollout, 'physical_trace_from_step', trace_step):
            kernel = rollout._make_kernel(context, spec, 1, early_stop=True, light_metrics=True, trace_counter=[0])
            _, ticks, tape = kernel(np.array([spec['seed']], np.uint32), q, v,
                                    np.array(spec['pulse_start_schedule'], np.int32),
                                    np.zeros((count, 4), np.float32))
        keys = ['prefix_mask', 'qpos', 'qvel', 'success', 'end_code', 'finite', 'capacity_exceeded',
                'time_after', 'front_wheel_clearance', 'rear_wheel_clearance',
                'valid_contact_seen', 'contact_x', 'recovery_ticks']
        keys += [prefix + str(i) for prefix in ('all_load', 'step_load') for i in range(3)]
        tape = jax.device_get({k: tape[k][:int(ticks)] for k in keys})
        if np.any(tape['prefix_mask'] & (~tape['finite'] | tape['capacity_exceeded'])):
            raise RuntimeError('Nonfinite state or physics capacity exceeded')
        np.savez_compressed(out / (name + '.npz'), **tape, initial_qpos=q, initial_qvel=v)
        provenance = dict(actor=spec['evaluation_payload']['actor_sha256'], bank=spec['initial_state_bank'],
                          spec=str(parent / 'specs/batch_000.json'), step_height_m=height,
                          dt=env.dt, n_substeps=env.n_substeps, physics_dt=m.opt.timestep,
                          integrator=int(m.opt.integrator), backend=jax.default_backend())
        (out / (name + '.json')).write_text(json.dumps(provenance, indent=2))
        print(name, 'native recovery', int((tape['success'] & tape['prefix_mask']).any(0).sum()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--dt-ms', type=int, choices=[1, 5], default=1)
    parser.add_argument('--integrator', choices=['implicitfast', 'rk4'], default='implicitfast')
    args = parser.parse_args()
    run(args.output.resolve(), args.count, args.dt_ms, args.integrator)
