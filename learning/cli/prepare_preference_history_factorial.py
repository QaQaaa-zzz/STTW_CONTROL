"""Prepare one 24-branch H75 entry-history x future-preference diagnostic."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import jax
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path.cwd()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    source = base / 'runs/tight_midturn_capability_v1'
    bundle = json.loads((source / 'prefix_bundle.json').read_text())
    declaration = Path(bundle['source_declaration'])
    decl = json.loads(declaration.read_text())
    model = decl['policies']['H75']
    registry = json.loads((base / 'learning/configs/frozen_middle_registry.json').read_text())
    frozen = registry['records']['STTW_MIDDLE_FULLTAIL_PPO75_U77']
    assert sha(model['actor']) == frozen['actor_sha256'] == model['actor_sha256']
    evidence = Path(json.loads((source / 'left/prepare_plan.json').read_text())['path_evidence_directory'])
    records = [r for r in bundle['cases'] if r['speed'] == 1.5]
    assert len(records) == 4
    directions, path_receipts = {}, {}
    for route in sorted({r['route'] for r in records}):
        path_file = evidence / (route + '_path.npz')
        with np.load(path_file) as p:
            xy = p['xy'].astype(float)
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        targets = [r['target_progress_m'] for r in records if r['route'] == route]
        assert max(targets) - min(targets) < 1e-6
        index = int(np.searchsorted(distance, targets[0]))
        tangent = xy[index + 1] - xy[index - 1]
        tangent /= np.linalg.norm(tangent)
        directions[route] = [-float(tangent[1]), float(tangent[0]), 0.]
        path_receipts[route] = dict(path=str(path_file), sha256=sha(path_file),
                                    target_progress_m=targets[0], tangent_index=index)
    lineage, cases = [], []
    snapshot_dir = output / 'snapshots'
    snapshot_dir.mkdir()
    for record in records:
        original = Path(record['snapshot'])
        assert sha(original) == record['snapshot_sha256']
        state = pickle.loads(original.read_bytes())
        assert float(state.alpha) == record['alpha']
        assert not bool(state.fault | state.domain_exit | state.physical.failed)
        for suffix in (0., 1.):
            changed = state.replace(alpha=np.asarray(suffix, dtype=np.asarray(state.alpha).dtype))
            restored = changed.replace(alpha=state.alpha)
            before, after = jax.tree.leaves(state), jax.tree.leaves(restored)
            assert len(before) == len(after)
            for a, b in zip(before, after):
                np.testing.assert_array_equal(a, b)
            target = snapshot_dir / (record['state_id'] + f'_future{suffix:g}.pkl')
            target.write_bytes(pickle.dumps(changed))
            lineage.append(dict(state_id=record['state_id'], original=str(original),
                                original_sha256=sha(original), snapshot=str(target),
                                snapshot_sha256=sha(target), entry_alpha=record['alpha'],
                                future_alpha=suffix, only_changed_field='alpha',
                                restored_leaf_count=len(before), original_task_tick=int(state.tick),
                                original_physical_tick=int(state.physical.physical_tick)))
            for force in (0., -32., 32.):
                cases.append(dict(state_id=record['state_id'], policy='H75',
                                  snapshot=str(target), route=record['route'],
                                  requested_speed=1.5, entry_alpha=record['alpha'],
                                  future_alpha=suffix, alpha=suffix, force_N=force,
                                  gate_mode=0, world_direction=directions[record['route']]))
    files = list((base / 'learning/src/sttw_control').glob('*.py'))
    files += [Path(__file__).resolve(), base / 'learning/cli/probe_sensor_gate_domain_hold.py',
              declaration, Path(model['actor']), Path(model['checkpoint']),
              source / 'prefix_bundle.json']
    files += [Path(r['snapshot']) for r in lineage]
    files += [Path(r['original']) for r in lineage]
    files += [Path(r['path']) for r in path_receipts.values()]
    sources = {str(p.resolve()): sha(p) for p in files}
    archive = output / 'source_at_launch'
    for p in files:
        if p.suffix == '.py':
            target = archive / p.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
    write(output / 'snapshot_lineage.json', dict(cases=lineage, paths=path_receipts))
    plan = dict(project='STTW_CONTROL', experiment_id='preference_history_factorial_v1',
                question='Entry-history versus future-alpha intervention within one frozen H75',
                source_declaration=str(declaration), path_evidence_directory=str(evidence),
                actors={'H75': dict(path=model['actor'], sha256=model['actor_sha256'])},
                cases=cases, source_sha256=sources,
                sensor_contract={'rear_wheel_radius_proxy_m': .1},
                pulse_duration_s=.1, observation_s=4., maximum_lower_ticks=19200,
                training_steps=0, declared_cases=24, new_prefix_ticks=0,
                direction_contract='same route fixed original path midpoint world normal; not snapshot heading',
                intervention='future Actor alpha only; original user preference not redefined',
                termination='contact/fault; domain exit keeps identical held-reference lower continuation',
                primary_outcome='real non-wheel contact or full four-second observation; fault/unknown separate',
                scope='posthoc DEV mechanism experiment, not deployment switching or task-arrival validation',
                stop_rule='One fixed 24-case panel; report saturation/no output contrast; no force/time/library expansion')
    write(output / 'plan.json', plan)
    write(output / 'status.json', dict(phase='prepared', declared_cases=24,
                                       active_lower_ticks=0, completed_cases=0, training_steps=0))
    print(json.dumps(dict(output=str(output), cases=len(cases), snapshots=len(lineage),
                          restored_carry_verified=True, new_physics_ticks=0)))


if __name__ == '__main__':
    main()
