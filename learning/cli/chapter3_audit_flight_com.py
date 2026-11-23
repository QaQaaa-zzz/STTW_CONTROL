"""Check the horizontal COM impulse law against an archived paired three-force experiment."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np


def write_rows(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def horizontal_com(model, body, qpos, qvel):
    data = mujoco.MjData(model)
    jacobian = np.zeros((3, model.nv))
    positions, velocities = [], []
    for q, v in zip(qpos, qvel, strict=True):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSubtreeCom(model, data, jacobian, body)
        positions.append(data.subtree_com[body, :2].copy())
        velocities.append(jacobian[:2] @ v)
    return np.array(positions), np.array(velocities)


def audit(oracle, layout, root, output):
    # Each planned exit shares the prefix. Use one measured exit and one gate label per attempt.
    with oracle.open() as stream:
        cases = [r for r in csv.DictReader(stream) if r['layout'] == layout and int(r['root']) == root
                 and r['skill'] == 'rb_run1' and r['planned_dep'] == '0' and r['section'] == 'mid'
                 and r['side'] == '-1']
    groups = defaultdict(dict)
    for r in cases:
        group = groups[int(r['approach']), float(r['approach_alpha'])]
        force = int(r['force'])
        if force in group:
            raise ValueError('Duplicate archived attempt')
        group[force] = r
    if len(groups) != 5 or any(set(g) != {-5, 0, 5} for g in groups.values()):
        raise ValueError('This panel requires all five original approaches and all three forces')
    models = {Path(r['trajectory']).parents[1] / 'terrain.mjb' for r in cases}
    hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in models}
    if len(hashes) != 1:
        raise ValueError('Compared histories use different model binaries')
    model_path = sorted(models)[0]
    model = mujoco.MjModel.from_binary_path(str(model_path))
    joint, = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    body = model.jnt_bodyid[joint]
    start = model.jnt_dofadr[joint]
    linear = slice(start, start + 3)
    if (np.any(model.opt.gravity[:2]) or model.opt.density or model.opt.viscosity
            or np.any(model.dof_damping[linear]) or np.any(model.dof_armature[linear])
            or np.any(model.dof_frictionloss[linear]) or model.jnt_stiffness[joint]):
        raise ValueError('Model has additional translational inertia or passive forces')
    mass = float(model.body_subtreemass[body])
    dt = json.loads((model_path.parent / 'declaration.json').read_text())['lower_dt']
    steps = 20  # Archived rising-edge force lasts 20 control updates.
    seconds = np.arange(steps + 1) * dt
    records, traces, contrasts = [], [], []
    for (approach, alpha), group in sorted(groups.items()):
        series, initial, onsets, summaries = {}, {}, [], []
        for force, r in sorted(group.items()):
            path = Path(r['trajectory'])
            onset = int(r['force_onset'])
            onsets.append(onset)
            frames = slice(onset, onset + steps + 1)
            after = slice(onset + 1, onset + steps + 1)
            with np.load(path) as z:
                loads = z['contact_substep'][after]
                ground = z['wheel_contact_flags'][after, 0]
                other = z['world_flags'][after, 1:].any(axis=1)
                disturbed = (np.abs(loads).max(axis=(1, 2)) > 0) | ~np.isfinite(loads).all(axis=(1, 2))
                flagged = np.flatnonzero(disturbed | ground | other)
                clean_ticks = int(flagged[0]) if len(flagged) else len(loads)
                applied = int(r['force_steps'])
                row = dict(approach=approach, alpha=alpha, force_n=force, onset_tick=onset,
                           applied_ticks=applied, clean_ticks_from_onset=clean_ticks if onset >= 0 else 0,
                           trajectory=str(path))
                if onset >= 0 and clean_ticks == steps and applied == (steps if force else 0):
                    np.testing.assert_allclose(z['trace'][frames, 0], onset * dt + seconds, rtol=0, atol=1e-6)
                    q, v = z['qpos'][frames], z['qvel'][frames]
                    initial[force] = (q[0], v[0])
                    series[force] = horizontal_com(model, body, q, v)
                summaries.append(row)
        matched = (len(series) == 3 and len(set(onsets)) == 1 and all(
            np.array_equal(a, b) for f in (-5, 5) for a, b in zip(initial[f], initial[0], strict=True)))
        for row in summaries:
            row['eligible_triple'] = matched
            records.append(row)
        if not matched:
            continue
        for force, (xy, vel) in sorted(series.items()):
            tangent = xy[0] + seconds[:, None] * vel[0]
            ideal = tangent.copy()
            ideal[:, 1] += .5 * force / mass * seconds**2
            for k, t in enumerate(seconds):
                traces.append(dict(approach=approach, alpha=alpha, force_n=force, step=k, t_s=t,
                     com_x_m=xy[k, 0], com_y_m=xy[k, 1], ideal_x_m=ideal[k, 0], ideal_y_m=ideal[k, 1],
                     extra_y_m=xy[k, 1] - tangent[k, 1], ideal_extra_y_m=.5 * force / mass * t**2,
                     com_vy_m_s=vel[k, 1], ideal_vy_m_s=vel[0, 1] + force / mass * t))
        t = seconds[-1]
        contrasts.append(dict(approach=approach, alpha=alpha, onset_tick=onsets[0], duration_s=t,
                              measured_plus_minus_y_m=series[5][0][-1, 1] - series[-5][0][-1, 1],
                              ideal_plus_minus_y_m=5*t*t/mass,
                              measured_plus_minus_vy_m_s=series[5][1][-1, 1] - series[-5][1][-1, 1],
                              ideal_plus_minus_vy_m_s=10*t/mass))
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/'flight_com_windows.csv', records, list(records[0]))
    trace_fields = ['approach','alpha','force_n','step','t_s','com_x_m','com_y_m','ideal_x_m','ideal_y_m',
                    'extra_y_m','ideal_extra_y_m','com_vy_m_s','ideal_vy_m_s']
    write_rows(output/'flight_com_trace.csv', traces, trace_fields)
    receipt = dict(layout=layout, root=root, oracle=str(oracle), model=str(model_path), model_sha256=hashes.pop(),
                   mass_kg=mass, body=model.body(body).name, mujoco=mujoco.__version__, control_dt_s=dt,
                   archived_attempts=len(records), complete_matched_triples=len(contrasts), contrasts=contrasts,
                   max_abs_y_residual_m=max((abs(r['com_y_m']-r['ideal_y_m']) for r in traces), default=None),
                   selection='All five approaches for one diagnostic root; zero recorded substep loads/body flag, zero terrain and world contact flags during each of the 20 intervals. Identical onset qpos/qvel for all three forces. Reconstructed from archived qpos and qvel without integrating dynamics.',
                   new_physics=0)
    (output/'flight_com_summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('oracle', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--layout', required=True)
    parser.add_argument('--root', type=int, required=True)
    args = parser.parse_args()
    audit(args.oracle, args.layout, args.root, args.output)
