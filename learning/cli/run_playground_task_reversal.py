"""Execute fixed departure commands from measured STTW handbacks on one DEV map."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_playground_wide_gap_oracle import DIRECTIONS, run, save


def write_rows(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def branch_route(reference, bank, lane, direction, output):
    # Keep the executed approach verbatim. The frozen departure bank shares a
    # straight entry and all three commands start changing only beyond x=12.
    ref = {key: value.copy() for key, value in reference.items()}
    template = bank['xy'][direction]
    nominal = bank['xy'][0]
    split = np.flatnonzero(abs(template[:, 1] - nominal[:, 1]) > 1e-8)[0] - 1
    s = ref['s']
    xy = ref['xy'][lane]
    join = np.searchsorted(xy[:, 0], template[split, 0])
    local = s[join:] - s[join] + bank['s'][split] + xy[join, 0] - template[split, 0]
    for k in ('heading', 'curvature'):
        ref[k][lane, join:] = np.interp(local, bank['s'], bank[k][direction])
    for dim in (0, 1):
        xy[join:, dim] = np.interp(local, bank['s'], template[:, dim])
    xy[join:, 0] += np.maximum(0, local - bank['s'][-1])
    np.savez_compressed(output, **ref)


def crossings_by_section(trace, handoff, sections):
    t, x, y = trace[:, :3].T
    for name, section in zip(('near', 'mid', 'far'), sections, strict=True):
        crossings = np.flatnonzero((x[:-1] < section) & (x[1:] >= section) & (t[1:] >= handoff))
        cross_t = cross_y = None
        if len(crossings):
            k = crossings[0]
            w = (section - x[k]) / (x[k+1] - x[k])
            cross_t = float(t[k] + w * (t[k+1] - t[k]))
            cross_y = float(y[k] + w * (y[k+1] - y[k]))
        yield name, section, cross_t, cross_y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('route_bank', type=Path)
    parser.add_argument('--capture', action='store_true', help='Replay the frozen screen to capture the first locked ticks before branching')
    args = parser.parse_args()
    out = args.output.resolve()
    plan = json.loads((out/'plan.json').read_text())
    screen = Path(plan['input'])
    cfg = json.loads((screen/'screen.json').read_text())
    repeat = out/'repeat'
    if args.capture:
        run(dict(cfg, snapshot_times_s=plan['snapshots_s'], snapshot_loaded_far=False), repeat, 'source', 10, out)
    world = json.loads(Path(cfg['world_map']).read_text())
    original = json.loads((repeat/'status.json').read_text())['results']
    attempts = json.loads((repeat/'jump_validation.json').read_text())['cases']
    ref = dict(np.load(cfg['oracle_branch_reference']))
    bank = dict(np.load(args.route_bank))
    contract = plan['contracts']
    rows = []
    lanes = plan['lanes']
    same_root = [np.load(repeat/original[i]['case_dir']/'trajectory.npz') for i in lanes]
    for z in same_root[1:]:
        for field in ('qpos', 'qvel', 'controller_state', 'rb_state'):
            np.testing.assert_array_equal(z[field][0], same_root[0][field][0])
    for z in same_root:
        z.close()
    for lane in plan['lanes']:
        case = original[lane]
        hand = case['handoff_s']
        audit = attempts[lane]
        land = min(v for v in (audit['solid_bank_landing_s'], audit['both_far_after_flight_s']) if v is not None)
        assert land <= hand and case['case_dir'] == audit['case']
        filename = Path(case['case_dir'])/'trajectory.npz'
        with np.load(screen/'screen'/filename) as first, np.load(repeat/filename) as again:
            for field in ('qpos', 'qvel', 'controller_state', 'rb_state', 'applied_targets', 'trace'):
                np.testing.assert_array_equal(first[field], again[field])
            k = int(np.flatnonzero(again['trace'][:, -1] == 2)[0])
            hand_state = again['trace'][k].copy()
            q0, v0 = again['qpos'][k+1], again['qvel'][k+1]
            prefix = again['trace'][:k+1]
        # Snapshots precede the discrete handback update at tick k. At k+1 the
        # runner already owns the actuator in phase 2 and preserves the lock/ESO.
        tick = round(hand/.005) + 1
        anchor = repeat/f'source_anchor_{tick:05d}.pkl'
        for direction, name in enumerate(DIRECTIONS):
            path = repeat/filename
            result = case
            if direction:
                child = out/f'lane{lane}_{name}'
                route_path = out/(child.name+'_routes.npz')
                branch_route(ref, bank, lane, direction, route_path)
                suffix = {key: value for key, value in cfg.items()
                          if key not in ('oracle_initial_speed_m_s', 'oracle_initial_alpha')}
                suffix.update(initial_snapshot_path=str(anchor), oracle_branch_reference=str(route_path),
                              ground_handoff=dict(cfg['ground_handoff'], reference_path=str(route_path)),
                              snapshot_loaded_far=False, verify_active_jump=False)
                if (child/'status.json').exists() and json.loads((child/'status.json').read_text()).get('results'):
                    result = json.loads((child/'status.json').read_text())['results'][lane]
                else:
                    result = run(suffix, child, 'source', 10, out)[lane]
                path = child/filename
            with np.load(path) as z:
                trace = z['trace']
                if direction:
                    np.testing.assert_array_equal(q0, z['qpos'][0])
                    np.testing.assert_array_equal(v0, z['qvel'][0])
                    assert trace[0,-1] == 2 and abs(trace[0,0] - tick*.005)<1e-6
                    trace = np.vstack((prefix, trace))
                wall = bool(z['world_flags'][:,2].any())
                for label, x, at, y in crossings_by_section(trace, hand, contract['x_sections_m']):
                    for side in ('left', 'right'):
                        lo, hi = contract[side+'_y_interval_m']
                        gate = y is not None and lo <= y <= hi
                        rows.append(dict(map_id=world['map_id'], root=case['variant']['root_variant'],
                            lane=lane, bin=label, section_x_m=x, task=side, departure=name,
                            landing_s=land, handoff_s=hand, handoff_y_m=hand_state[2],
                            handoff_yaw_rad=hand_state[6],handoff_vy_m_s=hand_state[8],handoff_roll_rad=hand_state[4],
                            cross_s=at, measured_T_s=None if at is None else at-hand, cross_y_m=y,
                            checkpoint=gate, forbidden_contact=wall, goal=result['goal'], hold_s=result['goal_hold_s'],
                            success=bool(gate and not wall and result['goal']), termination=result['termination'],
                            end_s=float(trace[-1,0]), prefix=str(repeat/filename), trajectory=str(path)))
            write_rows(out/'oracle_task_reversal.csv', rows)
    save(out/'complete.json',dict(physical_trajectories=len(plan['lanes'])*3,scored_rows=len(rows)))


if __name__ == '__main__':
    main()
