"""Measure flight across a level trench from recorded forces and vehicle poses."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def measure(model, path, terrain):
    ids = [model.geom(f'{name}wheel_collision').id for name in ('front', 'rear')]
    if any(model.geom_type[i] != mujoco.mjtGeom.mjGEOM_ELLIPSOID for i in ids):
        raise ValueError('Wheel clearance calculation requires ellipsoid collision geoms')
    near = terrain['front_x_m']
    far = near + terrain['width_m']
    bank_z = max(0., terrain['far_bank_height_m'])
    data = mujoco.MjData(model)
    with np.load(path) as archive:
        q = archive['qpos']
        time = archive['trace'][:, 0]
        force = archive['contact_substep']
        both_far = archive['wheel_contact_flags'][:, 2].astype(bool)
        dt = float(archive['contact_physics_dt_s'])
    nsub = force.shape[1]
    x, y, bottom, radius_x = (np.empty((len(q), 2)) for _ in range(4))
    com = np.empty(len(q))
    for k, pose in enumerate(q):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        com[k] = np.average(data.xipos[:, 2], weights=model.body_mass)
        centers = data.geom_xpos[ids]
        x[k], y[k] = centers[:, 0], centers[:, 1]
        axes = data.geom_xmat[ids].reshape(2, 3, 3)
        radius_x[k] = np.linalg.norm(axes[:, 0, :] * model.geom_size[ids], axis=1)
        axes_z = axes[:, 2, :]
        bottom[k] = centers[:, 2] - np.linalg.norm(axes_z * model.geom_size[ids], axis=1)
    unloaded = ((force[:, :, :3] <= 1e-5).all(axis=2) & (force[:, :, 3] == 0)).ravel()
    edges = np.diff(np.r_[False, unloaded, False].astype(int))
    flights = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        # Force row k covers the interval ending at pose k. Use only poses
        # strictly inside the force-free interval for wheel and COM evidence.
        first = (start + nsub - 1) // nsub
        last = end // nsub - 1
        if last <= first or x[last, 0] < near or x[first, 1] > far:
            continue
        sl = slice(first, last + 1)
        center = x[sl].mean(axis=1)
        over_gap = (center >= near) & (center <= far)
        if not over_gap.any():
            continue
        vz = (com[first + 1] - com[first]) / (time[first + 1] - time[first])
        clearance = bottom[sl][over_gap].min() - bank_z
        rise = com[sl].max() - com[first]
        bridge = terrain.get('land_bridge_y_interval_m')
        outside_bridge = bridge is None or np.all(y[sl] < bridge[0]) or np.all(y[sl] > bridge[1])
        qualifies = x[first, 0] < near and x[last, 1] > far and vz > 0 and clearance > 0 and outside_bridge
        # The front wheel may already extend past the lip when the rear wheel pushes off.
        prior_pose = start // nsub - 1
        previous_load = force.reshape(-1, 4)[start - 1] if start else np.zeros(4)
        # Each bank is contained in its x/z half-spaces in the flat lip strip.
        # Distance between bounding boxes is a lower bound on wheel-to-bank
        # clearance. A wheel may dip below lip elevation inside the open gap.
        near_dx = np.maximum(x[sl] - radius_x[sl] - near, 0.)
        far_dx = np.maximum(far - x[sl] - radius_x[sl], 0.)
        near_dist = np.hypot(near_dx, np.maximum(bottom[sl], 0.))
        far_dist = np.hypot(far_dx, np.maximum(bottom[sl] - terrain['far_bank_height_m'], 0.))
        solid_clearance = float(np.minimum(near_dist, far_dist).min())
        rear_takeoff = (prior_pose >= 0 and previous_load[1] > 1e-5 and previous_load[2] <= 1e-5
                       and previous_load[3] == 0 and x[prior_pose, 1] < near
                       and x[first, 1] < near and x[first, 0] < far and x[last, 1] > far
                       and vz > 0 and rise > model.geom_size[ids[1]].max()
                       and outside_bridge)
        rear_launch = rear_takeoff and clearance > 0
        solid_crossing = rear_takeoff and solid_clearance > 0
        flights.append(dict(rear_supported_launch=bool(rear_launch),solid_bank_rear_launch=bool(solid_crossing),
                            wheel_to_solid_bank_lower_bound_m=solid_clearance,start_s=float(time[first]), end_s=float(time[last]), unloaded_ms=(end-start)*dt*1000,
                            initial_com_vz_m_s=float(vz), com_rise_m=float(rise),
                            clearance_over_gap_m=float(clearance), front_start_x=float(x[first,0]),
                            rear_end_x=float(x[last,1]), bank_to_bank_flight=bool(qualifies)))
    loaded_far = both_far & (force[:, -1, :2] > 1e-5).all(axis=1)
    return flights, time[loaded_far]


def assess(evaluation):
    model = mujoco.MjModel.from_binary_path(str(evaluation/'terrain.mjb'))
    terrain = json.loads((evaluation/'resolved_config.json').read_text())['ditch_terrain']
    runs = json.loads((evaluation/'status.json').read_text())['results']
    records = []
    for row in runs:
        flights, supported_times = measure(model, evaluation/row['case_dir']/'trajectory.npz', terrain)
        flight = any(f['bank_to_bank_flight'] for f in flights)
        landing_time = rear_landing_time = solid_landing_time = None
        for event in flights:
            times = supported_times[supported_times >= event['end_s']]
            if len(times):
                if event['bank_to_bank_flight'] and landing_time is None:
                    landing_time = float(times[0])
                if event['solid_bank_rear_launch'] and solid_landing_time is None:
                    solid_landing_time = float(times[0])
                if event['rear_supported_launch'] and rear_landing_time is None:
                    rear_landing_time = float(times[0])
        handoff = row['handoff_s']
        task_success = (landing_time is not None and handoff is not None
                        and landing_time <= handoff + 1e-5 and row['goal'])
        rear_handoff = rear_landing_time is not None and handoff is not None and rear_landing_time <= handoff + 1e-5
        solid_handoff = solid_landing_time is not None and handoff is not None and solid_landing_time <= handoff + 1e-5
        records.append(dict(solid_bank_landing_s=solid_landing_time,solid_bank_goal=bool(solid_handoff and row['goal']),
                            rear_supported_launch=any(e['rear_supported_launch'] for e in flights),
                            rear_launch_both_far_s=rear_landing_time, rear_launch_handoff=rear_handoff,
                            rear_launch_goal=bool(rear_handoff and row['goal']), task_success=bool(task_success), both_far_after_flight_s=landing_time, case=row['case_dir'], skill=row['skill'], flight=flight,
                            far_contact=row['far_contact'], handoff_s=row['handoff_s'], goal=row['goal'],
                            termination=row['termination'], events=flights))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    records = assess(args.evaluation)
    args.output.write_text(json.dumps(records, indent=2)+'\n')
    print(f'{sum(r["flight"] for r in records)}/{len(records)} bank-to-bank flights, '
          f'{sum(r["task_success"] for r in records)} strict full tasks, '
          f'{sum(r["solid_bank_goal"] for r in records)} rear-launch full tasks with solid-bank clearance')


if __name__ == '__main__':
    main()
