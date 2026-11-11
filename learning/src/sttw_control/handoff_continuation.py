"""Retrospective nominal-envelope labels; not a physical viability certificate."""
import numpy as np


def mounted_features(root, mount_wxyz, requested_speed=1.5):
    q = root[..., 3:7]
    m = np.asarray(mount_wxyz)
    xyz = q[..., 3:] * m[1:] + m[0] * q[..., :3] + np.cross(q[..., :3], m[1:])
    w = q[..., 3] * m[0] - (q[..., :3] * m[1:]).sum(-1)
    x, y, z = np.moveaxis(xyz, -1, 0)
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return np.stack((roll, pitch, yaw, root[..., 1], root[..., 7] - requested_speed), -1)


def label_window(time, root, features, road_bad, bounds, handoff_s, end_code,
                 hold_s=.2, capture_deadline_s=1., horizon_s=2., min_progress_m=1.):
    """Arrays contain actual valid post-step samples, including terminal sample.

    Entry requires observed endpoints separated by hold_s, not ten instantaneous
    samples mistaken for .2s. Goal or timeout before horizon is right-censored.
    """
    after = np.flatnonzero(time >= handoff_s - 1e-6)
    t = time[after] - handoff_s
    inside = (np.abs(features[after]) <= bounds).all(-1) & ~road_bad[after]
    capture = None
    start = None
    for i, ok in enumerate(inside):
        if not ok:
            start = None
        elif start is None:
            start = i
        if start is not None and t[i] - t[start] >= hold_s - 1e-6:
            if t[i] <= capture_deadline_s + 1e-6:
                capture = float(t[i])
            break
    failed = end_code not in (9, 13, 0)
    capture_status = 'entered_held' if capture is not None else ('not_entered_held' if t[-1] >= capture_deadline_s - 1e-6 or failed else 'censored')
    target = np.flatnonzero(t >= horizon_s - 1e-6)
    if len(target):
        k = target[0]
        progress = float(root[after[k], 0] - root[after[0], 0])
        terminal_in_window = failed and t[-1] <= horizon_s + 1e-6
        continued = bool(not road_bad[after[:k+1]].any() and not terminal_in_window and progress >= min_progress_m)
    else:
        progress = None
        continued = False if failed or road_bad[after].any() else None
    return dict(envelope_entry_status=capture_status,envelope_entry_s=capture,
                continued_2s=continued,forward_progress_2s_m=progress,
                posthandoff_observed_s=float(t[-1]))
