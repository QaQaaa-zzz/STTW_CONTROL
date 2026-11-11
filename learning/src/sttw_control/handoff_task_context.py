"""Known downstream road geometry sampled at a measured or predicted handoff."""
import numpy as np
from .handoff_continuation import mounted_features

OFFSETS_M = np.array([0., .5, 1., 2., 3., 4.], dtype=np.float32)


def route_context(root, route, half_width_m, mount_wxyz):
    """Return six lateral offsets, heading errors, curvatures and road half-width.

    This interface covers the monotonically increasing world-x downstream roads
    used in the current experiment; it is not a general nearest-path projection.
    """
    x = root[:, :1] + OFFSETS_M[None]
    xy = route['xy']
    lateral = np.interp(x, xy[:, 0], xy[:, 1]) - root[:, 1:2]
    heading = np.interp(x, xy[:, 0], route['heading'])
    yaw = mounted_features(root, mount_wxyz)[:, 2:3]
    error = np.arctan2(np.sin(heading-yaw), np.cos(heading-yaw))
    curvature = np.interp(x, xy[:, 0], route['curvature'])
    return np.column_stack((lateral, error, curvature,
                           np.full(len(root), half_width_m))).astype(np.float32)
