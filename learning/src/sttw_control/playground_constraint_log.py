"""Explain sampled body-box clearance without changing the planner's predicate."""
import numpy as np


def clearance_witness(lower, upper, margin, bounds, boxes, geom_names):
    """Return the limiting sampled constraint and raw clearance at that same witness."""
    low, high = lower - margin[:, None, :], upper + margin[:, None, :]
    road = np.stack((low[..., 0]-bounds[0], bounds[1]-high[..., 0],
                     low[..., 1]-bounds[2], bounds[3]-high[..., 1]), -1)
    index = np.unravel_index(road.argmin(), road.shape)
    step, geom, side = (int(x) for x in index)
    raw = np.stack((lower[..., 0]-bounds[0], bounds[1]-upper[..., 0],
                    lower[..., 1]-bounds[2], bounds[3]-upper[..., 1]), -1)
    result = dict(step=step, geometry=geom_names[geom], constraint=f'road_{side}',
                  raw_clearance_m=float(raw[index]), expanded_clearance_m=float(road[index]),
                  margin_xyz_m=margin[step].tolist())
    for obstacle, box in enumerate(boxes):
        separation = np.maximum(box[:3]-high, low-box[3:]).max(-1)
        step, geom = (int(x) for x in np.unravel_index(separation.argmin(), separation.shape))
        value = float(separation[step, geom])
        if value < result['expanded_clearance_m']:
            raw = np.maximum(box[:3]-upper[step, geom], lower[step, geom]-box[3:]).max()
            result = dict(step=step, geometry=geom_names[geom], constraint=f'obstacle_{obstacle}',
                          raw_clearance_m=float(raw), expanded_clearance_m=value,
                          margin_xyz_m=margin[step].tolist())
    return result
