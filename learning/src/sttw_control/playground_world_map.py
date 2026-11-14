"""Fixed physical map checks independent of the controller's reference curve."""
import jax.numpy as jp
from jit_dvgc.geometry import collision_support_bounds


def body_outside(data, geometry, bounds):
    """Test the whole collision body's planar support against a fixed rectangle."""
    ids=geometry.robot_geom_ids
    support=collision_support_bounds(data.geom_xpos[:,ids],data.geom_xmat[:,ids],
                                     geometry.robot_geom_types,geometry.robot_geom_sizes)
    xmin,xmax,ymin,ymax=bounds
    return ((jp.min(support.min_x,axis=1)<xmin)|(jp.max(support.max_x,axis=1)>xmax)
            |(jp.min(support.min_y,axis=1)<ymin)|(jp.max(support.max_y,axis=1)>ymax))
