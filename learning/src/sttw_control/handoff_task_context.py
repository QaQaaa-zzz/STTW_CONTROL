"""Known downstream road geometry sampled at a measured or predicted handoff."""
import numpy as np
from .handoff_continuation import mounted_features

OFFSETS_M = np.array([0., .5, 1., 2., 3., 4.], dtype=np.float32)


def route_context(root, route, half_width_m, mount_wxyz, representation="world_x"):
    """Return the selected 19-field road representation.

    The legacy world-x representation assumes monotone roads. The Frenet
    representation uses nearest-segment projection and route-relative dynamics.
    """
    if representation=="frenet":return frenet_context(root,route,half_width_m,mount_wxyz)
    x = root[:, :1] + OFFSETS_M[None]
    xy = route['xy']
    lateral = np.interp(x, xy[:, 0], xy[:, 1]) - root[:, 1:2]
    heading = np.interp(x, xy[:, 0], route['heading'])
    yaw = mounted_features(root, mount_wxyz)[:, 2:3]
    error = np.arctan2(np.sin(heading-yaw), np.cos(heading-yaw))
    curvature = np.interp(x, xy[:, 0], route['curvature'])
    return np.column_stack((lateral, error, curvature,
                           np.full(len(root), half_width_m))).astype(np.float32)


def frenet_context(root, route, half_width_m, mount_wxyz):
    """Route-relative motion plus preview; ballistic margins are not safety bounds."""
    progress,ey,tangent,arc=project_polyline(root[:,:2],route);yaw=mounted_features(root,mount_wxyz)[:,2]
    ahead=progress[:,None]+OFFSETS_M[None]
    heading=np.interp(ahead,arc,route['heading']);heading_error=np.arctan2(np.sin(yaw[:,None]-heading),np.cos(yaw[:,None]-heading))
    curvature=np.interp(ahead,arc,route['curvature'])
    velocity=root[:,7:9];parallel=(velocity*tangent).sum(1);normal=-tangent[:,1]*velocity[:,0]+tangent[:,0]*velocity[:,1]
    _,ey_future,_,_=project_polyline(root[:,:2]+.4*velocity,route)
    return np.column_stack((ey,heading_error[:,2],parallel,normal,half_width_m-ey_future,half_width_m+ey_future,heading_error,curvature,np.full(len(root),half_width_m))).astype(np.float32)


def project_polyline(points, route):
    """Exact nearest-segment projection using a conservative endpoint search radius."""
    from scipy.spatial import cKDTree
    if '_projection' not in route:
        xy=route['xy'];segment=np.diff(xy,axis=0);length=np.linalg.norm(segment,axis=1)
        route['_projection']=(xy,segment,length,np.r_[0.,np.cumsum(length)],cKDTree(xy))
    xy,segment,length,arc,tree=route['_projection']
    nearest,_=tree.query(points)
    # An endpoint of the closest segment is within sqrt(d_vertex^2+L_max^2).
    neighbors=tree.query_ball_point(points,np.sqrt(nearest**2+length.max()**2)+1e-6)
    counts=np.array([len(n) for n in neighbors]);owner=np.repeat(np.arange(len(points)),counts)
    vertices=np.concatenate(neighbors).astype(int);owner=np.tile(owner,2);indices=np.r_[vertices-1,vertices]
    valid=(indices>=0)&(indices<len(segment));owner=owner[valid];indices=indices[valid]
    delta=points[owner]-xy[indices];t=((delta*segment[indices]).sum(1)/(length[indices]**2)).clip(0,1)
    residual=delta-t[:,None]*segment[indices];distance=(residual**2).sum(1)
    minimum=np.full(len(points),np.inf);np.minimum.at(minimum,owner,distance)
    candidate=np.flatnonzero(distance==minimum[owner]);best=np.full(len(points),len(distance),int)
    np.minimum.at(best,owner[candidate],candidate)
    i=indices[best];tangent=segment[i]/length[i,None];residual=residual[best]
    lateral=-tangent[:,1]*residual[:,0]+tangent[:,0]*residual[:,1]
    return arc[i]+t[best]*length[i],lateral,tangent,arc
