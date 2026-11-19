"""Recorded-pose obstacle distance via native MuJoCo geometry; no physics replay."""
import mujoco


def obstacle_clearance(model, qpos, times):
    robot=[i for i in range(model.ngeom) if model.geom_bodyid[i]>0 and (model.geom_contype[i] or model.geom_conaffinity[i])]
    obstacles=[i for i in range(model.ngeom) if model.geom(i).name.startswith('planning_obstacle_')]
    data=mujoco.MjData(model);witness=None;minimum=float('inf')
    for q,t in zip(qpos,times):
        data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        for body in robot:
            for obstacle in obstacles:
                distance=float(mujoco.mj_geomDistance(model,data,body,obstacle,100.,None))
                if distance<minimum:
                    minimum=distance;witness=dict(distance_m=distance,time_s=float(t),geometry=model.geom(body).name,obstacle=model.geom(obstacle).name)
    return dict(minimum=witness,samples=len(qpos),method='mj_geomDistance at saved poses; 5ms samples, not continuous collision proof')


def obstacle_aabb_bounds(model, poses):
    """World XY body envelope and conservative 3-D obstacle distance at saved poses."""
    import numpy as np

    robot = (model.geom_bodyid > 0) & ((model.geom_contype | model.geom_conaffinity) != 0)
    obstacle = np.array([model.geom(i).name.startswith('planning_obstacle_') for i in range(model.ngeom)])
    data = mujoco.MjData(model)
    centers, half = np.hsplit(model.geom_aabb, 2)
    envelopes = np.empty((len(poses), 4))
    distance = np.inf
    for i,q in enumerate(poses):
        data.qpos[:] = q
        mujoco.mj_kinematics(model,data)
        rotation = data.geom_xmat.reshape(-1,3,3)
        mid = data.geom_xpos + (rotation @ centers[...,None])[...,0]
        extent = (abs(rotation) @ half[...,None])[...,0]
        lo,hi = mid - extent,mid + extent
        separation = np.maximum(lo[robot,None]-hi[obstacle],lo[obstacle]-hi[robot,None])
        distance = min(distance,np.linalg.norm(np.maximum(separation,0.),axis=-1).min())
        envelopes[i] = [*lo[robot,:2].min(axis=0),*hi[robot,:2].max(axis=0)]
    return envelopes, float(distance)
