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
