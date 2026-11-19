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


def update_wall_minimum(m,md,robot,wall,rot,lb,best,time_s):
 """Refine the signed minimum against an axis-aligned box at the current pose."""
 import numpy as np
 from scipy.optimize import minimize
 for j in np.argsort(lb):
  if best and lb[j]>max(0,best[0]):break
  geom=robot[j];dist=mujoco.mj_geomDistance(m,md,geom,wall,2.,None)
  if dist<=0 or dist+1e-7<lb[j]:
   # Some native pairs return zero when separated. Independently minimize the
   # squared distance from a point of the convex robot geom to the wall box.
   size=m.geom_size[geom].copy();typ=int(m.geom_type[geom]);R=rot[j];center=md.geom_xpos[geom]-md.geom_xpos[wall]
   if typ==5:size=np.array([size[0],size[0],size[1]])
   M=R*size
   def objective(u):
    pos=center+M@u;delta=pos-np.clip(pos,-m.geom_size[wall],m.geom_size[wall])
    return np.dot(delta,delta),2*M.T@delta
   dim={4:3,5:2,6:0}[typ]
   cons=({'type':'ineq','fun':lambda u:1-u[:dim]@u[:dim]},) if dim else ()
   solution=minimize(objective,np.zeros(3),jac=True,bounds=[(-1,1)]*3,constraints=cons,method='SLSQP',options={'ftol':1e-14})
   # Convex first-order lower bound from exact support on the primitive.
   u=solution.x;val,grad=objective(u)
   support=np.linalg.norm(grad[:dim])+abs(grad[dim:]).sum()
   lower=np.sqrt(max(0,val-grad@u-support));upper=np.sqrt(val)
   if upper-max(lb[j],lower)>0.0001:raise ValueError(('unresolved distance',geom,lower,upper))
   if upper>1e-6:dist=upper
  if best is None or dist<best[0]:best=(dist,float(time_s),m.geom(geom).name)
 return best
