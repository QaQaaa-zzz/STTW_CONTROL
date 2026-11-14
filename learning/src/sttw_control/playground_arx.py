"""Nominal 20 ms ARX(2) closed-loop dynamics; causal free rollout."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def wrap(x):
    return np.arctan2(np.sin(x),np.cos(x))


class Reference:
    def __init__(self, bank, index, follower):
        self.s=bank['s'];self.xy=bank['xy'][index];self.heading=bank['heading'][index]
        self.curvature=bank['curvature'][index];self.tree=cKDTree(self.xy);self.follower=follower

    def features(self, pose, xi, speed=1.8):
        # Candidate reference is fixed; only predicted current pose is queried.
        nearest=self.tree.query(pose[:,:2])[1]
        indices=np.stack((np.maximum(nearest-1,0),np.minimum(nearest,len(self.s)-2)),1)
        a=self.xy[indices];v=self.xy[indices+1]-a
        u=np.clip(((pose[:,None,:2]-a)*v).sum(-1)/(v*v).sum(-1),0,1)
        point=a+u[...,None]*v;j=np.argmin(((pose[:,None,:2]-point)**2).sum(-1),1);rows=np.arange(len(pose))
        i=indices[rows,j];point=point[rows,j];s=self.s[i]+u[rows,j]*(self.s[i+1]-self.s[i])
        h=np.interp(s,self.s,self.heading);delta=pose[:,:2]-point
        ey=-np.sin(h)*delta[:,0]+np.cos(h)*delta[:,1]
        c=self.follower
        look=np.clip(c['lookahead_time_s']*np.maximum(xi[:,0],c['lookahead_min_speed_m_s']),c['lookahead_min_m'],c['lookahead_max_m'])
        look*=c.get('lookahead_scale',1.)
        ahead=np.stack([np.interp(s+look,self.s,self.xy[:,k]) for k in (0,1)],1)-pose[:,:2]
        yaw=Rotation.from_quat(pose[:,3:7][:,[1,2,3,0]]).as_euler('xyz')
        cs=np.cos(yaw[:,2]);sn=np.sin(yaw[:,2])
        preview=np.stack((cs*ahead[:,0]+sn*ahead[:,1],-sn*ahead[:,0]+cs*ahead[:,1]),1)
        return np.column_stack((pose[:,2],yaw[:,:2],np.sin(yaw[:,2]),np.cos(yaw[:,2]),pose[:,7],
            *[np.interp(s+d,self.s,self.curvature) for d in (0,.5,1.)],ey,wrap(h-yaw[:,2]),preview,np.full(len(pose),speed),look))


def states(q, v, schema):
    rotation=Rotation.from_quat(q[:,3:7][:,[1,2,3,0]])
    xi=np.column_stack((rotation.inv().apply(v[:,:3]),v[:,3:6],v[:,schema['steer_dof']],v[:,schema['front_dof']],v[:,schema['rear_dof']]))
    names=schema['qpos_names']
    pose=np.column_stack((q[:,:7],q[:,schema['steer_qpos']],q[:,names.index('frontwheel_joint')],q[:,names.index('rearwheel_joint')],q[:,names.index('hip_joint')],q[:,names.index('knee_joint')]))
    return pose,xi


def integrate(pose, xi, next_xi, dt=.02):
    # Trapezoidal velocity; body angular rates advance the quaternion locally.
    mid=(xi+next_xi)*.5;rotation=Rotation.from_quat(pose[:,3:7][:,[1,2,3,0]])
    next_rotation=rotation*Rotation.from_rotvec(mid[:,3:6]*dt)
    q=next_rotation.as_quat()[:,[3,0,1,2]]
    xyz=pose[:,:3]+.5*(rotation.apply(xi[:,:3])+next_rotation.apply(next_xi[:,:3]))*dt
    return np.column_stack((xyz,q,pose[:,7:10]+mid[:,6:9]*dt,pose[:,10:12]))


def design(xi, previous, eta):
    return np.column_stack((xi,previous,eta))


def fit(x, y, ridge=.001):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1.
    z=np.column_stack(((x-mean)/scale,np.ones(len(x))))
    penalty=np.eye(z.shape[1])*ridge*len(z);penalty[-1,-1]=0
    weights=np.linalg.solve(z.T@z+penalty,z.T@y)
    return dict(mean=mean,scale=scale,weights=weights,ridge=np.array(ridge))


def predict(model, xi, previous, eta):
    x=design(xi,previous,eta)
    return np.column_stack(((x-model['mean'])/model['scale'],np.ones(len(x))))@model['weights']


def rollout(model, reference, pose, xi, previous, steps):
    poses=[pose.copy()];velocities=[xi.copy()]
    for _ in range(steps):
        next_xi=predict(model,xi,previous,reference.features(pose,xi))
        pose=integrate(pose,xi,next_xi);previous,xi=xi,next_xi
        poses.append(pose.copy());velocities.append(xi.copy())
    return np.stack(poses,1),np.stack(velocities,1)


def robot_radius_bound(model):
    """Conservative radii; axisymmetric leaf wheels are spin-invariant."""
    geoms=np.flatnonzero((model.geom_bodyid>0)&((model.geom_contype!=0)|(model.geom_conaffinity!=0)))
    root_radii=[];joint_radii=np.zeros(model.njnt)
    for geom in geoms:
        body=int(model.geom_bodyid[geom]);radius=float(np.linalg.norm(model.geom_pos[geom])+model.geom_rbound[geom])
        while model.body_parentid[body]!=0:
            start=model.body_jntadr[body]
            for j in range(start,start+model.body_jntnum[body]):
                joint_radii[j]=max(joint_radii[j],radius+np.linalg.norm(model.jnt_pos[j]))
                radius+=2*np.linalg.norm(model.jnt_pos[j])
            radius+=np.linalg.norm(model.body_pos[body]);body=int(model.body_parentid[body])
        root_radii.append(radius)
    for j in range(1,model.njnt):
        body=model.jnt_bodyid[j];local=geoms[model.geom_bodyid[geoms]==body]
        if np.any(model.body_parentid[1:]==body) or len(local)!=1:continue
        g=local[0];axis=Rotation.from_quat(model.geom_quat[g][[1,2,3,0]]).apply([0,0,1])
        centered=np.allclose(model.geom_pos[g],model.jnt_pos[j])
        aligned=np.isclose(abs(axis@model.jnt_axis[j]),1.)
        symmetric=model.geom_type[g]==5 or (model.geom_type[g]==4 and np.isclose(model.geom_size[g,0],model.geom_size[g,1]))
        if centered and aligned and symmetric:joint_radii[j]=0.
    order=[model.joint(name).id for name in ['steering_joint','frontwheel_joint','rearwheel_joint','hip_joint','knee_joint']]
    return float(max(root_radii)),joint_radii[order]


def geometry_error_bound(predicted, actual, radius, joint_radii):
    """Point-correspondence upper bound, hence a conservative Hausdorff bound.

    Translation + root rotation + a telescoping sum of hinge rotations.
    Each hinge uses its descendant radius; invariant leaf wheel spin costs zero.
    This is not an exact surface distance or a calibrated probability guarantee.
    """
    shape=predicted.shape[:-1]
    rotation=Rotation.from_quat(predicted[...,3:7].reshape(-1,4)[:,[1,2,3,0]]).inv()*Rotation.from_quat(actual[...,3:7].reshape(-1,4)[:,[1,2,3,0]])
    angle=rotation.magnitude().reshape(shape)
    return np.linalg.norm(predicted[...,:3]-actual[...,:3],axis=-1)+2*radius*np.sin(angle/2)+(2*joint_radii*np.abs(np.sin((predicted[...,7:]-actual[...,7:])/2))).sum(-1)
