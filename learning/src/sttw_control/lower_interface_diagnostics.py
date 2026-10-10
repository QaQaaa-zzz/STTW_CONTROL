"""Read-only logging of the executed R196 chain; never advances controller state."""
import jax.numpy as jp

def ecbc_contributions(measurement,gains,output,enable_eso):
    _,steer,_,roll,roll_rate,reference=measurement
    return jp.array([gains[0]*(reference-steer),gains[1]*(output.reference_roll-roll),
                     gains[1]*jp.where(enable_eso,output.equilibrium_shift,0.),-gains[2]*roll_rate])

def projection_details(s,pose):
    """Mirror the adapter's global segment / strict endpoint-ray tie rule for logging."""
    a,b=s.path_points[:-1],s.path_points[1:];v=b[:,:2]-a[:,:2];l2=jp.sum(v*v,axis=1)
    u=jp.clip(jp.sum((pose[:2]-a[:,:2])*v,axis=1)/jp.maximum(l2,1e-12),0,1)
    feet=a[:,:2]+u[:,None]*v;valid=(jp.arange(len(a))<s.valid_count-1)&(l2>1e-12)
    ds=jp.where(valid,jp.sum((pose[:2]-feet)**2,axis=1),jp.inf);i=jp.argmin(ds)
    end=s.path_points[jp.maximum(s.valid_count-1,0)];tan=jp.array([jp.cos(end[2]),jp.sin(end[2])])
    forward=jp.maximum(jp.dot(pose[:2]-end[:2],tan),0);rd=jp.sum((pose[:2]-end[:2]-forward*tan)**2);ray=rd<ds[i]
    lengths=jp.where(valid,jp.sqrt(l2),0);arc=jp.cumsum(lengths)-lengths
    return dict(segment_index=jp.where(ray,s.valid_count-1,i),segment_u=jp.where(ray,0.,u[i]),
                progress=jp.where(ray,jp.sum(lengths)+forward,arc[i]+u[i]*lengths[i]),
                distance=jp.sqrt(jp.where(ray,rd,ds[i])),endpoint_ray=ray,ray_forward=forward,
                path_count=s.valid_count,path_overflow=s.overflow)

def capture(env,s,prepared,post,measurement,pose,governed,preview,obs,flags):
    cs,_,output=preview;enabled=s.physical.physical_tick*env.cc.dt>3.
    m=jp.array([measurement[5]*.1,measurement[2],measurement[3],measurement[0],measurement[1],governed[1]])
    terms=ecbc_contributions(m,cs.gains,output,enabled)
    path=flags['path_features'];tr=env.lower.source['timed_reference']
    requested_yaw=governed[0]*path[2]-tr['yaw_feedback']*path[1]+tr['lateral_feedback']*path[0]
    old_ref=jp.clip(jp.arctan(env.cc.wheelbase*requested_yaw/(jp.maximum(governed[0],.1)*jp.cos(env.cc.caster))),-tr['max_steer'],tr['max_steer'])
    return dict(pre_time=s.tick*env.cc.dt,post_time=(s.tick+1)*env.cc.dt,
        pre_measurement=measurement,pre_pose=pose,pre_forward_speed=env.physics.observe(s.physical.data)[2],
        lower_input_raw=obs,lower_input_normalized=(obs-env.diagnostic_mean)/env.diagnostic_std,
        lower_frame=obs[270:300],lower_mask=obs[300:],lower_reference_pose=prepared.inner.reference_pose,
        projection=projection_details(prepared.inner,pose),
        return_pre=prepared.inner.return_state._asdict(),return_post=post.inner.return_state._asdict(),
        eso_pre=s.physical.controller.eso,eso_post=cs.eso,disturbance_pre=s.physical.controller.disturbance,
        disturbance_post=cs.disturbance,gains=cs.gains,reference_roll=output.reference_roll,
        equilibrium_shift=output.equilibrium_shift,eso_enabled=enabled,ecbc_terms=terms,
        ecbc_preclip=jp.sum(terms),ecbc_output=output.steer_rate,
        # Counterfactual old relationship evaluated on current adapter geometry,
        # not the old continuous projection and not an executed command.
        old_reference_relationship=old_ref)
