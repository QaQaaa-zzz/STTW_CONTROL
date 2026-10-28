"""Causal periodic geometry adapter for fixed-endpoint evaluation.

No policy/physics reset; the original world goal and diagnostic route survive.
This is ordinary Dubins replanning, not a new control or safety algorithm.
"""
import numpy as np
import jax.numpy as j
from .geometric_path import Path
from .goal_replanning import dubins_goal_path


def pad_original_path(path, *, count=4001, spacing_m=.02):
    """Only extend the existing straight support; never resample old geometry."""
    s,xy,h,k=map(np.asarray,(path.s,path.xy,path.heading,path.curvature))
    if len(s)>count or len(s)<2 or not np.isfinite(np.r_[s,xy.ravel(),h,k]).all():
        raise ValueError('finite original path within fixed support required')
    if not np.all(np.diff(s)>0) or abs(k[-1])>1e-7 or spacing_m<=0:
        raise ValueError('positive arclength and straight terminal support required')
    extra=np.arange(1,count-len(s)+1,dtype=s.dtype)*spacing_m
    out=Path(j.asarray(np.r_[s,s[-1]+extra]),
        j.asarray(np.vstack([xy,xy[-1]+extra[:,None]*np.array([np.cos(h[-1]),np.sin(h[-1])])])),
        j.asarray(np.r_[h,np.full(len(extra),h[-1])]),j.asarray(np.r_[k,np.zeros(len(extra))]),
        path.turn_end,path.goal)
    return out


def periodic_plan(pose, original_goal, *, task_tick, arrived, contact, fault,
                  period_ticks=200, horizon_ticks=2000):
    """Check every1s, with no disturbance labels or future-state input.

    Every live pre-goal case gets a route, including nominal cases. At most9
    plans within H10. Stops at first actual goal entry or physical/fault end.
    """
    if period_ticks<=0 or horizon_ticks<=period_ticks:
        raise ValueError('positive planning period shorter than horizon required')
    if (arrived or contact or fault or task_tick<=0 or task_tick>=horizon_ticks
            or task_tick%period_ticks):return None
    pose=np.asarray(pose,float);goal=np.asarray(original_goal,float)
    if pose.shape!=(3,) or goal.shape!=(2,) or not np.isfinite(np.r_[pose,goal]).all():
        raise ValueError('finite current measured pose and original goal required')
    if np.linalg.norm(pose[:2]-goal)<=1.:return None
    path,meta=dubins_goal_path(pose,goal,radius_m=1.8,length_m=80.,spacing_m=.02)
    return path,dict(meta,task_tick=int(task_tick),trigger='periodic measured-state pre-goal update; no force/onset information')


def make_rebinding(env, lanes):
    """Compile geometry-only binding once; caller retains all other carry."""
    import jax
    from .goal_replanning import rebind_geometry
    @jax.jit
    def apply(carry,paths,speeds,mask):
        rebound=jax.vmap(lambda c,p,v:rebind_geometry(c,p,v,env))(carry,paths,speeds)
        return jax.tree.map(lambda a,b:j.where(mask.reshape((lanes,)+(1,)*(a.ndim-1)),a,b),rebound,carry)
    @jax.jit
    def observe(carry):
        return jax.vmap(lambda d:env.physics.observe(d)[1])(carry.state.physical.data)
    return apply,observe


def update_paths(root, *, carry,paths,speeds,goals,cases,offset,block,rebinding,observe,events):
    """Save actual planned geometry/CPU time; no hidden force input to planner."""
    import jax,time
    from .preference_best import atomic_json
    started=time.monotonic()
    poses=np.asarray(jax.device_get(observe(carry)))
    arrived,failed,fault,ticks=jax.device_get((carry.arrived,carry.state.physical.failed,carry.state.fault,carry.state.tick))
    changed=np.zeros(len(cases),bool)
    for i,c in enumerate(cases):
        plan=periodic_plan(poses[i],goals[i],task_tick=int(ticks[i]),arrived=bool(arrived[i]),contact=bool(failed[i]),fault=bool(fault[i]))
        if plan is None:continue
        path,meta=plan;paths[i]=path;changed[i]=True
        filename=f'replanned_{offset+i:03d}_{block:02d}.npz'
        np.savez_compressed(root/filename,**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
        events.append(dict(case_id=c['case_id'],case_index=offset+i,file=filename,**meta))
    batched=jax.tree.map(lambda *x:j.stack(x),*paths)
    if changed.any():carry=rebinding(carry,batched,speeds,j.asarray(changed))
    atomic_json(root/'replanning_decisions.json',events)
    return carry,batched,time.monotonic()-started
