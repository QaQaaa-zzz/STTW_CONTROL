"""Adapter for pinned PythonRobotics Dubins planning to an unchanged goal.

Upstream MIT implementation supplies the six Dubins path families. This module
only selects a finite free-goal-heading set and adapts its geometry to Path.
Curvature steps still use the ORIGINAL reference slew/filter.
"""
import numpy as np
import jax.numpy as j
from .geometric_path import Path


def dubins_goal_path(pose, goal_xy, *, radius_m=1.8, length_m=80., spacing_m=.02):
    """Upstream planner with eight world headings plus the current goal bearing.

    This finite heading set avoids imposing one final heading as task success;
    it is not a globally optimal free-heading solver. No event input exists.
    """
    from ._vendor.pythonrobotics.dubins import plan_dubins_path, _interpolate
    pose=np.asarray(pose,float);goal=np.asarray(goal_xy,float)
    if pose.shape!=(3,) or goal.shape!=(2,) or not np.isfinite(np.r_[pose,goal]).all():
        raise ValueError('finite measured XY/yaw and original world goal required')
    if not all(np.isfinite(x) and x>0 for x in (radius_m,length_m,spacing_m)):
        raise ValueError('positive geometric radius, length and spacing required')
    distance=float(np.linalg.norm(goal-pose[:2]))
    if distance<1e-6:raise ValueError('already at goal; retain arrival rather than build zero-length route')
    bearing=np.arctan2(goal[1]-pose[1],goal[0]-pose[0])
    headings=np.unique(np.r_[np.arange(8)*np.pi/4,bearing])
    candidates=[]
    for heading in headings:
        x,y,yaw,modes,lengths=plan_dubins_path(*pose,*goal,float(heading),1/radius_m,
                                              step_size=spacing_m/radius_m)
        if not np.isfinite(np.r_[x,y,yaw,lengths]).all():raise ValueError('nonfinite upstream plan')
        np.testing.assert_allclose([x[-1],y[-1]],goal,atol=1e-6,rtol=0)
        candidates.append((float(sum(lengths)),float(heading),modes,lengths))
    total,goal_heading,modes,lengths=min(candidates,key=lambda x:x[0])
    if total>=length_m-spacing_m:raise ValueError('goal beyond declared fixed path support; do not clip or move goal')
    s=np.linspace(0,length_m,int(np.ceil(length_m/spacing_m))+1)
    # Include the original goal exactly; keep the same array count and positive
    # arclength intervals. No endpoint shortening is allowed to fake arrival.
    nearest=int(np.argmin(abs(s-total)))
    if nearest==0:nearest=1
    s[nearest]=total
    if np.any(np.diff(s)<=0):raise ValueError('goal insertion broke arclength ordering')
    # Use upstream's exact segment interpolator on our fixed arclength grid,
    # including straight support beyond the original destination. Its length
    # argument is normalized by curvature, despite the public step-size docs.
    xy=np.empty((len(s),2));heading=np.empty(len(s));curvature=np.zeros(len(s))
    origin=pose.copy();start=0.;turn_end=0.
    for mode,length in [*zip(modes,lengths),('S',length_m-total)]:
        end=start+length
        indices=np.flatnonzero((s>=start)&(s<=end+1e-10))
        for i in indices:
            xx,yy,hh=_interpolate(max(0.,s[i]-start)/radius_m,mode,1/radius_m,*origin,[],[],[])
            xy[i]=[xx[-1],yy[-1]];heading[i]=hh[-1]
            curvature[i]=0. if mode=='S' else (1. if mode=='L' else -1.)/radius_m
        xx,yy,hh=_interpolate(length/radius_m,mode,1/radius_m,*origin,[],[],[])
        origin=np.array([xx[-1],yy[-1],hh[-1]])
        if mode!='S' and length>0:turn_end=end
        start=end
    xy[0]=pose[:2];xy[nearest]=goal
    result=Path(*(j.asarray(x) for x in (s,xy,heading,curvature,turn_end,total)))
    metadata=dict(planner='PythonRobotics Dubins; six path families and finite goal-heading candidates',
                  upstream_commit='8c392897c7f41a6e4d9aab9cb2e9737a7a6fa031',
                  radius_m=float(radius_m),modes=modes,segment_lengths_m=list(lengths),
                  goal_heading_rad=goal_heading,goal_heading_candidates=len(headings),
                  goal_arclength_m=float(total),goal_xy=goal.tolist(),start_pose=pose.tolist(),
                  curvature_continuous=False,position_heading_continuous=True,
                  shortest_path_certificate=False,safety_certificate=False)
    return result,metadata


def rebind_geometry(carry, path, speed, env):
    """Replace ONLY the geometry projection/follower in the incoming carry.

    Original clocks, alpha, physical state, history, ESO, lower state, offsets,
    reference filters/raw reference, relay and domain-exit record survive.
    The ordinary publish function changes references on subsequent real ticks.
    """
    from .finite_value_evaluation import rebind_path_context
    return carry.replace(state=rebind_path_context(carry.state,path,speed,env))


def replan_requested(*, xy, goal_xy, cross_track_m, follower_invalid, already_replanned,
                     contact, fault, ever_arrived=False, arrival_radius_m=1., departure_m=1.,
                     scheduled_time_s=None, task_time_s=None, continuous_replanning=False):
    """Causal departure trigger; optionally keep updating an active reconnection.

    Default remains one-shot. Continuous mode preserves the original route until
    departure/invalidity, then requests updates at the caller-declared cadence.

    Invoke at a declared ordinary planning cadence, not at a privileged pulse
    end. The thresholds define this baseline, not success or safety gates.
    """
    values=np.r_[xy,goal_xy,cross_track_m,arrival_radius_m,departure_m]
    if not np.isfinite(values).all() or min(arrival_radius_m,departure_m)<=0:
        raise ValueError('finite measured geometry and positive thresholds required')
    if continuous_replanning and scheduled_time_s is not None:raise ValueError('scheduled one-shot and continuous replanning are distinct protocols')
    if scheduled_time_s is not None:
        if task_time_s is None or not np.isfinite([scheduled_time_s,task_time_s]).all() or scheduled_time_s<=0:
            raise ValueError('finite positive scheduled time and measured task time required')
        return bool(not (already_replanned or contact or fault or ever_arrived)
                    and np.linalg.norm(np.asarray(xy)-goal_xy)>arrival_radius_m
                    and abs(task_time_s-scheduled_time_s)<1e-8)
    return bool(not ((already_replanned and not continuous_replanning) or contact or fault or ever_arrived)
                and np.linalg.norm(np.asarray(xy)-goal_xy)>arrival_radius_m
                and ((continuous_replanning and already_replanned) or follower_invalid or abs(cross_track_m)>departure_m))
