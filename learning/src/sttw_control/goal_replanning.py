"""Adapter for pinned PythonRobotics Dubins planning to an unchanged goal.

Upstream MIT implementation supplies the six Dubins path families. This module
only selects a finite free-goal-heading set and adapts its geometry to Path.
Curvature steps still use the ORIGINAL reference slew/filter.
"""
import numpy as np


def dubins_goal_path(pose, goal_xy, *, radius_m=1.8, length_m=80., spacing_m=.02, goal_heading_rad=None, fallback_max_length_m=None, numpy_output=False):
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
    if goal_heading_rad is not None and not np.isfinite(goal_heading_rad):
        raise ValueError('finite optional goal heading required')
    if fallback_max_length_m is not None and (goal_heading_rad is None or not np.isfinite(fallback_max_length_m) or fallback_max_length_m<0):
        raise ValueError('heading fallback requires a preferred heading and finite nonnegative length allowance')
    finite_headings=np.unique(np.r_[np.arange(8)*np.pi/4,bearing])
    headings=finite_headings if goal_heading_rad is None else np.r_[goal_heading_rad,finite_headings] if fallback_max_length_m is not None else np.array([goal_heading_rad])
    candidates=[]
    for heading_index,heading in enumerate(headings):
        if heading_index==1 and fallback_max_length_m is not None and candidates[0][0]<=fallback_max_length_m:break
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
    values=dict(s=s,xy=xy,heading=heading,curvature=curvature,turn_end=turn_end,goal=total)
    if numpy_output:
        from types import SimpleNamespace
        result=SimpleNamespace(**values)
    else:
        import jax.numpy as j
        from .geometric_path import Path
        result=Path(**{key:j.asarray(value) for key,value in values.items()})
    metadata=dict(planner='PythonRobotics Dubins; six path families and finite goal-heading candidates',
                  upstream_commit='8c392897c7f41a6e4d9aab9cb2e9737a7a6fa031',
                  radius_m=float(radius_m),modes=modes,segment_lengths_m=list(lengths),
                  goal_heading_rad=goal_heading,goal_heading_candidates=len(candidates),
                  goal_heading_mode='free' if goal_heading_rad is None else 'prescribed',
                  goal_arclength_m=float(total),goal_xy=goal.tolist(),start_pose=pose.tolist(),
                  curvature_continuous=False,position_heading_continuous=True,
                  shortest_path_certificate=False,safety_certificate=False)
    if fallback_max_length_m is not None:
        metadata.update(goal_heading_mode='finite_fallback' if goal_heading!=goal_heading_rad else 'prescribed',heading_fallback_attempted=len(candidates)>1,heading_fallback_selected=goal_heading!=goal_heading_rad,
                        preferred_goal_heading_rad=float(goal_heading_rad),preferred_candidate_length_m=candidates[0][0],maximum_goal_length_m=float(fallback_max_length_m))
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


class IsaacGoalReplanning:
    """Evaluation adapter: reuse Dubins geometry, retain all controller state."""
    def __init__(self, env, config, output):
        import torch
        from .torch_path_reference import TorchPathReference
        self.env=env;self.config=config;self.output=output;self.events=[]
        original=env.route
        self.grid=original.s.cpu().numpy()
        route_data=np.load(config['original_route'])
        self.goal_s=float(route_data['turn_end'])+config['goal_after_turn_m'] if 'goal_after_turn_m' in config else float(route_data['goal'])
        self.goal=original.at(torch.tensor([self.goal_s],device=env.device))[0][0].cpu().numpy()
        self.arrived=np.zeros(env.n,dtype=bool);self.exit_bound=np.zeros(env.n,dtype=bool);self.replanned=np.zeros(env.n,dtype=bool)
        route=dict(s=self.grid,**{key:getattr(original,key).cpu().numpy()[None].repeat(env.n,axis=0) for key in ('xy','heading','curvature')})
        env.route=TorchPathReference(route,env.config,env.device)
        env.middle.route=env.route
        self.period=round(config['period_s']/.02)

    def update(self):
        import json,time,torch
        env=self.env;c=self.config;_,pose,speed=env.plant.observe()
        poses=pose.cpu().numpy()
        self.arrived|=np.linalg.norm(poses[:,:2]-self.goal,axis=1)<=c['arrival_radius_m']
        due=env.tick>0 and env.tick%self.period==0
        cross=env.middle.projection['cross_track'].cpu().numpy()
        invalid=env.middle.follower['invalid'].cpu().numpy();alive=env.alive.cpu().numpy()
        changed=[]
        for i,enabled in enumerate(c['enabled']):
            if enabled and alive[i] and self.arrived[i] and self.replanned[i] and not self.exit_bound[i]:
                # Finite-goal handoff: do not keep following a pre-arrival loop.
                xy=poses[i,:2]+self.grid[:,None]*np.array([np.cos(poses[i,2]),np.sin(poses[i,2])])
                heading=np.full_like(self.grid,poses[i,2]);curvature=np.zeros_like(self.grid)
                for key,value in dict(xy=xy,heading=heading,curvature=curvature).items():getattr(env.route,key)[i]=torch.as_tensor(value,device=env.device,dtype=torch.float32)
                file=f'forward_exit_{i:02d}_tick{env.tick*4:04d}.npz'
                np.savez_compressed(self.output/file,s=self.grid,xy=xy,heading=heading,curvature=curvature,goal=0.,original_goal_xy=self.goal)
                self.events.append(dict(lane=i,task_tick=env.tick*4,time_s=env.tick*.02,event='goal_arrival_forward_exit',file=file,start_pose=poses[i].tolist(),goal_xy=self.goal.tolist(),planning_seconds=0.))
                self.exit_bound[i]=True;changed.append(i)
                continue
            if not due or not enabled or not replan_requested(xy=poses[i,:2],goal_xy=self.goal,cross_track_m=cross[i],follower_invalid=invalid[i],already_replanned=self.replanned[i],contact=not alive[i],fault=bool(env.middle.fault[i]),ever_arrived=self.arrived[i],arrival_radius_m=c['arrival_radius_m'],departure_m=c['departure_m'],continuous_replanning=True):continue
            started=time.monotonic()
            path,meta=dubins_goal_path(poses[i],self.goal,radius_m=c['radius_m'],numpy_output=True)
            xy=np.column_stack([np.interp(self.grid,path.s,path.xy[:,k]) for k in (0,1)])
            heading=np.interp(self.grid,path.s,np.unwrap(path.heading));curvature=np.interp(self.grid,path.s,path.curvature)
            for key,value in dict(xy=xy,heading=heading,curvature=curvature).items():getattr(env.route,key)[i]=torch.as_tensor(value,device=env.device,dtype=torch.float32)
            file=f'replanned_path_{i:02d}_tick{env.tick*4:04d}.npz'
            np.savez_compressed(self.output/file,s=self.grid,xy=xy,heading=heading,curvature=curvature,goal=path.goal,original_goal_xy=self.goal)
            self.events.append(dict(meta,lane=i,task_tick=env.tick*4,time_s=env.tick*.02,cross_track_before=float(cross[i]),file=file,planning_seconds=time.monotonic()-started))
            self.replanned[i]=True;changed.append(i)
        if changed:
            ids=torch.tensor(changed,device=env.device)
            previous=env.middle.projection['progress'].clone();previous[ids]=0
            projection=env.route.project(pose[:,:2],previous,torch.zeros_like(speed))
            follower=env.route.pursuit(projection['progress'],pose,speed,env.speed)
            for key,value in projection.items():env.middle.projection[key][ids]=value[ids]
            for key,value in follower.items():env.middle.follower[key][ids]=value[ids]
            # No policy/filter/history/ESO reset. Normal publish retains slew.
            (self.output/'replanning_decisions.json').write_text(json.dumps(self.events,indent=2))
