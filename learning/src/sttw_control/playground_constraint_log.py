"""Explain sampled body-box clearance without changing the planner's predicate."""
import numpy as np


def clearance_witness(lower, upper, margin, bounds, boxes, geom_names):
    """Return the limiting sampled constraint and raw clearance at that same witness."""
    low, high = lower - margin[:, None, :], upper + margin[:, None, :]
    road = np.stack((low[..., 0]-bounds[0], bounds[1]-high[..., 0],
                     low[..., 1]-bounds[2], bounds[3]-high[..., 1]), -1)
    index = np.unravel_index(road.argmin(), road.shape)
    step, geom, side = (int(x) for x in index)
    raw = np.stack((lower[..., 0]-bounds[0], bounds[1]-upper[..., 0],
                    lower[..., 1]-bounds[2], bounds[3]-upper[..., 1]), -1)
    result = dict(step=step, geometry=geom_names[geom], constraint=f'road_{side}',
                  raw_clearance_m=float(raw[index]), expanded_clearance_m=float(road[index]),
                  margin_xyz_m=margin[step].tolist())
    for obstacle, box in enumerate(boxes):
        separation = np.maximum(box[:3]-high, low-box[3:]).max(-1)
        step, geom = (int(x) for x in np.unravel_index(separation.argmin(), separation.shape))
        value = float(separation[step, geom])
        if value < result['expanded_clearance_m']:
            raw = np.maximum(box[:3]-upper[step, geom], lower[step, geom]-box[3:]).max()
            result = dict(step=step, geometry=geom_names[geom], constraint=f'obstacle_{obstacle}',
                          raw_clearance_m=float(raw), expanded_clearance_m=value,
                          margin_xyz_m=margin[step].tolist())
    return result


def dynamic_witnesses(pose, velocity, angles, domain, world_vx, speed, goal, road_width, anchor):
    """Explain original first-26-sample hold and domain gates at 20 ms spacing."""
    goal_slacks = dict(goal_x_min=pose[:,0]-goal['x_min_m'],
        goal_x_max=goal['x_max_m']-pose[:,0],goal_y=goal['abs_y_max_m']-abs(pose[:,1]),
        goal_forward=world_vx-goal['min_vx_m_s'],goal_path=goal['path_tolerance_m']-abs(domain[:,0]),
        goal_heading=goal['heading_tolerance_rad']-abs(domain[:,1]),
        goal_speed=goal['speed_tolerance_m_s']-abs(velocity[:,0]-speed))
    good=np.logical_and.reduce([x>=0 for x in goal_slacks.values()])
    holds=np.flatnonzero(np.convolve(good.astype(int),np.ones(26,dtype=int),'valid')==26) if len(good)>=26 else np.array([],int)
    end=int(holds[0]+25) if len(holds) else len(pose)-1
    slacks=dict(roll=.7-abs(angles[:,0]),pitch=1.309-abs(angles[:,1]),
        tracking=road_width-abs(domain[:,0]),heading_domain=np.pi/2-abs(domain[:,1]),
        preview=domain[:,2]-.05,forward=velocity[:,0]-.5)
    checks={'finite':bool(np.isfinite(pose[:end+1]).all())};events=[]
    for name,values in slacks.items():
        values=values[:end+1];bad=np.flatnonzero(values<=0 if name in ('heading_domain','preview') else values<0)
        checks[name]=not len(bad)
        if len(bad):
            worst=int(np.argmin(values));events.append(dict(constraint=name,phase=2,geometry='root_or_controller',first_time_s=(anchor+int(bad[0]))*.02,worst_time_s=(anchor+worst)*.02,signed_slack=float(values[worst]),clearance_m=None,margin_source='frozen execution contract; no spatial expansion'))
    if not checks['finite']:
        step=int(np.flatnonzero(~np.isfinite(pose[:end+1]).all(1))[0]);events.append(dict(constraint='finite',phase=2,first_time_s=(anchor+step)*.02))
    if not len(holds):
        starts=np.flatnonzero(np.diff(np.r_[False,good,False].astype(int))==1);stops=np.flatnonzero(np.diff(np.r_[False,good,False].astype(int))==-1)
        events.append(dict(constraint='goal_hold',phase=2,time_s=(anchor+end)*.02,required_consecutive_samples=26,longest_consecutive_samples=int(max(stops-starts,default=0)),failed_final_constraints=[k for k,v in goal_slacks.items() if v[-1]<0],reason='no complete hold within prediction horizon'))
    elif (anchor+end)*.02>10.:
        events.append(dict(constraint='task_deadline',phase=2,time_s=(anchor+end)*.02,limit_s=10.))
    return dict(checks=checks,goal_time_s=end*.02 if len(holds) else None,valid=all(checks.values()) and bool(len(holds)) and (anchor+end)*.02<=10.,events=events)
