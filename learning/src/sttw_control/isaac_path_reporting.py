"""Read saved path cases and reuse their immutable physical Zero trajectory."""
import json
from pathlib import Path
import numpy as np


def load_case(run,baseline_run=None):
    run=Path(run);archive=np.load(run/'trajectory.npz');contract=json.loads((run/'contract.json').read_text())
    if baseline_run is None:return archive,contract
    baseline_run=Path(baseline_run);baseline=np.load(baseline_run/'trajectory.npz');bc=json.loads((baseline_run/'contract.json').read_text())
    policy=np.flatnonzero(~np.asarray(contract['settings']['zero_middle'],dtype=bool));zero=np.flatnonzero(bc['settings']['zero_middle'])
    data={}
    for key in archive.files:
        if key not in baseline.files:continue
        axis=2 if key in ('lower_trace','wheel_contact_points','wheel_contact_forces') else 1
        data[key]=np.concatenate((np.take(archive[key],policy,axis=axis),np.take(baseline[key],zero,axis=axis)),axis=axis)
    contract['origin']=[contract['origin'][i] for i in policy]+[bc['origin'][i] for i in zero]
    contract['baseline_reuse']=dict(run=str(baseline_run.resolve()),identity=bc['identity'],scope='saved local-coordinate matched case; global clone translations may differ with lane count')
    return data,contract


def replay_rewards(data,config,offset_reset=None):
    """Reuse saved effective costs and rebuild the policy-rate correction cost."""
    c=config['path_reward'];count=data['reward'].shape[1]
    offsets=data['offsets'].reshape(-1,4,count,2)[:,-1]
    previous=np.concatenate((np.zeros_like(offsets[:1]),offsets[:-1]))
    if offset_reset is not None:
        step,lanes,value=offset_reset;previous[step,lanes]=value
    rate=(offsets-previous)/.02
    rate_cost=np.minimum(c['offset_rate_weight']*((rate/np.asarray(c['offset_rate_scales']))**2).sum(-1),c['component_caps']['offset_rate'])
    keys=[key for key in data.keys() if key.startswith('effective_')]
    components={key.removeprefix('effective_'):-c['scale']*.005*np.asarray(data[key]).copy() for key in keys}
    components['offset_rate']=-c['scale']*.005*np.repeat(rate_cost,4,axis=0)
    total=sum(components.values()).reshape(-1,4,count)
    trace=data['lower_trace']
    for step,lane in np.argwhere(data['failure']):
        total[step,:,lane]=0
        active=np.flatnonzero(trace[step,:,lane,6]);last=int(active[-1]) if len(active) else 0
        total[step,last,lane]=data['reward'][step,lane]
    total=np.where(data['valid'][:,None,:],total,0)
    closure=float(abs(total.sum(1)-data['reward']).max())
    np.testing.assert_allclose(total.sum(1),data['reward'],atol=1e-4,rtol=1e-5)
    total=total.reshape(-1,count)
    components['terminal_adjustment']=total-sum(components.values())
    return total,components,closure


def write_zero_scores(run):
    """Rescore the saved Zero lane with the original shared five-alpha reward."""
    import jax
    import jax.numpy as jnp
    from .controller import ControllerConfig, _system
    from .path_command_reward import costs
    run=Path(run);data,contract=load_case(run)
    cfg=json.loads((Path(contract['bundle'])/'config.json').read_text())
    route=np.load(Path(contract['bundle'])/'route.npz')
    lane=int(np.flatnonzero(contract['settings']['zero_middle'])[0])
    trace=data['lower_trace'][:,:,lane].reshape(-1,11);ids=np.flatnonzero(trace[:,6])
    time=ids*.005+trace[ids,7]*.0002;speed=float(contract['settings'].get('user_speed_m_s',2.3))
    progress=data['progress'][ids,lane];heading=np.interp(progress,route['s'],route['heading'])
    eh=(heading-data['pose'][ids,lane,2]+np.pi)%(2*np.pi)-np.pi
    cc=ControllerConfig(**json.loads(Path(contract['settings']['controller_config']).read_text()))
    ratio=float(_system(jnp.float32(speed),jnp.zeros(3),cc)[3])
    curvature=np.interp(progress,route['s'],route['curvature'])
    chi=np.clip((abs(np.arctan(cc.wheelbase*curvature/np.cos(cc.caster))/ratio)-.18)/.12,0,1)
    for alpha,stem in zip([0,.25,.5,.75,1],['alpha0','alpha025','alpha05','alpha075','alpha1']):
        args=dict(alpha=jnp.full(len(ids),alpha),chi=jnp.asarray(chi),ev=jnp.asarray(trace[ids,2]-speed),ey=jnp.asarray(trace[ids,3]),eh=jnp.asarray(eh),peak_roll=jnp.asarray(trace[ids,4]),roll_rate=jnp.asarray(data['measurement'][ids,lane,1]),speed=jnp.asarray(trace[ids,2]),offsets=jnp.zeros((len(ids),2)),offset_rate=jnp.zeros((len(ids),2)))
        terms=jax.vmap(lambda **x:costs(**x,config=cfg))(**args);reward=np.asarray(terms['reward']).copy()
        for step in np.flatnonzero(data['failure'][:,lane]):
            active=np.flatnonzero(ids//4==step)
            if len(active):
                reward[active]=0;reward[active[-1]]=data['reward'][step,lane]
        np.savez_compressed(run/(stem+'_zero_rescored.npz'),time=time,reward=reward,**{key:np.asarray(value) for key,value in terms['effective'].items()})


def static_case_metrics(data,contract):
    """Original bend-report task criteria, evaluated on each actual trajectory."""
    cfg=json.loads((Path(contract['bundle'])/'config.json').read_text())
    route=np.load(Path(contract['bundle'])/'route.npz');b=cfg['stage_B_baseline']
    speed=float(contract['settings'].get('user_speed_m_s',2.3))
    goal=float(route['goal']);goal_xy=np.array([np.interp(goal,route['s'],route['xy'][:,k]) for k in (0,1)])
    yaw=np.interp(goal,route['s'],route['heading']);rows=[]
    for lane in range(data['reward'].shape[1]):
        trace=data['lower_trace'][:,:,lane].reshape(-1,11);ids=np.flatnonzero(trace[:,6]);time=ids*.005+trace[ids,7]*.0002
        progress=data['progress'][ids,lane];heading=(np.interp(progress,route['s'],route['heading'])-data['pose'][ids,lane,2]+np.pi)%(2*np.pi)-np.pi
        signed=(trace[ids,:2]-goal_xy)@np.array([np.cos(yaw),np.sin(yaw)])
        failed=bool(data['failure'][:,lane].any());tail=time>contract['settings']['episode_s']-b['hold_s']+1e-5
        hold=bool(tail.sum()==round(b['hold_s']/.005) and np.all(abs(trace[ids,3][tail])<=b['final_y_error_m']) and np.all(abs(heading[tail])<=b['final_path_heading_error_rad']) and np.all(abs(trace[ids,2][tail]-speed)<=b['final_speed_error_m_s']) and not failed)
        rows.append(dict(lane=lane,failed=failed,observed_seconds=float(time[-1]),goal_crossed=bool(np.any((signed>=0)&(progress>=goal))),final_hold=hold,working_roll_exceeded=bool(np.any(trace[ids,4]>b['working_roll_rad'])),path_rmse_m=float(np.sqrt(np.mean(trace[ids,3]**2))),speed_rmse_m_s=float(np.sqrt(np.mean((trace[ids,2]-speed)**2))),peak_roll_rad=float(trace[ids,4].max()),requested_speed_m_s=speed))
    return rows


def raised_contact_phases(data,contract,lane):
    """Reuse the original switch report's measured mean-point convention."""
    terrain=Path(next(iter(contract['settings']['terrains'].values())));meta=json.loads((terrain/'identity.json').read_text());vertices=np.load(terrain/'terrain.npz')['vertices'];side=int(np.sqrt(len(vertices)));height=vertices[:,2].reshape(side,side);lo=meta['bounds_m'][0];spacing=meta['horizontal_scale_m']
    trace=data['lower_trace'][:,:,lane].reshape(-1,11)
    points=data['wheel_contact_points'][:,:,lane].reshape(-1,2,data['wheel_contact_points'].shape[-2],3)-np.asarray(contract['origin'][lane]);forces=data['wheel_contact_forces'][:,:,lane].reshape(points.shape)
    active=np.repeat(trace[:,6].astype(bool),25)&(np.tile(np.arange(25),len(trace))<np.repeat(trace[:,7],25));time=(np.arange(len(points))+1)*.0002;result={}
    for wheel,index in [('front',0),('rear',1)]:
        xy=points[:,index,:,:2];finite=np.isfinite(xy).all(-1);grid=np.rint((np.where(finite[...,None],xy,lo)-lo)/spacing).astype(int);inside=finite&(grid>=0).all(-1)&(grid<side).all(-1);grid=np.clip(grid,0,side-1)
        raised=(inside&(height[grid[...,0],grid[...,1]]>0)&(np.linalg.norm(forces[:,index],axis=-1)>1e-3)&(points[:,index,:,2]>.0025)).any(-1)&active;observed=time[raised]
        result[wheel+'_first_s']=float(observed[0]) if len(observed) else None;result[wheel+'_last_s']=float(observed[-1]) if len(observed) else None
    return result
