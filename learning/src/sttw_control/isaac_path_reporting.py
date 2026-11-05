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
