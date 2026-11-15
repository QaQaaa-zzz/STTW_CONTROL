"""Offline attribution of full-chain calibration and geometry rejections."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch,mujoco
from sttw_control.playground_arx import states
from sttw_control.playground_prediction import BodyEnvelope

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--calibration',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);start=time.perf_counter()
cfg=json.loads((a.calibration/'contract.json').read_text())
cal=json.loads((a.calibration/'calibration.json').read_text())
edges=np.array(cal['edges']);sources=[];cases=[]
(a.output/'contract.json').write_text(json.dumps(dict(source=str(a.calibration),role='Known DEV offline diagnosis; actual future used only for error attribution',budget='CPU only, no physical episodes or training; stop within25min',changes='No policy, predictor, calibration rule, task or geometry changes',diagnostics='Axis-only expansion and actual-path expansion are counterfactual diagnostics, not proposed safe margins'),indent=2)+'\n')

def motion(z,i,full=False):
    return np.concatenate((z['approach'][i,:int(z['entry_steps'][i])+1],z['jump'][i,:int(z['jump_steps'][i])],z['departure'][i,:len(z['departure'][i]) if full else int(z['departure_steps'][i])+1]))

def support(env,pose):
    return tuple(v.numpy() for v in env(torch.tensor(pose,dtype=torch.float32)))

for method in ['C','I']:
    peak=np.full((len(edges)-1,3),-1.);origin={}
    for map_id in ['map06','map07']:
        run=Path(cfg['dev'])/map_id/method
        rc=json.loads((run/'contract.json').read_text());physical=Path(rc['evaluation'])
        schema=json.loads((physical/'model_state_schema.json').read_text())
        env=BodyEnvelope(mujoco.MjModel.from_binary_path(str(physical/'terrain.mjb')),device='cpu',dimensions=3)
        pred=np.load(run/'predictions.npz')
        for i,row in enumerate(json.loads((run/'cases.json').read_text())):
            if cfg['supported_only'] and not row['events_found']:continue
            z=np.load(row['prefix']);truth,_=states(z['qpos'][::4],z['qvel'][::4],schema)
            alo,ahi=support(env,truth);pose=motion(pred,i,True);lo,hi=support(env,pose)
            order=np.argsort(pose[:,0],kind='stable');x=pose[order,0]
            inside=(truth[:,0]>=x[0])&(truth[:,0]<=x[-1]);tx=truth[inside,0]
            aligned=[]
            for val in [lo,hi]:
                flat=val[order].reshape(len(order),-1)
                aligned.append(np.stack([np.interp(tx,x,flat[:,j]) for j in range(flat.shape[1])],1).reshape(-1,*lo.shape[1:]))
            miss=np.maximum(np.maximum(aligned[0]-alo[inside],ahi[inside]-aligned[1]),0)
            center=np.stack([np.interp(tx,x,pose[order,j]) for j in range(3)],1)
            delta=truth[inside,:3]-center
            shape_miss=np.maximum(np.maximum(aligned[0]+delta[:,None,:]-alo[inside],ahi[inside]-aligned[1]-delta[:,None,:]),0)
            region=np.clip(np.searchsorted(edges,tx,side='right')-1,0,len(edges)-2)
            for r in range(len(edges)-1):
                ids=np.flatnonzero(region==r)
                if not len(ids):continue
                for axis in range(3):
                    local,geom=np.unravel_index(miss[ids,:,axis].argmax(),miss[ids,:,axis].shape)
                    sample=int(ids[local]);value=float(miss[sample,geom,axis])
                    if value>peak[r,axis]:
                        peak[r,axis]=value
                        origin[r,axis]=dict(method=method,region=r,axis='xyz'[axis],map=map_id,root=row['root'],route=row['route'],speed=row['speed'],actual_x_m=float(tx[sample]),actual_time_s=float(z['trace'][::4,0][inside][sample]),geometry_index=int(geom),support_miss_m=value,root_signed_error_m=float(delta[sample,axis]),after_root_translation_m=float(shape_miss[sample,geom,axis]),predicted_goal_x_m=float(pred['departure'][i,int(pred['departure_steps'][i]),0]),nonmonotonic_steps=int((np.diff(pose[:,0])<0).sum()))
    sources.extend(origin.values())
    np.testing.assert_allclose(np.maximum(peak,0)+.02,np.array(cal['axis_margin_m'][method]),atol=1e-6)
    run=Path(cfg['evaluation'])/method;rc=json.loads((run/'contract.json').read_text());base=Path(rc['oracle'])/'map48'
    model=mujoco.MjModel.from_binary_path(str(base/'base/terrain.mjb'));env=BodyEnvelope(model,device='cpu',dimensions=3)
    schema=json.loads((base/'base/model_state_schema.json').read_text());world=json.loads((base/'world.json').read_text())
    boxes=np.array([[r[0],r[2],0,r[1],r[3],world['terrain'].get('obstacle_height_m',.5)] for r in world['terrain']['forbidden_rectangles']])
    bounds=world['bounds_xy_m'];margin=np.array(cal['axis_margin_m'][method]);pred=np.load(run/'predictions.npz')
    for i,row in enumerate(json.loads((run/'cases.json').read_text())):
        prefix=np.load(row['prefix']);qpos=prefix['qpos'];qvel=prefix['qvel']
        if row['suffix']:
            tail=np.load(row['suffix']);keep=prefix['trace'][:,0]<tail['trace'][0,0]-1e-6
            qpos=np.r_[qpos[keep],tail['qpos']];qvel=np.r_[qvel[keep],tail['qvel']]
        truth,_=states(qpos,qvel,schema)
        for trajectory,pose in [('predicted',motion(pred,i)),('actual',truth)]:
            lo,hi=support(env,pose);regions=np.clip(np.searchsorted(edges,pose[:,0],side='right')-1,0,len(edges)-2)
            for mode,scale in [('raw',[0,0,0]),('own',[1,1,1]),('x_only',[1,0,0]),('y_only',[0,1,0]),('z_only',[0,0,1])]:
                expansion=margin[regions,None,:]*np.array(scale);l=lo-expansion;h=hi+expansion
                obstacle=np.maximum(boxes[None,None,:,:3]-h[:,:,None,:],l[:,:,None,:]-boxes[None,None,:,3:]).max(-1)
                road=np.stack((l[:,:,0]-bounds[0],bounds[1]-h[:,:,0],l[:,:,1]-bounds[2],bounds[3]-h[:,:,1]),-1)
                step,geom,obj=np.unravel_index(obstacle.argmin(),obstacle.shape)
                cases.append(dict(method=method,root=row['root'],route=row['route'],actual_success=row['actual_success'],trajectory=trajectory,margin=mode,obstacle_clearance_m=float(obstacle.min()),world_clearance_m=float(road.min()),obstacle_index=int(obj),geometry_index=int(geom),root_x_at_min_m=float(pose[step,0]),root_y_at_min_m=float(pose[step,1])))
for name,values in [('sources',sources),('cases',cases)]:
    (a.output/(name+'.json')).write_text(json.dumps(values,indent=2)+'\n')
summary=[]
for method in ['C','I']:
    for trajectory in ['predicted','actual']:
        for mode in ['raw','own','x_only','y_only','z_only']:
            selected=[r for r in cases if r['method']==method and r['trajectory']==trajectory and r['margin']==mode and r['actual_success']]
            summary.append(dict(method=method,trajectory=trajectory,margin=mode,actual_successes=len(selected),geometry_clear=sum(min(r['obstacle_clearance_m'],r['world_clearance_m'])>=0 for r in selected)))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(a.output/'timing.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-start,physical_episodes=0,training_updates=0))+'\n')
print(json.dumps(summary))
