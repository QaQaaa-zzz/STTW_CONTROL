"""Full-chain DEV spatial support calibration and matched own/shared/cross margins."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch,mujoco
from sttw_control.playground_arx import states
from sttw_control.playground_prediction import BodyEnvelope

p=argparse.ArgumentParser(description=__doc__)
for name in ['dev','evaluation','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--supported-only',action='store_true');p.add_argument('--external-calibration',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter()
edges=np.array([-5.,8.3,10.3,10.6,12.6,25.]);margins={};counts={};records=[];fit_cases=[]
contract=dict(supported_only=a.supported_only,fit_domain='Only candidates with predicted handoff events; excluded cases remain rejected in evaluation' if a.supported_only else 'Historical all trajectories including unsupported suffix diagnostics',role='DEV calibration only; map48 is known DEV diagnostic, not TEST',maps=['map06','map07'],methods=['C','I'],rule='Same existing spatial support rule: per-region per-axis maximum lower/upper AABB support miss plus0.02m. Stable x-sort/interpolation; actual samples outside predicted x range explicitly uncovered, not clipped into calibration.',edges=edges.tolist(),coordinate='World x spatial alignment, not same-time error or phase-aligned duration; nonmonotonic prediction counts retained',scope='Per collision geometry3D axis supports; empirical observed bound only; no probabilistic or continuous-time safety claim',budget='CPU geometry/analysis, 0 training and 0 physical episodes, stop25min',dev=str(a.dev),evaluation=str(a.evaluation),external_calibration=str(a.external_calibration) if a.external_calibration else None)
(a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')

def motion(z,i,full=False):
    return np.concatenate((z['approach'][i,:int(z['entry_steps'][i])+1],z['jump'][i,:int(z['jump_steps'][i])],z['departure'][i,:len(z['departure'][i]) if full else int(z['departure_steps'][i])+1]))

def supports(envelope,pose):
    lo,hi=envelope(torch.as_tensor(pose,dtype=torch.float32))
    return lo.numpy(),hi.numpy()

for method in ['C','I']:
    maximum=np.zeros((len(edges)-1,3));count=np.zeros(len(edges)-1,int)
    for map_id in ['map06','map07']:
        run=a.dev/map_id/method;cfg=json.loads((run/'contract.json').read_text());physical=Path(cfg['evaluation']);schema=json.loads((physical/'model_state_schema.json').read_text());envelope=BodyEnvelope(mujoco.MjModel.from_binary_path(str(physical/'terrain.mjb')),device='cpu',dimensions=3)
        rows=json.loads((run/'cases.json').read_text());pred=np.load(run/'predictions.npz')
        for i,row in enumerate(rows):
            included=not a.supported_only or bool(row['events_found'])
            fit_cases.append(dict(method=method,map=map_id,root=row['root'],route=row['route'],speed=row['speed'],included=included,reason='included' if included else 'missing_predicted_handoff'))
            if not included:continue
            z=np.load(row['prefix']);truth,_=states(z['qpos'][::4],z['qvel'][::4],schema);alo,ahi=supports(envelope,truth);pose=motion(pred,i,full=True);lo,hi=supports(envelope,pose)
            order=np.argsort(pose[:,0],kind='stable');x=pose[order,0];inside=(truth[:,0]>=x[0])&(truth[:,0]<=x[-1]);actual_x=truth[inside,0]
            interpolated=[]
            for values in [lo,hi]:
                flat=values[order].reshape(len(order),-1);interpolated.append(np.stack([np.interp(actual_x,x,flat[:,j]) for j in range(flat.shape[1])],1).reshape(-1,*lo.shape[1:]))
            miss=np.maximum(np.maximum(interpolated[0]-alo[inside],ahi[inside]-interpolated[1]),0)
            regions=np.clip(np.searchsorted(edges,actual_x,side='right')-1,0,len(edges)-2)
            for region in range(len(edges)-1):
                selected=regions==region
                if selected.any():maximum[region]=np.maximum(maximum[region],miss[selected].max((0,1)));count[region]+=int(selected.sum())
            phase=z['trace'][::4,-1][inside]
            for stage,mask in [('approach',phase==0),('jump',np.isin(phase,[1,3,4])),('departure',phase==2)]:
                records.append(dict(method=method,map=map_id,root=row['root'],route=row['route'],speed=row['speed'],stage=stage,samples=int(mask.sum()),uncovered_samples=int((~inside).sum()),nonmonotonic_predicted_steps=int((np.diff(pose[:,0])<0).sum()),max_x_m=float(miss[mask,:,0].max()) if mask.any() else None,max_y_m=float(miss[mask,:,1].max()) if mask.any() else None,max_z_m=float(miss[mask,:,2].max()) if mask.any() else None))
    if not count.any():raise ValueError(f'{method}: no supported calibration samples')
    maximum[count==0]=maximum.max(0);margins[method]=maximum+.02;counts[method]=count
(a.output/'fit_cases.json').write_text(json.dumps(fit_cases,indent=2)+'\n')
margins['shared']=np.maximum(margins['C'],margins['I'])
if a.external_calibration:
    external=json.loads(a.external_calibration.read_text())
    assert external['edges']==edges.tolist()
    for method in ['C','I']:margins['external_'+method]=np.array(external['axis_margin_m'][method])
    margins['shared_all']=np.maximum.reduce([margins[k] for k in ['C','I','external_C','external_I']])
(a.output/'calibration.json').write_text(json.dumps(dict(edges=edges.tolist(),axis_margin_m={k:v.tolist() for k,v in margins.items()},counts={k:v.tolist() for k,v in counts.items()}),indent=2)+'\n')
with (a.output/'support_errors.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=records[0],lineterminator='\n');w.writeheader();w.writerows(records)
results=[];rejections=[]
for method in ['C','I']:
    run=a.evaluation/method;cfg=json.loads((run/'contract.json').read_text());base=Path(cfg['oracle'])/'map48';world=json.loads((base/'world.json').read_text());envelope=BodyEnvelope(mujoco.MjModel.from_binary_path(str(base/'base/terrain.mjb')),device='cpu',dimensions=3);rows=json.loads((run/'cases.json').read_text());pred=np.load(run/'predictions.npz');rect=world['terrain']['forbidden_rectangles'];boxes=np.array([[r[0],r[2],0,r[1],r[3],world['terrain'].get('obstacle_height_m',.5)] for r in rect]);bounds=world['bounds_xy_m']
    gate_fields=['has_goal','posture_valid','departure_domain_valid','events_found','approach_domain_valid']+[k for k in ['handoff_joints_ready','handoff_support_ready','handoff_body_ready','handoff_support_geometry'] if k in rows[0]]
    for i,row in enumerate(rows):
        pose=motion(pred,i);lo,hi=supports(envelope,pose);regions=np.clip(np.searchsorted(edges,pose[:,0],side='right')-1,0,len(edges)-2)
        # Event/domain/goal gates stay fixed; only geometry expansion changes.
        nongeometric=all(bool(row[k]) for k in gate_fields) and row['goal_time_s']<=10
        modes=[('raw',np.zeros_like(margins[method])),('own',margins[method]),('shared',margins['shared']),('cross',margins['I' if method=='C' else 'C'])]
        if a.external_calibration:modes += [('structure_cross',margins['external_'+method]),('shared_all',margins['shared_all'])]
        for mode,margin in modes:
            l=lo-margin[regions,None,:];h=hi+margin[regions,None,:];obstacle=np.maximum(boxes[None,None,:,:3]-h[:,:,None,:],l[:,:,None,:]-boxes[None,None,:,3:]).max(-1)
            road=np.stack((l[:,:,0]-bounds[0],bounds[1]-h[:,:,0],l[:,:,1]-bounds[2],bounds[3]-h[:,:,1]),-1)
            obstacle_min=float(obstacle.min());road_min=float(road.min());clear=min(obstacle_min,road_min)
            admitted=bool(nongeometric and clear>=0)
            results.append(dict(method=method,margin=mode,root=row['root'],route=row['route'],actual_success=row['actual_success'],admitted=admitted,clearance_m=clear,goal_time_s=row['goal_time_s'],nongeometric_valid=nongeometric))
            if clear<0:
                if obstacle_min<=road_min:
                    step,geom,obj=np.unravel_index(obstacle.argmin(),obstacle.shape);reason='obstacle';constraint=int(obj)
                else:
                    step,geom,obj=np.unravel_index(road.argmin(),road.shape);reason='world_bound';constraint=int(obj)
                stage='approach' if step<int(pred['entry_steps'][i])+1 else 'jump' if step<int(pred['entry_steps'][i])+1+int(pred['jump_steps'][i]) else 'departure'
                rejections.append(dict(method=method,margin=mode,root=row['root'],route=row['route'],stage=stage,prediction_index=int(step),geometry_index=int(geom),constraint=reason,constraint_index=constraint,clearance_m=clear,margin_xyz_m=margin[regions[step]].tolist()))
summary=[]
for method in ['C','I']:
    for mode in (['raw','own','shared','cross','structure_cross','shared_all'] if a.external_calibration else ['raw','own','shared','cross']):
        rows=[r for r in results if r['method']==method and r['margin']==mode]
        summary.append(dict(method=method,margin=mode,**{key:sum(r['admitted']==ad and r['actual_success']==ok for r in rows) for key,ad,ok in [('true_admit',True,True),('false_admit',True,False),('false_reject',False,True),('true_reject',False,False)]}))
for name,value in [('cases',results),('rejections',rejections),('summary',summary)]:
    (a.output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
(a.output/'timing.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-started,physical_episodes=0,training_updates=0))+'\n');print(json.dumps(summary))
