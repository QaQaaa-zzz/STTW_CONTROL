"""Matched recorded-data attribution after predicted versus real RB exits."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
from sttw_control.playground_arx import states

p=argparse.ArgumentParser(description=__doc__)
for name in ['run','baseline','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);records=[];parity=[];event_records=[]
for model in ['seed0_update3475','seed1_update3450']:
 for mapid in ['map06','map07','map48']:
  old=a.baseline/'calibration_eval'/model/'C' if mapid=='map48' else a.baseline/'calibration_input'/model/mapid/'C'
  rows=json.loads((old/'cases.json').read_text());zold=np.load(old/'predictions.npz');keep=[i for i,r in enumerate(rows) if mapid!='map48' or r['upstream']!=0]
  base=a.run/model/mapid/'baseline';base.mkdir(exist_ok=True)
  selected=[rows[i] for i in keep];np.savez_compressed(base/'predictions.npz',**{k:zold[k][keep] for k in zold.files})
  (base/'cases.json').write_text(json.dumps(selected,indent=2)+'\n');shutil.copyfile(old/'contract.json',base/'contract.json')
  (base/'summary.json').write_text(json.dumps({k:sum(bool(r['admitted'])==ad and r['actual_success']==ok for r in selected) for k,ad,ok in [('true_admit',True,True),('false_admit',True,False),('false_reject',False,True),('true_reject',False,False)]},indent=2)+'\n')
  if mapid=='map06':
   new=np.load(a.run/model/mapid/'parity/predictions.npz')
   errors={k:float(np.max(np.abs(new[k]-zold[k]))) for k in zold.files}
   assert max(errors.values())==0,errors
   parity.append(dict(model=model,map=mapid,max_array_difference=max(errors.values()),fields=len(errors)))
  contract=json.loads((old/'contract.json').read_text());physical=Path(contract['evaluation']) if contract['evaluation'] else Path(contract['oracle'])/'map48/base'
  schema=json.loads((physical/'model_state_schema.json').read_text())
  for mode in ['baseline','physical','history','both']:
   path=a.run/model/mapid/mode;rows=json.loads((path/'cases.json').read_text());pred=np.load(path/'predictions.npz')
   for i,row in enumerate(rows):
    prefix=np.load(row['prefix']);q=prefix['qpos'];v=prefix['qvel'];trace=prefix['trace'];idx=int(np.flatnonzero(trace[:,-1]==2)[0]);entry=q[idx,:3].copy()
    if mode=='baseline':
     event_time=float(trace[idx,0]);j=int(round((event_time-row['entry_time_s'])/.02))-1;available=0<=j<125
     selected=pred['handoff_qpos'][i];aligned=pred['jump'][i,j] if available else None
     event_records.append(dict(model=model,map=mapid,root=row['root'],route=row['route'],speed=row['speed'],event_found=bool(row['events_found']),event_dt_s=row['handoff_time_s']-event_time,event_x_error_m=float(selected[0]-entry[0]),event_y_error_m=float(selected[1]-entry[1]),time_grid_available=available,time_grid_x_error_m=float(aligned[0]-entry[0]) if available else None,time_grid_y_error_m=float(aligned[1]-entry[1]) if available else None,time_grid_error_s=row['entry_time_s']+(j+1)*.02-event_time if available else None))
    if row['suffix']:
     tail=np.load(row['suffix']);mask=trace[:,0]<tail['trace'][0,0]-1e-6
     q=np.r_[q[mask],tail['qpos']];v=np.r_[v[mask],tail['qvel']];trace=np.r_[trace[mask],tail['trace']]
    after=trace[:,-1]==2;truth,_=states(q[after],v[after],schema);times=trace[after,0]-trace[after,0][0]
    estimate=pred['departure'][i];order=np.argsort(estimate[:,0],kind='stable');px=estimate[order,0]
    inside=(truth[:,0]>=px[0])&(truth[:,0]<=px[-1]);tx=truth[inside,0]
    aligned=np.stack([np.interp(tx,px,estimate[order,j]) for j in range(3)],1);error=aligned-truth[inside,:3]
    near=(tx>=11)&(tx<=14);valid_time=times<=5.;tt=times[valid_time]
    temporal=np.stack([np.interp(tt,np.arange(251)*.02,estimate[:,j]) for j in range(3)],1)-truth[valid_time,:3]
    records.append(dict(model=model,map=mapid,mode=mode,root=row['root'],route=row['route'],speed=row['speed'],actual_success=row['actual_success'],events_found=bool(row['events_found']),admitted=bool(row['admitted']),handoff_time_error_s=float(row['handoff_time_s']-trace[after,0][0]),initial_xyz_error_m=float(np.linalg.norm(estimate[0,:3]-entry)),initial_y_error_m=float(estimate[0,1]-entry[1]),spatial_y_rmse_m=float(np.sqrt(np.mean(error[:,1]**2))) if len(error) else None,spatial_y_max_m=float(np.abs(error[:,1]).max()) if len(error) else None,near_obstacle_y_max_m=float(np.abs(error[near,1]).max()) if near.any() else None,temporal_xy_rmse_m=float(np.sqrt(np.mean(np.sum(temporal[:,:2]**2,1)))),uncovered_spatial_samples=int((~inside).sum()),nonmonotonic_predicted_steps=int((np.diff(estimate[:,0])<0).sum()),physical_samples=len(truth)))
summary=[]
for model in ['seed0_update3475','seed1_update3450']:
 for maps in [('map06','map07'),('map48',)]:
  for mode in ['baseline','physical','history','both']:
   rows=[r for r in records if r['model']==model and r['map'] in maps and r['mode']==mode];supported=[r for r in rows if r['events_found']];overlap=[r for r in supported if r['spatial_y_rmse_m'] is not None]
   summary.append(dict(model=model,maps=list(maps),mode=mode,cases=len(rows),supported=len(supported),supported_spatial_overlap=len(overlap),true_admit=sum(r['admitted'] and r['actual_success'] for r in rows),false_admit=sum(r['admitted'] and not r['actual_success'] for r in rows),mean_spatial_y_rmse_m=float(np.mean([r['spatial_y_rmse_m'] for r in overlap])),max_spatial_y_m=max(r['spatial_y_max_m'] for r in overlap),max_near_obstacle_y_m=max(r['near_obstacle_y_max_m'] for r in supported if r['near_obstacle_y_max_m'] is not None),mean_temporal_xy_rmse_m=float(np.mean([r['temporal_xy_rmse_m'] for r in supported]))))
with (a.output/'cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
for name,value in [('summary',summary),('parity',parity)]:
 (a.output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
with (a.output/'event_time_diagnostic.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=event_records[0]);w.writeheader();w.writerows(event_records)
event_summary=[]
for model in ['seed0_update3475','seed1_update3450']:
 for maps in [['map06','map07'],['map48']]:
  rows=[r for r in event_records if r['model']==model and r['map'] in maps and r['event_found'] and r['time_grid_available']]
  event_summary.append(dict(model=model,maps=maps,n=len(rows),**{k:float(np.mean([abs(r[k]) for r in rows])) for k in ['event_dt_s','event_x_error_m','event_y_error_m','time_grid_x_error_m','time_grid_y_error_m']}))
(a.output/'event_time_summary.json').write_text(json.dumps(event_summary,indent=2)+'\n')
shutil.copyfile(a.run/'contract.json',a.output/'contract.json')
print(json.dumps(summary))
