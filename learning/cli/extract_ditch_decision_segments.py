"""Extract actual post-publication segments; never restore or simulate a snapshot."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);out=a.output/'evaluation';out.mkdir()
source=a.run/'evaluation';cfg=json.loads((a.run/'config.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text());summary=json.loads((source/'summary.json').read_text());decisions=json.loads((source/'moving_planning.json').read_text());records=sorted([r for r in decisions['records'] if r['status']=='published'],key=lambda r:r['lane'])
d=dict(np.load(source/'trajectories.npz'));lanes=np.array([r['lane'] for r in records]);starts=np.array([r['published_step'] for r in records]);tmax=len(d['time'])-starts.min();pmax=len(d['physics_time'])-starts.min()*8;n=len(records);arrays={}
policy_fields=['root_xyzw','valid','reward','stage','phase','observation','action','end_code','controller_history','road_violation_history','reference_route_id']
physics_fields=['physics_root_xyzw','physics_contact_mode','physics_valid']
for name in policy_fields+physics_fields:
 physical=name in physics_fields;length=pmax if physical else tmax;values=d[name];result=np.empty((length,n,*values.shape[2:]),dtype=values.dtype)
 for j,(lane,start) in enumerate(zip(lanes,starts)):
  portion=values[start*8 if physical else start:,lane];result[:len(portion),j]=portion;result[len(portion):,j]=portion[-1]
  if name in ('valid','physics_valid'):result[len(portion):,j]=False
 arrays[name]=result
arrays.update(time=(np.arange(tmax)+1)*.02,physics_time=(np.arange(pmax)+1)*.0025,initial_root_xyzw=np.asarray([r['published_root'] for r in records],np.float32),initial_observation=np.stack([d['observation'][start,lane] for lane,start in zip(lanes,starts)]),event_names=d['event_names'],event_time=d['event_time'][lanes]-starts[:,None]*.02,event_root_xyzw=d['event_root_xyzw'][lanes],event_road_violation=d['event_road_violation'][lanes],road_violation=d['road_violation'][lanes])
old_event=arrays['event_time']<0;arrays['event_time'][old_event]=np.nan;arrays['event_root_xyzw'][old_event]=np.nan
np.savez_compressed(out/'trajectories.npz',**arrays)
cases=[];metadata=[]
for j,(lane,start,record) in enumerate(zip(lanes,starts,records)):
 row=dict(summary['cases'][lane]);row.update(lane=j,steps=row['steps']-int(start))
 for field in ('entry_s','handoff_s'):
  if row[field] is not None:row[field]-=float(start)*.02
 cases.append(row);meta=dict(protocol['cases'][lane]);meta.update(lane=j,source_lane=int(lane),candidate=protocol['route_names'][record['selected']],published_time_s=float(start)*.02);metadata.append(meta)
cfg['ground_handoff']['reference_route_ids']=[r['selected'] for r in records];cfg['ground_handoff']['ground_only_lanes']=[r['selected']==0 for r in records];cfg['num_envs']=n;cfg['derived_dataset_only']=True;cfg['source_run']=str(a.run);cfg.pop('moving_event_planner',None)
(out/'terrain').mkdir();(out/'terrain/terrain.npz').symlink_to((source/'terrain/terrain.npz').resolve())
(a.output/'config.json').write_text(json.dumps(cfg,indent=2));(a.output/'protocol.json').write_text(json.dumps(dict(roots=protocol['roots'],cases=metadata,candidates=protocol['route_names'],scope='Observed states after actual route publication with uninterrupted physics/controller history; not simulation resets; incomplete counterfactual panel'),indent=2));(out/'summary.json').write_text(json.dumps(dict(cases=cases,source_run=str(a.run)),indent=2));(a.output/'extraction.json').write_text(json.dumps(dict(included_source_lanes=lanes.tolist(),excluded_unpublished=[r['lane'] for r in decisions['records'] if r['status']!='published'],unqueried=decisions['unqueried'],policy_dt_s=.02,physics_dt_s=.0025,source=str(a.run)),indent=2));print('Observed segments',n)
