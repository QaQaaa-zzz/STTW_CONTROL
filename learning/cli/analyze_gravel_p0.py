"""Task A/B/C and actual wheel events for the completed paired gravel P0."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sttw_control.gravel_contact_events import raised_contact_and_load,intervals
from sttw_control.contact_static_reference import static_wheel_loads
p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
spec=json.loads(a.config.read_text());index=json.loads((a.evaluation/'index.json').read_text());records=[]
for case in index:
 run=Path(case['run']);c=json.loads((run/'contract.json').read_text());base=json.loads((run/'condition_metrics.json').read_text())
 with np.load(run/'trajectory.npz') as f:tr=f['lower_trace'].reshape(-1,6,11)
 for lane,b in enumerate(base):
  valid=tr[:,lane,6].astype(bool);peak=float(abs(tr[valid,lane,3]).max())
  r=dict(terrain=case['terrain'],speed_m_s=case['user_speed_m_s'],parameter_seed=case['seed'],method=['alpha0','alpha025','alpha05','alpha075','alpha1','Zero'][lane],**b,peak_path_error_m=peak)
  for name,t in spec['tasks'].items():r['task_'+name]=bool(b['goal_crossed'] and b['final_hold'] and not b['failed'] and peak<=t['W_m'] and b['speed_rmse_m_s']<=t['speed_rmse_m_s'])
  records.append(r)
(a.evaluation/'gravel_task_metrics.json').write_text(json.dumps(records,indent=2)+'\n')
with (a.evaluation/'gravel_task_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
# Reuse shared raw physics chunks, preserving the six-lane case packing.
events=[]
for batch in sorted((a.evaluation/'batches').glob('case*')):
 run=batch/'evaluation';c=json.loads((run/'contract.json').read_text());settings=c['settings'];cases=json.loads((batch/'cases.json').read_text());n=settings['num_envs'];raw=run/'physics_contacts';meta=json.loads((raw/'index.json').read_text());dt=meta['physics_dt_s'];static=static_wheel_loads(c);collected=[]
 terrains=[None if case['terrain'] in ('flat','single_6cm') else np.load(Path(case['terrain_path'])/'terrain.npz') for case in cases]
 for chunk in meta['chunks']:
  with np.load(raw/chunk['file']) as f:d={k:f[k] for k in f.files}
  ticks=len(d['physics_step']);contact=np.zeros((ticks,n,2),bool);vertical=np.zeros((ticks,n,2))
  for lane in range(n):
   caseid=lane//6;terrain=terrains[caseid]
   for wheel_id,wheel in enumerate(['frontwheel','rearwheel']):
    if terrain is not None:
     raised,_,load,_=raised_contact_and_load(d,wheel,lane,np.asarray(meta['origins'][lane]),terrain)
    else:
     count=d[wheel+'_normal_count'][:,lane,0];start=d[wheel+'_normal_start'][:,lane,0];width=int(count.max(initial=0));ids=np.clip(start[:,None]+np.arange(width),0,d[wheel+'_normal_force'].shape[1]-1);rr=np.arange(ticks)[:,None];valid=np.arange(width)[None]<count[:,None];force=d[wheel+'_normal_force'][rr,ids,0]*valid
     load=(force*d[wheel+'_normal_direction'][rr,ids,2]).sum(1)
     raised=np.zeros(ticks,bool) if cases[caseid]['terrain']=='flat' else ((force>.01)&(d[wheel+'_normal_point'][rr,ids,2]-meta['origins'][lane][2]>.003)).any(1)
    contact[:,lane,wheel_id]=raised&d['alive_before'][:,lane];vertical[:,lane,wheel_id]=load
  steer=meta['joint_order'].index('steering_joint')
  collected.append(dict(time=d['physics_step']*dt,contact=contact,vertical=vertical,alive=d['alive_before'],saturated=abs(d['motor_torque'][:,:,steer])>=.999))
 x={k:np.concatenate([q[k] for q in collected]) for k in collected[0]}
 for lane in range(n):
  case=cases[lane//6];ct=x['contact'][:,lane];live=x['alive'][:,lane].astype(bool);low=(x['vertical'][:,lane,0]<.2*static['front_load_N'][lane])&live;sat=x['saturated'][:,lane]&live;front=intervals(ct[:,0],x['time'],dt);rear=intervals(ct[:,1],x['time'],dt);lowevents=intervals(low,x['time'],dt);satevents=intervals(sat,x['time'],dt)
  events.append(dict(terrain=case['terrain'],speed_m_s=case['user_speed_m_s'],parameter_seed=case['seed'],method=['alpha0','alpha025','alpha05','alpha075','alpha1','Zero'][lane%6],recorded_live_s=float(live.sum()*dt),front_events=front,rear_events=rear,front_interevent_onsets_s=np.diff([r['on_s'] for r in front]).tolist(),overlap_ratio=float(ct.all(1).sum()/max(1,ct.any(1).sum())),front_low_load_s=float(low.sum()*dt),front_low_load_longest_s=max([v['duration_s'] for v in lowevents],default=0),steering_saturated_s=float(sat.sum()*dt),steering_saturated_longest_s=max([v['duration_s'] for v in satevents],default=0),load_reference='20% exported-reference-pose quasi-static front load using actual frozen mass/COM',front_static_load_N=static['front_load_N'][lane],raw_batch=str(run)))
(a.evaluation/'gravel_events.json').write_text(json.dumps(dict(role='Actual physics events in declared0.8-5s record window, stopped at physical termination; flat has zero raised events by definition',lanes=events),indent=2)+'\n')
summary=[]
for terrain in ['flat','single_6cm','S2','S5']:
 group=[r for r in records if r['terrain']==terrain];summary.append(dict(terrain=terrain,n=len(group),survived=sum(not r['failed'] for r in group),arrived=sum(r['goal_crossed'] for r in group),final_hold=sum(r['final_hold'] for r in group),**{k:sum(r['task_'+k] for r in group) for k in spec['tasks']}))
(a.evaluation/'gravel_screening_summary.json').write_text(json.dumps(dict(role='P0 DEV screening; no automatic P1 acceptance and no new selector training',by_terrain=summary,conditions=len(records),physical_parameter_seeds=spec['seeds']),indent=2)+'\n')
with (a.evaluation/'RESULTS.md').open('a') as f:
 f.write('\n\n## Gravel P0 task screening\n\n'+json.dumps(summary,indent=2)+'\n\nFull per-condition data: gravel_task_metrics.json/CSV and gravel_events.json. Task success requires original goal + final hold + declared peak-path/speed limits; early failure never becomes qualified through a shorter error window. Inspect all XY/paired reward figures linked above before expanding P1.\n')
print(json.dumps(summary))
