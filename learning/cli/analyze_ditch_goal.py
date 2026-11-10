"""Compare every declared ditch candidate against its frozen goal and fixed route."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
contracts=[json.loads((r/'protocol.json').read_text()) for r in a.runs]
goal=contracts[0]['goal'];width=contracts[0]['road_half_width_m']
if any(c['goal']!=goal or c['road_half_width_m']!=width for c in contracts):raise ValueError('Compared runs require identical goal and corridor contracts')
records=[];plot=[];arrays={}
for base,contract in zip(a.runs,contracts):
 for name in contract['cases']:
  src=base/name;cfg=json.loads((src/'config.json').read_text());summary=json.loads((src/'summary.json').read_text())['cases'][0];lane=src/summary['candidate']['name']
  phys=np.load(lane/'physics_contact.npz');t=phys['time'];r=phys['root_state_wxyz'][:,0];tr=np.load(lane/'trajectory.npz');diag=np.load(lane/'verified_diagnostics.npz')
  reference_path=Path(cfg['ground_handoff']['reference_path']);reference=np.load(reference_path)['xy'];v=np.diff(reference,axis=0)
  delta=r[:,None,:2]-reference[None,:-1];u=np.clip((delta*v).sum(-1)/(v*v).sum(-1),0,1);distance=np.linalg.norm(delta-u[:,:,None]*v,axis=-1).min(1)
  hand=json.loads((src/'handoff.json').read_text()) if (src/'handoff.json').exists() else None
  eligible=(r[:,0]>=goal['x_min_m'])&(r[:,0]<=goal['x_max_m'])&(abs(r[:,1])<=goal['abs_y_max_m'])&(r[:,7]>=goal['min_vx_m_s'])
  eligible &= t>=hand['time_s'] if hand else False
  hit=np.flatnonzero(eligible);arrival=int(hit[0]) if len(hit) else None;end=arrival+1 if arrival is not None else len(t)
  failures=diag['failed'][diag['time']<=t[end-1]].any();corridor=bool((distance[:end]<=width).all());outside=np.flatnonzero(distance>width)
  entry=np.load(src/'approach_entry.npz') if (src/'approach_entry.npz').exists() else None
  features=entry['observation'][0,-4:] if entry is not None else None
  condition=cfg['ground_handoff']['approach'];label=f"v{condition['speed_m_s']:g} alpha{condition['alpha']:g}"
  record=dict(case=name,source=str(src),route=reference_path.stem,condition=condition,entry_yaw_deg=float(np.degrees(np.arctan2(features[1],1+features[2]))) if features is not None else None,entry_vy_m_s=float(features[3]) if features is not None else None,entry_y_m=float(features[0]) if features is not None else None,handoff_s=hand['time_s'] if hand else None,arrival_s=float(t[arrival]) if arrival is not None else None,goal_completed=bool(arrival is not None and corridor and not failures),corridor_until_end_or_arrival=corridor,first_corridor_exit_s=float(t[outside[0]]) if len(outside) else None,max_centerline_distance_m=float(distance.max()),end_code=summary['terminal_code'],duration_s=summary['duration_s'],final_xyz=summary['final_xyz'])
  records.append(record);plot.append((record,label,t,r,tr,reference));key=base.name+'__'+name
  arrays.update({key+'_time':t,key+'_root_wxyz':r,key+'_reference':reference,key+'_distance':distance,key+'_reward_time':tr['time'],key+'_reward':tr['reward'][:,0]})
routes=list(dict.fromkeys(r['route'] for r in records));fig,axes=plt.subplots(len(routes),4,figsize=(18,4*len(routes)),layout='constrained',squeeze=False)
for i,route in enumerate(routes):
 members=[d for d in plot if d[0]['route']==route];ref=members[0][-1];ax=axes[i,0];ax.plot(*ref.T,'k:',label='Fixed reference');ax.add_patch(plt.Rectangle((goal['x_min_m'],-goal['abs_y_max_m']),goal['x_max_m']-goal['x_min_m'],2*goal['abs_y_max_m'],fill=False,edgecolor='green',label='Common goal'));ax.set_aspect('equal',adjustable='box');ax.set(xlim=(1,20),ylim=(-2,2))
 for j,(rec,label,t,r,tr,_) in enumerate(members):
  color=['#0072B2','#D55E00','#009E73'][j%3];style=['-','--','-.'][j%3]
  for panel,x,y in [(0,r[:,0],r[:,1]),(1,t,r[:,7]),(2,tr['time'],tr['reward'][:,0]),(3,tr['time'],np.cumsum(tr['reward'][:,0]))]:
   ax=axes[i,panel];ax.plot(x,y,color=color,ls=style,label=label);ax.plot(x[-1],y[-1],'o' if rec['end_code']==9 else 'x',color=color)
 for j,(title,x,y) in enumerate([('Actual XY','World x (m)','World y (m)'),('Forward velocity','Time (s)','World vx (m/s)'),('Diagnostic reward','Time (s)','Reward / step'),('Cumulative diagnostic reward','Time (s)','Reward')]):
  ax=axes[i,j];ax.set(title=route+' | '+title,xlabel=x,ylabel=y);ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Isaac heading125/H75 | same map, reset, goal and handoff | nominal DEV,0training\nPredeclared goal/corridor evaluated at physics frequency; o=horizon, x=failure')
fig.savefig(a.output/'xy_and_rewards.png',dpi=150);np.savez_compressed(a.output/'plot_data.npz',**arrays);(a.output/'summary.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
