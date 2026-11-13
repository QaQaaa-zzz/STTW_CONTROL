"""Freeze turning tasks on the unchanged Isaac chapter-three terrain."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from sttw_control.geometric_path import build_path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
cfg=json.loads(a.base.read_text());source=json.loads(a.source.read_text());cfg['ditch_terrain']=source['ditch_terrain']
h=cfg['ground_handoff'];h['nominal_steer_slew_rad_s']=source['ground_handoff']['nominal_steer_slew_rad_s']
network=json.loads((Path(h['bundle'])/'config.json').read_text());network['path'].update(straight_prefix_m=.5,straight_tail_m=1.3,goal_section_offset_after_turn_m=.6,turn_radius_m=1.8)
cases=[];routes=[]
for terrain,x in [('stones',16.5),('ridge20mm',2.5),('ridge80mm',11.5)]:
 for shape in ['straight','left90','right90','s_bend']:
  local=copy.deepcopy(network);pose=np.array([x,0.,0.]);path=build_path('right90_R2' if shape=='right90' else 'straight' if shape=='straight' else 'left90_R2',pose,local)
  route={key:np.asarray(getattr(path,key)) for key in ('s','xy','heading','curvature')};end=float(path.turn_end)
  if shape=='straight':end=3.;mask=route['s']<=end+1.3;route={k:v[mask] for k,v in route.items()}
  if shape=='s_bend':
   local['path'].update(turn_angle_rad=np.pi/4,straight_tail_m=0.)
   first=build_path('left90_R2',pose,local);local['path'].update(straight_prefix_m=0.,straight_tail_m=1.3)
   second=build_path('right90_R2',np.r_[np.asarray(first.xy[-1]),float(first.heading[-1])],local)
   route={key:np.concatenate((np.asarray(getattr(first,key)),np.asarray(getattr(second,key))[1:]+(float(first.s[-1]) if key=='s' else 0)),axis=0) for key in ('s','xy','heading','curvature')}
   end=float(first.s[-1])+float(second.turn_end)
  goal=end+.6
  cases.append(dict(name=f'{terrain}_{shape}',terrain=terrain,shape=shape,start_xy=[x,0.],start_yaw=0.,turn_end_s=end,goal_s=goal,speed_m_s=2.))
  routes.append(route)
# A common grid permits one GPU batch; extrapolate only the original straight tail.
ss=np.arange(0,max(r['s'][-1] for r in routes)+.02001,.02)
arrays={k:[] for k in ('xy','heading','curvature')}
for route in routes:
 xy=np.column_stack([np.interp(ss,route['s'],route['xy'][:,k]) for k in (0,1)])
 beyond=np.maximum(0,ss-route['s'][-1]);heading=route['heading'][-1];xy+=beyond[:,None]*np.array([np.cos(heading),np.sin(heading)])
 arrays['xy'].append(xy);arrays['heading'].append(np.interp(ss,route['s'],route['heading']));arrays['curvature'].append(np.interp(ss,route['s'],route['curvature']))
np.savez(a.output/'routes.npz',s=ss,**{k:np.array(v) for k,v in arrays.items()})
for case,route in zip(cases,routes):
 visible=route['xy'][route['s']<=case['goal_s']+.1]
 assert (abs(visible[:,1])<3.9).all() and not ((visible[:,0]>10.3)&(visible[:,0]<10.6)).any()
cfg['ground_suite']=dict(cases=cases,reference_path=str((a.output/'routes.npz').resolve()),alphas=[0,.25,.5,.75,1],zero_middle=True,seconds=10.,final_hold_s=.2,final_path_m=.3,final_heading_rad=.2,final_speed_m_s=.4,road_half_width_m=.6,ground_lateral_limit_m=3.9)
(a.output/'config.json').write_text(json.dumps(cfg,indent=2))
(a.output/'protocol.json').write_text(json.dumps(dict(role='Nominal DEV capability evaluation; no training, no independent seeds',terrain_source=str(a.source),terrain_unchanged=True,cases=cases,methods='Five H75 alpha values and one zero-middle ECBC+ESO baseline per case',episodes=72,maximum_seconds_each=10,physics_dt=.001,lower_dt=.005,middle_dt=.02,maximum_physics_steps=720000,optimizer_updates=0,pendulum='Locked hip/knee, native wheels and position servos, implicitfast and mu0.8',acceptance='Reach original route goal progress with final speed/path/heading hold, never exceed physical/corridor limits; failures retained',bounds=cfg['ground_suite'],limits='Same chapter map; fresh ground starts, no RB or jumping. Straight/left90/right90/S across stone, 2cm and 8cm ridge regions. Existing negative transfer cases remain.'),indent=2))
print(a.output/'config.json')
