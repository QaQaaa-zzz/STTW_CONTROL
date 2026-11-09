"""Add declared offroad terrain and single/reverse bends before the RB platform."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--base',type=Path,default=Path('learning/configs/isaac_rb_ground_jump_ground.json'))
p.add_argument('--platform-offset-x',type=float,default=0.)
p.add_argument('--turn-end-x',type=float,default=-2.)
p.add_argument('--seconds',type=float,default=18.)
p.add_argument('--route-end-x',type=float,default=80.,help='Straight continuation endpoint beyond the task goal and observation window')
p.add_argument('--start-y',type=float,default=-7.)
p.add_argument('--bump-start-y',type=float,default=-6.2)
p.add_argument('--bump-length',type=float,default=1.6)
p.add_argument('--bump-height',type=float,default=.02)
p.add_argument('--terrain-profile',choices=['rolling','ridge','ramp','cross_slope','gravel'],default='rolling')
p.add_argument('--terrain-seed',type=int,default=20261108)
p.add_argument('--bend',choices=['right','left','s','reverse_s'],default='right')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
if not 0<=a.bump_height<=.15 or a.bump_length<=0:raise ValueError('Declare terrain height within [0,.15]m and positive length')
if a.route_end_x<=max(a.turn_end_x,10.6+a.platform_offset_x):raise ValueError('Route must continue beyond the bend and final goal')
def terrain_height(phase,lateral):
 # Longitudinal phase is measured along the initial straight, before bend mirroring.
 taper=np.clip((1.2-np.abs(lateral))/.2,0,1)
 if a.terrain_profile=='rolling':height=np.where((phase>0)&(phase<1),a.bump_height*np.sin(2*np.pi*phase)**2,0.)
 elif a.terrain_profile=='ridge':height=np.where((phase>0)&(phase<1),a.bump_height*np.sin(np.pi*phase)**2,0.)
 elif a.terrain_profile=='ramp':height=a.bump_height*np.clip(np.minimum(phase/.25,(1-phase)/.25),0,1)
 elif a.terrain_profile=='cross_slope':
  height=a.bump_height*np.clip((lateral+.6)/1.2,0,1)*np.clip(np.minimum(phase/.2,(1-phase)/.2),0,1)
 else:
  rng=np.random.default_rng(a.terrain_seed);height=np.zeros_like(phase)
  # Rounded, fixed stones; density30/m² across the central1.6m strip.
  count=round(a.bump_length*1.6*30)
  for along,across,radius,scale in zip(rng.uniform(.1,.9,count),rng.uniform(-.8,.8,count),rng.uniform(.055,.08,count),rng.uniform(.6,1.,count)):
   rho=np.sqrt(((phase-along)*a.bump_length)**2+(lateral-across)**2)/radius
   height=np.maximum(height,np.where(rho<1,a.bump_height*scale*np.cos(np.pi*np.minimum(rho,1)/2)**2,0.))
 return height*taper

c=json.loads(a.base.read_text());source=Path(c['scene_terrain'])
mesh=np.load(source/'terrain.npz');vertices=mesh['vertices'];triangles=mesh['triangles']
if c.get('terrain_base_plane'):vertices=vertices.copy();vertices[:,2]=np.maximum(vertices[:,2],0.)
vertices=vertices.copy();vertices[:,0]+=a.platform_offset_x
# Existing platform and its approach are retained. Only the southern approach strip is added.
spacing=.01 if a.terrain_profile=='gravel' else .05
xmin=min(-5,a.turn_end_x-3);xmax=25. if a.terrain_profile=='rolling' else a.turn_end_x-.8
xmin=xmin if a.terrain_profile=='rolling' else a.turn_end_x-3.2
x=np.linspace(xmin,xmax,round((xmax-xmin)/spacing)+1);y=np.linspace(-10,-4,round(6/spacing)+1)
xx,yy=np.meshgrid(x,y,indexing='ij')
phase=(yy-a.bump_start_y)/a.bump_length
z=terrain_height(phase,xx-(a.turn_end_x-2))
v=np.stack((xx,yy,z),-1).reshape(-1,3).astype(np.float32)
i,j=np.meshgrid(np.arange(len(x)-1),np.arange(len(y)-1),indexing='ij');q=(i*len(y)+j).ravel()
t=np.concatenate((np.stack((q,q+len(y),q+1),-1),np.stack((q+1,q+len(y),q+len(y)+1),-1)))+len(vertices)
terrain=a.output/'terrain';terrain.mkdir()
identity=json.loads((source/'identity.json').read_text());identity.update(source=str(source),platform_front_x=3.6+a.platform_offset_x,platform_back_x=7.6+a.platform_offset_x,added_terrain=dict(bounds=[xmin,25,-10,-4],height_m=a.bump_height,undulation_y_m=[a.bump_start_y,a.bump_start_y+a.bump_length],width_m=2.4,flat_takeoff=f'Turn ends x={a.turn_end_x},y=0; {3.6+a.platform_offset_x-a.turn_end_x}m flat to edge'))
# Northbound straight -> clockwise R2 quarter-circle -> eastbound platform approach.
a1=np.linspace(0,-2-a.start_y,round((-2-a.start_y)/.02)+1);ang=np.linspace(np.pi,np.pi/2,159)[1:];a3=np.linspace(0,a.route_end_x-a.turn_end_x,round((a.route_end_x-a.turn_end_x)/.02)+1)[1:]
xy=np.concatenate((np.stack((np.full_like(a1,a.turn_end_x-2),a.start_y+a1),-1),np.stack((a.turn_end_x+2*np.cos(ang),-2+2*np.sin(ang)),-1),np.stack((a.turn_end_x+a3,np.zeros_like(a3)),-1)))
heading=np.r_[np.full_like(a1,np.pi/2),ang-np.pi/2,np.zeros_like(a3)]
curvature=np.r_[np.zeros_like(a1),np.full_like(ang,-.5),np.zeros_like(a3)]
if a.bend in ('s','reverse_s'):
 # Two opposite R2, 60-degree arcs; same eastbound exit and fixed platform.
 points=[np.array([0.,0.])];heads=[0.];curves=[0.]
 for length,kappa in [(4.,0.),(2*np.pi/3,.5),(2*np.pi/3,-.5)]:
  start=points[-1].copy();h=heads[-1]
  for u in np.linspace(0,length,round(length/.02)+1)[1:]:
   delta=np.array([u*np.cos(h),u*np.sin(h)]) if kappa==0 else np.array([(np.sin(h+kappa*u)-np.sin(h))/kappa,(-np.cos(h+kappa*u)+np.cos(h))/kappa])
   points.append(start+delta);heads.append(h+kappa*u);curves.append(kappa)
 xy=np.array(points);xy+=np.array([a.turn_end_x,0.])-xy[-1]
 prefix_start=xy[0].copy()
 tail=np.stack((a.turn_end_x+a3,np.zeros_like(a3)),-1)
 xy=np.concatenate((xy,tail));heading=np.r_[heads,np.zeros_like(a3)];curvature=np.r_[curves,np.zeros_like(a3)]
 # A separate prefix patch before the platform mesh's x-min; plane carries the turns.
 px=np.linspace(prefix_start[0]-.5,prefix_start[0]+4.,round(4.5/spacing)+1);py=np.linspace(prefix_start[1]-1.2,prefix_start[1]+1.2,round(2.4/spacing)+1)
 xx,yy=np.meshgrid(px,py,indexing='ij');phase=(xx-prefix_start[0]-1.)/a.bump_length
 z=terrain_height(phase,yy-prefix_start[1])
 v=np.stack((xx,yy,z),-1).reshape(-1,3).astype(np.float32)
 i,j=np.meshgrid(np.arange(len(px)-1),np.arange(len(py)-1),indexing='ij');q=(i*len(py)+j).ravel()
 t=np.concatenate((np.stack((q,q+len(py),q+1),-1),np.stack((q+1,q+len(py),q+len(py)+1),-1)))+len(vertices)
 identity['added_terrain']=dict(bounds=[float(px[0]),float(px[-1]),float(py[0]),float(py[-1])],height_m=a.bump_height,width_m=2.4,profile='two sin-squared bumps along initial straight',bump_start_world_x=float(prefix_start[0]+1),bump_length_m=a.bump_length)
if a.bend in ('left','reverse_s'):
 xy[:,1]*=-1;heading*=-1;curvature*=-1
 vertices=vertices.copy();vertices[:,1]*=-1;v[:,1]*=-1
 triangles=triangles[:,[0,2,1]];t=t[:,[0,2,1]]
patch_bounds=[float(v[:,0].min()),float(v[:,0].max()),float(v[:,1].min()),float(v[:,1].max())]
vertices=np.concatenate((vertices,v));triangles=np.concatenate((triangles,t))
np.savez(terrain/'terrain.npz',vertices=vertices,triangles=triangles)
identity.update(sha256=hashlib.sha256((terrain/'terrain.npz').read_bytes()).hexdigest(),bend=a.bend,bump_height_m=a.bump_height,
                patch_mirrored_y=a.bend in ('left','reverse_s'),route_scope='new geometry; frozen vehicle and controllers unchanged')
identity['added_terrain'].update(height_m=a.bump_height,actual_peak_m=float(v[:,2].max()),bounds=patch_bounds,profile=a.terrain_profile,grid_spacing_m=spacing,seed=a.terrain_seed,length_m=a.bump_length,gravel_density_m2=30 if a.terrain_profile=='gravel' else None,gravel_radius_m=[.055,.08] if a.terrain_profile=='gravel' else None)
(terrain/'identity.json').write_text(json.dumps(identity,indent=2))
s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
np.savez(a.output/'route.npz',s=s,xy=xy,heading=heading,curvature=curvature)
c.update(output=str(a.output/'evaluation'),scene_terrain=str(terrain),ground_route=str(a.output/'route.npz'),start_x=a.turn_end_x-2,start_y=a.start_y,start_heading_rad=float(np.pi/2),ground_after_landing_only=False,seconds=a.seconds,maximum_physics_lane_steps=round(a.seconds/c['physics_dt']),role='Nominal DEV full rolling terrain, right turn, flat approach, platform jump and resumed path tracking; same frozen controllers; no training')
c['platform_front_x']=3.6+a.platform_offset_x;c['platform_back_x']=7.6+a.platform_offset_x;c['handoff_x']=2.+a.platform_offset_x
c['completion']=dict(corridor_half_width_m=1.,waypoints=[[a.turn_end_x-2,a.bump_start_y],[a.turn_end_x-2,a.bump_start_y+a.bump_length],[a.turn_end_x-2,-2],[a.turn_end_x,0],[2+a.platform_offset_x,0]],waypoint_radius_m=1.,minimum_post_handoff_s=2.,minimum_post_handoff_forward_m=2.)
if a.bend!='right':
 c.update(start_x=float(xy[0,0]),start_y=float(xy[0,1]),start_heading_rad=float(heading[0]))
 c['completion']['waypoints']=[xy[np.argmin(abs(s-distance))].tolist() for distance in (1.,3.,4.)]+[[a.turn_end_x,0.],[c['handoff_x'],0.]]
c['completion']['final_goal_xy']=[c['platform_back_x']+3.,0.]
c['role']=f'Engineering integration: {a.bend} bend, {a.bump_height}m {a.terrain_profile} terrain, jump and continuation; not a success-rate study'
(a.output/'config.json').write_text(json.dumps(c,indent=2)+'\n')
print(a.output/'config.json')
