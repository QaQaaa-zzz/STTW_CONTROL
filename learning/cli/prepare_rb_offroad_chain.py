"""Add a rolling approach and right turn to the existing RB platform scene."""
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
p.add_argument('--start-y',type=float,default=-7.)
p.add_argument('--bump-start-y',type=float,default=-6.2)
p.add_argument('--bump-length',type=float,default=1.6)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
c=json.loads(a.base.read_text());source=Path(c['scene_terrain'])
mesh=np.load(source/'terrain.npz');vertices=mesh['vertices'];triangles=mesh['triangles']
if c.get('terrain_base_plane'):vertices=vertices.copy();vertices[:,2]=np.maximum(vertices[:,2],0.)
vertices=vertices.copy();vertices[:,0]+=a.platform_offset_x
# Existing platform and its approach are retained. Only the southern approach strip is added.
xmin=min(-5,a.turn_end_x-3);x=np.linspace(xmin,25,round((25-xmin)/.05)+1);y=np.linspace(-10,-4,121)
xx,yy=np.meshgrid(x,y,indexing='ij')
phase=(yy-a.bump_start_y)/a.bump_length
z=np.where((phase>0)&(phase<1),.02*np.sin(2*np.pi*phase)**2,0.)
z*=np.clip((1.2-np.abs(xx-(a.turn_end_x-2)))/.2,0,1)
v=np.stack((xx,yy,z),-1).reshape(-1,3).astype(np.float32)
i,j=np.meshgrid(np.arange(len(x)-1),np.arange(len(y)-1),indexing='ij');q=(i*len(y)+j).ravel()
t=np.concatenate((np.stack((q,q+len(y),q+1),-1),np.stack((q+1,q+len(y),q+len(y)+1),-1)))+len(vertices)
terrain=a.output/'terrain';terrain.mkdir()
np.savez(terrain/'terrain.npz',vertices=np.concatenate((vertices,v)),triangles=np.concatenate((triangles,t)))
identity=json.loads((source/'identity.json').read_text());identity.update(sha256=hashlib.sha256((terrain/'terrain.npz').read_bytes()).hexdigest(),source=str(source),platform_front_x=3.6+a.platform_offset_x,platform_back_x=7.6+a.platform_offset_x,added_terrain=dict(bounds=[xmin,25,-10,-4],height_m=.02,undulation_y_m=[a.bump_start_y,a.bump_start_y+a.bump_length],width_m=2.4,flat_takeoff=f'Turn ends x={a.turn_end_x},y=0; {3.6+a.platform_offset_x-a.turn_end_x}m flat to edge'))
(terrain/'identity.json').write_text(json.dumps(identity,indent=2))
# Northbound straight -> clockwise R2 quarter-circle -> eastbound platform approach.
a1=np.linspace(0,-2-a.start_y,round((-2-a.start_y)/.02)+1);ang=np.linspace(np.pi,np.pi/2,159)[1:];a3=np.linspace(0,25-a.turn_end_x,round((25-a.turn_end_x)/.02)+1)[1:]
xy=np.concatenate((np.stack((np.full_like(a1,a.turn_end_x-2),a.start_y+a1),-1),np.stack((a.turn_end_x+2*np.cos(ang),-2+2*np.sin(ang)),-1),np.stack((a.turn_end_x+a3,np.zeros_like(a3)),-1)))
heading=np.r_[np.full_like(a1,np.pi/2),ang-np.pi/2,np.zeros_like(a3)]
curvature=np.r_[np.zeros_like(a1),np.full_like(ang,-.5),np.zeros_like(a3)]
s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
np.savez(a.output/'route.npz',s=s,xy=xy,heading=heading,curvature=curvature)
c.update(output=str(a.output/'evaluation'),scene_terrain=str(terrain),ground_route=str(a.output/'route.npz'),start_x=a.turn_end_x-2,start_y=a.start_y,start_heading_rad=float(np.pi/2),ground_after_landing_only=False,seconds=a.seconds,maximum_physics_lane_steps=round(a.seconds/c['physics_dt']),role='Nominal DEV full rolling terrain, right turn, flat approach, platform jump and resumed path tracking; same frozen controllers; no training')
c['platform_front_x']=3.6+a.platform_offset_x;c['platform_back_x']=7.6+a.platform_offset_x;c['handoff_x']=2.+a.platform_offset_x
c['completion']=dict(corridor_half_width_m=1.,waypoints=[[a.turn_end_x-2,a.bump_start_y],[a.turn_end_x-2,a.bump_start_y+a.bump_length],[a.turn_end_x-2,-2],[a.turn_end_x,0],[2+a.platform_offset_x,0]],waypoint_radius_m=1.,minimum_post_handoff_s=2.,minimum_post_handoff_forward_m=2.)
(a.output/'config.json').write_text(json.dumps(c,indent=2)+'\n')
print(a.output/'config.json')
