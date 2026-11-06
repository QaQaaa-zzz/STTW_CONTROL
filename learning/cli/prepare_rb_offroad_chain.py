"""Add a rolling approach and right turn to the existing RB platform scene."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--base',type=Path,default=Path('learning/configs/isaac_rb_ground_jump_ground.json'))
p.add_argument('--start-y',type=float,default=-7.)
p.add_argument('--bump-start-y',type=float,default=-6.2)
p.add_argument('--bump-length',type=float,default=1.6)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
c=json.loads(a.base.read_text());source=Path(c['scene_terrain'])
mesh=np.load(source/'terrain.npz');vertices=mesh['vertices'];triangles=mesh['triangles']
if c.get('terrain_base_plane'):vertices=vertices.copy();vertices[:,2]=np.maximum(vertices[:,2],0.)
# Existing platform and its approach are retained. Only the southern approach strip is added.
x=np.linspace(-5,25,601);y=np.linspace(-10,-4,121)
xx,yy=np.meshgrid(x,y,indexing='ij')
phase=(yy-a.bump_start_y)/a.bump_length
z=np.where((phase>0)&(phase<1),.02*np.sin(2*np.pi*phase)**2,0.)
z*=np.clip((1.2-np.abs(xx+4))/.2,0,1)
v=np.stack((xx,yy,z),-1).reshape(-1,3).astype(np.float32)
i,j=np.meshgrid(np.arange(len(x)-1),np.arange(len(y)-1),indexing='ij');q=(i*len(y)+j).ravel()
t=np.concatenate((np.stack((q,q+len(y),q+1),-1),np.stack((q+1,q+len(y),q+len(y)+1),-1)))+len(vertices)
terrain=a.output/'terrain';terrain.mkdir()
np.savez(terrain/'terrain.npz',vertices=np.concatenate((vertices,v)),triangles=np.concatenate((triangles,t)))
identity=json.loads((source/'identity.json').read_text());identity.update(sha256=hashlib.sha256((terrain/'terrain.npz').read_bytes()).hexdigest(),source=str(source),added_terrain=dict(bounds=[-5,25,-10,-4],height_m=.02,undulation_y_m=[a.bump_start_y,a.bump_start_y+a.bump_length],width_m=2.4,flat_takeoff='Original platform approach unchanged; turn ends x=-2,y=0, then 5.6m flat to edge'))
(terrain/'identity.json').write_text(json.dumps(identity,indent=2))
# Northbound straight -> clockwise R2 quarter-circle -> eastbound platform approach.
a1=np.linspace(0,-2-a.start_y,round((-2-a.start_y)/.02)+1);ang=np.linspace(np.pi,np.pi/2,159)[1:];a3=np.linspace(0,27,1351)[1:]
xy=np.concatenate((np.stack((np.full_like(a1,-4),a.start_y+a1),-1),np.stack((-2+2*np.cos(ang),-2+2*np.sin(ang)),-1),np.stack((-2+a3,np.zeros_like(a3)),-1)))
heading=np.r_[np.full_like(a1,np.pi/2),ang-np.pi/2,np.zeros_like(a3)]
curvature=np.r_[np.zeros_like(a1),np.full_like(ang,-.5),np.zeros_like(a3)]
s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
np.savez(a.output/'route.npz',s=s,xy=xy,heading=heading,curvature=curvature)
c.update(output=str(a.output/'evaluation'),scene_terrain=str(terrain),ground_route=str(a.output/'route.npz'),start_x=-4.,start_y=a.start_y,start_heading_rad=float(np.pi/2),ground_after_landing_only=False,seconds=18.,maximum_physics_lane_steps=3600,role='Nominal DEV full rolling terrain, right turn, flat approach, platform jump and resumed path tracking; same frozen controllers; no training')
c['completion']=dict(corridor_half_width_m=1.,waypoints=[[-4,a.bump_start_y],[-4,a.bump_start_y+a.bump_length],[-4,-2],[-2,0],[2,0]],waypoint_radius_m=1.,minimum_post_handoff_s=2.,minimum_post_handoff_forward_m=2.)
(a.output/'config.json').write_text(json.dumps(c,indent=2)+'\n')
print(a.output/'config.json')
