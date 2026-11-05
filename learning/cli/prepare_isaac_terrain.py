"""Prepare an explicit shallow raised terrain mesh; no physics execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--height',type=float,default=.012);p.add_argument('--spacing',type=float,default=.05);a=p.parse_args()
if not 0<=a.height<=.03 or not .02<=a.spacing<=.1:raise ValueError('shallow terrain height/spacing out of declared range')
a.output.mkdir(parents=True,exist_ok=False)
# Two smooth longitudinal ridges around curve entry, tapered across Y. Grid
# includes the whole declared five-second route; no overlapping ground plane.
axis=np.linspace(-2,12,int(np.ceil(14/a.spacing))+1,dtype=np.float32);x,y=np.meshgrid(axis,axis,indexing='ij')
z=np.zeros_like(x)
for start,end in [(2.8,3.4),(3.7,4.3)]:
 u=(x-start)/(end-start);profile=np.where((u>0)&(u<1),np.sin(np.pi*u)**2,0)
 taper=np.sin(np.pi*np.clip((y+1)/4,0,1))**2
 z+=a.height*profile*taper
vertices=np.column_stack((x.ravel(),y.ravel(),z.ravel())).astype(np.float32)
n=len(axis);i,j=np.meshgrid(np.arange(n-1),np.arange(n-1),indexing='ij');q=(i*n+j).ravel()
triangles=np.concatenate((np.column_stack((q,q+n,q+n+1)),np.column_stack((q,q+n+1,q+1)))).astype(np.uint32)
normal=np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]])
assert np.all(normal[:,2]>0) and np.all(np.isfinite(vertices))
assert np.max(z)<=a.height+1e-7 and np.min(z)==0
np.savez_compressed(a.output/'terrain.npz',vertices=vertices,triangles=triangles)
meta=dict(schema='sttw_shallow_two_ridge_mesh_v1',height_m=a.height,actual_max_height_m=float(z.max()),grid_spacing_m=float(axis[1]-axis[0]),xy_bounds_m=[-2,12],ridge_x_intervals_m=[[2.8,3.4],[3.7,4.3]],lateral_taper_y_bounds_m=[-1,3],surface='height*sin(pi*u)^2*sin(pi*(y+1)/4)^2 within support; zero elsewhere',vertices=len(vertices),triangles=len(triangles),friction='source model sliding friction, not changed',overlapping_plane=False,sha256=hashlib.sha256((a.output/'terrain.npz').read_bytes()).hexdigest())
(a.output/'identity.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
