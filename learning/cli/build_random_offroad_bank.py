"""Seeded independent terrain banks with flat spawn discs and random features."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(a.seed);assets={}
for i in range(13):
 kind='flat' if i==0 else ['pebbles','ridges','waves'][(i-1)%3];dest=a.output/f'{kind}_{i:02d}';dest.mkdir(exist_ok=True)
 axis=np.linspace(-12,12,241,dtype=np.float32);x,y=np.meshgrid(axis,axis,indexing='ij');radius=np.hypot(x,y);fade=np.clip((radius-2)/.5,0,1)*np.clip((11-radius),0,1);z=np.zeros_like(x)
 parameters={}
 if kind=='pebbles':
  vertices=[];triangles=[]
  count=int(rng.integers(800,1601));height=float(rng.uniform(.005,.02));parameters=dict(count=count,height_max_m=height,radius_range_m=[.055,.08])
  # Independent embedded caps; triangle surfaces, no decorative rocks.
  for k in range(count):
   angle=(k*2*np.pi/480 if k<480 else rng.uniform(0,2*np.pi));distance=(3.0 if k<480 else np.sqrt(rng.uniform(2.6**2,5.5**2)));cx,cy=distance*np.cos(angle),distance*np.sin(angle);r=rng.uniform(.055,.08);h=rng.uniform(.5,1)*height;start=len(vertices)
   vertices.append([cx,cy,h])
   for ring in range(1,9):
    rr=r*ring/8
    for j in range(20):vertices.append([cx+rr*np.cos(j*np.pi/10),cy+rr*np.sin(j*np.pi/10),h*np.sqrt(max(0,1-(rr/r)**2))])
   for j in range(20):triangles.append([start,start+1+j,start+1+(j+1)%20])
   for ring in range(7):
    lo=start+1+ring*20;hi=lo+20
    for j in range(20):
     q=(j+1)%20;triangles.extend([[lo+j,hi+j,hi+q],[lo+j,hi+q,lo+q]])
  vertices=np.asarray(vertices,np.float32);triangles=np.asarray(triangles,np.uint32)
 else:
  if kind=='ridges':
   params=[]
   for k in range(int(rng.integers(2,6))):
    angle=rng.uniform(0,np.pi);center=rng.uniform(-8,8);width=rng.uniform(.6,1.2);height=rng.uniform(.02,.1);u=(x*np.cos(angle)+y*np.sin(angle)-center)/width
    z=np.maximum(z,np.where(abs(u)<.5,height*np.cos(np.pi*u)**2,0));params.append([angle,center,width,height])
   parameters['ridges_angle_center_width_height']=params
  if kind=='waves':
   height=rng.uniform(.01,.06);wavelength=rng.uniform(.6,2);angle=rng.uniform(0,np.pi);z=height*(1+np.sin((x*np.cos(angle)+y*np.sin(angle))*2*np.pi/wavelength))/2;parameters=dict(height_m=height,wavelength_m=wavelength,angle_rad=angle)
  
  if kind!='flat':
   ring=np.where(abs(radius-3)<.4,rng.uniform(.02,.06)*np.cos(np.pi*(radius-3)/.8)**2,0)
   z=np.maximum(z,ring)
  z*=fade;vertices=np.column_stack((x.ravel(),y.ravel(),z.ravel())).astype(np.float32);q=(np.arange(240)[:,None]*241+np.arange(240)).ravel();triangles=np.concatenate([np.column_stack((q,q+241,q+242)),np.column_stack((q,q+242,q+1))]).astype(np.uint32)
 np.savez_compressed(dest/'terrain.npz',vertices=vertices,triangles=triangles)
 identity=dict(terrain_kind=kind,seed=a.seed,index=i,parameters=parameters,spawn_clear_radius_m=2,mandatory_crossing_band_m=[2.6,3.4],crossing_design='continuous annular ridge, or480 overlapping pebble caps at radius3m; no planar bypass from inner disc to outside',terrain_extent_m=[-12,12],sha256=hashlib.sha256((dest/'terrain.npz').read_bytes()).hexdigest(),role='random procedural mesh; base plane supplied by environment; new explicit geometry, not old frozen mesh')
 (dest/'identity.json').write_text(json.dumps(identity,indent=2));assets[f'{kind}_{i:02d}']=str(dest.resolve())
(a.output/'assets.json').write_text(json.dumps(assets,indent=2));print(json.dumps(dict(assets=len(assets),seed=a.seed)))
