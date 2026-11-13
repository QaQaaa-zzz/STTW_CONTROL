import json
from pathlib import Path
import mujoco
import numpy as np
import argparse
p=argparse.ArgumentParser(description='Compare compiled Playground heightfield rays against original Isaac triangles')
p.add_argument('folder',type=Path)
folder=p.parse_args().folder
m=mujoco.MjModel.from_binary_path(str(folder/'terrain.mjb'))
d=mujoco.MjData(m);mujoco.mj_forward(m,d)
for i in range(m.ngeom):
 m.geom_group[i]=0 if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_HFIELD else 5
z=np.load(folder/'terrain_source.npz');v=z['vertices'];f=z['triangles']
rng=np.random.default_rng(92115);ids=rng.choice(len(f),512,replace=False)
w=rng.dirichlet(np.ones(3),len(ids));points=(v[f[ids]]*w[:,:,None]).sum(1)
actual=[]
for p in points:
 gid=np.zeros(1,dtype=np.int32)
 distance=mujoco.mj_ray(m,d,np.array([p[0],p[1],2.]),np.array([0.,0.,-1.]),np.array([1,0,0,0,0,0],dtype=np.uint8),1,-1,gid)
 actual.append(2-distance)
error=np.abs(np.array(actual)-points[:,2]);result=dict(samples=len(points),max_surface_error_m=float(error.max()),mean_surface_error_m=float(error.mean()),tolerance_m=1e-6,passed=bool((error<1e-6).all()),method='Barycentric samples on original Isaac triangles versus rays through compiled MuJoCo heightfields; all nonterrain geoms excluded')
(folder/'terrain_audit.json').write_text(json.dumps(result,indent=2));print(result)
assert result['passed']
