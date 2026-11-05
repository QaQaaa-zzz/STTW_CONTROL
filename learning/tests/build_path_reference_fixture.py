"""Original JAX geometry oracle on fixed straight/left/right routes."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from sttw_control.geometric_path import build_path,project,at
from sttw_control.path_reference import pursuit,publish
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
c=json.load(open('runs/h75_torch_parity/config.json'));(a.output/'config.json').write_text(json.dumps(c,indent=2))
rng=np.random.default_rng(20261105);n=256
for name in ('straight','left90_R2','right90_R2'):
    path=build_path(name,[.2,-.3,.4],c)
    route={k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')};np.savez(a.output/(name+'_route.npz'),**route)
    previous=rng.uniform(0,float(path.s[-1]),n).astype('float32');distance=rng.uniform(0,.04,n).astype('float32')
    point,heading,_=jax.vmap(lambda s:at(path,s))(j.asarray(previous))
    pose=np.column_stack((np.asarray(point)+rng.normal(0,.15,(n,2)),np.asarray(heading)+rng.uniform(-1.7,1.7,n))).astype('float32')
    speed=rng.uniform(-.2,4,n).astype('float32');v_user=np.full(n,2.3,dtype='float32');last=rng.uniform([1,-.3],[3,.3],(n,2)).astype('float32')
    projection=jax.jit(jax.vmap(lambda xy,s,d:project(path,xy,s,d,c)))(j.asarray(pose[:,:2]),j.asarray(previous),j.asarray(distance))
    follower=jax.jit(jax.vmap(lambda s,p,v,u:pursuit(path,s,p,v,u,c)))(projection.progress,j.asarray(pose),j.asarray(speed),j.asarray(v_user))
    issued,rates=jax.vmap(lambda p,t:publish(p,t,c))(j.asarray(last),follower.command)
    result=dict(pose=pose,previous=previous,distance=distance,speed=speed,v_user=v_user,last=last,issued=np.asarray(issued),rates=np.asarray(rates))
    result.update({'projection_'+k:np.asarray(getattr(projection,k)) for k in ('progress','unclamped','point','cross_track','heading')})
    result.update({'follower_'+k:np.asarray(getattr(follower,k)) for k in ('command','preview_body','preview_world','lookahead','kappa','invalid','numerical_fault')})
    np.savez_compressed(a.output/(name+'_oracle.npz'),**result)
print('saved',a.output)
