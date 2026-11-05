"""Prepare a hash-bound H75 fixed-route bundle, without simulation."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from sttw_control.geometric_path import build_path
p=argparse.ArgumentParser();p.add_argument('--anchor',nargs=3,type=float,default=[0,0,0]);p.add_argument('--actor-bundle',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--route',choices=('straight','left90_R2','right90_R2'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cdata=a.config.read_bytes()
if hashlib.sha256(cdata).hexdigest()!=a.config_sha256:raise ValueError('config identity mismatch')
identity=json.loads((a.actor_bundle/'identity.json').read_text());actor=a.actor_bundle/identity['artifact']
if hashlib.sha256(actor.read_bytes()).hexdigest()!=identity['artifact_sha256']:raise ValueError('portable actor mismatch')
c=json.loads(cdata);path=build_path(a.route,a.anchor,c)
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'config.json').write_bytes(cdata);shutil.copyfile(actor,a.output/'actor.npz')
np.savez(a.output/'route.npz',**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
scales=[.3,1.5,.35,3.,3.,2.,3.,3.,3.,.35,.8,.45,3.,.35,3.,60.,1.5,10.,.3,.3]
(a.output/'identity.json').write_text(json.dumps(dict(source_actor=identity,route_name=a.route,world_anchor=a.anchor,frame_scales=scales,files={name:hashlib.sha256((a.output/name).read_bytes()).hexdigest() for name in ('actor.npz','config.json','route.npz')}),indent=2))
