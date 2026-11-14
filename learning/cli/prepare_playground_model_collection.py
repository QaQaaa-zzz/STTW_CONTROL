"""Freeze a bounded P1 nominal/rough collection with root and map splits."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,default=Path('learning/configs/playground_sttw_rb_sttw.json'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
base=json.loads(a.config.read_text());world=json.loads(Path(base['world_map']).read_text())
s=np.linspace(0,30,1501);x=np.linspace(-5,26,6201);curves=[]
for amplitude,reversal in [(-.2,False),(.2,False),(-.4,False),(.4,False),(-.6,False),(.6,False),(-.35,True),(.35,True)]:
    t=np.clip((x-3.5)/4.5,0,1);y=amplitude*np.sin(np.pi*t)**4
    if reversal:y*=2*np.cos(np.pi*t)
    arc=np.r_[0,np.cumsum(np.hypot(np.diff(x),np.diff(y)))];curves.append(np.stack((np.interp(s,arc,x),np.interp(s,arc,y)),1))
xy=np.asarray(curves);heading=np.unwrap(np.arctan2(np.gradient(xy[:,:,1],s,axis=1),np.gradient(xy[:,:,0],s,axis=1)),axis=1)
np.savez_compressed(a.output/'routes.npz',s=s,xy=xy,heading=heading,curvature=np.gradient(heading,s,axis=1))
entries=[]
for i,(kind,split) in enumerate([('flat','TRAIN'),('flat','DEV_ROOT'),('rough','TRAIN'),('rough','TRAIN'),('rough','DEV_MAP'),('rough','DEV_MAP')]):
    name=f'{kind}_{i:02d}_{split.lower()}';folder=a.output/name;folder.mkdir()
    w=copy.deepcopy(world);w['map_id']='nominal_flat' if kind=='flat' else name;w['terrain']['seed']=2026111400+i
    if kind=='flat':w['terrain'].update(stones_per_m2=0,ridges=[])
    else:
        w['terrain']['ridges'][0]['x_interval_m']=[3.8+.3*(i-2),4.8+.3*(i-2)]
        w['terrain']['ridges'][0]['height_m']=[.01,.02,.015,.025][i-2]
    (folder/'world_map.json').write_text(json.dumps(w,indent=2))
    c=copy.deepcopy(base);c.update(world_map=str(folder/'world_map.json'),contact_substep_telemetry=True,collect_model_state=True,audit_actuator_owner=False)
    c['ground_handoff']['reference_path']=str(a.output/'routes.npz')
    rng=np.random.default_rng(2026111450+i);variants=[]
    for root in range(8):
        v=copy.deepcopy(base['ground_handoff']['handoff_variants'][0]);v.update(start_y_m=float(rng.uniform(-.05,.05)),start_yaw_rad=float(rng.uniform(-.03,.03)),start_roll_rad=float(rng.uniform(-.015,.015)),root_id=f'{name}_root{root}',data_split=split)
        variants.append(v)
    c['ground_handoff']['handoff_variants']=variants
    c['role']=f'P1 {kind} {split}; fixed nominal plant, not randomized policy training; all eight curves of each root share one split'
    (folder/'config.json').write_text(json.dumps(c,indent=2))
    entries.append(dict(name=name,kind=kind,split=split,map_id=w['map_id'],config=str(folder/'config.json'),output=str(folder/'evaluation'),roots=8,curves=8,episode_s=4))
manifest=dict(role='P1 first bounded collection: 48 roots x 8 curves x 4 s; no policy optimization',entries=entries,maximum_20ms_transitions=76800,
    split='Frozen before windows. Flat TRAIN/DEV share the same nominal planar geometry but disjoint roots; DEV_ROOT is not unseen-map evidence. Rough maps grouped TRAIN/DEV_MAP. No independent TEST consumed.',
    state='Full initialized runtime snapshot per batch; 5 ms physics/controller history; 1 ms contact normals; 20 ms model windows. Common route prefix through world x3.5; verify actual initial equality across candidate lanes from saved data.',
    termination='4 s timeout is completed collection slice if no earlier physical/domain/map exit; not task-goal success. Keep failures through actual terminal state.',
    model='First fit nominal ARX only on flat TRAIN. Rough data reserved for out-of-domain diagnostics and later paired residual/black-box models.',
    budget='Six batches only, no automatic retry/extension. Session pause deadline 2026-11-14T08:00:00Z.')
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(a.output/'manifest.json')
