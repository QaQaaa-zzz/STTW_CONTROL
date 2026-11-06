"""Offline sampled collision-envelope diagnostics; no change to frozen task labels."""
import argparse,json,xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from sttw_control.polyline_distance import PolylineDistance

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--task-run',type=Path,required=True)
p.add_argument('--half-width',type=float,required=True)
a=p.parse_args()
if a.half_width<=0:p.error('half-width must be positive')
d=np.load(a.task_run/'body_poses.npz');tr=np.load(a.task_run/'trajectory.npz')
urdf=Path(str(d['urdf']));robot=ET.parse(urdf).getroot();links={l.attrib['name']:l for l in robot.findall('link')}
poses=d['poses'];origin=d['env_origins'];route=tr['original_xy']
distance=PolylineDistance(route)

body_errors=[];names=[];geometry=[]
for b,name in enumerate(d['body_names'].tolist()):
    for j,collision in enumerate(links[name].findall('collision')):
        shape=list(collision.find('geometry'))[0];kind=shape.tag
        if kind=='box':
            half=np.fromstring(shape.attrib['size'],sep=' ')/2
            points=np.array(list(product((-1.,1.),repeat=3)))*half
        elif kind=='mesh':
            mesh=trimesh.load(urdf.parent/shape.attrib['filename'],force='mesh',process=False)
            points=np.asarray(mesh.vertices)*np.fromstring(shape.attrib.get('scale','1 1 1'),sep=' ')
        else:raise ValueError(f'Unsupported collision shape {kind}; do not approximate silently')
        transform=collision.find('origin')
        if transform is not None:
            points=Rotation.from_euler('xyz',np.fromstring(transform.attrib.get('rpy','0 0 0'),sep=' ')).apply(points)+np.fromstring(transform.attrib.get('xyz','0 0 0'),sep=' ')
        # Rotate each actual body-local collision vertex, including steer/roll/suspension.
        rotation=Rotation.from_quat(poses[:,:,b,3:7].reshape(-1,4)).as_matrix().reshape(*poses.shape[:2],3,3)
        world=np.einsum('tnij,pj->tnpi',rotation,points)+poses[:,:,b,None,:3]-origin[None,:,None,:]
        body_errors.append(distance(world[...,:2]).max(-1));names.append(f'{name}:{j}')
        geometry.append(dict(body=name,collision=j,kind=kind,vertices=len(points),local_min=points.min(0).tolist(),local_max=points.max(0).tolist()))
errors=np.stack(body_errors,axis=-1);maximum=errors.max(-1);valid=d['valid']
wheel_ids=[i for i,n in enumerate(names) if n.split(':')[0] in ('frontwheel','rearwheel')]
wheel=errors[:,:,wheel_ids].max(-1);center=tr['original_path_error']
np.savez_compressed(a.task_run/'envelope_errors.npz',body_vertex_distance_m=errors,collision_names=np.array(names),vehicle_vertex_distance_m=maximum,wheel_vertex_distance_m=wheel,center_distance_m=center,valid=valid,sample_dt_s=d['sample_dt_s'])
rows=[]
for lane in range(valid.shape[1]):
    mask=valid[:,lane]
    rows.append(dict(lane=lane,valid_samples=int(mask.sum()),center_max_m=float(center[mask,lane].max()),vehicle_sampled_max_m=float(maximum[mask,lane].max()),wheel_sampled_max_m=float(wheel[mask,lane].max()),center_within=bool((center[mask,lane]<=a.half_width).all()),vehicle_samples_within=bool((maximum[mask,lane]<=a.half_width).all()),wheel_samples_within=bool((wheel[mask,lane]<=a.half_width).all())))
result=dict(half_width_m=a.half_width,urdf=str(urdf),geometry=geometry,rows=rows,scope='200 Hz sampled authored URDF collision vertices projected into world XY; boxes use corners. Not contact-patch measurements, exact cooked SDF surfaces, between-sample clearance, or a continuous whole-volume containment guarantee. Existing task success labels are unchanged.')
(a.task_run/'envelope_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(rows))
