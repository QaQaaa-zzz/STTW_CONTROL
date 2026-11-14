"""Check the kinematic bound on saved predictions, using forward kinematics only."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.playground_arx import robot_radius_bound,geometry_error_bound

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);a=p.parse_args()
root=a.collection/'model';manifest=json.loads((a.collection/'manifest.json').read_text());examples=np.load(root/'prediction_examples.npz');bank=np.load(a.collection/'routes.npz')
model=mujoco.MjModel.from_binary_path(str(Path(manifest['entries'][0]['output'])/'terrain.mjb'));data=mujoco.MjData(model)
geoms=np.flatnonzero((model.geom_bodyid>0)&((model.geom_contype!=0)|(model.geom_conaffinity!=0)))
directions=np.array([[x,y,z] for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if (x,y,z)!=(0,0,0)],float);directions/=np.linalg.norm(directions,axis=1)[:,None]
addresses=[model.joint(name).qposadr[0] for name in ['steering_joint','frontwheel_joint','rearwheel_joint','hip_joint','knee_joint']]
radius,joint_radii=robot_radius_bound(model)
def support(pose):
    output=[]
    for q in pose:
        data.qpos[:7]=q[:7];data.qpos[addresses]=q[7:];mujoco.mj_kinematics(model,data)
        matrix=data.geom_xmat[geoms].reshape(-1,3,3);local=np.einsum('di,gij->gdj',directions,matrix)
        size=model.geom_size[geoms];kind=model.geom_type[geoms]
        box=np.sum(np.abs(local)*size[:,None,:],axis=-1)
        ellipsoid=np.sqrt(np.sum((local*size[:,None,:])**2,axis=-1))
        cylinder=size[:,None,0]*np.linalg.norm(local[:,:,:2],axis=-1)+size[:,None,1]*np.abs(local[:,:,2])
        extent=np.where((kind==6)[:,None],box,np.where((kind==4)[:,None],ellipsoid,cylinder))
        output.append(data.geom_xpos[geoms]@directions.T+extent)
    return np.asarray(output)
rows=[]
for entry in manifest['entries']:
    fig,axes=plt.subplots(4,2,figsize=(12,7),layout='constrained')
    for route,ax in enumerate(axes.flat):
        key=f'{entry["name"]}_r{route}';pred=examples[key+'_predicted'];actual=examples[key+'_actual']
        gap=np.max(np.abs(support(pred)-support(actual)),axis=(1,2));bound=geometry_error_bound(pred,actual,radius,joint_radii)
        rows.append(dict(group=entry['name'],route=route,root=0,window_s=1.,frames=len(pred),max_directional_support_gap_m=float(gap.max()),max_conservative_bound_m=float(bound.max()),max_bound_violation_m=float(np.maximum(gap-bound,0).max())))
        ref=bank['xy'][route];mask=(ref[:,0]>=min(pred[:,0].min(),actual[:,0].min())-.1)&(ref[:,0]<=max(pred[:,0].max(),actual[:,0].max())+.1)
        ax.plot(*ref[mask].T,'k:',label='Fixed reference');ax.plot(*actual[:,:2].T,color='#0072B2',label='Actual');ax.plot(*pred[:,:2].T,'--',color='#D55E00',label='ARX free rollout')
        ax.set(title=f'Candidate {route}, root 0, 1 s',xlabel='World x (m)',ylabel='World y (m)',aspect='equal');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7);fig.suptitle(entry['name']+' | illustrative last complete 1 s window; all-window errors in calibration.json')
    fig.savefig(root/(entry['name']+'_prediction_xy.png'),dpi=160);plt.close(fig)
report=dict(method='MuJoCo kinematics, seven robot collision primitives, 26 directions. A sampled support gap is not exact surface distance. This audits the upper bound on illustrative root-0 windows, not independent validation.',cases=rows,maximum_violation_m=max(r['max_bound_violation_m'] for r in rows))
(root/'geometry_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
