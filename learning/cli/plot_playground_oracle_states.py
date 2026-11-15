"""Plot saved Oracle branch states and conservative collision-geometry XY AABBs."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mujoco
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--map',type=int,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
folder=a.root/f'map{a.map:02d}'
model=mujoco.MjModel.from_binary_path(str(folder/'base/terrain.mjb'))
data=mujoco.MjData(model)
robot=[g for g in range(model.ngeom) if model.geom_bodyid[g]>0 and (model.geom_contype[g] or model.geom_conaffinity[g])]
obstacles=[g for g in range(model.ngeom) if model.geom(g).name.startswith('planning_obstacle_')]
records=[];sources=[]
colors=['#0072B2','#D55E00'];styles=['-','--']
for lane,root in enumerate([1001,1002]):
 fig,axes=plt.subplots(2,1,figsize=(11,6),layout='constrained')
 state,axs=plt.subplots(3,2,figsize=(11,8),sharex=True,layout='constrained')
 for k,up in enumerate([1,2]):
  prefix=folder/f'approach{up}/evaluation/rb_run3_route0_h75_v{lane*2}/trajectory.npz'
  suffix=folder/f'departure{up}_0/evaluation/rb_run3_route0_h75_v{lane*2}/trajectory.npz'
  before=np.load(prefix);after=np.load(suffix);start=float(after['trace'][0,0]);mask=before['trace'][:,0]<start-1e-6
  trace=np.concatenate([before['trace'][mask],after['trace']]);qpos=np.concatenate([before['qpos'][mask],after['qpos']]);qvel=np.concatenate([before['qvel'][mask],after['qvel']])
  rows=json.loads((suffix.parent.parent/'status.json').read_text())['results'];row=next(r for r in rows if r['case_dir']==suffix.parent.name)
  label=f'Approach {up}, departure 0: {row["termination"]}'
  ax=axes[k];ax.plot(trace[:,1],trace[:,2],styles[k],color=colors[k],label=label)
  ax.plot(after['reference_xy'][:,0],after['reference_xy'][:,1],':',color='#555555',label='Original reference')
  indices=[int(np.argmin(abs(trace[:,0]-t))) for t in [row['handoff_s'],row['handoff_s']+.4,row['handoff_s']+.8]]
  indices=sorted(set(indices+[len(trace)-1]))
  for sample in indices:
   data.qpos[:]=qpos[sample];mujoco.mj_kinematics(model,data)
   for g in robot:
    rot=data.geom_xmat[g].reshape(3,3);center=data.geom_xpos[g]+rot@model.geom_aabb[g,:3];extent=abs(rot)@model.geom_aabb[g,3:]
    ax.add_patch(Rectangle(center[:2]-extent[:2],2*extent[0],2*extent[1],fill=False,edgecolor=colors[k],alpha=.4,lw=.7))
   ax.annotate(f'{trace[sample,0]:.3f}s',trace[sample,1:3],xytext=(0,14),textcoords='offset points',fontsize=8)
   records.append(dict(root=root,upstream=up,time_s=float(trace[sample,0]),x=float(trace[sample,1]),y=float(trace[sample,2]),roll_deg=float(np.rad2deg(trace[sample,4])),yaw_deg=float(np.rad2deg(trace[sample,6])),vx=float(trace[sample,7]),source=str(suffix)))
  for g in obstacles:
   rot=data.geom_xmat[g].reshape(3,3);center=data.geom_xpos[g]+rot@model.geom_aabb[g,:3];extent=abs(rot)@model.geom_aabb[g,3:]
   ax.add_patch(Rectangle(center[:2]-extent[:2],2*extent[0],2*extent[1],color='#777777',alpha=.5))
  ax.scatter(trace[-1,1],trace[-1,2],marker='X',color=colors[k]);ax.axvspan(10.3,10.6,color='#56B4E9',alpha=.15)
  ax.set(xlim=(10,14),ylim=(-1.5,1.1),xlabel='World x (m); travel toward +x',ylabel='World y (m)');ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=8,loc='upper left')
  schema=json.loads((suffix.parent.parent/'model_state_schema.json').read_text())
  values=[trace[:,2],np.rad2deg(trace[:,4]),np.rad2deg(trace[:,6]),trace[:,7],np.rad2deg(qpos[:,schema['steer_qpos']]),np.rad2deg(qvel[:,5])]
  labels=['World y (m)','Roll (deg)','Yaw (deg)','World vx (m/s)','Steering angle (deg)','Body wz (deg/s)']
  select=trace[:,0]>=4.7
  for axis,value,ylabel in zip(axs.flat,values,labels):
   axis.plot(trace[select,0],value[select],styles[k],color=colors[k],label=f'Approach {up}');axis.axvline(row['first_far_contact_s'],color=colors[k],ls=':',alpha=.6);axis.axvline(row['handoff_s'],color=colors[k],ls='-.',alpha=.6);axis.set_ylabel(ylabel);axis.grid(alpha=.2)
   axis.scatter(trace[-1,0],value[-1],marker='X',color=colors[k],s=18)
  sources.append(dict(root=root,upstream=up,prefix=str(prefix),suffix=str(suffix),handoff_s=row['handoff_s'],first_far_contact_s=row['first_far_contact_s']))
 fig.suptitle(f'Map {a.map}, root {root}: actual XY and collision-geometry AABBs\nConservative projected boxes, not mesh footprints or a collision certificate')
 fig.savefig(a.output/f'occupancy_root{root}.png',dpi=180);plt.close(fig)
 for ax in axs[-1]:ax.set_xlabel('Original simulation time (s)')
 axs[0,0].legend();state.suptitle(f'Map {a.map}, root {root}: measured states, identical departure request\nDotted: first far contact; dash-dot: STTW handoff; X: real terminal state')
 state.savefig(a.output/f'states_root{root}.png',dpi=180);plt.close(state)
with (a.output/'state_samples.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
(a.output/'state_figure_sources.json').write_text(json.dumps(dict(sources=sources,transformations='No smoothing; exact recorded prefix before suffix start, then actual suffix; angles converted rad to deg. XY boxes use transformed local geom_aabb; full terminal states retained.',role='Known-map DEV diagnostic; no uncertainty or generalization claim'),indent=2)+'\n')
