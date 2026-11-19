"""Map-local signed clearance from recorded poses after handback; no integration."""
import argparse,csv,json
from pathlib import Path
import numpy as np,mujoco
from sttw_control.playground_actual_clearance import update_wall_minimum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

p=argparse.ArgumentParser(description=__doc__)
for key in ('evidence','output'):p.add_argument(key,type=Path)
p.add_argument('--obstacle',default='planning_obstacle_2');a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
rows=list(csv.DictReader((a.evidence/'oracle_matrix.csv').open()));anchors={(r['root'],r['upstream']):r for r in json.loads((a.evidence/'state_figure_sources.json').read_text())['sources']}
records=[];fig,axes=plt.subplot_mosaic('AAA;BCD;EFG',figsize=(12,8),layout='constrained')
for row in rows:
 if not row['suffix_trajectory']:continue
 root,up,down=[int(row[k]) for k in ('root','upstream','downstream')]
 prefix,suffix=map(Path,(row['prefix_trajectory'],row['suffix_trajectory']))
 with np.load(prefix) as z,np.load(suffix) as y:
  keep=z['trace'][:,0]<y['trace'][0,0]
  q,v,tr=[np.concatenate((z[k][keep],y[k])) for k in ('qpos','qvel','trace')]
  ref=y['reference_xy']
 h=anchors[root,up]['handoff_s'];m=mujoco.MjModel.from_binary_path(str(suffix.parents[1]/'terrain.mjb'));md=mujoco.MjData(m)
 wall=m.geom(a.obstacle).id;robot=np.flatnonzero((m.geom_bodyid>0)&((m.geom_contype|m.geom_conaffinity)!=0))
 ox,oy,oz=m.geom_pos[wall];hx,hy,hz=m.geom_size[wall];best=None;xs=[]
 for k in np.flatnonzero((tr[:,0]>=h)&(abs(q[:,0]-ox)<=hx+1.0)):
  md.qpos[:]=q[k];mujoco.mj_forward(m,md)
  rot=md.geom_xmat[robot].reshape(-1,3,3);mid=md.geom_xpos[robot]+(rot@m.geom_aabb[robot,:3,None])[...,0];ex=(abs(rot)@m.geom_aabb[robot,3:,None])[...,0]
  lb=np.linalg.norm(np.maximum(abs(mid-md.geom_xpos[wall])-ex-m.geom_size[wall],0),axis=1)
  best=update_wall_minimum(m,md,robot,wall,rot,lb,best,tr[k,0])
  if k%20==0 and root==1001 and down==0:xs.append((np.min(mid-ex,0),np.max(mid+ex,0)))
 records.append(dict(root=root,approach=up,departure=down,outcome=row['termination'],handoff_s=h,wall=a.obstacle,signed_min_m=None if best is None else best[0],time_s=None if best is None else best[1],robot_geom=None if best is None else best[2]))
 if root==1001 and down==0:
  ax=axes['A'];col={1:'#0072b2',2:'#c05500'}[up];ax.plot(*ref.T,':',color=col,lw=.8);ax.plot(*q[:,:2].T,color=col,label=f'approach {up}');ax.plot(*q[-1,:2],'x',color=col)
  for lo,hi in xs:ax.add_patch(Rectangle(lo[:2],*(hi-lo)[:2],facecolor=col,alpha=.14))
  if up==1:
   for j in range(m.ngeom):
    if m.geom(j).name.startswith('planning_obstacle_'):ax.add_patch(Rectangle(m.geom_pos[j,:2]-m.geom_size[j,:2],* (2*m.geom_size[j,:2]),facecolor='0.35'))
  steer=json.loads((suffix.parents[1]/'model_state_schema.json').read_text())['steer_qpos']
  for id,series,label in zip('BCDEFG',[q[:,1],np.rad2deg(tr[:,6]),np.rad2deg(tr[:,4]),tr[:,7],np.rad2deg(q[:,steer]),np.rad2deg(v[:,5])],['World y (m)','Yaw (deg)','Roll (deg)','World vx (m/s)','Steering (deg)','Body wz (deg/s)']):
   ax=axes[id];sel=tr[:,0]>=4.7;ax.plot(tr[sel,0],series[sel],color=col);ax.axvline(h,color=col,ls='-.',lw=.8);ax.plot(tr[-1,0],series[-1],'x',color=col);ax.set(ylabel=label,xlabel='Simulation time (s)')
axes['A'].set(xlim=(10.5,14),ylim=(-1.45,1),aspect='equal',xlabel='World x (m)',ylabel='World y (m)');axes['A'].legend()
fig.suptitle('Historical short-window motivation only (active jump unverified) | Map 48, root 1001\nDotted: reference; shaded: robot AABB / walls; dash-dot: handback; X: recorded terminal pose')
fig.savefig(a.output/'wall_motion.png',dpi=145)
with (a.output/'map48_local_clearance.csv').open('w') as f:
 w=csv.DictWriter(f,records[0],lineterminator='\n');w.writeheader();w.writerows(records)
