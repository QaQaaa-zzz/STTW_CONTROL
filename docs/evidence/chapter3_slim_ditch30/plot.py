from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_slim_reproduction');out=Path(__file__).parent
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained');data={}
for name,label in [('physics20','Continuous ground'),('ditch30','Ditch + 1mm cap'),('ditch30_flat','Ditch + flat bank')]:
 d=np.load(base/name/'handoff_20us/trajectory.npz');q=d['qpos'];t=d['time_s'];v=d['qvel'];data[name]=dict(time=t,xy=q[:,:2],z=q[:,2],vx=v[:,0])
 axs[0,0].plot(q[:,0],q[:,1],label=label);axs[0,0].plot(*q[-1,:2],'x')
 axs[0,1].plot(q[:,0],q[:,2],label=label)
 axs[1,0].plot(t,v[:,0],label=label)
 axs[1,1].plot(t,q[:,1],label=label)
 print(name,'final xyz',q[-1,:3].tolist(),'x min/max',q[:,0].min(),q[:,0].max())
axs[0,0].axhline(0,color='k',ls=':',label='original y=0 reference');axs[0,0].set(xlabel='World x (m)',ylabel='World y (m)',title='Actual XY (same entry4)');axs[0,0].set_aspect('equal',adjustable='box')
axs[0,1].set(xlabel='World x (m)',ylabel='Root z (m)',title='Measured root XZ')
for ax in axs[0]:ax.axvspan(3.3,3.6,color='0.8')
axs[1,0].set(xlabel='Time (s)',ylabel='World vx (m/s)',title='Actual longitudinal velocity')
axs[1,1].set(xlabel='Time (s)',ylabel='World y (m)',title='Lateral position')
for ax in axs[1]:ax.axvline(2.36,ls=':',color='k',label='H75 handoff')
for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('heading125 + H75 | MuJoCo 3.6.0 / 20us | single-entry DEV, no training')
fig.savefig(out/'xy.png',dpi=160)
np.savez_compressed(out/'plot_data.npz',**{f'{k}_{n}':a for k,d in data.items() for n,a in d.items()})
