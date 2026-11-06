"""Plot all selected jump policies from saved physical trajectories."""
from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
p=args.run;d=np.load(p/'trajectory.npz');t=d['trace'];v=d['valid'];r=json.loads((p/'results.json').read_text());fig,ax=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
for i,c in enumerate(r['cases']):
 a=t[v[:,i],i];label=c['skill'];color=plt.get_cmap('tab10')(i)
 ax[0].plot(a[:,1],a[:,2],label=label,color=color);ax[0].plot(a[-1,1],a[-1,2],'x',color=color)
 ax[1].plot(a[:,1],a[:,3],label=label,color=color);ax[1].plot(a[-1,1],a[-1,3],'x',color=color)
 ax[2].plot(a[:,0],np.rad2deg(a[:,5]),label=label,color=color)
ax[0].axhline(0,color='gray',ls=':',label='Straight approach reference');ax[0].set(xlabel='World x (m)',ylabel='World y (m)');ax[0].set_aspect('equal',adjustable='datalim');ax[0].legend(fontsize=8)
ax[1].plot([1,3.6,3.6,7.6],[0,0,.16,.16],color='gray',ls=':',label='Platform surface');ax[1].set(xlabel='World x (m)',ylabel='Root height (m)');ax[1].legend(fontsize=8)
ax[2].set(xlabel='Time (s)',ylabel='Roll (deg)');ax[2].legend(fontsize=8)
for a in ax:a.grid(alpha=.25)
fig.suptitle('Common pendulum bicycle | selected frozen checkpoints | nominal paired DEV\nActual trajectories end at physical termination; root height is not wheel clearance')
fig.savefig(p/'xy_jump.png',dpi=150)
