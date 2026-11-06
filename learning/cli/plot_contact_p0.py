"""Actual/reference XY overview and full-observed-window P0 task metrics."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--terrain',required=True);a=p.parse_args()
run=a.root/a.terrain;lanes=json.loads((a.root/'lanes.json').read_text());c=json.loads((run/'contract.json').read_text())
archive=np.load(run/'trajectory.npz');pose=archive['pose'];alive=archive['alive'];trace=archive['lower_trace'].reshape(len(pose),len(lanes),11);measurement=archive['measurement']
route=np.load(Path(c['bundle'])/'route.npz')['xy'];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500'];styles=['-','--','-.',':','-']
fig,axes=plt.subplots(5,2,figsize=(12,19),layout='constrained');rows=[]
for lane,identity in enumerate(lanes):
 m=alive[:,lane].astype(bool);v=trace[m,lane,2];cross=trace[m,lane,3]
 rows.append(dict(**identity,observed_s=float(m.sum()*.005),path_rmse_m=float(np.sqrt(np.mean(cross**2))) if len(cross) else None,speed_rmse_m_s=float(np.sqrt(np.mean((v-identity['speed_m_s'])**2))) if len(v) else None,peak_roll_deg=float(np.rad2deg(abs(measurement[m,lane,0])).max()) if m.any() else None))
 ax=axes[(identity['seed']-101),0 if identity['speed_m_s']==2.3 else 1];ai=lane%5
 if ai==0:ax.plot(*route.T,color='gray',ls=':',label='Original reference')
 # pose contains solved endpoints; active masks terminate observations at actual failure.
 xy=pose[m,lane,:2];ax.plot(*xy.T,color=colors[ai],ls=styles[ai],label=f"alpha={identity['alpha']:g}")
 if len(xy):ax.plot(*xy[-1],marker='x',color=colors[ai])
 ax.set(title=f"{a.terrain}, seed{identity['seed']}, {identity['speed_m_s']} m/s",xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
for group,ax in enumerate(axes.flat):
 xy=pose[:,group*5:group*5+5,:2][alive[:,group*5:group*5+5].astype(bool)]
 ax.set_xlim(float(xy[:,0].min())-.5,float(xy[:,0].max())+.5)
 ax.set_ylim(float(xy[:,1].min())-.5,float(xy[:,1].max())+.5)
axes[0,0].legend(fontsize=8)
fig.suptitle('Frozen H75 + Local300 | P0 natural trajectories | DEV\nCrosses: actual observation endpoints; all five alphas shown per condition')
fig.savefig(run/'xy_p0.png',dpi=140);plt.close(fig)
(run/'task_metrics.json').write_text(json.dumps(dict(role='full observed window per lane, failures retained; not arrival success statistics',lanes=rows),indent=2)+'\n')
