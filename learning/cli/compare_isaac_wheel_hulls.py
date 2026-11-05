"""Compare declared wheel OBJ approximations and saved matched H75 trajectories."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
p=argparse.ArgumentParser();p.add_argument('--labels',nargs='+');p.add_argument('--assets',type=Path,nargs='+',required=True);p.add_argument('--runs',type=Path,nargs='*',default=[]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);meshes=[]
for asset in a.assets:
 lines=(asset/'ellipsoid_0.1_0.1_0.025.obj').read_text().splitlines();v=np.array([[float(x) for x in l.split()[1:]] for l in lines if l.startswith('v ')]);f=np.array([[int(x)-1 for x in l.split()[1:]] for l in lines if l.startswith('f ')]);meshes.append((v,f))
fig=plt.figure(figsize=(5*len(meshes),8))
for i,(v,f) in enumerate(meshes):
 ax=fig.add_subplot(2,len(meshes),i+1,projection='3d');ax.add_collection3d(Poly3DCollection(v[f]*1000,facecolors='#56B4E9',edgecolors='#333333',linewidths=.35,alpha=.45));ax.set(xlim=(-105,105),ylim=(-105,105),zlim=(-30,30),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',title=f'{len(v)} vertices, {len(f)} triangles');ax.set_box_aspect((1,1,.3));ax.view_init(25,35)
 ax=fig.add_subplot(2,len(meshes),len(meshes)+i+1);theta=np.linspace(0,2*np.pi,500);ax.plot(100*np.cos(theta),100*np.sin(theta),color='gray',linestyle=':',label='Analytic rim');rim=v[np.abs(v[:,2])<1e-8,:2];rim=rim[np.argsort(np.arctan2(rim[:,1],rim[:,0]))];rim=np.vstack((rim,rim[0]));ax.plot(*(rim*1000).T,'o-',markersize=3,label='Mesh rim');ax.set_aspect('equal',adjustable='box');ax.set(xlim=(-105,105),ylim=(-105,105),xlabel='x (mm)',ylabel='y (mm)');ax.legend(fontsize=8)
fig.suptitle('Source collision mesh comparison — radius 100 mm, half-thickness 25 mm\nSource OBJ surfaces, not a rendering of internal GPU-cooked shapes');fig.tight_layout(rect=(0,0,1,.93));fig.savefig(a.output/'wheel_geometry.png',dpi=160);plt.close(fig)
if a.runs:
 data=[np.load(run/'trajectory.npz') for run in a.runs];contracts=[json.loads((run/'contract.json').read_text()) for run in a.runs];route=np.load(Path(contracts[0]['bundle'])/'route.npz');labels=a.labels or [f"{len(v)} vertices" for v,_ in meshes];colors=['#D55E00','#0072B2','#009E73'];styles=['--','-','-.'];fig,axes=plt.subplots(2,3,figsize=(15,9));metrics=[]
 for lane,ax in enumerate(axes.ravel()):
  series=[d['lower_trace'][:,:,lane].reshape(-1,11) for d in data];counts=[int(x[:,6].sum()) for x in series];common=min(counts);maxprogress=max(float(d['progress'][n-1,lane]) for d,n in zip(data,counts));ref=route['s']<=maxprogress+.3;ax.plot(*route['xy'][ref].T,':',color='gray',label='Original reference')
  for i,(trace,count) in enumerate(zip(series,counts)):
   t=trace[:count];ax.plot(t[:,0],t[:,1],styles[i%3],color=colors[i%3],label=labels[i]);ax.scatter(*t[-1,:2],marker='x' if data[i]['failure'][:,lane].any() else 'o',color=colors[i%3]);metrics.append(dict(alpha=[0,.25,.5,.75,1,None][lane],method='H75' if lane<5 else 'Zero',variant=labels[i],vertices=len(meshes[i][0]),failed=bool(data[i]['failure'][:,lane].any()),end_s=(count-1)*.005+float(t[-1,7])*.0002,progress_m=float(data[i]['progress'][count-1,lane]),common_window_s=common*.005,common_speed_rmse=float(np.sqrt(np.mean((t[:common,2]-2.3)**2))),common_cross_rmse=float(np.sqrt(np.mean(t[:common,3]**2)))))
  ax.set(title=f'H75 alpha={[0,.25,.5,.75,1][lane]}' if lane<5 else 'Zero: LOCAL300 + ECBC/ESO',xlabel='World x (m)',ylabel='World y (m)');ax.axis('equal');ax.legend(fontsize=8)
 fig.suptitle('Same original H75, bend bump, seed 1 — GPU wheel-collision comparison\nCross: failure endpoint; circle: 6 s observation end (not task success)');fig.tight_layout(rect=(0,0,1,.94));fig.savefig(a.output/'h75_xy_comparison.png',dpi=170);plt.close(fig);(a.output/'metrics.json').write_text(json.dumps(metrics,indent=2))
(a.output/'manifest.json').write_text(json.dumps(dict(labels=a.labels,assets=[str(x.resolve()) for x in a.assets],runs=[str(x.resolve()) for x in a.runs],role='DEV one seed, geometry approximation comparison, no smoothing',geometry='actual source OBJ vertices/faces; equal dimensional axes',trajectory='mask inactive lower steps; retain endpoints; common-window RMSE per alpha across variants'),indent=2))
