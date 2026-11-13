"""Summarize the declared two-terrain native RB chassis ground test."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path)
p.add_argument('--compare-runs',nargs='+',help='Named runs NAME=PATH, overlaid by route; original frozen results are read only')
a=p.parse_args()
if a.compare_runs:
 runs={name:Path(path) for name,path in (value.split('=',1) for value in a.compare_runs)}
 records=[]
 for name,folder in runs.items():
  state=json.loads((folder/'status.json').read_text())
  for r in state['results']:
   z=np.load(folder/f"{r['skill']}_route{r['route']}_{r['continuation']}"/'trajectory.npz')
   records.append((name,r,z['trace'],z['reference_xy']))
 routes=sorted({r['route'] for _,r,_,_ in records})
 fig,axes=plt.subplots(3,len(routes),figsize=(5*len(routes),10),squeeze=False,layout='constrained')
 styles=['-','--','-.',':'];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#000000']
 metrics=[]
 for j,route in enumerate(routes):
  subset=[item for item in records if item[1]['route']==route]
  common_end=min(t[-1,0] for _,_,t,_ in subset)
  for i,(name,r,t,ref) in enumerate(subset):
   speed=t[:,7]*np.cos(t[:,6])+t[:,8]*np.sin(t[:,6])
   common=t[:,0]<=common_end+1e-8
   color=colors[i%len(colors)];style=styles[i%len(styles)]
   axes[0,j].plot(t[:,1],t[:,2],color=color,linestyle=style,label=name)
   axes[0,j].scatter(t[-1,1],t[-1,2],marker='o' if r['goal'] else 'x',color=color)
   axes[1,j].plot(t[:,0],speed,color=color,linestyle=style,label=name)
   axes[2,j].plot(t[:,0],t[:,4],color=color,linestyle=style,label=name)
   metrics.append(dict(run=name,route=route,end=r['termination'],goal=r['goal'],duration_s=r['duration_s'],common_window_s=float(common_end),common_speed_rmse_m_s=float(np.sqrt(np.mean((speed[common]-2)**2))),common_path_rmse_m=float(np.sqrt(np.mean(t[common,11]**2))),max_path_m=float(t[:,11].max()),peak_roll_rad=float(abs(t[:,4]).max())))
  xmin=min(t[:,1].min() for _,_,t,_ in subset)-.3;xmax=max(t[:,1].max() for _,_,t,_ in subset)+.3
  visible_ref=ref[(ref[:,0]>=xmin)&(ref[:,0]<=xmax)]
  yvalues=np.concatenate([t[:,2] for _,_,t,_ in subset]+[visible_ref[:,1]])
  axes[0,j].plot(ref[:,0],ref[:,1],'k:',label='Original reference');axes[0,j].set_xlim(xmin,xmax);axes[0,j].set_ylim(yvalues.min()-.3,yvalues.max()+.3);axes[0,j].set_aspect('equal',adjustable='box')
  axes[0,j].set(title=f'Route {route}',xlabel='X (m)',ylabel='Y (m)')
  axes[1,j].axhline(2,color='black',linewidth=.8);axes[1,j].set(xlabel='Time (s)',ylabel='Body forward speed (m/s)')
  axes[2,j].axhline(.7,color='gray',linewidth=.8);axes[2,j].axhline(-.7,color='gray',linewidth=.8);axes[2,j].set(xlabel='Time (s)',ylabel='Roll (rad)')
  for ax in axes[:,j]:ax.grid(alpha=.2);ax.legend(fontsize=6)
 fig.suptitle('Frozen H75, alpha 0.5, request 2 m/s; nominal DEV; actual termination endpoints')
 a.root.mkdir(parents=True,exist_ok=True);fig.savefig(a.root/'xy_comparison.png',dpi=160)
 (a.root/'comparison_metrics.json').write_text(json.dumps(metrics,indent=2))
 lines=['# Playground ground recovery diagnosis','','![Actual/reference XY and unsmoothed speed/roll](xy_comparison.png)','','All declared runs and failures retained. Each route uses a common actual observation window for paired RMSE; goal and physical failure are separate. No training, no task-reward implementation, no independent TEST claim.','','|Run|Route|End|Duration s|Common window s|Speed RMSE m/s|Path RMSE m|','|---|---:|---|---:|---:|---:|---:|']
 for r in metrics:lines.append(f"|{r['run']}|{r['route']}|{r['end']}|{r['duration_s']:.3f}|{r['common_window_s']:.3f}|{r['common_speed_rmse_m_s']:.4f}|{r['common_path_rmse_m']:.4f}|")
 lines+=['','Sources:']+[f'- {name}: {folder}' for name,folder in runs.items()]
 (a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 raise SystemExit(0)
fig,axes=plt.subplots(2,3,figsize=(15,8));speedfig,saxes=plt.subplots(2,3,figsize=(15,8))
rows=[]
for i,terrain in enumerate(('flat','rough')):
 root=a.root/terrain;state=json.loads((root/'status.json').read_text())
 for r in state['results']:
  z=np.load(root/f"{r['skill']}_route{r['route']}_{r['continuation']}"/'trajectory.npz');t=z['trace'];j=r['route']-60
  ax=axes[i,j];sx=saxes[i,j];mode=r['continuation'];color='tab:blue' if mode=='h75' else 'tab:orange'
  ax.plot(t[:,1],t[:,2],label=mode,color=color);ax.scatter(t[-1,1],t[-1,2],marker='x',color=color)
  if mode=='h75':ax.plot(z['reference_xy'][:,0],z['reference_xy'][:,1],'k:',label='original route')
  actual=t[:,7]*np.cos(t[:,6])+t[:,8]*np.sin(t[:,6])
  sx.plot(t[:,0],actual,color=color,label=mode+' body speed')
  sx.plot(t[:,0],t[:,17]*.07,color=color,linestyle=':',label=mode+' wheel speed')
  sx.plot(t[:,0],t[:,13]*.07,color=color,linestyle='--',label=mode+' delivered ref')
  ax.set(title=f'{terrain}, route {r["route"]}',xlabel='x (m)',ylabel='y (m)');ax.axis('equal');ax.grid(alpha=.2);ax.legend(fontsize=7)
  sx.set(title=f'{terrain}, route {r["route"]}',xlabel='time (s)',ylabel='speed (m/s)');sx.grid(alpha=.2)
  if mode=='h75':sx.axhline(2,color='black',linewidth=.8,label='request 2 m/s');sx.legend(fontsize=6)
  rows.append(dict(terrain=terrain,route=r['route'],method=mode,end=r['termination'],duration_s=r['duration_s'],goal=r['goal'],max_path_error_m=float(t[:,11].max()),speed_mean_m_s=float(actual.mean()),speed_rmse_m_s=float(np.sqrt(((actual-2)**2).mean())),max_abs_roll_rad=float(abs(t[:,4]).max()),max_abs_pitch_rad=float(abs(t[:,5]).max()),failure_axis='roll' if abs(t[-1,4])>.7 else 'pitch' if abs(t[-1,5])>1.309 else 'other',jump_activated=bool((t[:,16]==0).any())))
for ax in list(axes.flat)+list(saxes.flat):
 if not ax.lines:ax.set_visible(False)
fig.tight_layout();fig.savefig(a.root/'xy.png',dpi=160);speedfig.tight_layout();speedfig.savefig(a.root/'speed.png',dpi=160)
(a.root/'metrics.json').write_text(json.dumps(rows,indent=2))
lines=['# RB native chassis: STTW ground-only verification','','![All actual and reference XY](xy.png)','','![Body, wheel, delivered and requested speeds](speed.png)','',f'Nominal DEV integration diagnostic, {len(rows)} episodes; actual coverage is listed below. Requested speed 2 m/s, H75 alpha 0.5. Same nominal initialization; no guarantee of bitwise identical GPU dynamics. No jumping or training. Zero-middle retains ECBC+ESO and the reference filters; neither method includes Local300. Different actual observation lengths end at each real failure; averages are descriptive, not matched-window rankings.','','|Terrain|Route|Method|End|Duration s|Mean speed m/s|Speed RMSE m/s|Max path error m|Peak roll rad|Peak pitch rad|Failure axis|','|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in rows:lines.append(f"|{r['terrain']}|{r['route']}|{r['method']}|{r['end']}|{r['duration_s']:.3f}|{r['speed_mean_m_s']:.3f}|{r['speed_rmse_m_s']:.3f}|{r['max_path_error_m']:.3f}|{r['max_abs_roll_rad']:.3f}|{r['max_abs_pitch_rad']:.3f}|{r['failure_axis']}|")
lines+=['','The old floor/step explicit contact pairs were removed. The trench is at x=40 m, outside this test. Flat has no stones/ridges; rough preserves the declared millimeter stones and ridges. All actions and physical trajectories are in the terrain subdirectories. No reward comparison: this inference-only runner does not implement a task reward.']
(a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
